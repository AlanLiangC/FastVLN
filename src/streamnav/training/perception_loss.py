"""Confidence-weighted spatial losses and globally balanced arrival supervision."""

import math
from collections.abc import Callable
from typing import Any

import torch
import torch.nn.functional as F

from streamnav.contracts.perception import GRID_HEIGHT, GRID_WIDTH, POINT_CELLS
from streamnav.models.policy.perception import (
    decode_point_logits,
    point_type_logits,
    point_type_targets,
)
from streamnav.training import distributed as parallel


def loss_config(config=None):
    result: dict[str, Any] = {
        "apos_coef": 0.05,
        "opos_coef": 0.10,
        "arrival_coef": 0.10,
        "spatial_mode": "flat",
        "location_coef": 1.0,
        "prior_correction": False,
    }
    if config:
        if set(config) - set(result):
            raise ValueError("Unknown perception loss options")
        result.update(config)
    if not isinstance(result["prior_correction"], bool):
        raise ValueError("prior_correction must be boolean")
    if any(
        not math.isfinite(v) or v < 0
        for k, v in result.items()
        if k not in ("spatial_mode", "prior_correction")
    ):
        raise ValueError("Perception coefficients must be finite and nonnegative")
    if result["spatial_mode"] not in ("flat", "hierarchical"):
        raise ValueError("Unknown spatial loss mode")
    return result


def spatial_cross_entropy(logits, target):
    """Smooth positional labels over neighboring cells, never over sentinels."""
    log_prob = logits.float().log_softmax(-1)
    exact = F.nll_loss(log_prob.flatten(0, 1), target.flatten(), reduction="none").view_as(target)
    smooth = spatial_location_nll(log_prob[..., 1 : POINT_CELLS + 1], target)
    return torch.where((target > 0) & (target <= POINT_CELLS), smooth, exact)


def spatial_location_nll(point_log_prob, target):
    """Conditional position loss, with no competition against sentinel classes."""
    cell = (target - 1).clamp(0, POINT_CELLS - 1)
    row, col = cell // GRID_WIDTH, cell % GRID_WIDTH
    offsets = torch.tensor([(y, x) for y in (-1, 0, 1) for x in (-1, 0, 1)], device=target.device)
    rows, cols = row[..., None] + offsets[:, 0], col[..., None] + offsets[:, 1]
    valid = (rows >= 0) & (rows < GRID_HEIGHT) & (cols >= 0) & (cols < GRID_WIDTH)
    indices = (rows * GRID_WIDTH + cols).clamp(0, POINT_CELLS - 1)
    weights = torch.exp(-offsets.float().square().sum(-1)) * valid
    weights = weights / weights.sum(-1, keepdim=True).clamp_min(1e-8)
    return -(point_log_prob.gather(-1, indices) * weights).sum(-1)


def balanced_mean(loss, target, valid, confidence, groups):
    # DDP averages local gradients; divide by global densities, not a per-rank
    # valid count. Empty ranks/classes contribute connected zero gradients.
    total, present = loss.sum() * 0, 0
    for group in groups:
        mask = valid & group(target)
        fraction = parallel.average_scalar(mask.float().mean())
        if fraction.item() > 0:
            total = total + (loss * mask * confidence).mean() / fraction
            present += 1
    return total / max(present, 1)


def masked_mean(values, mask):
    density = parallel.average_scalar(mask.float().mean()).clamp_min(1e-8)
    return (values * mask).mean() / density


def prior_corrected_logits(logits, target, valid):
    """Keep balanced supervision without teaching an artificial uniform prior.

    Balanced CE learns posterior odds divided by empirical class priors. Apply
    that correction inside the training loss so inference keeps natural odds.
    Counts are shared across ranks; absent classes receive no logit shift.
    """
    counts = torch.stack([((target == k) & valid).sum() for k in range(logits.shape[-1])]).float()
    counts = parallel.average_scalar(counts)
    priors = counts / counts.sum().clamp_min(1e-8)
    log_prior = torch.where(counts > 0, priors.clamp_min(1e-8).log(), torch.zeros_like(priors))
    return logits.float() - log_prior


def perception_losses(predictions, targets, config=None):
    coefficients = loss_config(config)
    total = predictions["arrival"].sum() * 0
    metrics = {}
    for name in ("apos", "opos", "arrival"):
        groups: list[Callable[[torch.Tensor], torch.Tensor]]
        logits, labels = predictions[name], targets[name]
        target, valid, confidence = labels["target"], labels["valid"], labels["confidence"]
        if bool((((target < 0) | (target >= logits.shape[-1])) & valid).any()):
            raise ValueError(f"Invalid {name} target")
        if not bool(torch.isfinite(confidence).all()) or bool(
            ((confidence < 0) | (confidence > 1)).any()
        ):
            raise ValueError("Invalid perception confidence")
        target = torch.where(valid, target, 0)
        if name == "arrival":
            training_logits = (
                prior_corrected_logits(logits, target, valid)
                if coefficients["prior_correction"]
                else logits
            )
            loss = F.cross_entropy(
                training_logits.flatten(0, 1), target.flatten(), reduction="none"
            ).view_as(target)
            groups = [class_group(k) for k in range(3)]
        elif coefficients["spatial_mode"] == "hierarchical":
            types = point_type_logits(logits)
            kinds = point_type_targets(target)
            if coefficients["prior_correction"]:
                types = prior_corrected_logits(types, kinds, valid)
            type_loss = F.cross_entropy(
                types.flatten(0, 1), kinds.flatten(), reduction="none"
            ).view_as(target)
            type_mean = balanced_mean(
                type_loss,
                kinds,
                valid,
                confidence,
                [class_group(k) for k in range(types.shape[-1])],
            )
            positive = valid & (target > 0) & (target <= POINT_CELLS)
            location_loss = spatial_location_nll(
                logits[..., 1 : POINT_CELLS + 1].float().log_softmax(-1), target
            )
            location_mean = masked_mean(location_loss * confidence, positive)
            mean = type_mean + coefficients["location_coef"] * location_mean
            metrics["perception_" + name + "_type_loss"] = type_mean.detach().item()
            metrics["perception_" + name + "_location_loss"] = location_mean.detach().item()
        else:
            loss = spatial_cross_entropy(logits, target)
            groups = [lambda t: (t > 0) & (t <= POINT_CELLS), lambda t: t == 0]
            if name == "apos":
                groups.extend(class_group(k) for k in range(POINT_CELLS + 1, logits.shape[-1]))
        if name == "arrival" or coefficients["spatial_mode"] == "flat":
            mean = balanced_mean(loss, target, valid, confidence, groups)
        total = total + coefficients[name + "_coef"] * mean
        metrics["perception_" + name + "_loss"] = mean.detach().item()
        metrics["perception_" + name + "_label_fraction"] = valid.float().mean().item()
        predicted = (
            logits.argmax(-1)
            if name == "arrival"
            else decode_point_logits(
                logits, hierarchical=coefficients["spatial_mode"] == "hierarchical"
            )
        )
        fraction = parallel.average_scalar(valid.float().mean()).clamp_min(1e-8)
        accuracy = (
            parallel.average_scalar(((predicted == target) & valid).float().mean()) / fraction
        )
        metrics["perception_" + name + "_accuracy"] = accuracy.item()
        if name in ("apos", "opos"):
            positive = valid & (target > 0) & (target <= POINT_CELLS)
            metrics["perception_" + name + "_point_label_fraction"] = positive.float().mean().item()
            with torch.no_grad():
                kinds = point_type_targets(target)
                probabilities = point_type_logits(logits.detach()).softmax(-1)
                predicted_kind = probabilities.argmax(-1)
                location = logits[..., 1 : POINT_CELLS + 1].argmax(-1)
                expected = (target - 1).clamp(0, POINT_CELLS - 1)
                dx = (location % GRID_WIDTH - expected % GRID_WIDTH).float()
                dy = (location // GRID_WIDTH - expected // GRID_WIDTH).float()
                statistics = {
                    "type_accuracy": ((predicted_kind == kinds).float(), valid),
                    "point_recall": ((predicted_kind == 1).float(), positive),
                    "point_probability_on_point": (probabilities[..., 1], positive),
                    "point_probability_on_none": (probabilities[..., 1], valid & (target == 0)),
                    "location_error_cells": ((dx.square() + dy.square()).sqrt(), positive),
                    "location_within_one_cell": (
                        ((dx.abs() <= 1) & (dy.abs() <= 1)).float(),
                        positive,
                    ),
                }
                for key, (values, mask) in statistics.items():
                    metrics["perception_" + name + "_" + key] = parallel.average_scalar(
                        masked_mean(values, mask)
                    ).item()
        else:
            ready = valid & (target == 2)
            near = valid & (target == 1)
            metrics["perception_arrival_near_label_fraction"] = near.float().mean().item()
            metrics["perception_arrival_ready_label_fraction"] = ready.float().mean().item()
            density = parallel.average_scalar(ready.float().mean()).clamp_min(1e-8)
            recall = parallel.average_scalar(((predicted == 2) & ready).float().mean()) / density
            metrics["perception_arrival_ready_recall"] = recall.item()
    metrics["perception_loss"] = total.detach().item()
    return total, metrics


def class_group(index):
    def group(target: torch.Tensor) -> torch.Tensor:
        return target == index

    return group
