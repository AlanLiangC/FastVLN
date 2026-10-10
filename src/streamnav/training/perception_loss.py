"""Confidence-weighted spatial losses and globally balanced arrival supervision."""

import math
from collections.abc import Callable

import torch
import torch.nn.functional as F

from streamnav.contracts.perception import GRID_HEIGHT, GRID_WIDTH, POINT_CELLS
from streamnav.training import distributed as parallel


def loss_config(config=None):
    result = {"apos_coef": 0.05, "opos_coef": 0.10, "arrival_coef": 0.10}
    if config:
        if set(config) - set(result):
            raise ValueError("Unknown perception loss options")
        result.update(config)
    if any(not math.isfinite(v) or v < 0 for v in result.values()):
        raise ValueError("Perception coefficients must be finite and nonnegative")
    return result


def spatial_cross_entropy(logits, target):
    """Smooth positional labels over neighboring cells, never over sentinels."""
    log_prob = logits.float().log_softmax(-1)
    exact = F.nll_loss(log_prob.flatten(0, 1), target.flatten(), reduction="none").view_as(target)
    cell = (target - 1).clamp(0, POINT_CELLS - 1)
    row, col = cell // GRID_WIDTH, cell % GRID_WIDTH
    offsets = torch.tensor([(y, x) for y in (-1, 0, 1) for x in (-1, 0, 1)], device=target.device)
    rows, cols = row[..., None] + offsets[:, 0], col[..., None] + offsets[:, 1]
    valid = (rows >= 0) & (rows < GRID_HEIGHT) & (cols >= 0) & (cols < GRID_WIDTH)
    indices = (1 + rows * GRID_WIDTH + cols).clamp(1, POINT_CELLS)
    weights = torch.exp(-offsets.float().square().sum(-1)) * valid
    weights = weights / weights.sum(-1, keepdim=True).clamp_min(1e-8)
    smooth = -(log_prob.gather(-1, indices) * weights).sum(-1)
    return torch.where((target > 0) & (target <= POINT_CELLS), smooth, exact)


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
            loss = F.cross_entropy(
                logits.flatten(0, 1), target.flatten(), reduction="none"
            ).view_as(target)
            groups = [class_group(k) for k in range(3)]
        else:
            loss = spatial_cross_entropy(logits, target)
            groups = [lambda t: (t > 0) & (t <= POINT_CELLS), lambda t: t == 0]
            if name == "apos":
                groups.extend(class_group(k) for k in range(POINT_CELLS + 1, logits.shape[-1]))
        mean = balanced_mean(loss, target, valid, confidence, groups)
        total = total + coefficients[name + "_coef"] * mean
        metrics["perception_" + name + "_loss"] = mean.detach().item()
        metrics["perception_" + name + "_label_fraction"] = valid.float().mean().item()
        predicted = logits.argmax(-1)
        fraction = parallel.average_scalar(valid.float().mean()).clamp_min(1e-8)
        accuracy = (
            parallel.average_scalar(((predicted == target) & valid).float().mean()) / fraction
        )
        metrics["perception_" + name + "_accuracy"] = accuracy.item()
        if name in ("apos", "opos"):
            positive = valid & (target > 0) & (target <= POINT_CELLS)
            metrics["perception_" + name + "_point_label_fraction"] = positive.float().mean().item()
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
