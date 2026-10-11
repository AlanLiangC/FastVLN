"""Class-balanced calibration of the policy STOP probability using teacher labels."""

import torch
import torch.nn.functional as F

from streamnav.training import distributed as parallel
from streamnav.training.perception_loss import (
    balanced_mean,
    class_group,
    masked_mean,
    prior_corrected_logits,
)


def stop_calibration_loss(logits, expert, valid, *, prior_correction=False):
    stop_log_odds = logits[..., 0].float() - logits[..., 1:].float().logsumexp(-1)
    target = (expert == 0).long()
    training_odds = stop_log_odds
    if prior_correction:
        binary_logits = prior_corrected_logits(
            torch.stack((torch.zeros_like(stop_log_odds), stop_log_odds), dim=-1), target, valid
        )
        training_odds = binary_logits[..., 1] - binary_logits[..., 0]
    loss = F.binary_cross_entropy_with_logits(training_odds, target.float(), reduction="none")
    mean = balanced_mean(
        loss, target, valid, torch.ones_like(loss), [class_group(0), class_group(1)]
    )
    probability = stop_log_odds.detach().sigmoid()
    metrics = {"stop_calibration_loss": mean.detach().item()}
    for name, mask in (
        ("positive", valid & (target == 1)),
        ("negative", valid & (target == 0)),
    ):
        metrics["stop_calibration_" + name + "_probability"] = parallel.average_scalar(
            masked_mean(probability, mask)
        ).item()
    return mean, metrics


def stop_hard_negative_loss(logits, expert, valid, executed_actions=None):
    """Focus navigation supervision on unsafe predicted or executed STOPs."""
    odds = logits[..., 0].float() - logits[..., 1:].float().logsumexp(-1)
    proposed_stop = logits.detach().argmax(-1) == 0
    if executed_actions is not None:
        proposed_stop = proposed_stop | (executed_actions == 0)
    hard = valid & (expert != 0) & proposed_stop
    loss = F.softplus(odds)
    mean = masked_mean(loss, hard)
    return mean, {
        "stop_hard_negative_loss": mean.detach().item(),
        "stop_hard_negative_fraction": hard.float().mean().item(),
        "stop_hard_negative_probability": parallel.average_scalar(
            masked_mean(odds.detach().sigmoid(), hard)
        ).item(),
    }
