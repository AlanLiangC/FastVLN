"""Arrival and STOP supervision on actual policy prefixes, normalized per rollout."""

import math

import torch
from torch.nn import functional as F

from streamnav.contracts.perception import APOS_STOP, ARRIVAL_READY
from streamnav.training import distributed as parallel

ARRIVAL_METRICS = (
    "arrival_loss",
    "stop_calibration_loss",
    "apos_stop_calibration_loss",
    "stop_hard_negative_loss",
    "arrival_accuracy",
    "arrival_ready_recall",
    "arrival_label_fraction",
    "arrival_ready_fraction",
    "stop_hard_negative_fraction",
    "stop_probability_on_ready",
    "stop_probability_on_not_ready",
)


def prepare_arrival_supervision(records, device, *, success_distance=0.25, near_distance=0.75):
    """One collective for the full rollout; fragments never rebalance their own classes.

    Mild sqrt balancing helps rare arrival labels without a uniform-prior binary
    STOP objective. Both sides of the stopping boundary get the same emphasis.
    """
    rows = [r for row in records for r in row]
    stats = torch.zeros(5)
    for r in rows:
        label = r.get("perception_label") or {}
        target = label.get("arrival", -1) if label.get("arrival_valid") else -1
        if target not in (-1, 0, 1, 2):
            raise ValueError("Invalid arrival target")
        valid, ready = target >= 0, target == ARRIVAL_READY
        if valid:
            if (r["expert"] == 0) != ready:
                raise ValueError("Teacher STOP and arrival readiness disagree")
            if "distance_before" in r:
                distance = r["distance_before"]
                if not math.isfinite(distance) or (distance < success_distance) != ready:
                    raise ValueError("Arrival label disagrees with success distance")
            stats[target] += 1
        proposed_stop = r.get("conditional_greedy_action", -1) == 0 or int(r["tokens"][2]) == 0
        hard = valid and not ready and proposed_stop
        # No extra positive-only STOP prior: close positives and negatives both count twice.
        calibration = (2.0 if r.get("distance_before", math.inf) < near_distance else 1.0) * valid
        stats[3] += calibration
        stats[4] += hard
        r["arrival_target"] = max(target, 0)
        r["arrival_valid"] = valid
        r["stop_calibration_weight"] = calibration
        r["stop_hard_negative_weight"] = float(hard)
    density = parallel.average_scalar((stats / max(1, len(rows))).to(device)).cpu()
    frequencies = density[:3]
    class_weights = frequencies.clamp_min(1e-8).rsqrt().clamp_max(4)
    class_weights /= (class_weights * frequencies).sum().clamp_min(1e-8)
    for r in rows:
        target, valid = r["arrival_target"], r["arrival_valid"]
        r["arrival_weight"] = float(class_weights[target]) * valid
        r["arrival_valid_weight"] = valid / max(float(frequencies.sum()), 1e-8)
        r["arrival_ready_weight"] = (valid and target == ARRIVAL_READY) / max(
            float(frequencies[ARRIVAL_READY]), 1e-8
        )
        r["arrival_nonready_weight"] = (valid and target != ARRIVAL_READY) / max(
            float(frequencies[:2].sum()), 1e-8
        )
        r["stop_calibration_weight"] /= max(float(density[3]), 1e-8)
        # Cap rare selected mistakes at 50x, so one isolated frame cannot dominate.
        r["stop_hard_negative_weight"] /= max(float(density[4]), 0.02)


def arrival_losses(step, records, cfg):
    zero = step.action_logits.sum(-1) * 0
    metrics = dict.fromkeys(ARRIVAL_METRICS, zero)
    if not any(
        cfg.get(k, 0)
        for k in (
            "arrival_loss_coef",
            "stop_calibration_coef",
            "apos_stop_calibration_coef",
            "stop_hard_negative_coef",
        )
    ):
        return zero, metrics
    if step.arrival_logits is None:
        raise ValueError("Arrival supervision requires the arrival head")
    if any("arrival_weight" not in r for r in records):
        raise ValueError("Prepare arrival supervision over the complete rollout before replay")
    device = step.action_logits.device

    def weights(name):
        return step.action_logits.new_tensor([r.get(name, 0.0) for r in records])

    targets = torch.tensor([r.get("arrival_target", 0) for r in records], device=device)
    ready = targets == ARRIVAL_READY
    odds = step.action_logits[:, 0].float() - step.action_logits[:, 1:].float().logsumexp(-1)
    apos = step.apos_logits.float()
    apos_odds = apos[:, APOS_STOP] - apos[:, :APOS_STOP].logsumexp(-1)
    metrics.update(
        arrival_loss=F.cross_entropy(step.arrival_logits.float(), targets, reduction="none")
        * weights("arrival_weight"),
        stop_calibration_loss=F.binary_cross_entropy_with_logits(
            odds, ready.float(), reduction="none"
        )
        * weights("stop_calibration_weight"),
        apos_stop_calibration_loss=F.binary_cross_entropy_with_logits(
            apos_odds, ready.float(), reduction="none"
        )
        * weights("stop_calibration_weight"),
        stop_hard_negative_loss=F.softplus(odds) * weights("stop_hard_negative_weight"),
        arrival_accuracy=(step.arrival_logits.argmax(-1) == targets).float()
        * weights("arrival_valid_weight"),
        arrival_ready_recall=(step.arrival_logits.argmax(-1) == ARRIVAL_READY).float()
        * weights("arrival_ready_weight"),
        arrival_label_fraction=weights("arrival_valid"),
        arrival_ready_fraction=weights("arrival_valid") * ready,
        stop_hard_negative_fraction=step.action_logits.new_tensor(
            [bool(r.get("stop_hard_negative_weight", 0)) for r in records]
        ),
        stop_probability_on_ready=odds.detach().sigmoid() * weights("arrival_ready_weight"),
        stop_probability_on_not_ready=odds.detach().sigmoid() * weights("arrival_nonready_weight"),
    )
    loss = (
        cfg.get("arrival_loss_coef", 0) * metrics["arrival_loss"]
        + cfg.get("stop_calibration_coef", 0) * metrics["stop_calibration_loss"]
        + cfg.get("apos_stop_calibration_coef", 0) * metrics["apos_stop_calibration_loss"]
        + cfg.get("stop_hard_negative_coef", 0) * metrics["stop_hard_negative_loss"]
    )
    return loss, metrics
