"""Training-only geometric APOS labels and legacy action-intent supervision."""

import math

import torch
from torch.nn import functional as F

from streamnav.models.policy.perception import point_type_logits
from streamnav.training.perception_loss import spatial_cross_entropy


def validate_affordance_label(label, episode_id, frame_id):
    if label["episode_id"] != episode_id or label["frame_id"] != frame_id:
        raise ValueError("Affordance label belongs to a different episode or camera frame")
    if label["apos_valid"] and not 0 <= label["apos"] < 1300:
        raise ValueError("Invalid geometric APOS label")
    confidence = label["apos_confidence"]
    if not math.isfinite(confidence) or not 0 <= confidence <= 1:
        raise ValueError("Invalid geometric APOS confidence")
    if label.get("opos_valid"):
        targets = label.get("opos_targets") or [label["opos"]]
        if not 0 <= label["opos"] <= 1296 or any(
            type(t) is not int or not 0 <= t <= 1296 for t in targets
        ):
            raise ValueError("Invalid instance OPOS label")
        if label["opos"] not in targets or (0 in targets and targets != [0]):
            raise ValueError("OPOS absence and visible points cannot share a label")
    return label


def geometric_affordance_loss(apos_logits, labels):
    """Supervise actual visible waypoints; mask unknown geometry, smooth pixels.

    An oracle turn can have a visible waypoint. Its label is that pixel, not a
    turn sentinel. The action remains a separate DAgger target.
    """
    target = torch.tensor(
        [label["apos"] if label and label["apos_valid"] else 0 for label in labels],
        device=apos_logits.device,
    )
    weight = apos_logits.new_tensor(
        [label["apos_confidence"] if label and label["apos_valid"] else 0 for label in labels]
    )
    return spatial_cross_entropy(apos_logits.unsqueeze(0), target.unsqueeze(0)).squeeze(0) * weight


def imitation_weight(record, config):
    weight = config.get("stop_il_weight", 1.0) if record["expert"] == 0 else 1.0
    if record.get("collision_before", False) and record["expert"] != 1:
        weight *= config.get("recovery_il_weight", 1.0)
    return weight


def affordance_intent_loss(apos_logits, expert_actions):
    # Type order: none, point, left, right, stop. Forward supervises total
    # point probability; its pixel location remains supervised by LightNav KL.
    # Looking up/down has no corresponding APOS intent and is ignored.
    mapping = torch.tensor([4, 1, 2, 3, -1, -1], device=expert_actions.device)
    return F.cross_entropy(
        point_type_logits(apos_logits),
        mapping[expert_actions],
        ignore_index=-1,
        reduction="none",
    )
