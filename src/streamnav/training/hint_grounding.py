"""Matched image/query supervision for accepting and rejecting point hints.

Only complete instance masks and explicit absence annotations license quality
labels. A point outside a weak box or a sparse point annotation is not a negative.
This auxiliary task never invents a navigation action or arrival label.
"""

import random

import numpy as np
from torch.nn import functional as F

from streamnav.contracts.perception import POINT_CELLS
from streamnav.data.grounding_support import unpack_grid


def sample_grounding_hints(rows):
    packets = []
    for i, row in enumerate(rows):
        if row.get("opos_targets") == [0]:
            packets.append((i, random.randint(1, POINT_CELLS), 0.0))
        elif row.get("opos_support"):
            support = unpack_grid(row["opos_support"])
            if random.random() < 0.5:
                token = random.choice(row["opos_targets"])
                packets.append((i, token, 1.0))
            else:
                # Exclude a one-cell margin around every target instance, so
                # quantization at a silhouette cannot become a hard negative.
                padded = np.pad(support, 1)
                nearby = np.zeros_like(support)
                h, w = support.shape
                for y in range(3):
                    for x in range(3):
                        nearby |= padded[y : y + h, x : x + w]
                candidates = np.flatnonzero(~nearby)
                if candidates.size:
                    packets.append((i, int(random.choice(candidates)) + 1, 0.0))
    return packets


def hint_grounding_loss(policy, features, rows, prefix_tokens, *, anchor_weight, point_radius):
    from streamnav.training.point_supervision import grounding_opos_losses, supervised_fragment

    packets = sample_grounding_hints(rows)
    metrics = dict.fromkeys(
        ("hint_opos_loss", "hint_quality_loss", "hint_fraction", "hint_positive_fraction"), 0.0
    )
    if not packets:
        zero = policy.opos_rows.sum() * 0
        return (zero, zero), metrics
    selected = [rows[i] for i, _, _ in packets]
    hinted = policy.condition_features(
        [features[i] for i, _, _ in packets], [t for _, t, _ in packets]
    )
    state = policy.start_episode(selected[0]["image_id"], selected[0]["instruction"])
    resets = {i: (r["image_id"], r["instruction"]) for i, r in enumerate(selected)}
    step, _, _, _ = supervised_fragment(
        policy, hinted, state, [prefix_tokens[i] for i, _, _ in packets], selected, resets=resets
    )
    if step.hint_quality_logits is None:
        raise ValueError("Counterfactual grounding requires the hint quality head")
    opos, _, _ = grounding_opos_losses(
        step.opos_logits, selected, anchor_weight=anchor_weight, point_radius=point_radius
    )
    weights = opos.new_tensor([r.get("loss_weight", 1.0) for r in selected])
    targets = opos.new_tensor([v for _, _, v in packets])
    quality = F.binary_cross_entropy_with_logits(
        step.hint_quality_logits.float(), targets, reduction="none"
    )
    # Normalize by the original batch: weak/unlicensed rows contribute zero,
    # rather than amplifying one eligible example into a complete batch.
    opos_loss = (opos * weights).sum() / len(rows)
    quality_loss = (quality * weights).sum() / len(rows)
    metrics.update(
        hint_opos_loss=float(opos_loss.detach()),
        hint_quality_loss=float(quality_loss.detach()),
        hint_fraction=len(packets) / len(rows),
        hint_positive_fraction=float(targets.sum()) / len(rows),
    )
    return (opos_loss, quality_loss), metrics
