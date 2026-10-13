"""Separate teacher-forced point CoT from on-policy PPO token replay."""

import torch
from torch.nn import functional as F

from streamnav.data.grounding_support import support_targets


def targets_from_label(label):
    label = label or {}
    return {
        name + "_targets": [int(label[name])] if label.get(name + "_valid", False) else []
        for name in ("apos", "opos")
    }


def marginal_point_loss(logits, targets):
    # Equivalent goal instances are alternatives, not conflicting one-hot labels.
    # Missing annotations yield zero; token 0 is used only for explicit absence.
    losses = []
    for row, valid in zip(logits.float(), targets, strict=True):
        if valid:
            indices = torch.tensor(sorted(set(valid)), device=row.device)
            losses.append(-torch.log_softmax(row, -1).index_select(0, indices).logsumexp(0))
        else:
            losses.append(row.sum() * 0)
    return torch.stack(losses)


def supervised_fragment(policy, features, state, tokens, labels, *, resets=None, experts=None):
    forced = []
    for token, label in zip(tokens, labels, strict=True):
        points = [
            torch.tensor(label[name + "_targets"][0], device=policy.device)
            if label.get(name + "_targets")
            else token[j].to(policy.device)
            for j, name in enumerate(("apos", "opos"))
        ]
        forced.append((*points, token[2].to(policy.device)))
    step = policy.replay_fragment(features, state, forced, resets=resets)
    apos = marginal_point_loss(
        step.apos_logits, [label.get("apos_targets", []) for label in labels]
    )
    opos = marginal_point_loss(
        step.opos_logits, [label.get("opos_targets", []) for label in labels]
    )
    action = (
        F.cross_entropy(step.action_logits, experts, reduction="none")
        if experts is not None
        else torch.zeros_like(apos)
    )
    return step, apos, opos, action


def grounding_opos_losses(logits, rows, *, anchor_weight=1.0, point_radius=0.0):
    if not 0 <= anchor_weight <= 1:
        raise ValueError("Point anchor weight must lie in [0, 1]")
    anchor = marginal_point_loss(logits, [r.get("opos_targets", []) for r in rows])
    support = marginal_point_loss(logits, [support_targets(r, point_radius) for r in rows])
    return anchor_weight * anchor + (1 - anchor_weight) * support, anchor, support


def grounding_loss_options(config):
    return {
        "anchor_weight": config.get("anchor_weight", 1.0),
        "point_radius": config.get("point_radius", 0.0),
    }


def grounding_loss(policy, rows, *, anchor_weight=1.0, point_radius=0.0):
    # Each still image is an independent episode: grounding never writes into
    # a navigation cache or assumes a navigation action / STOP label.
    from streamnav.data.grounding import GroundingSource

    features = GroundingSource.features_many(policy, rows)
    state = policy.start_episode(rows[0]["image_id"], rows[0]["instruction"])
    resets = {i: (r["image_id"], r["instruction"]) for i, r in enumerate(rows)}
    zero = torch.zeros((), device=policy.device, dtype=torch.long)
    with torch.no_grad():
        prefix = policy.replay_fragment(
            features, state, [(zero, zero, zero)] * len(rows), resets=resets
        )
        prefix_tokens = [(a, zero, zero) for a in prefix.apos_logits.argmax(-1)]
    step, apos, opos, _ = supervised_fragment(
        policy, features, state, prefix_tokens, rows, resets=resets
    )
    opos, anchor, support = grounding_opos_losses(
        step.opos_logits, rows, anchor_weight=anchor_weight, point_radius=point_radius
    )
    weights = torch.tensor([r.get("loss_weight", 1.0) for r in rows], device=policy.device)
    active = sum(bool(r.get("apos_targets")) + bool(r.get("opos_targets")) for r in rows)
    return (
        ((apos + opos) * weights).sum() / active,
        {
            "apos_loss": float(apos.detach().sum())
            / max(1, sum(bool(r.get("apos_targets")) for r in rows)),
            "opos_loss": float(opos.detach().sum())
            / max(1, sum(bool(r.get("opos_targets")) for r in rows)),
            "opos_anchor_loss": float(anchor.detach().mean()),
            "opos_support_loss": float(support.detach().mean()),
        },
        step,
    )
