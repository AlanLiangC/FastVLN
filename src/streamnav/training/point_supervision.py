"""Separate teacher-forced point CoT from on-policy PPO token replay."""

import torch
from torch.nn import functional as F

from streamnav.data.grounding_support import support_targets
from streamnav.models.policy.perception import decode_point_logits, point_type_logits


def targets_from_label(label):
    label = label or {}
    return {
        name + "_targets": (label.get(name + "_targets") or [int(label[name])])
        if label.get(name + "_valid", False)
        else []
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


def navigation_object_losses(logits, targets, *, anchor_weight=0.0):
    """Factorized token CE: visibility plus conditional location, counted once."""
    if not 0 <= anchor_weight <= 1:
        raise ValueError("Navigation OPOS anchor weight must lie in [0, 1]")
    types = point_type_logits(logits).float().log_softmax(-1)
    valid = torch.tensor([bool(t) for t in targets], device=logits.device)
    visible = torch.tensor([bool(t) and t != [0] for t in targets], device=logits.device)
    visibility = -types.gather(-1, visible.long()[:, None]).squeeze(-1) * valid
    total = marginal_point_loss(logits, targets)
    if anchor_weight:
        anchor = marginal_point_loss(logits, [[t[0]] if t else [] for t in targets])
        total = (1 - anchor_weight) * total + anchor_weight * anchor
    location = (total - visibility) * visible
    return visibility + location, visibility, location


def forward_intent_loss(logits, labels, experts):
    """Unknown coordinates still permit forward-compatible NONE/POINT intent.

    This does not invent a pixel or assert visibility. Reliable APOS labels,
    including visible waypoints during turns, retain their original semantics.
    """
    mask = torch.tensor([not label["apos_targets"] for label in labels], device=logits.device)
    mask &= experts == 1
    log_probs = point_type_logits(logits).float().log_softmax(-1)
    return -log_probs[:, :2].logsumexp(-1) * mask, mask


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
        apos_tokens = decode_point_logits(
            prefix.apos_logits,
            hierarchical=getattr(policy, "apos_decoding", "flat") == "hierarchical",
        )
        prefix_tokens = [(a, zero, zero) for a in apos_tokens]
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
