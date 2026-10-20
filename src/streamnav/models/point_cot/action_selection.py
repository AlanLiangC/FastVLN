"""Current-frame arrival calibration and causal, greedy-only plan fusion."""

import torch


def calibrated_action_logits(logits, arrival_logits, weight):
    if not 0 <= weight <= 1:
        raise ValueError("Arrival STOP weight must lie in [0, 1]")
    if weight == 0:
        return logits
    if arrival_logits is None:
        raise ValueError("Arrival STOP calibration requires an arrival head")
    probabilities = logits.float().softmax(-1)
    stop = (1 - weight) * probabilities[..., :1] + weight * arrival_logits.float().softmax(-1)[
        ..., 2:3
    ]
    motion = logits[..., 1:].float().softmax(-1) * (1 - stop)
    return torch.cat((stop, motion), -1).clamp_min(1e-30).log()


def temporal_greedy_logits(logits, previous_probabilities, weight):
    """Mix only the previous frame's next-step forecast, conditioned on its actual action.

    STOP mass is preserved. Fusion is disabled when p(STOP) >= 1 / action_dim:
    below that bound at least one non-STOP action must exceed STOP after any
    redistribution of motion mass. Thus this function cannot create or remove a
    greedy STOP on a fixed observation. Past forecast STOPs only reduce its weight.
    """
    if not 0 <= weight < 1:
        raise ValueError("Temporal action weight must lie in [0, 1)")
    if weight == 0 or previous_probabilities is None:
        return logits
    previous = previous_probabilities.to(device=logits.device, dtype=torch.float32)
    if previous.shape != logits.shape:
        raise ValueError("One previous forecast is required per current action distribution")
    current = logits.float().softmax(-1)
    mass = previous[..., 1:].sum(-1, keepdim=True)
    effective = weight * mass.clamp(0, 1)
    old_motion = previous[..., 1:] / mass.clamp_min(1e-30)
    new_motion = logits[..., 1:].float().softmax(-1)
    mixed = ((1 - effective) * new_motion + effective * old_motion) * (1 - current[..., :1])
    result = torch.cat((current[..., :1], mixed), -1).clamp_min(1e-30).log()
    eligible = (current[..., :1] < 1 / logits.shape[-1]) & (mass > 0)
    return torch.where(eligible, result, logits)
