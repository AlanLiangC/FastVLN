"""Future labels are valid only along actual expert-executed trajectories."""

import torch
from torch.nn import functional as F


def attach_future_actions(records, horizon=4):
    if not 1 <= horizon <= 8:
        raise ValueError("Action horizon must be in [1, 8]")
    for t, row in enumerate(records):
        for i, record in enumerate(row):
            targets = [-100] * horizon
            if record.get("expert_stream", False):
                for k in range(min(horizon, len(records) - t)):
                    following = records[t + k][i]
                    if following["episode_id"] != record["episode_id"]:
                        break
                    if not following.get("expert_stream", False):
                        break
                    if int(following["tokens"][2]) != following["expert"]:
                        raise ValueError("Future-action labels require expert-executed actions")
                    targets[k] = following["expert"]
                    if following["done"] or following["expert"] == 0:
                        break
            record["future_actions"] = targets


def future_action_loss(logits, records):
    """The current action already has IL/PPO; supervise only later horizons."""
    targets = torch.tensor(
        [r.get("future_actions", [-100] * logits.shape[-2]) for r in records],
        device=logits.device,
    )[:, 1:]
    valid = targets != -100
    loss = F.cross_entropy(
        logits[:, 1:].float().flatten(0, 1), targets.flatten(), reduction="none", ignore_index=-100
    ).view_as(targets)
    per_frame = loss.sum(-1) / valid.sum(-1).clamp_min(1)
    return per_frame * logits.new_tensor([r.get("future_loss_weight", 1.0) for r in records])
