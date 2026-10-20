"""Future teacher labels respect the actual action-conditioned execution prefix."""

import torch
from torch.nn import functional as F


def attach_future_actions(records, horizon=4, *, mode="expert_only"):
    if not 1 <= horizon <= 8:
        raise ValueError("Action horizon must be in [1, 8]")
    if mode not in ("expert_only", "executed_prefix"):
        raise ValueError("Unknown future-action supervision mode")
    for t, row in enumerate(records):
        for i, record in enumerate(row):
            targets = [-100] * horizon
            if mode == "executed_prefix":
                if (
                    record.get("expert_stream", False)
                    and int(record["tokens"][2]) != record["expert"]
                ):
                    raise ValueError("Future-action labels require expert-executed actions")
                targets[0] = record["expert"]
                # The planning head reads a_t, so the teacher at the real s_(t+1)
                # is a valid next-step label even when a_t was a student action.
                # Beyond that, the intervening execution must follow the teacher
                # prefix; arbitrary future learner states are not expert plans.
                for k in range(1, min(horizon, len(records) - t)):
                    previous = records[t + k - 1][i]
                    following = records[t + k][i]
                    if previous["done"] or int(previous["tokens"][2]) == 0:
                        break
                    if k > 1 and int(previous["tokens"][2]) != previous["expert"]:
                        break
                    if following["episode_id"] != record["episode_id"]:
                        break
                    if following.get("hint_goal_key") != record.get("hint_goal_key"):
                        break
                    targets[k] = following["expert"]
                    if following["expert"] == 0:
                        break
                record["future_actions"] = targets
                continue
            if record.get("expert_stream", False):
                for k in range(min(horizon, len(records) - t)):
                    following = records[t + k][i]
                    if following["episode_id"] != record["episode_id"]:
                        break
                    if following.get("hint_goal_key") != record.get("hint_goal_key"):
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
