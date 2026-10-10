"""Synchronous data parallel training for the explicit recurrent policy API."""

import math
import os
from datetime import timedelta

import torch
import torch.distributed as dist


def rank():
    return dist.get_rank() if dist.is_initialized() else 0


def world_size():
    return dist.get_world_size() if dist.is_initialized() else 1


def initialize(config):
    size = int(os.environ.get("WORLD_SIZE", "1"))
    if size > 1:
        device = int(os.environ["LOCAL_RANK"]) + config.get("distributed", {}).get("gpu_offset", 0)
        config["device"] = f"cuda:{device}"
        config["habitat"]["gpu_device_id"] = device
        torch.cuda.set_device(device)
        dist.init_process_group(
            "nccl", timeout=timedelta(minutes=20), device_id=torch.device(config["device"])
        )
    config.setdefault("distributed", {})["world_size"] = size


def gather(value):
    if world_size() == 1:
        return [value]
    values = [None] * world_size()
    dist.all_gather_object(values, value)
    return values


def barrier():
    if dist.is_initialized():
        dist.barrier()


def any_rank(flag, device):
    value = torch.tensor(int(flag), device=device)
    if world_size() > 1:
        dist.all_reduce(value, op=dist.ReduceOp.MAX)
    return bool(value.item())


def average_scalar(value):
    result = value.detach().clone()
    if world_size() > 1:
        dist.all_reduce(result)
        result /= world_size()
    return result


def normalize_advantages(values, device, mask=None):
    eligible = values if mask is None else values[mask]
    stats = torch.stack(
        (eligible.sum(), eligible.square().sum(), values.new_tensor(eligible.numel()))
    ).to(device)
    if world_size() > 1:
        dist.all_reduce(stats)
    count = stats[2].clamp_min(1)
    mean = stats[0] / count
    std = (stats[1] / count - mean.square()).clamp_min(0).sqrt().clamp_min(1e-8)
    normalized = (values - mean.cpu()) / std.cpu()
    return normalized if mask is None else torch.where(mask, normalized, 0)


def average_gradients(parameters):
    """Reduce before clipping/Adam, including parameters unused on some ranks.

    A contiguous buffer permits one collective for this sub-billion-parameter
    model; optimizer tensors stay local and identical after the same update.
    """
    if world_size() == 1:
        return
    params = [p for p in parameters if p.requires_grad]
    used = torch.tensor(
        [p.grad is not None for p in params], device=params[0].device, dtype=torch.int
    )
    dist.all_reduce(used, op=dist.ReduceOp.MAX)
    gradients = []
    for p, active in zip(params, used.tolist(), strict=True):
        if active:
            if p.grad is None:
                p.grad = torch.zeros_like(p)
            gradients.append(p.grad)
    if not gradients:
        return
    flat = torch.cat([g.reshape(-1) for g in gradients])
    dist.all_reduce(flat)
    flat.div_(world_size())
    offset = 0
    for gradient in gradients:
        gradient.copy_(flat[offset : offset + gradient.numel()].view_as(gradient))
        offset += gradient.numel()


def combine_metrics(rows):
    result = dict(rows[0])
    count_fields = {
        "episodes_completed",
        "oracle_skipped_episodes",
        "invalid_forward_labels_filtered",
        "oracle_navigation_repairs",
        "curriculum_warmup_steps",
        "curriculum_fallbacks",
        "auxiliary_episodes_completed",
        "auxiliary_expert_steps",
        "auxiliary_recovery_triggers",
        "on_policy_transitions",
        "auxiliary_il_transitions",
        "teacher_stop_count_on_policy",
        "teacher_stop_count_auxiliary",
    }
    for key, value in result.items():
        if key in {"stop_probability_on_teacher_stop", "stop_probability_on_teacher_nonstop"}:
            # Nullable conditional means must be combined with their class
            # counts below; a rank can have no examples of the conditioning class.
            continue
        if isinstance(value, bool):
            result[key] = any(r[key] for r in rows)
        elif isinstance(value, (float, int)):
            if key in count_fields or key == "training_fps":
                result[key] = sum(r[key] for r in rows)
            elif key in {
                "update_seconds",
                "optimization_seconds",
                "rollout_seconds",
                "reset_seconds",
                "gpu_memory_bytes",
                "preupdate_replay_log_prob_error_max",
                "preupdate_replay_initial_error_max",
                "replay_preflight_seconds",
            }:
                result[key] = max(r[key] for r in rows)
            elif key != "update":
                result[key] = sum(r[key] for r in rows) / len(rows)
        elif key.endswith("action_histogram"):
            result[key] = [sum(r[key][a] for r in rows) / len(rows) for a in range(len(value))]
    completed = sum(r["episodes_completed"] for r in rows)
    for key in ("success", "spl"):
        result[key] = sum(r[key] * r["episodes_completed"] for r in rows) / max(completed, 1)
    if "auxiliary_success" in result:
        count = sum(r["auxiliary_episodes_completed"] for r in rows)
        result["auxiliary_success"] = sum(
            r["auxiliary_success"] * r["auxiliary_episodes_completed"] for r in rows
        ) / max(count, 1)
    result["oracle_class_recall"] = []
    for action in range(len(result["expert_action_histogram"])):
        weight = sum(r["expert_action_histogram"][action] for r in rows)
        result["oracle_class_recall"].append(
            sum(
                (r["oracle_class_recall"][action] or 0) * r["expert_action_histogram"][action]
                for r in rows
            )
            / weight
            if weight
            else None
        )
    if all("expert_action_counts" in r and "greedy_action_confusion" in r for r in rows):
        actions = len(result["expert_action_histogram"])
        counts = [sum(r["expert_action_counts"][a] for r in rows) for a in range(actions)]
        confusion = [
            [sum(r["greedy_action_confusion"][a][p] for r in rows) for p in range(actions)]
            for a in range(actions)
        ]
        result["expert_action_counts"] = counts
        result["greedy_action_confusion"] = confusion
        result["expert_action_histogram"] = [n / sum(counts) for n in counts]
        result["oracle_class_recall"] = [
            confusion[a][a] / counts[a] if counts[a] else None for a in range(actions)
        ]
        result["teacher_stop_count"] = counts[0]
        result["greedy_stop_count"] = sum(row[0] for row in confusion)
        result["greedy_stop_precision"] = (
            confusion[0][0] / result["greedy_stop_count"] if result["greedy_stop_count"] else None
        )
        for key, positive in (
            ("stop_probability_on_teacher_stop", True),
            ("stop_probability_on_teacher_nonstop", False),
        ):
            weights = [
                r["expert_action_counts"][0]
                if positive
                else sum(r["expert_action_counts"]) - r["expert_action_counts"][0]
                for r in rows
            ]
            result[key] = (
                sum((r[key] or 0) * n for r, n in zip(rows, weights, strict=True)) / sum(weights)
                if sum(weights)
                else None
            )
    # A global constant-action baseline uses the global label distribution,
    # not the average entropy of independently estimated per-rank priors.
    if "oracle_prior_cross_entropy" in result:
        prior_ce = -sum(p * math.log(p) for p in result["expert_action_histogram"] if p > 0)
        result["oracle_prior_cross_entropy"] = prior_ce
        result["il_gain_over_prior"] = prior_ce - result["il_loss"]
    return result
