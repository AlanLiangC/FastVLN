"""Synchronous data parallel training for the explicit recurrent policy API."""

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


def normalize_advantages(values, device):
    stats = torch.stack(
        (values.sum(), values.square().sum(), values.new_tensor(values.numel()))
    ).to(device)
    if world_size() > 1:
        dist.all_reduce(stats)
    mean = stats[0] / stats[2]
    std = (stats[1] / stats[2] - mean.square()).clamp_min(0).sqrt().clamp_min(1e-8)
    return (values - mean.cpu()) / std.cpu()


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
        "curriculum_warmup_steps",
        "curriculum_fallbacks",
    }
    for key, value in result.items():
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
            }:
                result[key] = max(r[key] for r in rows)
            elif key != "update":
                result[key] = sum(r[key] for r in rows) / len(rows)
        elif key.endswith("action_histogram"):
            result[key] = [sum(r[key][a] for r in rows) / len(rows) for a in range(4)]
    completed = sum(r["episodes_completed"] for r in rows)
    for key in ("success", "spl"):
        result[key] = sum(r[key] * r["episodes_completed"] for r in rows) / max(completed, 1)
    result["oracle_class_recall"] = []
    for action in range(4):
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
    return result
