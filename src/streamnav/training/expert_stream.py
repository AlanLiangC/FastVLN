"""Separate ongoing expert imitation trajectories from on-policy student RL."""

import torch


def validate_expert_streams(config):
    count, experts = config["num_envs"], config.get("num_expert_envs", 0)
    if type(experts) is not int or not 0 <= experts < count:
        raise ValueError("num_expert_envs must leave at least one student environment per rank")
    daggers = config.get("num_dagger_envs", 0)
    if type(daggers) is not int or not 0 <= daggers < count - experts:
        raise ValueError("num_dagger_envs must leave at least one stochastic student environment")
    return experts


def expert_environment(slot, config):
    return slot >= config["num_envs"] - config.get("num_expert_envs", 0)


def dagger_environment(slot, config):
    end = config["num_envs"] - config.get("num_expert_envs", 0)
    return end - config.get("num_dagger_envs", 0) <= slot < end


def split_episode_metrics(metrics):
    return {
        "episodes": sum(
            not m["expert_stream"] and not m.get("dagger_stream", False) for m in metrics
        ),
        "successes": sum(
            m["success"]
            for m in metrics
            if not m["expert_stream"] and not m.get("dagger_stream", False)
        ),
        "expert_episodes": sum(m["expert_stream"] for m in metrics),
        "expert_successes": sum(m["success"] for m in metrics if m["expert_stream"]),
        "dagger_episodes": sum(m.get("dagger_stream", False) for m in metrics),
        "dagger_successes": sum(m["success"] for m in metrics if m.get("dagger_stream", False)),
    }


def student_rl_weights(records, config, device):
    """Normalize RL by student stream size, independent of replay fragmentation.

    A fragment contains a single environment; its local eligible count cannot
    serve as the denominator for a full multi-environment rollout objective.
    Legacy configurations retain their original unmasked weighting.
    """
    excluded = config.get("num_expert_envs", 0) + config.get("num_dagger_envs", 0)
    if not excluded:
        return torch.ones(len(records), device=device)
    scale = config["num_envs"] / (config["num_envs"] - excluded)
    return torch.tensor(
        [
            0.0 if r.get("expert_stream", False) or r.get("dagger_stream", False) else scale
            for r in records
        ],
        device=device,
    )
