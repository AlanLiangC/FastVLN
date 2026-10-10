from dataclasses import dataclass

import torch

from streamnav.models.qwen35_kda.cache import clone_state


@dataclass(frozen=True)
class SequenceIndex:
    env: int
    start: int
    stop: int


class RecurrentRolloutBuffer:
    def __init__(self, steps, num_envs, rgb_shape, sequence_length, beta):
        if steps % sequence_length:
            raise ValueError("rollout_steps must be divisible by sequence_length")
        self.steps, self.num_envs, self.sequence_length = steps, num_envs, sequence_length
        self.beta = beta
        self.observations = torch.empty((steps, num_envs, *rgb_shape), dtype=torch.uint8)
        self.actions = torch.empty(steps, num_envs, dtype=torch.long)
        self.greedy_actions = torch.empty_like(self.actions)
        self.executed_actions = torch.empty_like(self.actions)
        self.expert_actions = torch.empty_like(self.actions)
        self.used_expert = torch.empty(steps, num_envs, dtype=torch.bool)
        self.ppo_mask = torch.ones(steps, num_envs, dtype=torch.bool)
        self.il_mask = torch.ones(steps, num_envs, dtype=torch.bool)
        self.replay_log_probs = None
        self.dones = torch.empty_like(self.used_expert)
        self.rewards = torch.empty(steps, num_envs)
        self.old_log_probs = torch.empty_like(self.rewards)
        self.old_policy_log_probs = torch.empty_like(self.rewards)
        self.old_values = torch.empty_like(self.rewards)
        self.entropies = torch.empty_like(self.rewards)
        self.perception_targets = None
        self.stop_probabilities = torch.empty_like(self.rewards)
        self.visual_embeddings = None
        self.timeout_bootstrap = torch.zeros_like(self.rewards)
        self.initial_states = {}
        self.resets = {}
        self.episode_metrics = []
        self.auxiliary_episode_metrics = []
        self.auxiliary_expert_steps = 0
        self.auxiliary_recovery_triggers = 0
        self.collisions = 0
        self.auxiliary_collisions = 0
        self.oracle_failures = []
        self.oracle_navigation_repairs = 0
        self.last_values = torch.zeros(num_envs)
        self.elapsed_s = 0.0
        self.curriculum_warmup_steps = 0
        self.curriculum_fallbacks = 0
        self.reset_seconds = 0.0
        self.advantages = torch.empty_like(self.rewards)
        self.returns = torch.empty_like(self.rewards)

    def enable_perception(self):
        self.perception_targets = {
            name: {
                "target": torch.zeros(self.steps, self.num_envs, dtype=torch.long),
                "valid": torch.zeros(self.steps, self.num_envs, dtype=torch.bool),
                "confidence": torch.zeros(self.steps, self.num_envs),
            }
            for name in ("apos", "opos", "arrival")
        }

    def store_perception(self, step, env, labels):
        assert self.perception_targets is not None
        for name, tensors in self.perception_targets.items():
            tensors["target"][step, env] = labels[name]
            tensors["valid"][step, env] = labels[name + "_valid"]
            tensors["confidence"][step, env] = labels[name + "_confidence"]

    def gather_perception(self, sequences, device):
        if self.perception_targets is None:
            raise ValueError("Rollout lacks perception supervision")
        return {
            name: {
                key: torch.stack([tensor[s.start : s.stop, s.env] for s in sequences], dim=1).to(
                    device
                )
                for key, tensor in tensors.items()
            }
            for name, tensors in self.perception_targets.items()
        }

    def save_boundary(self, step, states):
        if step % self.sequence_length == 0:
            for i, state in enumerate(states):
                self.initial_states[(i, step)] = clone_state(state)

    def sequence_batches(self, batch_size, shuffle=True):
        indices = [
            SequenceIndex(e, s, s + self.sequence_length)
            for e in range(self.num_envs)
            for s in range(0, self.steps, self.sequence_length)
        ]
        order = torch.randperm(len(indices)).tolist() if shuffle else list(range(len(indices)))
        for start in range(0, len(order), batch_size):
            yield [indices[i] for i in order[start : start + batch_size]]

    def gather(self, name, sequences, device):
        tensor = getattr(self, name)
        return torch.stack([tensor[s.start : s.stop, s.env] for s in sequences], dim=1).to(device)
