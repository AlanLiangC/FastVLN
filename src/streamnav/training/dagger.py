from dataclasses import dataclass

import torch


@dataclass
class DaggerBetaScheduler:
    beta_start: float = 0.8
    beta_end: float = 0.05
    decay_updates: int = 10000
    update: int = 0

    def __post_init__(self):
        if not 0 <= self.beta_start <= 1 or not 0 <= self.beta_end <= 1 or self.decay_updates < 1:
            raise ValueError("Invalid DAgger schedule")

    @property
    def beta(self):
        fraction = min(self.update / self.decay_updates, 1.0)
        return self.beta_start + (self.beta_end - self.beta_start) * fraction

    def step(self):
        self.update += 1


def select_env_action(policy_action, expert_action, beta, generator=None):
    use_expert = (
        torch.rand(policy_action.shape, device=policy_action.device, generator=generator) < beta
    )
    return torch.where(use_expert, expert_action, policy_action), use_expert


def behavior_log_prob(logits, executed_actions, expert_actions, beta):
    """Exact DAgger behavior distribution mu=(1-beta)*pi + beta*delta_expert.

    PPO compares this distribution at fixed beta on the action actually executed.
    This avoids crediting oracle rewards to an unexecuted policy action.
    """
    log_pi = (
        logits.float().log_softmax(-1).gather(-1, executed_actions.long().unsqueeze(-1)).squeeze(-1)
    )
    probability = (1.0 - beta) * log_pi.exp() + beta * (executed_actions == expert_actions).float()
    return probability.clamp_min(torch.finfo(torch.float32).tiny).log()
