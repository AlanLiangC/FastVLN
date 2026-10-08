import torch


def compute_gae(
    rewards, values, dones, last_values, gamma=0.99, gae_lambda=0.95, timeout_bootstrap=None
):
    if rewards.shape != values.shape or rewards.shape != dones.shape:
        raise ValueError("Rewards, values and dones must have shape [T,N]")
    advantages = torch.zeros_like(rewards)
    carry = torch.zeros_like(last_values)
    for t in reversed(range(rewards.shape[0])):
        next_value = last_values if t == rewards.shape[0] - 1 else values[t + 1]
        alive = (~dones[t]).to(rewards.dtype)
        bootstrap = next_value * alive
        if timeout_bootstrap is not None:
            bootstrap = bootstrap + timeout_bootstrap[t]
        delta = rewards[t] + gamma * bootstrap - values[t]
        carry = delta + gamma * gae_lambda * alive * carry
        advantages[t] = carry
    return advantages, advantages + values
