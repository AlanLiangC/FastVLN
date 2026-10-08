import torch


def replay_consistency_metrics(new_log_prob, old_log_prob):
    difference = (new_log_prob.detach() - old_log_prob).abs()
    return {
        "replay_log_prob_error_max": difference.max().item(),
        "replay_log_prob_error_mean": difference.mean().item(),
    }


def ovsegdt_value_loss(values, old_values, returns, clip_eps, use_clipped_value_loss=True):
    """Match DAgger_PPO's detached clipped target, including its gradient mask."""
    values = values.float()
    if use_clipped_value_loss:
        delta = values.detach() - old_values
        clipped = old_values + delta.clamp(-clip_eps, clip_eps)
        values = torch.where(delta.abs() < clip_eps, values, clipped)
    return 0.5 * (values - returns).square()


def clipped_ppo_loss(new_log_prob, old_log_prob, advantages, clip_eps=0.2):
    ratio = (new_log_prob - old_log_prob).exp()
    loss = -torch.minimum(ratio * advantages, ratio.clamp(1 - clip_eps, 1 + clip_eps) * advantages)
    return loss, ratio
