import torch


def clipped_ppo_loss(new_log_prob, old_log_prob, advantages, clip_eps=0.2):
    ratio = (new_log_prob - old_log_prob).exp()
    loss = -torch.minimum(ratio * advantages, ratio.clamp(1 - clip_eps, 1 + clip_eps) * advantages)
    return loss, ratio
