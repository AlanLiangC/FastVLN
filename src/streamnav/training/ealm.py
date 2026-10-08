import torch.nn as nn


class EntropyAdaptiveLossMixer(nn.Module):
    def __init__(self, entropy_low=0.2, entropy_high=1.2, enabled=True, fixed_alpha=0.5):
        super().__init__()
        if entropy_high <= entropy_low or not 0 <= fixed_alpha <= 1:
            raise ValueError("Invalid entropy normalization or fixed alpha")
        self.low, self.high = entropy_low, entropy_high
        self.enabled, self.fixed_alpha = enabled, fixed_alpha

    def forward(self, il_loss, ppo_loss, entropy):
        # Loss weights are scheduling signals, not a route to game entropy.
        alpha = ((entropy.detach() - self.low) / (self.high - self.low)).clamp(0, 1)
        if not self.enabled:
            alpha = alpha * 0 + self.fixed_alpha
        return alpha * il_loss + (1 - alpha) * ppo_loss, alpha
