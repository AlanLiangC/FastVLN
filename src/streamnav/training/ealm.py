import math

import torch
import torch.nn as nn


class EntropyAdaptiveLossMixer(nn.Module):
    entropy_ema: torch.Tensor

    def __init__(
        self,
        entropy_low=0.35,
        entropy_high=0.75,
        entropy_ema_decay=0.95,
        enabled=True,
        fixed_alpha=0.5,
    ):
        super().__init__()
        if (
            entropy_high <= entropy_low
            or not 0 <= fixed_alpha <= 1
            or not 0 <= entropy_ema_decay < 1
        ):
            raise ValueError("Invalid entropy normalization or fixed alpha")
        self.low, self.high = entropy_low, entropy_high
        self.enabled, self.fixed_alpha = enabled, fixed_alpha
        self.decay = entropy_ema_decay
        self.register_buffer("entropy_ema", torch.tensor(float("nan"), dtype=torch.float64))

    def forward(self, il_loss, ppo_loss, entropy):
        # OVSegDT uses the PREVIOUS minibatch EMA for one common batch weight,
        # then updates EMA for the next minibatch. The first batch is pure IL.
        previous = self.entropy_ema.item()
        weight = (
            1.0
            if math.isnan(previous)
            else min(max((previous - self.low) / (self.high - self.low), 0.0), 1.0)
        )
        alpha = torch.full_like(entropy, weight if self.enabled else self.fixed_alpha)
        return alpha * il_loss + (1 - alpha) * ppo_loss, alpha

    @torch.no_grad()
    def observe_entropy(self, entropy_mean):
        # Called only after a valid optimizer step, so rejected replay checks
        # cannot mutate the schedule. Saved separately for each DDP rank.
        value = float(entropy_mean)
        previous = self.entropy_ema.item()
        self.entropy_ema.fill_(
            value if math.isnan(previous) else self.decay * previous + (1 - self.decay) * value
        )
