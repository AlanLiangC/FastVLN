"""Small horizon-conditioned future features; no writes to navigation cache."""

import torch
from torch import nn


class FutureActionPlan(nn.Module):
    def __init__(self, hidden_size, future_steps=3, bottleneck=128, device=None):
        super().__init__()
        self.future_steps = future_steps
        self.norm = nn.LayerNorm(hidden_size, device=device)
        self.down = nn.Linear(hidden_size, bottleneck, device=device)
        self.up = nn.Linear(bottleneck, future_steps * hidden_size, device=device)
        nn.init.zeros_(self.up.weight)
        nn.init.zeros_(self.up.bias)

    def forward(self, hidden):
        delta = self.up(torch.nn.functional.silu(self.down(self.norm(hidden.float()))))
        delta = delta.view(*hidden.shape[:-1], self.future_steps, hidden.shape[-1])
        return hidden.float().unsqueeze(-2) + delta
