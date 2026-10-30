"""Zero-output input adapters preserve the parent's initial action distribution."""

import torch
from torch import nn


def residual_mlp(inputs, hidden):
    output = nn.Linear(128, hidden)
    module = nn.Sequential(nn.Linear(inputs, 128), nn.SiLU(), output)
    nn.init.zeros_(output.weight)
    nn.init.zeros_(output.bias)
    return module


class GeneralizationAdapters(nn.Module):
    def __init__(self, hidden):
        super().__init__()
        self.spatial = residual_mlp(2, hidden)
        self.motion = residual_mlp(12, hidden)
        self.quality = nn.Linear(hidden, 1)
        nn.init.zeros_(self.quality.weight)
        nn.init.zeros_(self.quality.bias)

    def forward(self, tokens, feedback):
        indices = (tokens - 1).clamp_min(0)
        coordinates = torch.stack(
            (((indices % 48).float() + 0.5) / 48, ((indices // 48).float() + 0.5) / 27), -1
        )
        return self.spatial(coordinates * 2 - 1) * (tokens > 0)[..., None] + self.motion(
            feedback.float()
        )
