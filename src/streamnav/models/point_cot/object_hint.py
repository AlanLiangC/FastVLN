"""One input embedding, sharing spatial rows without sharing NULL semantics."""

import torch
from torch import nn


class ObjectHintToken(nn.Module):
    def __init__(self, null_embedding):
        super().__init__()
        self.null = nn.Parameter(null_embedding.detach().clone())
        self.source = nn.Parameter(torch.zeros_like(null_embedding))

    def forward(self, tokens, point_rows):
        points = point_rows[tokens] + self.source
        return torch.where(tokens[..., None] > 0, points, self.null)
