"""Lightweight pointing and arrival supervision with predicted-feature action fusion."""

import torch
from torch import nn

from streamnav.contracts.perception import (
    APOS_SIZE,
    GRID_HEIGHT,
    GRID_WIDTH,
    OPOS_SIZE,
    POINT_CELLS,
)


class NavigationPerception(nn.Module):
    def __init__(self, hidden_dim, action_dim, embedding_dim=64):
        super().__init__()
        self.norm = nn.LayerNorm(hidden_dim)
        self.apos = nn.Linear(hidden_dim, APOS_SIZE)
        self.opos = nn.Linear(hidden_dim, OPOS_SIZE)
        self.arrival = nn.Linear(hidden_dim, 3)
        self.apos_embedding = nn.Embedding(APOS_SIZE, embedding_dim)
        self.opos_embedding = nn.Embedding(OPOS_SIZE, embedding_dim)
        self.arrival_embedding = nn.Embedding(3, embedding_dim)
        # Give the actor metric image coordinates and sentinel probabilities as
        # well as learned embeddings; grid adjacency need not be learned anew.
        apos_geometry = torch.zeros(APOS_SIZE, 7)
        opos_geometry = torch.zeros(OPOS_SIZE, 4)
        cells = torch.arange(POINT_CELLS)
        coordinates = torch.stack(
            (
                (cells % GRID_WIDTH + 0.5) / GRID_WIDTH * 2 - 1,
                (cells // GRID_WIDTH + 0.5) / GRID_HEIGHT * 2 - 1,
            ),
            dim=-1,
        )
        for geometry in (apos_geometry, opos_geometry):
            geometry[1 : POINT_CELLS + 1, :2] = coordinates
            geometry[1 : POINT_CELLS + 1, 2] = 1
            geometry[0, 3] = 1
        for index in range(3):
            apos_geometry[POINT_CELLS + 1 + index, 4 + index] = 1
        self.register_buffer("apos_geometry", apos_geometry)
        self.register_buffer("opos_geometry", opos_geometry)
        self.action_residual = nn.Linear(3 * embedding_dim + 14, action_dim)
        for head in (self.apos, self.opos, self.arrival):
            nn.init.normal_(head.weight, std=0.01)
            nn.init.zeros_(head.bias)
        for table in (self.apos_embedding, self.opos_embedding, self.arrival_embedding):
            nn.init.normal_(table.weight, std=0.02)
        # Adding this module to an existing policy preserves its initial decisions.
        nn.init.zeros_(self.action_residual.weight)
        nn.init.zeros_(self.action_residual.bias)

    def forward(self, hidden):
        features = self.norm(hidden)
        predictions = {
            "apos": self.apos(features).float(),
            "opos": self.opos(features).float(),
            "arrival": self.arrival(features).float(),
        }
        probabilities = {name: logits.softmax(-1) for name, logits in predictions.items()}
        expected = [
            probabilities[name].to(table.weight.dtype) @ table.weight
            for name, table in (
                ("apos", self.apos_embedding),
                ("opos", self.opos_embedding),
                ("arrival", self.arrival_embedding),
            )
        ]
        expected.extend(
            (
                probabilities["apos"].to(self.apos_geometry.dtype) @ self.apos_geometry,
                probabilities["opos"].to(self.opos_geometry.dtype) @ self.opos_geometry,
                probabilities["arrival"].to(hidden.dtype),
            )
        )
        # No teacher-forced coordinates, hard STOP override or extra body forward.
        residual = self.action_residual(torch.cat(expected, dim=-1)).float()
        return predictions, residual
