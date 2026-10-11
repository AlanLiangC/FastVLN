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


def point_type_logits(logits):
    """Compare aggregate point probability with each singleton sentinel."""
    if logits.shape[-1] not in (APOS_SIZE, OPOS_SIZE):
        raise ValueError("Invalid pointing vocabulary")
    return torch.cat(
        (
            logits[..., :1],
            logits[..., 1 : POINT_CELLS + 1].float().logsumexp(-1, keepdim=True),
            logits[..., POINT_CELLS + 1 :],
        ),
        dim=-1,
    )


def point_type_targets(target):
    return torch.where(
        target > POINT_CELLS,
        target - POINT_CELLS + 1,
        (target > 0).long(),
    )


def decode_point_logits(logits, *, hierarchical=False):
    if not hierarchical:
        return logits.argmax(-1)
    kind = point_type_logits(logits).argmax(-1)
    location = logits[..., 1 : POINT_CELLS + 1].argmax(-1) + 1
    return torch.where(kind == 1, location, torch.where(kind == 0, 0, POINT_CELLS + kind - 1))


class NavigationPerception(nn.Module):
    def __init__(
        self,
        hidden_dim,
        action_dim,
        embedding_dim=64,
        fusion_scale=1.0,
        supervision_backbone_scale=1.0,
    ):
        super().__init__()
        self.fusion_scale = fusion_scale
        self.supervision_backbone_scale = supervision_backbone_scale
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

    def predict(self, hidden):
        features = self.norm(hidden)
        return {
            "apos": self.apos(features).float(),
            "opos": self.opos(features).float(),
            "arrival": self.arrival(features).float(),
        }

    def forward(self, hidden):
        predictions = self.predict(hidden)
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
        residual = self.action_residual(torch.cat(expected, dim=-1)).float() * self.fusion_scale
        supervision = predictions
        if torch.is_grad_enabled() and self.supervision_backbone_scale != 1:
            # Weak auxiliary labels must not overwhelm navigation features.
            # Keep full head gradients and the original IL/PPO fusion gradient;
            # scale only the teacher loss gradient entering the shared backbone.
            teacher_hidden = (
                hidden.detach() + (hidden - hidden.detach()) * self.supervision_backbone_scale
            )
            supervision = self.predict(teacher_hidden)
        return supervision, residual

    def load_scaled_weights(self, state, source_scale=1.0):
        # A power-of-two reparameterization preserves initial policy decisions
        # while increasing the effective learning rate of the fusion branch.
        state = dict(state)
        ratio = source_scale / self.fusion_scale
        for name in ("action_residual.weight", "action_residual.bias"):
            state[name] = state[name] * ratio
        self.load_state_dict(state)

    def rescale_optimizer_moments(self, optimizer, source_scale=1.0):
        # dL/dW_new = (new_scale / old_scale) * dL/dW_old.
        ratio = self.fusion_scale / source_scale
        for parameter in self.action_residual.parameters():
            state = optimizer.state.get(parameter, {})
            for name in ("exp_avg", "exp_avg_sq", "max_exp_avg_sq"):
                if name in state:
                    state[name].mul_(ratio if name == "exp_avg" else ratio**2)
