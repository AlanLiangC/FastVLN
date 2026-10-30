import random
from types import SimpleNamespace

import numpy as np
import torch

from streamnav.data.grounding import GroundingSource
from streamnav.data.grounding_support import pack_grid, unpack_grid
from streamnav.training.hint_grounding import sample_grounding_hints
from streamnav.training.point_supervision import grounding_loss


def instance():
    grid = np.zeros((27, 48), dtype=bool)
    grid[10:15, 20:25] = True
    return {
        "image_id": "a",
        "instruction": "Find a chair.",
        "opos_targets": [12 * 48 + 23],
        "opos_support": pack_grid(grid),
    }


def test_only_complete_masks_and_explicit_absence_license_hint_quality():
    positive = instance()
    rows = [
        positive,
        {**positive, "opos_targets": [0], "opos_support": None},
        {"opos_targets": [649]},
        {"opos_targets": [649], "supervision": "region"},
    ]
    support = unpack_grid(positive["opos_support"])
    observed = set()
    saved_rng = random.getstate()
    try:
        random.seed(17)
        for _ in range(100):
            packets = sample_grounding_hints(rows)
            assert {p[0] for p in packets} == {0, 1}
            for index, token, target in packets:
                assert 1 <= token <= 1296
                if index == 1:
                    assert target == 0
                else:
                    y, x = divmod(token - 1, 48)
                    assert bool(support[y, x]) == bool(target)
                    if not target:
                        assert not support[max(0, y - 1) : y + 2, max(0, x - 1) : x + 2].any()
                    observed.add(target)
        assert observed == {0.0, 1.0}
    finally:
        random.setstate(saved_rng)


def test_matched_null_and_wrong_hint_have_same_true_opos_and_no_action_labels(monkeypatch):
    row = {"image_id": "x", "instruction": "Find a chair.", "opos_targets": [0]}
    monkeypatch.setattr(GroundingSource, "features_many", lambda policy, rows: [None])
    logits = torch.zeros(1, 1297, requires_grad=True)
    quality = torch.zeros(1, requires_grad=True)
    calls = []

    class Policy:
        device = torch.device("cpu")
        apos_decoding = "flat"

        def condition_features(self, features, tokens):
            return tokens

        def start_episode(self, *args):
            return None

        def replay_fragment(self, features, state, tokens, resets=None):
            calls.append((features, tokens))
            return SimpleNamespace(
                apos_logits=torch.zeros(1, 1300), opos_logits=logits, hint_quality_logits=quality
            )

    loss, metrics, _ = grounding_loss(Policy(), [row], hint_training=True)
    loss.backward()
    assert len(calls) == 3 and calls[0][0] == calls[1][0] == [0]
    assert 1 <= calls[2][0][0] <= 1296
    assert int(calls[1][1][0][1]) == int(calls[2][1][0][1]) == 0
    assert logits.grad[0, 0] < 0 and quality.grad.item() > 0
    assert metrics["hint_fraction"] == 1 and metrics["hint_positive_fraction"] == 0


def test_hint_sampler_reproduces_checkpointed_python_rng():
    rows = [instance()] * 32
    state = random.getstate()
    first = sample_grounding_hints(rows)
    random.setstate(state)
    assert sample_grounding_hints(rows) == first
