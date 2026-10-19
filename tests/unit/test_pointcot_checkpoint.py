import hashlib
import json

import pytest
import torch
from safetensors.torch import save_file
from torch import nn

from streamnav.training.pointcot_trainer import (
    added_navigation_rank_seed,
    load_adapters,
    navigation_rank_state,
    policy_schedule_update,
)


class Policy(nn.Module):
    def __init__(self):
        super().__init__()
        self.row = nn.Parameter(torch.zeros(3))
        self.device = torch.device("cpu")

    def parameter_summary(self):
        return {"source_revision": {"revision": "pinned"}}


def test_checkpoint_checks_integrity_and_base(tmp_path):
    expected = torch.tensor([1.0, 2.0, 3.0])
    digest = hashlib.sha256(b"row" + expected.numpy().tobytes()).hexdigest()
    save_file({"row": expected}, str(tmp_path / "adapters.safetensors"))
    manifest = {
        "format": "qwen35-pointcot-adapters-v1",
        "base": {"revision": "pinned"},
        "adapter_sha256": digest,
    }
    (tmp_path / "manifest.json").write_text(json.dumps(manifest))
    policy = Policy()
    load_adapters(policy, tmp_path)
    torch.testing.assert_close(policy.row, expected)
    save_file({"row": expected + 1}, str(tmp_path / "adapters.safetensors"))
    with pytest.raises(ValueError, match="checksum"):
        load_adapters(policy, tmp_path)
    manifest["base"] = {"revision": "different"}
    (tmp_path / "manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="base"):
        load_adapters(policy, tmp_path)


def test_legacy_checkpoint_can_only_add_an_explicit_planning_branch(tmp_path):
    expected = torch.tensor([1.0, 2.0, 3.0])
    save_file({"row": expected}, str(tmp_path / "adapters.safetensors"))
    (tmp_path / "manifest.json").write_text(
        json.dumps(
            {
                "format": "qwen35-pointcot-adapters-v1",
                "base": {"revision": "pinned"},
                "adapter_sha256": hashlib.sha256(b"row" + expected.numpy().tobytes()).hexdigest(),
            }
        )
    )
    policy = Policy()
    policy.future_plan = nn.Linear(3, 3)
    policy.arrival_head = nn.Linear(3, 3)
    with pytest.raises(ValueError, match="recipe"):
        load_adapters(policy, tmp_path)
    before = policy.future_plan.weight.detach().clone()
    load_adapters(policy, tmp_path, allow_new_plan=True)
    torch.testing.assert_close(policy.row, expected)
    torch.testing.assert_close(policy.future_plan.weight, before)
    policy.other = nn.Parameter(torch.zeros(2))
    with pytest.raises(ValueError, match="recipe"):
        load_adapters(policy, tmp_path, allow_new_plan=True)


def test_hint_recipe_fork_can_keep_original_ppo_schedule_across_resume():
    parent = {"trainer": {"schedule_from_fork": True}, "fork_origin": {"update": 1403}}
    repaired = {
        **parent,
        "fork_origin": {"update": 1515},
        "schedule_origin_update": 1403,
    }
    assert policy_schedule_update(parent, 1516) == policy_schedule_update(repaired, 1516) == 113
    assert policy_schedule_update(repaired, 1600) == 197


def test_expansion_preserves_existing_streams_and_adds_independent_ranks():
    saved = {"world_size": 6, "ranks": [{"rank": i, "source": f"stream-{i}"} for i in range(6)]}
    with pytest.raises(ValueError, match="allow-world-size-change"):
        navigation_rank_state(saved, 0, 8)
    for rank in range(6):
        assert (
            navigation_rank_state(saved, rank, 8, allow_world_size_change=True)
            is saved["ranks"][rank]
        )
    for rank in (6, 7):
        assert navigation_rank_state(saved, rank, 8, allow_world_size_change=True) is None
    assert navigation_rank_state(saved, 5, 6) is saved["ranks"][5]
    with pytest.raises(ValueError, match="worker group"):
        navigation_rank_state(saved, 8, 8, allow_world_size_change=True)
    corrupt = {**saved, "ranks": saved["ranks"][:-1]}
    with pytest.raises(ValueError, match="rank states"):
        navigation_rank_state(corrupt, 6, 8, allow_world_size_change=True)


def test_added_stream_seeds_are_stable_distinct_and_numpy_compatible():
    seeds = [
        added_navigation_rank_seed(17, update, rank) for update in (2040, 10000) for rank in (6, 7)
    ]
    assert len(set(seeds)) == 4
    assert seeds[0] == added_navigation_rank_seed(17, 2040, 6)
    assert all(0 <= seed < 2**32 for seed in seeds)


def test_an_expanded_checkpoint_can_resume_normally_and_shrink_explicitly():
    saved = {"world_size": 8, "ranks": [{"rank": i, "source": i} for i in range(8)]}
    for rank in range(8):
        assert navigation_rank_state(saved, rank, 8) is saved["ranks"][rank]
    with pytest.raises(ValueError):
        navigation_rank_state(saved, 0, 6)
    for rank in range(6):
        assert (
            navigation_rank_state(saved, rank, 6, allow_world_size_change=True)
            is saved["ranks"][rank]
        )
