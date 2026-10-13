import math

import pytest

from streamnav.training.grounding_trainer import (
    configure_sample_schedule,
    resumed_rank_state,
    sample_lr_scale,
)


def test_shrink_preserves_remaining_samples_and_learning_rate():
    old = {"batch_size": 32, "pretrain_updates": 12023, "warmup_updates": 100, "min_lr_ratio": 0.1}
    saved = {"world_size": 8, "update": 417, "frames": 417 * 256}
    cfg = dict(old)
    configure_sample_schedule(cfg, 6, saved=saved, previous={"grounding": old})
    assert cfg["sample_budget"] == 12023 * 256
    assert cfg["warmup_samples"] == 100 * 256
    assert cfg["pretrain_updates"] == 15892
    expected = 0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * (417 - 100) / (12023 - 100)))
    assert sample_lr_scale(cfg, saved["frames"], 192) == pytest.approx(expected)
    assert sample_lr_scale(cfg, cfg["sample_budget"], 192) == pytest.approx(0.1)
    same = dict(cfg)
    later = {**saved, "world_size": 6, "update": 500, "frames": saved["frames"] + 83 * 192}
    configure_sample_schedule(same, 6, saved=later, previous={"grounding": cfg})
    assert same["pretrain_updates"] == 15892 and same["sample_budget"] == cfg["sample_budget"]


def test_shrink_requires_explicit_flag_and_keeps_surviving_rank_streams():
    saved = {"world_size": 8, "ranks": [{"source": i, "rng": i} for i in range(8)]}
    with pytest.raises(ValueError, match="same world size"):
        resumed_rank_state(saved, 0, 6)
    for rank in range(6):
        assert (
            resumed_rank_state(saved, rank, 6, allow_world_size_change=True) is saved["ranks"][rank]
        )
    with pytest.raises(ValueError):
        resumed_rank_state(saved, 0, 9, allow_world_size_change=True)
    assert resumed_rank_state(saved, 7, 8) is saved["ranks"][7]


def test_new_run_warmup_and_explicit_extension():
    cfg = {"batch_size": 32, "pretrain_updates": 1000, "warmup_updates": 100}
    configure_sample_schedule(cfg, 8)
    assert sample_lr_scale(cfg, 0, 256) == pytest.approx(0.01)
    assert sample_lr_scale(cfg, 99 * 256, 256) == 1
    assert sample_lr_scale(cfg, 100 * 256, 256) == 1
    saved = {"world_size": 8, "update": 1000, "frames": 1000 * 256}
    extended = dict(cfg)
    configure_sample_schedule(extended, 6, saved=saved, previous={"grounding": cfg}, updates=1100)
    assert extended["pretrain_updates"] == 1100
    assert extended["sample_budget"] == saved["frames"] + 100 * 192
