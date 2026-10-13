import json

import numpy as np
import pytest

from streamnav.contracts.perception import decode_pixel
from streamnav.data.grounding import GroundingSource, check_grounding_split, validate_record
from streamnav.data.grounding_preparation import (
    box_tokens,
    goal_instruction,
    group_split,
    mask_point,
)


def test_quantized_point_is_inside_concave_visible_mask():
    mask = np.zeros((270, 480), bool)
    mask[20:250, 20:80] = True
    mask[180:250, 20:400] = True
    assert not mask[135, 240]  # A box center would supervise the background.
    token = mask_point(mask)
    u, v = decode_pixel(token, 480, 270)
    assert mask[int(v), int(u)]
    assert mask_point(np.zeros_like(mask)) is None


def test_regions_remain_weak_and_expand_only_when_sampled(tmp_path):
    row = {
        "image_id": "x",
        "image": "x.jpg",
        "instruction": "Find a chair.",
        "source": "boxes",
        "supervision": "region",
        "opos_targets": [],
        "opos_boxes": [[0.1, 0.2, 0.4, 0.6]],
    }
    assert validate_record(row) is row
    path = tmp_path / "rows.jsonl"
    path.write_text(json.dumps(row) + "\n")
    source = GroundingSource(path)
    assert source.rows[0]["opos_targets"] == []
    expanded = source.sample(1)[0]
    assert expanded["opos_targets"] and 0 not in expanded["opos_targets"]
    for token in expanded["opos_targets"]:
        u, v = decode_pixel(token, 1, 1)
        assert 0.1 <= u < 0.4 and 0.2 <= v < 0.6
    assert box_tokens([[0, 0, 1, 1]], 1, 1) == list(range(1, 1297))
    with pytest.raises(ValueError):
        validate_record({**row, "opos_boxes": [[0.1, 0.2, 2, 0.6]]})


def test_absence_balance_and_resume_do_not_follow_raw_negative_ratio(tmp_path):
    rows = [
        {
            "image_id": str(i),
            "image": "x.jpg",
            "instruction": "Find a chair.",
            "source": "real",
            "opos_targets": [0] if i < 90 else [100],
        }
        for i in range(100)
    ]
    path = tmp_path / "rows.jsonl"
    path.write_text("".join(json.dumps(r) + "\n" for r in rows))
    source = GroundingSource(path, seed=2, source_weights={"real": 1}, absence_probability=0.1)
    state = source.state_dict()
    samples = source.sample(2000)
    assert 150 < sum(r["opos_targets"] == [0] for r in samples) < 250
    source.load_state_dict(state)
    assert source.sample(2000) == samples


def test_related_scans_cannot_cross_validation(tmp_path):
    sources = []
    for split in ("train", "val"):
        path = tmp_path / (split + ".jsonl")
        path.write_text(
            json.dumps(
                {
                    "image_id": split,
                    "image": "x.jpg",
                    "instruction": "Find a chair.",
                    "opos_targets": [5],
                    "split_group": "scene0000",
                }
            )
            + "\n"
        )
        sources.append(GroundingSource(path))
    with pytest.raises(ValueError, match="physical spaces"):
        check_grounding_split(*sources)
    assert group_split("scene0000") == group_split("scene0000")


def test_grounding_goals_match_navigation_and_preserve_referring_text():
    assert goal_instruction("chair") == "Find a chair."
    assert (
        goal_instruction("the chair on the left.", referring=True) == "Find the chair on the left."
    )
    assert goal_instruction("chair on the left", referring=True) == "Find the chair on the left."


def test_weak_regions_cannot_inflate_strong_point_validation(monkeypatch):
    from types import SimpleNamespace

    import torch

    from streamnav.training.grounding_trainer import validate_grounding

    rows = [
        {
            "image_id": str(i),
            "image": "unused",
            "instruction": str(i),
            "source": s,
            "opos_targets": t,
            **extra,
        }
        for i, (s, t, extra) in enumerate(
            [
                ("points", [1], {}),
                ("boxes", [], {"supervision": "region", "opos_boxes": [[0, 0, 1, 1]]}),
                ("points", [0], {}),
            ]
        )
    ]

    class Policy:
        def start_episode(self, *args):
            return None

        def step(self, feature, state, deterministic):
            return SimpleNamespace(
                apos=torch.tensor(0), opos=torch.tensor(0 if feature == "2" else 1296)
            )

    monkeypatch.setattr(GroundingSource, "features", lambda p, r: r["image_id"])
    result = validate_grounding(Policy(), SimpleNamespace(rows=rows, order=[0, 1, 2]), 3)
    assert result["visible_count"] == 1 and result["visible_point_accuracy_5pct"] == 0
    assert result["region_count"] == 1 and result["region_accuracy"] == 1
    assert result["absence_accuracy"] == 1 and result["macro_visible_point_accuracy_5pct"] == 0
