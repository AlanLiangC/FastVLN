import json

import pytest
import yaml

from streamnav.training.pipeline import completed_grounding


def make_run(root):
    best = root / "checkpoints" / "update_0000003"
    best.mkdir(parents=True)
    (best.parent / "best").symlink_to(best.name)
    (best / "adapters.safetensors").write_bytes(b"adapter")
    (best / "manifest.json").write_text(
        json.dumps({"format": "qwen35-pointcot-adapters-v1", "update": 3})
    )
    (root / "resolved_config.yaml").write_text(
        yaml.safe_dump({"stage": "grounding", "grounding": {"pretrain_updates": 5}})
    )
    (root / "latest_checkpoint.json").write_text(json.dumps({"update": 5}))
    (root / "best_grounding.json").write_text(json.dumps({"update": 3}))
    return best


def test_finished_grounding_uses_best_not_latest(tmp_path):
    best = make_run(tmp_path)
    assert completed_grounding(tmp_path) == best


def test_explicit_stop_does_not_advance_pipeline(tmp_path):
    make_run(tmp_path)
    (tmp_path / "stop_request.json").write_text("{}")
    with pytest.raises(ValueError, match="explicitly stopped"):
        completed_grounding(tmp_path)


def test_incomplete_grounding_does_not_advance_pipeline(tmp_path):
    make_run(tmp_path)
    (tmp_path / "latest_checkpoint.json").write_text(json.dumps({"update": 4}))
    with pytest.raises(ValueError, match="sample budget"):
        completed_grounding(tmp_path)


def test_mismatched_best_checkpoint_does_not_advance_pipeline(tmp_path):
    best = make_run(tmp_path)
    (best / "manifest.json").write_text(
        json.dumps({"format": "qwen35-pointcot-adapters-v1", "update": 4})
    )
    with pytest.raises(ValueError, match="inconsistent"):
        completed_grounding(tmp_path)
