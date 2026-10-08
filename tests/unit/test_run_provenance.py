import json

import pytest

from streamnav.training.trainer import EndToEndObjectNavTrainer


def test_existing_run_cannot_be_silently_restarted(tmp_path):
    metrics = tmp_path / "train_metrics.jsonl"
    metrics.write_text(json.dumps({"update": 100}) + "\n")
    config = {"trainer": {}, "run_dir": str(tmp_path)}
    with pytest.raises(ValueError, match="already contains training"):
        EndToEndObjectNavTrainer(config)
    assert json.loads(metrics.read_text())["update"] == 100
    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir()
    (checkpoint / "manifest.json").write_text(json.dumps({"update": 75}))
    with pytest.raises(ValueError, match="Checkpoint differs"):
        EndToEndObjectNavTrainer({**config, "checkpoint": str(checkpoint)})
