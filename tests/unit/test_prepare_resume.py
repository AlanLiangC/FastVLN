import importlib.util
import json
from pathlib import Path

import pytest

spec = importlib.util.spec_from_file_location("prepare_resume", Path("tools/prepare_resume.py"))
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def test_resume_preserves_unsaved_tail_and_inherits_only_saved_history(tmp_path):
    source, target = tmp_path / "original", tmp_path / "continuation"
    checkpoint = source / "checkpoints/update_0000025"
    checkpoint.mkdir(parents=True)
    for name in (
        "model.safetensors",
        "actor_critic.safetensors",
        "optimizer.pt",
        "resolved_config.yaml",
        "dagger_scheduler.json",
        "source.zip",
    ):
        (checkpoint / name).write_text("fixture")
    (checkpoint / "manifest.json").write_text('{"update": 25}')
    (source / "checkpoints/latest").symlink_to(checkpoint)
    (source / "checkpoints/best").symlink_to(checkpoint)
    (source / "best_evaluation.json").write_text('{"update": 25, "score": [0.1, 0.05]}')
    log = source / "train_metrics.jsonl"
    log.write_text("".join(json.dumps({"update": u}) + "\n" for u in (24, 25, 26)))
    original_bytes = log.read_bytes()
    (source / "eval_metrics.jsonl").write_text('{"update": 25}\n')
    evaluation = source / "evaluation/update_0000025"
    evaluation.mkdir(parents=True)
    lineage = module.prepare(source, target)
    assert log.read_bytes() == original_bytes
    assert lineage["uncheckpointed_updates_preserved_in_source"] == [26]
    assert [r["update"] for r in module.read_rows(target / log.name)] == [24, 25]
    assert (target / "checkpoints/best").resolve() == checkpoint
    assert (target / "evaluation/update_0000025").resolve() == evaluation
    with pytest.raises(ValueError, match="already exists"):
        module.prepare(source, target)
