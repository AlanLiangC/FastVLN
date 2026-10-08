import json
from types import SimpleNamespace

import torch

from streamnav.training.checkpoint import promote_best_checkpoint, save_checkpoint
from streamnav.training.dagger import DaggerBetaScheduler


def test_best_model_requires_positive_sr_and_complete_model_snapshot(tmp_path):
    def result(sr, spl):
        return {"seen": {"success": sr, "spl": spl}, "unseen": {"success": 0, "spl": 0}}

    assert not promote_best_checkpoint(tmp_path, 1, result(0, 0))
    for update in (25, 50, 75):
        target = tmp_path / "checkpoints" / f"update_{update:07d}"
        target.mkdir(parents=True)
        (target / "manifest.json").write_text(json.dumps({"update": update}))
    assert promote_best_checkpoint(tmp_path, 25, result(0.1, 0.05))
    assert not promote_best_checkpoint(tmp_path, 50, result(0.05, 0.04))
    assert (tmp_path / "checkpoints" / "best").resolve().name == "update_0000025"
    assert promote_best_checkpoint(tmp_path, 75, result(0.1, 0.08))
    assert (tmp_path / "checkpoints" / "best").resolve().name == "update_0000075"
    assert json.loads((tmp_path / "best_evaluation.json").read_text())["update"] == 75


def test_checkpoint_retention_protects_best_model(tmp_path):
    class Writer:
        def save_pretrained(self, path):
            path.mkdir(exist_ok=True)

    backbone = torch.nn.Linear(2, 2)
    backbone.config = Writer()
    backbone.tokenizer = Writer()
    policy = SimpleNamespace(backbone=backbone, actor_critic=torch.nn.Linear(2, 2))
    optimizer = torch.optim.AdamW(backbone.parameters())
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda step: 1.0)
    init = tmp_path / "init"
    init.mkdir()
    (init / "kda_layout.json").write_text("{}")
    config = {
        "run_dir": str(tmp_path),
        "model": {"checkpoint": str(init)},
        "trainer": {"keep_checkpoints": 3},
    }
    for update in (1, 25, 50, 75, 100):
        save_checkpoint(policy, optimizer, scheduler, DaggerBetaScheduler(), [], update, config, {})
        if update == 25:
            promote_best_checkpoint(tmp_path, update, {"seen": {"success": 0.1, "spl": 0.05}})
    root = tmp_path / "checkpoints"
    assert sorted(p.name for p in root.glob("update_*")) == [
        "update_0000025",
        "update_0000050",
        "update_0000075",
        "update_0000100",
    ]
    assert (root / "best").resolve().name == "update_0000025"
    assert (root / "latest").resolve().name == "update_0000100"
