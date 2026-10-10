import json
from types import SimpleNamespace

import pytest
import yaml

from tools import monitor_training


@pytest.mark.parametrize("workers_registered", [False, True])
def test_resume_does_not_signal_launcher_using_previous_worker_snapshot(
    tmp_path, monkeypatch, workers_registered
):
    # Old metrics already satisfy the old stopping rule when resume starts.
    (tmp_path / "resolved_config.yaml").write_text(
        yaml.safe_dump({"trainer": {"supervision": {"min_updates": 50, "zero_sr_patience": 1}}})
    )
    (tmp_path / "train_metrics.jsonl").write_text(json.dumps({"update": 50}) + "\n")
    (tmp_path / "eval_metrics.jsonl").write_text(
        "".join(json.dumps({"update": 50, "split": s, "success": 0}) + "\n" for s in "abc")
    )
    (tmp_path / "training_workers.json").write_text(
        json.dumps([{"pid": 11, "launcher_pid": 10 if workers_registered else 9}])
    )
    monkeypatch.setattr("sys.argv", ["monitor", "--pid", "10", "--run-dir", str(tmp_path)])
    monkeypatch.setattr(monitor_training, "worker_alive", lambda pid: True)
    monkeypatch.setattr(
        monitor_training, "training_stop_targets", lambda workers, pid: [11] if workers else []
    )
    signals = []
    monkeypatch.setattr(monitor_training.os, "kill", lambda pid, sig: signals.append(pid))
    monkeypatch.setattr(
        monitor_training.subprocess,
        "run",
        lambda *args, **kwargs: SimpleNamespace(returncode=0, stdout=""),
    )

    def finish_iteration(_):
        raise StopIteration

    monkeypatch.setattr(monitor_training.time, "sleep", finish_iteration)
    with pytest.raises(StopIteration):
        monitor_training.main()
    snapshot = json.loads((tmp_path / "health_status.json").read_text())
    assert signals == ([11] if workers_registered else [])
    assert snapshot["status"] == ("stopping_for_review" if workers_registered else "starting")
