import json
from types import SimpleNamespace

import pytest
import yaml

from tools import monitor_training


@pytest.mark.parametrize("workers_registered", [False, True])
@pytest.mark.parametrize("recipe_current", [False, True])
def test_resume_does_not_signal_launcher_using_previous_worker_snapshot(
    tmp_path, monkeypatch, workers_registered, recipe_current
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
    if recipe_current:
        (tmp_path / "resolved_config.yaml").write_text(
            yaml.safe_dump(
                {
                    "trainer": {"supervision": {"min_updates": 50, "zero_sr_patience": 1}},
                    "source_snapshot": {"sha256": "current"},
                }
            )
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
    assert signals == ([11] if workers_registered and recipe_current else [])
    assert snapshot["status"] == (
        "stopping_for_review" if workers_registered and recipe_current else "starting"
    )


def test_evaluation_heartbeat_from_other_active_rank_is_not_ignored(tmp_path):
    progress = tmp_path / "progress"
    progress.mkdir()
    (tmp_path / "training_progress.json").write_text(
        json.dumps({"worker_pid": 11, "time": 100, "phase": "evaluation_wait"})
    )
    (progress / "rank_001.json").write_text(
        json.dumps({"worker_pid": 12, "time": 990, "phase": "evaluation"})
    )
    # A previous launch and a corrupted/incomplete file must not count as progress.
    (progress / "rank_002.json").write_text(json.dumps({"worker_pid": 13, "time": 999}))
    (progress / "rank_003.json").write_text("{")
    selected = monitor_training.latest_worker_progress(tmp_path, [{"pid": 11}, {"pid": 12}], 1000)
    assert selected["worker_pid"] == 12
    assert selected["time"] == 990


@pytest.mark.parametrize("timestamp", [float("nan"), float("inf"), 1010])
def test_invalid_or_future_heartbeats_cannot_hide_a_stall(tmp_path, timestamp):
    (tmp_path / "training_progress.json").write_text(
        json.dumps({"worker_pid": 11, "time": timestamp})
    )
    assert monitor_training.latest_worker_progress(tmp_path, [{"pid": 11}], 1000) == {}


def test_manual_checkpoint_stop_does_not_report_expected_worker_exit_as_failure(
    tmp_path, monkeypatch
):
    (tmp_path / "training_workers.json").write_text(
        json.dumps([{"pid": 11, "launcher_pid": 10}, {"pid": 12, "launcher_pid": 10}])
    )
    (tmp_path / "stop_request.json").write_text(json.dumps({"training_pid": 10}))
    monkeypatch.setattr("sys.argv", ["monitor", "--pid", "10", "--run-dir", str(tmp_path)])
    monkeypatch.setattr(monitor_training, "worker_alive", lambda pid: pid != 11)
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
    report = json.loads((tmp_path / "health_status.json").read_text())
    assert report["status"] == "stopping"
    assert "training_worker_exited" not in report["warnings"]
    assert signals == []
