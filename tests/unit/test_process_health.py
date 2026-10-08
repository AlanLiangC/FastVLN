import json

from streamnav.utils.process_health import current_health, process_alive


def fake_process(root, pid, command, parent=1, state="S"):
    path = root / str(pid)
    path.mkdir(parents=True, exist_ok=True)
    (path / "stat").write_text(f"{pid} (python worker) {state} {parent} 0 0\n")
    (path / "cmdline").write_bytes(command.encode())


def status(tmp_path):
    run = tmp_path / "run"
    run.mkdir()
    (run / "health_status.json").write_text(
        json.dumps(
            {
                "time": 1000,
                "status": "training",
                "process_alive": True,
                "training_pid": 10,
                "warnings": [],
                "workers": [{"pid": 11, "launcher_pid": 10, "alive": True}],
            }
        )
    )
    (run / "monitor.pid").write_text("12")
    return run


def test_dead_container_snapshot_cannot_claim_running(tmp_path):
    run = status(tmp_path)
    report = current_health(run, now=1100, proc_root=tmp_path / "proc")
    assert report["status"] == "process_exited_early"
    assert not report["process_alive"]
    assert not report["workers"][0]["alive"]
    assert json.loads((run / "health_status.json").read_text())["process_alive"]


def test_pid_reuse_and_zombies_are_not_alive(tmp_path):
    fake_process(tmp_path, 10, "unrelated-program")
    assert not process_alive(10, "streamnav.training.trainer", proc_root=tmp_path)
    fake_process(tmp_path, 10, "streamnav.training.trainer", state="Z")
    assert not process_alive(10, "streamnav.training.trainer", proc_root=tmp_path)


def test_stale_monitor_and_wrong_worker_parent(tmp_path):
    run = status(tmp_path)
    proc = tmp_path / "proc"
    fake_process(proc, 10, "streamnav.training.trainer")
    fake_process(proc, 11, "streamnav.training.trainer", parent=10)
    fake_process(proc, 12, "tools/monitor_training.py")
    assert current_health(run, now=1050, proc_root=proc)["status"] == "training"
    assert current_health(run, now=1201, proc_root=proc)["status"] == "monitor_unavailable"
    fake_process(proc, 11, "streamnav.training.trainer", parent=99)
    assert current_health(run, now=1050, proc_root=proc)["status"] == "needs_review"
