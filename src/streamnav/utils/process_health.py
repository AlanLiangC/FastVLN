"""Check live process identity before presenting persisted monitor snapshots."""

import json
import time
from pathlib import Path


def process_alive(pid, marker, parent=None, proc_root=Path("/proc")):
    if not isinstance(pid, int) or pid <= 0:
        return False
    try:
        fields = (proc_root / str(pid) / "stat").read_text().rsplit(")", 1)[1].split()
        command = (proc_root / str(pid) / "cmdline").read_bytes()
        return (
            fields[0] not in {"Z", "X"}
            and marker.encode() in command
            and (parent is None or int(fields[1]) == parent)
        )
    except (OSError, IndexError, ValueError):
        return False


def training_process_alive(pid, parent=None, proc_root=Path("/proc")):
    return any(
        process_alive(pid, marker, parent=parent, proc_root=proc_root)
        for marker in (
            "streamnav.training.trainer",
            "tools/check_ovsegdt_training.py",
            "tools/benchmark_training_recipes.py",
        )
    )


def training_stop_targets(workers, launcher, proc_root=Path("/proc")):
    """One live worker coordinates a healthy group through any_rank(stop).

    Signalling every rank can race with handler removal during shutdown.
    A broken group cannot coordinate, so signal its remaining live workers.
    """
    current = [w for w in workers if w.get("launcher_pid") == launcher or w["pid"] == launcher]
    alive = [
        w
        for w in current
        if training_process_alive(
            w["pid"], parent=launcher if w["pid"] != launcher else None, proc_root=proc_root
        )
    ]
    if alive:
        if len(alive) == len(current):
            return [min(alive, key=lambda w: w.get("rank", 0))["pid"]]
        return [w["pid"] for w in alive]
    return (
        [launcher] if not current and training_process_alive(launcher, proc_root=proc_root) else []
    )


def current_health(run_dir, *, now=None, stale_seconds=120, proc_root=Path("/proc")):
    path = Path(run_dir) / "health_status.json"
    if not path.exists():
        return {"status": "unknown", "warnings": ["monitor_status_missing"], "workers": []}
    result = json.loads(path.read_text())
    result["recorded_status"] = result.get("status", "unknown")
    result["checked_at"] = time.time() if now is None else now
    result["snapshot_age_seconds"] = max(0, result["checked_at"] - result.get("time", 0))
    launcher = result.get("training_pid")
    result["process_alive"] = training_process_alive(launcher, proc_root=proc_root)
    for worker in result.get("workers", []):
        worker["alive"] = training_process_alive(
            worker.get("pid"),
            parent=launcher if worker.get("pid") != launcher else None,
            proc_root=proc_root,
        )
    monitor_file = Path(run_dir) / "monitor.pid"
    try:
        monitor_pid = int(monitor_file.read_text())
    except (OSError, ValueError):
        monitor_pid = None
    result["monitor_alive"] = process_alive(monitor_pid, "monitor_training.py", proc_root=proc_root)
    warnings = result.setdefault("warnings", [])
    if not result["process_alive"] and result["recorded_status"] not in {
        "complete",
        "stopped",
        "halted_for_review",
        "process_exited_early",
    }:
        result["status"] = "process_exited_early"
        warnings.append("training_process_missing_at_query")
    elif result["process_alive"] and (
        not result["monitor_alive"] or result["snapshot_age_seconds"] > stale_seconds
    ):
        result["status"] = "monitor_unavailable"
        warnings.append("monitor_missing_or_stale")
    elif result["process_alive"] and any(not w["alive"] for w in result.get("workers", [])):
        result["status"] = "needs_review"
        warnings.append("training_worker_exited")
    result["warnings"] = list(dict.fromkeys(warnings))
    return result
