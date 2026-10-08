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
        for marker in ("streamnav.training.trainer", "tools/check_ovsegdt_training.py")
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
