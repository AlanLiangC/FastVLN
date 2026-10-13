"""Atomic worker heartbeats shared by training, evaluation and supervision."""

import json
import os
import time
from pathlib import Path


def write_worker_progress(root, update, phase, completed, total, started, *, rank=0, **details):
    root = Path(root)
    data = {
        **details,
        "worker_pid": os.getpid(),
        "rank": rank,
        "time": time.time(),
        "update": update,
        "phase": phase,
        "completed": completed,
        "total": total,
        "elapsed_seconds": time.monotonic() - started,
    }
    directory = root / "progress"
    directory.mkdir(exist_ok=True)
    paths = [directory / f"rank_{rank:03d}.json"]
    if rank == 0:
        paths.append(root / "training_progress.json")
    for path in paths:
        temporary = path.with_suffix(".tmp")
        temporary.write_text(json.dumps(data))
        temporary.replace(path)
