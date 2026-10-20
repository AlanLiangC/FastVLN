"""Run grounding and hand its best checkpoint to navigation after normal completion."""

import argparse
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

import yaml

from streamnav.utils.process_health import training_process_alive, training_stop_targets


def completed_grounding(root):
    """An interrupted or incomplete run must never start navigation."""
    root = Path(root)
    if (root / "stop_request.json").exists():
        raise ValueError("Grounding was explicitly stopped")
    config = yaml.safe_load((root / "resolved_config.yaml").read_text())
    latest = json.loads((root / "latest_checkpoint.json").read_text())
    if (
        config.get("stage") != "grounding"
        or latest["update"] < config["grounding"]["pretrain_updates"]
    ):
        raise ValueError("Grounding did not finish its configured sample budget")
    best = root / "checkpoints" / "best"
    result = json.loads((root / "best_grounding.json").read_text())
    manifest = json.loads((best / "manifest.json").read_text())
    if (
        manifest.get("format") != "qwen35-pointcot-adapters-v1"
        or manifest["update"] != result["update"]
    ):
        raise ValueError("Grounding best checkpoint metadata is inconsistent")
    if not (best / "adapters.safetensors").is_file():
        raise ValueError("Grounding best weights are missing")
    return best.resolve()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/pointcot_temporal.yaml")
    parser.add_argument("--run-dir", default="runs/pointcot_pipeline")
    parser.add_argument("--grounding-run", default="runs/pointcot_grounding_refined")
    parser.add_argument("--navigation-run", default="runs/qwen35_pointcot_temporal")
    parser.add_argument("--world-size", type=int, default=1)
    args = parser.parse_args()
    root, grounding, navigation = map(Path, (args.run_dir, args.grounding_run, args.navigation_run))
    for directory in (grounding, navigation):
        pid_file = directory / "training.pid"
        if pid_file.exists() and training_process_alive(int(pid_file.read_text())):
            raise RuntimeError(f"Training already running: {directory}")
        if (directory / "checkpoints" / "latest").exists():
            raise ValueError(f"Use a new run directory or explicitly resume: {directory}")
    if args.world_size < 1:
        raise ValueError("world-size must be positive")
    root.mkdir(parents=True, exist_ok=True)
    grounding.mkdir(parents=True, exist_ok=True)

    def status(stage, **detail):
        record = {
            "stage": stage,
            "time": time.time(),
            "grounding_run": str(grounding.resolve()),
            "navigation_run": str(navigation.resolve()),
            "world_size": args.world_size,
            **detail,
        }
        temporary = root / ".status-writing.json"
        temporary.write_text(json.dumps(record, indent=2) + "\n")
        temporary.replace(root / "status.json")
        print(json.dumps(record), flush=True)

    stopped = [False]
    process = None

    def request_stop(signum, frame):
        stopped[0] = True
        if process is not None and process.poll() is None:
            workers_file = grounding / "training_workers.json"
            workers = json.loads(workers_file.read_text()) if workers_file.exists() else []
            for pid in training_stop_targets(workers, process.pid):
                try:
                    os.kill(pid, signal.SIGTERM)
                except ProcessLookupError:
                    pass

    signal.signal(signal.SIGTERM, request_stop)
    signal.signal(signal.SIGINT, request_stop)
    command = [
        sys.executable,
        "-m",
        "torch.distributed.run",
        "--standalone",
        "--nnodes=1",
        f"--nproc_per_node={args.world_size}",
        "--module",
        "streamnav.training.grounding_trainer",
        "--config",
        args.config,
        "--run-dir",
        str(grounding),
    ]
    with (grounding / "training.log").open("a") as log:
        process = subprocess.Popen(
            command,
            stdin=subprocess.DEVNULL,
            stdout=log,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        (grounding / "training.pid").write_text(str(process.pid) + "\n")
        status("grounding", training_pid=process.pid)
        returncode = process.wait()
    if stopped[0] or (grounding / "stop_request.json").exists():
        status("stopped", returncode=returncode)
        return
    if returncode:
        status("failed", returncode=returncode)
        raise RuntimeError(f"Grounding exited with status {returncode}")
    try:
        checkpoint = completed_grounding(grounding)
        if stopped[0] or (grounding / "stop_request.json").exists():
            status("stopped", returncode=returncode)
            return
        environment = dict(
            os.environ,
            STREAMNAV_RUN_DIR=str(navigation.resolve()),
            STREAMNAV_WORLD_SIZE=str(args.world_size),
        )
        subprocess.run(
            [
                "bash",
                "scripts/start_pointcot_training.sh",
                "--config",
                args.config,
                "--initialize-from",
                str(checkpoint),
                "--skip-initial-eval",
            ],
            env=environment,
            check=True,
        )
        status(
            "navigation_started",
            checkpoint=str(checkpoint),
            training_pid=int((navigation / "training.pid").read_text()),
        )
    except Exception as exc:
        status("failed", reason=str(exc))
        raise


if __name__ == "__main__":
    main()
