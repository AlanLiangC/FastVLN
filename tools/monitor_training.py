"""Monitor learning and GPU health; gracefully stop a persistently failed learner."""

import argparse
import json
import os
import signal
import subprocess
import time
from pathlib import Path

import yaml

from streamnav.training.health import learning_health
from streamnav.utils.process_health import process_alive


def read_rows(path):
    if not path.exists():
        return []
    rows = []
    for line in path.read_text().splitlines():
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            # The learner may still be writing the last line.
            continue
    return rows


def write_status(path, data):
    temp = path.with_suffix(".tmp")
    temp.write_text(json.dumps(data, indent=2))
    temp.replace(path)


def worker_alive(pid):
    return process_alive(pid, "streamnav.training.trainer")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pid", type=int, required=True)
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--interval", type=float, default=30)
    args = parser.parse_args()
    low_since = {}
    output = Path(args.run_dir)
    stopped_for_health = False
    start_time = time.time()
    while True:
        alive = process_alive(args.pid, "streamnav.training.trainer")
        now = time.time()
        rows = read_rows(output / "train_metrics.jsonl")
        evaluations = read_rows(output / "eval_metrics.jsonl")
        cfg_path = output / "resolved_config.yaml"
        cfg = (yaml.safe_load(cfg_path.read_text()) or {}) if cfg_path.exists() else {}
        supervision = cfg.get("trainer", {}).get("supervision", {})
        health = learning_health(
            rows,
            evaluations,
            expected_splits=len(cfg.get("eval", {}).get("manifests", [1, 2, 3])),
            min_updates=supervision.get("min_updates", 500),
            patience=supervision.get("zero_sr_patience", 5),
        )
        health.update(time=now, training_pid=args.pid, process_alive=alive)
        workers_file = output / "training_workers.json"
        workers = json.loads(workers_file.read_text()) if workers_file.exists() else []
        # A resumed job may still have the previous launcher's worker file.
        workers = [w for w in workers if w.get("launcher_pid") == args.pid or w["pid"] == args.pid]
        health["workers"] = [{**w, "alive": worker_alive(w["pid"])} for w in workers]
        health["expected_workers"] = cfg.get("distributed", {}).get("world_size", 1)
        if alive and workers and not all(w["alive"] for w in health["workers"]):
            health["warnings"].append("training_worker_exited")
            health["stop_recommended"] = True
        if alive:
            age = now - max(
                start_time, (output / "train_metrics.jsonl").stat().st_mtime if rows else start_time
            )
            health["seconds_since_update"] = age
            if age > supervision.get("stale_seconds", 1800):
                health["warnings"].append("training_has_stopped_advancing")
                health["stop_recommended"] = True
        if not alive:
            target = cfg.get("trainer", {}).get("num_updates", 10000)
            health["status"] = (
                "complete"
                if health["update"] >= target
                else ("halted_for_review" if stopped_for_health else "process_exited_early")
            )
            write_status(output / "health_status.json", health)
            print(json.dumps({"training_exit": health}), flush=True)
            return
        if (
            health["stop_recommended"]
            and not stopped_for_health
            and supervision.get("stop_on_failure", True)
        ):
            cmd = Path(f"/proc/{args.pid}/cmdline")
            if cmd.exists() and b"streamnav.training.trainer" in cmd.read_bytes():
                # Signal workers, not torchrun: torchrun escalates to SIGKILL
                # after a short timeout, possibly interrupting checkpoint I/O.
                targets = [w["pid"] for w in workers] or [args.pid]
                for pid in targets:
                    command = Path(f"/proc/{pid}/cmdline")
                    if command.exists() and b"streamnav.training.trainer" in command.read_bytes():
                        try:
                            os.kill(pid, signal.SIGTERM)
                        except ProcessLookupError:
                            pass
                stopped_for_health = True
                print(json.dumps({"graceful_stop_requested": health}), flush=True)
        health["graceful_stop_requested"] = stopped_for_health
        if stopped_for_health:
            health["status"] = "stopping_for_review"
        write_status(output / "health_status.json", health)
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=index,utilization.gpu,memory.used,power.draw",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
        )
        records = []
        if result.returncode == 0:
            for line in result.stdout.strip().splitlines():
                gpu, utilization, memory, power = [float(x.strip()) for x in line.split(",")]
                gpu = int(gpu)
                if utilization >= 50:
                    low_since[gpu] = now
                low_since.setdefault(gpu, now)
                seconds = now - low_since[gpu]
                records.append(
                    {
                        "gpu": gpu,
                        "utilization": utilization,
                        "memory_mib": memory,
                        "power_w": power,
                        "continuous_low_seconds": seconds,
                        "low_utilization_alert": seconds >= 9000,
                    }
                )
        record = {"time": now, "training_pid": args.pid, "gpus": records}
        with open(output / "gpu_metrics.jsonl", "a") as f:
            f.write(json.dumps(record) + "\n")
        write_status(output / "gpu_status.json", record)
        time.sleep(args.interval)


if __name__ == "__main__":
    main()
