"""Monitor learning and GPU health; gracefully stop a persistently failed learner."""

import argparse
import json
import math
import os
import signal
import subprocess
import time
from pathlib import Path

import yaml

from streamnav.training.health import grounding_learning_health, learning_health
from streamnav.utils.process_health import training_process_alive, training_stop_targets


def read_rows(path):
    if not path.exists():
        return []
    rows = []
    for line in path.read_text().splitlines():
        try:
            row = json.loads(line)
            if isinstance(row, dict):
                rows.append(row)
        except json.JSONDecodeError:
            # The learner may still be writing the last line.
            continue
    return rows


def write_status(path, data):
    temp = path.with_suffix(".tmp")
    temp.write_text(json.dumps(data, indent=2))
    temp.replace(path)


def worker_alive(pid):
    return training_process_alive(pid)


def read_json(path, default):
    try:
        value = json.loads(path.read_text())
        return value if isinstance(value, type(default)) else default
    except (FileNotFoundError, json.JSONDecodeError):
        # Worker registration may still be writing; never use another launch's PIDs.
        return default


def latest_worker_progress(output, workers, now):
    pids = {w["pid"] for w in workers}
    candidates = []
    for path in [
        output / "training_progress.json",
        *sorted((output / "progress").glob("rank_*.json")),
    ]:
        data = read_json(path, {})
        timestamp = data.get("time")
        if (
            data.get("worker_pid") in pids
            and isinstance(timestamp, (int, float))
            and math.isfinite(timestamp)
            and timestamp <= now + 5
        ):
            candidates.append(data)
    return max(candidates, key=lambda row: row["time"]) if candidates else {}


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
        alive = worker_alive(args.pid)
        now = time.time()
        rows = read_rows(output / "train_metrics.jsonl")
        evaluations = read_rows(output / "eval_metrics.jsonl")
        cfg_path = output / "resolved_config.yaml"
        try:
            cfg = yaml.safe_load(cfg_path.read_text()) or {}
        except (FileNotFoundError, yaml.YAMLError):
            cfg = {}
        supervision = cfg.get("trainer", {}).get("supervision", {})
        if cfg.get("stage") == "grounding":
            health = grounding_learning_health(rows, read_rows(output / "grounding_eval.jsonl"))
        else:
            health = learning_health(
                rows,
                evaluations,
                expected_splits=len(cfg.get("eval", {}).get("manifests", [1, 2, 3])),
                min_updates=supervision.get("min_updates", 500),
                patience=supervision.get("zero_sr_patience", 5),
            )
        health.update(time=now, training_pid=args.pid, process_alive=alive)
        manual_stop = read_json(output / "stop_request.json", {}).get("training_pid") == args.pid
        workers_file = output / "training_workers.json"
        workers = read_json(workers_file, [])
        # A resumed job may still have the previous launcher's worker file.
        workers = [w for w in workers if w.get("launcher_pid") == args.pid or w["pid"] == args.pid]
        health["workers"] = [{**w, "alive": worker_alive(w["pid"])} for w in workers]
        progress = latest_worker_progress(output, workers, now)
        if progress:
            health["progress"] = progress
            if health["status"] == "training" and progress.get("phase", "").startswith(
                ("evaluation", "grounding_eval")
            ):
                health["status"] = "evaluating"
        health["expected_workers"] = cfg.get("distributed", {}).get("world_size", 1)
        if alive and not workers:
            # During resume, config/metrics still belong to the prior run until
            # this launcher registers its workers. Do not terminate torchrun
            # using the old run's stopping rule before initialization finishes.
            health["status"] = "starting"
        if alive and workers and not manual_stop and not all(w["alive"] for w in health["workers"]):
            health["warnings"].append("training_worker_exited")
            health["stop_recommended"] = True
        if alive:
            age = now - max(
                start_time, (output / "train_metrics.jsonl").stat().st_mtime if rows else start_time
            )
            health["seconds_since_update"] = age
            progress_time = health.get("progress", {}).get("time", 0)
            progress_age = max(0, min(age, now - progress_time))
            health["seconds_since_progress"] = progress_age
            if progress_age > supervision.get("stale_seconds", 1800):
                health["warnings"].append("training_has_stopped_advancing")
                health["stop_recommended"] = True
        if not alive:
            target = (
                cfg["grounding"]["pretrain_updates"]
                if cfg.get("stage") == "grounding"
                else cfg.get("trainer", {}).get("num_updates", 10000)
            )
            health["status"] = (
                "complete"
                if health["update"] >= target
                else (
                    "halted_for_review"
                    if stopped_for_health
                    else ("stopped" if manual_stop else "process_exited_early")
                )
            )
            write_status(output / "health_status.json", health)
            print(json.dumps({"training_exit": health}), flush=True)
            return
        if (
            health["stop_recommended"]
            and not manual_stop
            and not stopped_for_health
            and (workers or "training_has_stopped_advancing" in health["warnings"])
            and supervision.get("stop_on_failure", True)
        ):
            if worker_alive(args.pid):
                # Signal workers, not torchrun: torchrun escalates to SIGKILL
                # after a short timeout, possibly interrupting checkpoint I/O.
                targets = training_stop_targets(workers, args.pid)
                for pid in targets:
                    if worker_alive(pid):
                        try:
                            os.kill(pid, signal.SIGTERM)
                        except ProcessLookupError:
                            pass
                stopped_for_health = True
                print(json.dumps({"graceful_stop_requested": health}), flush=True)
        health["graceful_stop_requested"] = stopped_for_health or manual_stop
        if manual_stop:
            health["status"] = "stopping"
        elif stopped_for_health:
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
