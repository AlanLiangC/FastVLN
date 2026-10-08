"""Read current learner, validation and monitor state without loading a model."""

import argparse
import json
import subprocess
import time
from pathlib import Path

from streamnav.utils.process_health import current_health


def rows(path):
    result = []
    if path.exists():
        for line in path.read_text().splitlines():
            try:
                result.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default="runs/streamnav_active")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()
    reports = []
    for directory in sorted(Path(args.root).iterdir()):
        if not directory.is_dir():
            continue
        train = rows(directory / "train_metrics.jsonl")
        evaluations = rows(directory / "eval_metrics.jsonl")
        health = current_health(directory)
        last = train[-1] if train else {}
        best_file = directory / "best_evaluation.json"
        best = json.loads(best_file.read_text()) if best_file.exists() else None
        latest = {}
        for evaluation in evaluations:
            latest[evaluation["split"]] = {
                k: evaluation.get(k)
                for k in (
                    "update",
                    "episodes",
                    "success",
                    "success_strict_0_1",
                    "oracle_success",
                    "spl",
                )
            }
        reports.append(
            {
                "experiment": directory.name,
                "update": last.get("update", 0),
                "total_env_steps": last.get("total_env_steps", 0),
                "last_update_seconds": last.get("update_seconds"),
                "autonomous_evaluation": latest,
                "best_evaluation": best,
                "health_age_seconds": round(time.time() - health.get("time", 0), 1),
                "health": health,
            }
        )
    if args.json:
        print(json.dumps(reports, indent=2))
        return
    for report in reports:
        health = report["health"]
        workers = health.get("workers", [])
        alive = sum(w.get("alive", False) for w in workers)
        print(
            f"{report['experiment']}: update={report['update']} "
            f"env_steps={report['total_env_steps']} workers={alive}/{health.get('expected_workers', '?')} "
            f"status={health.get('status', 'unknown')} monitor_age={report['health_age_seconds']}s"
        )
        print("  warnings:", ", ".join(health.get("warnings", [])) or "none")
        best = report["best_evaluation"]
        if best:
            print(f"  best: update={best['update']} macro_SR={best['score'][0]:.4f}")
        for split, evaluation in report["autonomous_evaluation"].items():
            print(f"  {split}: {json.dumps(evaluation)}")
    subprocess.run(
        ["nvidia-smi", "--query-gpu=index,utilization.gpu,memory.used", "--format=csv,noheader"],
        check=True,
    )


if __name__ == "__main__":
    main()
