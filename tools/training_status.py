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
    parser.add_argument("--root", default="runs/qwen35_pointcot_active")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()
    reports = []
    root = Path(args.root)
    directories = (
        [root]
        if (root / "resolved_config.yaml").exists() or (root / "training.pid").exists()
        else sorted(root.iterdir())
    )
    for directory in directories:
        if not directory.is_dir():
            continue
        train = rows(directory / "train_metrics.jsonl")
        evaluations = rows(directory / "eval_metrics.jsonl")
        grounding_evaluations = rows(directory / "grounding_eval.jsonl")
        health = current_health(directory)
        last = train[-1] if train else {}
        best_file = directory / (
            "best_grounding.json" if grounding_evaluations else "best_evaluation.json"
        )
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
                "total_env_steps": last.get("total_env_steps", last.get("frames", 0)),
                "last_update_seconds": last.get(
                    "update_seconds", last.get("elapsed_seconds", last.get("seconds"))
                ),
                "samples_per_second": last.get("samples_per_second"),
                "navigation_supervision": {
                    k: last.get(k)
                    for k in (
                        "apos_geometry_labels",
                        "apos_geometry_point_labels",
                        "apos_geometry_label_fraction",
                        "apos_loss",
                        "apos_geometry_greedy_accuracy",
                        "apos_geometry_pixel_greedy_accuracy_5pct",
                        "apos_parameter_gradient_norm",
                    )
                }
                if "apos_geometry_labels" in last
                else None,
                "grounding_evaluation": grounding_evaluations[-1]
                if grounding_evaluations
                else None,
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
        if report["navigation_supervision"]:
            print("  APOS geometry:", json.dumps(report["navigation_supervision"]))
        progress = health.get("progress", {})
        if progress:
            print(
                f"  progress: update={progress['update']} phase={progress['phase']} "
                f"{progress['completed']}/{progress['total']} "
                f"age={time.time() - progress['time']:.0f}s"
            )
        best = report["best_evaluation"]
        if best and report["grounding_evaluation"]:
            metric = best.get("selection_metric", "macro_visible_point_accuracy_5pct")
            print(f"  best: update={best['update']} {metric}={best[metric]:.4f}")
        elif best:
            score = best["success"] if "success" in best else best["score"][0]
            print(f"  best: update={best['update']} macro_SR={score:.4f}")
        if report["grounding_evaluation"]:
            evaluation = report["grounding_evaluation"]
            print(
                f"  grounding: update={evaluation['update']} visible_point_accuracy={evaluation['visible_point_accuracy_5pct']:.4f} absence_accuracy={evaluation['absence_accuracy']:.4f} region_accuracy={evaluation['region_accuracy']:.4f}"
            )
            if "visible_grounding_accuracy" in evaluation:
                print(
                    f"  localization: visible_accuracy={evaluation['visible_grounding_accuracy']:.4f} "
                    f"false_absence_rate={evaluation['false_absence_rate']:.4f} "
                    f"validation_loss={evaluation['validation_loss']:.4f}"
                )
        for split, evaluation in report["autonomous_evaluation"].items():
            print(f"  {split}: {json.dumps(evaluation)}")
    subprocess.run(
        ["nvidia-smi", "--query-gpu=index,utilization.gpu,memory.used", "--format=csv,noheader"],
        check=True,
    )


if __name__ == "__main__":
    main()
