"""Create a continuation with checkpoint-aligned history, preserving the source run."""

import argparse
import json
import shutil
from datetime import UTC, datetime
from pathlib import Path


def read_rows(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def prepare(source, target):
    source, target = Path(source).resolve(), Path(target).resolve()
    if target.exists():
        raise ValueError(f"Continuation already exists: {target}")
    checkpoint = (source / "checkpoints/latest").resolve(strict=True)
    update = json.loads((checkpoint / "manifest.json").read_text())["update"]
    required = (
        "model.safetensors",
        "actor_critic.safetensors",
        "optimizer.pt",
        "resolved_config.yaml",
        "dagger_scheduler.json",
        "source.zip",
    )
    for name in required:
        if not (checkpoint / name).is_file() or not (checkpoint / name).stat().st_size:
            raise ValueError(f"Incomplete checkpoint: {checkpoint / name}")
    train = read_rows(source / "train_metrics.jsonl")
    inherited = [r for r in train if r["update"] <= update]
    if not inherited or inherited[-1]["update"] != update:
        raise ValueError("Checkpoint has no matching training history")
    evaluations = read_rows(source / "eval_metrics.jsonl")
    best_file = source / "best_evaluation.json"
    best = json.loads(best_file.read_text()) if best_file.exists() else None
    if best and best["update"] > update:
        raise ValueError("Best evaluation is newer than the resume point")
    target.mkdir(parents=True)
    (target / "checkpoints").mkdir()
    (target / "checkpoints/latest").symlink_to(checkpoint, target_is_directory=True)
    if best:
        (target / "checkpoints/best").symlink_to(
            (source / "checkpoints/best").resolve(strict=True), target_is_directory=True
        )
        shutil.copy2(best_file, target / best_file.name)
    for name, rows in (
        ("train_metrics.jsonl", inherited),
        ("eval_metrics.jsonl", [r for r in evaluations if r["update"] <= update]),
    ):
        (target / name).write_text("".join(json.dumps(r) + "\n" for r in rows))
    (target / "evaluation").mkdir()
    for path in (source / "evaluation").glob("update_*"):
        if int(path.name.removeprefix("update_")) <= update:
            (target / "evaluation" / path.name).symlink_to(path.resolve(), target_is_directory=True)
    lineage = {
        "created_utc": datetime.now(UTC).isoformat(),
        "source_run": str(source),
        "source_checkpoint": str(checkpoint),
        "resume_update": update,
        "source_last_logged_update": train[-1]["update"],
        "uncheckpointed_updates_preserved_in_source": [
            r["update"] for r in train if r["update"] > update
        ],
        "inherited_history_through_update": update,
        "note": "Source is unchanged. Optimizer/RNG/sampler resume; simulator episodes restart.",
    }
    (target / "continuation.json").write_text(json.dumps(lineage, indent=2) + "\n")
    return lineage


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True)
    parser.add_argument("--target", required=True)
    args = parser.parse_args()
    print(json.dumps(prepare(args.source, args.target), indent=2))


if __name__ == "__main__":
    main()
