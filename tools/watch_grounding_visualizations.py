"""Backfill retained grounding eval checkpoints and follow the currently running job."""

import argparse
import json
import shutil
import tempfile
import time
from pathlib import Path

import torch
import yaml

from streamnav.data.grounding import GroundingSource
from streamnav.evaluation.grounding_cases import case_record, fixed_anchor_indices, write_gallery
from streamnav.models.point_cot.policy import Qwen35PointCoTPolicy
from streamnav.training.grounding_trainer import validation_indices
from streamnav.training.pointcot_trainer import load_adapters
from streamnav.utils.process_health import training_process_alive


@torch.no_grad()
def render_checkpoint(policy, source, root, result, count):
    update = result["update"]
    checkpoint = root / "checkpoints" / f"update_{update:07d}"
    # Retention can delete older checkpoints while training continues.
    with tempfile.TemporaryDirectory(prefix="grounding-cases-") as temporary:
        for name in ("manifest.json", "adapters.safetensors"):
            shutil.copyfile(checkpoint / name, Path(temporary) / name)
        load_adapters(policy, temporary)
    rows = [
        GroundingSource.materialize(source.rows[i])
        for i in validation_indices(source, result["count"])
    ]
    cases = []
    for index in fixed_anchor_indices(rows, count):
        row = rows[index]
        feature = source.features(policy, row)
        prediction = policy.step(
            feature, policy.start_episode(row["image_id"], row["instruction"]), deterministic=True
        )
        cases.append(case_record(row, prediction, index))
    directory = root / "grounding_visualizations" / f"update_{update:07d}"
    payload = write_gallery(directory, cases, update, result, count=count, anchors=count)
    payload["generation"] = "checkpoint backfill; fixed cases from the same validation subset"
    (directory / "cases.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n")
    print(
        json.dumps({"update": update, "cases": len(cases), "path": str(directory.resolve())}),
        flush=True,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", default="runs/pointcot_grounding")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--once", action="store_true")
    args = parser.parse_args()
    root = Path(args.run_dir)
    config = yaml.safe_load((root / "resolved_config.yaml").read_text())
    config["device"] = args.device
    policy = Qwen35PointCoTPolicy(config).eval()
    source = GroundingSource(config["grounding"]["val_manifest"])
    count = config["grounding"].get("visualization_cases", 24)
    launcher = int((root / "training.pid").read_text())
    final_scan = False
    while True:
        path = root / "grounding_eval.jsonl"
        results = (
            [json.loads(s) for s in path.read_text().splitlines() if s] if path.exists() else []
        )
        for result in results:
            update = result["update"]
            complete = root / "grounding_visualizations" / f"update_{update:07d}" / "index.html"
            if complete.exists() or not (root / "checkpoints" / f"update_{update:07d}").exists():
                continue
            try:
                render_checkpoint(policy, source, root, result, count)
            except FileNotFoundError:
                print(f"Checkpoint {update} removed by retention; skipping", flush=True)
        if args.once:
            break
        if not training_process_alive(launcher):
            if final_scan:
                break
            # The final eval can be appended between reading the log and checking liveness.
            final_scan = True
            continue
        time.sleep(5)


if __name__ == "__main__":
    torch.set_num_threads(2)
    main()
