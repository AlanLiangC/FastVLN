"""Measure end-to-end grounding updates and batch/single visual equivalence."""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
import yaml
from PIL import Image

from streamnav.data.grounding import GroundingSource
from streamnav.models.point_cot.policy import Qwen35PointCoTPolicy
from streamnav.training.grounding_trainer import optimizer_for
from streamnav.training.point_supervision import grounding_loss


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--batches", default="8,16,32,64")
    parser.add_argument("--checkpointing", choices=["on", "off"], default="off")
    parser.add_argument("--serial", action="store_true")
    parser.add_argument("--vision-dtype", choices=["float32", "bfloat16"], default="float32")
    parser.add_argument("--output", default="runtime/reports/grounding_benchmark.json")
    args = parser.parse_args()
    config = yaml.safe_load(Path("configs/pointcot.yaml").read_text())
    config["device"] = f"cuda:{args.gpu}"
    config["model"]["vision_compute_dtype"] = args.vision_dtype
    config["model"]["gradient_checkpointing"] = args.checkpointing == "on"
    torch.set_num_threads(4)
    policy = Qwen35PointCoTPolicy(config).train()
    path = Path("runtime/data/grounding/public_v1/coco.train.jsonl")
    if not path.exists():
        path = path.with_name("coco.train.jsonl.writing")
    if not path.exists():
        path = Path("runtime/data/grounding/processed/train.jsonl")
    rows = []
    for line in path.open():
        r = json.loads(line)
        if r.get("opos_targets") and r["opos_targets"] != [0] and r["source"] != "CoSyn-point":
            image = Path(r["image"])
            r["image"] = str(image if image.is_absolute() else path.parent / image)
            rows.append(r)
        if len(rows) >= 128:
            break
    assert len(rows) >= 64
    # Mixed aspect ratios and text lengths must retain per-image spatial RoPE.
    if args.serial:
        GroundingSource.features_many = staticmethod(
            lambda p, rs: [GroundingSource.features(p, r) for r in rs]
        )
    batch = GroundingSource.features_many(policy, rows[:3])
    differences = []
    for row, feature in zip(rows[:3], batch, strict=True):
        with Image.open(row["image"]) as im:
            rgb = torch.from_numpy(np.array(im.convert("RGB")))
        single = policy.frame_features(rgb, row["instruction"])[0]
        assert torch.equal(single.ids, feature.ids) and torch.equal(
            single.positions, feature.positions
        )
        error = (single.visual - feature.visual).abs().max().item()
        differences.append(error)
        # BF16 vision kernels change reduction order with batch shape. Test
        # whole-vector error in addition to exact IDs/RoPE, rather than a near-
        # zero element's relative error. Swapped images fail this decisively.
        relative = (
            single.visual.float() - feature.visual.float()
        ).norm() / single.visual.float().norm()
        assert relative < 0.01, float(relative)
    records = []
    for size in map(int, args.batches.split(",")):
        optimizer = optimizer_for(policy, config["grounding"]["learning_rates"])
        timings = []
        try:
            torch.cuda.reset_peak_memory_stats(policy.device)
            for iteration in range(4):
                optimizer.zero_grad(set_to_none=True)
                torch.cuda.synchronize(policy.device)
                started = time.monotonic()
                loss, _, _ = grounding_loss(policy, rows[:size])
                loss.backward()
                torch.nn.utils.clip_grad_norm_(policy._all_trainable, 1)
                optimizer.step()
                torch.cuda.synchronize(policy.device)
                if iteration:
                    timings.append(time.monotonic() - started)
            seconds = sum(timings) / len(timings)
            record = {
                "batch_size": size,
                "checkpointing": args.checkpointing,
                "seconds": seconds,
                "samples_per_second_per_gpu": size / seconds,
                "peak_vram_gib": torch.cuda.max_memory_allocated(policy.device) / 1024**3,
                "finite_loss": bool(torch.isfinite(loss)),
            }
            records.append(record)
            print(json.dumps(record), flush=True)
        except torch.OutOfMemoryError:
            records.append({"batch_size": size, "checkpointing": args.checkpointing, "oom": True})
            optimizer.zero_grad(set_to_none=True)
            torch.cuda.empty_cache()
            break
    Path(args.output).write_text(
        json.dumps({"visual_batch_single_max_abs_error": differences, "records": records}, indent=2)
        + "\n"
    )


if __name__ == "__main__":
    main()
