"""Measure full RGB-to-action batch throughput for evaluation sizing."""

import argparse
import json
import statistics
import time
from pathlib import Path

import numpy as np
import torch
import yaml

from streamnav.models.point_cot.policy import Qwen35PointCoTPolicy
from streamnav.training.pointcot_trainer import load_adapters


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch", type=int, required=True)
    parser.add_argument("--checkpoint", default="runs/qwen35_pointcot_opos_hint/checkpoints/latest")
    parser.add_argument("--inputs", default="runtime/fixtures/navigation/rgb_inputs.pt")
    parser.add_argument("--warmup", type=int, default=8)
    parser.add_argument("--iterations", type=int, default=40)
    parser.add_argument("--output")
    args = parser.parse_args()
    if args.batch < 1 or args.warmup < 0 or args.iterations < 1:
        parser.error("batch/iterations must be positive and warmup must be nonnegative")
    checkpoint = Path(args.checkpoint).resolve()
    config = yaml.safe_load((checkpoint / "resolved_config.yaml").read_text())
    config["device"] = "cuda:0"
    policy = Qwen35PointCoTPolicy(config).eval()
    load_adapters(policy, checkpoint)
    rows = torch.load(args.inputs, weights_only=False)
    selected = [rows[i % len(rows)] for i in range(args.batch)]
    rgb = torch.stack([r["rgb"] for r in selected])
    hints = [42 if i % 2 else 0 for i in range(args.batch)]
    times = []
    states = [policy.start_episode(str(i), r["instruction"]) for i, r in enumerate(selected)]
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        for iteration in range(args.warmup + args.iterations):
            torch.cuda.synchronize()
            started = time.perf_counter()
            output = policy.forward_batch(rgb, states, hint_tokens=hints, return_perception=True)
            torch.cuda.synchronize()
            if iteration >= args.warmup:
                times.append(time.perf_counter() - started)
            assert len(output[2]) == args.batch
            states = output[2]
    result = {
        "checkpoint": str(checkpoint),
        "inputs": args.inputs,
        "batch": args.batch,
        "warmup": args.warmup,
        "iterations": args.iterations,
        "mean_seconds": statistics.mean(times),
        "median_seconds": statistics.median(times),
        "latency_p50_ms": float(np.percentile(times, 50) * 1000),
        "latency_p95_ms": float(np.percentile(times, 95) * 1000),
        "latency_p99_ms": float(np.percentile(times, 99) * 1000),
        "decisions_per_second": args.batch / statistics.mean(times),
        "peak_vram_bytes": torch.cuda.max_memory_allocated(),
        "gpu": torch.cuda.get_device_name(),
        "parameters": policy.parameter_summary(),
        "scope": "Warm streaming RGB-to-points/action; repeated real frames; no simulator, detector or video cost. Engineering timing, not navigation quality.",
    }
    output = Path(args.output or f"runtime/reports/hint_eval_batch_{args.batch}.json")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result))


if __name__ == "__main__":
    torch.set_num_threads(4)
    main()
