import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from streamnav.evaluation.latency_metrics import latency_summary
from streamnav.models.policy.streaming_policy import StreamingObjectNavPolicy
from streamnav.models.qwen35_kda.backbone import Qwen35KDABackbone
from streamnav.models.qwen35_kda.cache import state_bytes


@torch.no_grad()
def benchmark(args):
    torch.manual_seed(17)
    device = torch.device(args.device)
    image_size = args.image_size or [args.image_height, args.image_width]
    height, width = (image_size, image_size) if isinstance(image_size, int) else image_size
    policy = StreamingObjectNavPolicy(
        Qwen35KDABackbone.from_converted(
            args.checkpoint,
            device=device,
            image_size=image_size,
            inference_mode=args.inference_mode,
        )
    ).eval()
    rgb = torch.randint(0, 256, (height, width, 3), dtype=torch.uint8)
    for _ in range(3):
        state = policy.start_episode("warmup", "Find a chair.")
        policy.forward_step(rgb, state)
    torch.cuda.synchronize(device)
    start = time.perf_counter()
    state = policy.start_episode("benchmark", "Find a chair.")
    torch.cuda.synchronize(device)
    prefill = time.perf_counter() - start
    torch.cuda.reset_peak_memory_stats(device)
    records, total_times, stages = (
        [],
        [],
        {"vision_ms": [], "recurrent_ms": [], "actor_critic_ms": []},
    )
    for step in range(1, args.steps + 1):
        torch.cuda.synchronize(device)
        start = time.perf_counter()
        tokens = policy.backbone.encode_rgb(rgb)
        torch.cuda.synchronize(device)
        encoded = time.perf_counter()
        hidden, cache = policy.backbone.recurrent_forward(tokens, state.kda_cache)
        torch.cuda.synchronize(device)
        recurrent = time.perf_counter()
        logits, _ = policy.actor_critic(policy.pooling(hidden))
        logits.argmax(-1).item()
        torch.cuda.synchronize(device)
        end = time.perf_counter()
        state = state.with_cache(cache)
        total_times.append(end - start)
        stages["vision_ms"].append((encoded - start) * 1000)
        stages["recurrent_ms"].append((recurrent - encoded) * 1000)
        stages["actor_critic_ms"].append((end - recurrent) * 1000)
        record = {
            "step": step,
            "state_bytes": state_bytes(state),
            "latency_ms": (end - start) * 1000,
            "allocated_bytes": torch.cuda.memory_allocated(device),
            "reserved_bytes": torch.cuda.memory_reserved(device),
        }
        records.append(record)
        if step in (10, 50, 100, 250, 500):
            print(json.dumps(record), flush=True)
    bounded = len({r["state_bytes"] for r in records}) == 1
    window = min(50, args.steps // 2)
    growth = float(np.median(total_times[-window:]) / np.median(total_times[:window]))
    result = {
        "checkpoint": str(Path(args.checkpoint).resolve()),
        "device": torch.cuda.get_device_name(device),
        "torch_version": torch.__version__,
        "image_size": image_size,
        "inference_mode": args.inference_mode,
        "steps": args.steps,
        "prefill_ms": prefill * 1000,
        **latency_summary(total_times),
        "bounded_cache": bounded,
        "state_bytes": state_bytes(state),
        "latency_growth_ratio": growth,
        "peak_vram_bytes": torch.cuda.max_memory_allocated(device),
        "stages": {
            name: {
                "p50": float(np.percentile(vals, 50)),
                "p95": float(np.percentile(vals, 95)),
                "p99": float(np.percentile(vals, 99)),
            }
            for name, vals in stages.items()
        },
        "records": records,
    }
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(result, indent=2))
    print(json.dumps({k: v for k, v in result.items() if k != "records"}, indent=2))
    if not bounded:
        raise AssertionError("Recurrent cache grew with episode length")
    if args.max_p95_ms is not None and result["latency_p95_ms"] > args.max_p95_ms:
        raise AssertionError("P95 latency exceeds configured budget")
    if growth > args.max_growth_ratio:
        raise AssertionError("Latency growth exceeds regression budget")
    if args.baseline:
        baseline = json.loads(Path(args.baseline).read_text())
        if baseline["image_size"] != image_size:
            raise ValueError("Baseline resolution differs")
        if result["latency_p95_ms"] > baseline["latency_p95_ms"] * args.baseline_tolerance:
            raise AssertionError("P95 regressed against baseline")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default="checkpoints/qwen35_0p8b_kda")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--image-size",
        type=int,
        help="Legacy square resize; default keeps native rectangular input",
    )
    parser.add_argument("--image-height", type=int, default=270)
    parser.add_argument("--image-width", type=int, default=480)
    parser.add_argument("--inference-mode", choices=["auto", "chunk", "recurrent"], default="auto")
    parser.add_argument("--steps", type=int, default=500)
    parser.add_argument("--output", default="runtime/reports/latency.json")
    parser.add_argument("--max-p95-ms", type=float)
    parser.add_argument("--max-growth-ratio", type=float, default=1.5)
    parser.add_argument("--baseline")
    parser.add_argument("--baseline-tolerance", type=float, default=1.25)
    arguments = parser.parse_args()
    if arguments.steps < 10:
        parser.error("At least 10 steps are required")
    benchmark(arguments)
