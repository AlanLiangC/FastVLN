"""Real-image inference profiling and activation-memory timing, without training writes."""

import argparse
import json
import time
from pathlib import Path

import torch
import yaml

from streamnav.models.point_cot.policy import Qwen35PointCoTPolicy
from streamnav.training.pointcot_trainer import load_adapters


def timed(function, repetitions):
    for _ in range(2):
        function()
    torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(repetitions):
        function()
    torch.cuda.synchronize()
    return (time.perf_counter() - start) / repetitions


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--inputs", default="runtime/fixtures/navigation/rgb_inputs.pt")
    parser.add_argument("--output", default="runtime/reports/navigation_efficiency_baseline.json")
    parser.add_argument("--batches", default="3,6,9")
    parser.add_argument("--iterations", type=int, default=6)
    parser.add_argument("--profile", action="store_true")
    parser.add_argument("--cuda-graphs", action="store_true")
    args = parser.parse_args()
    config = yaml.safe_load((Path(args.checkpoint) / "resolved_config.yaml").read_text())
    config["device"] = "cuda:0"
    config["model"]["inference_cuda_graphs"] = args.cuda_graphs
    policy = Qwen35PointCoTPolicy(config).eval()
    load_adapters(policy, args.checkpoint)
    rows = torch.load(args.inputs, weights_only=False)
    result = {"checkpoint": args.checkpoint, "inference": [], "backward": []}
    with torch.no_grad():
        for count in map(int, args.batches.split(",")):
            batch = [rows[i % len(rows)] for i in range(count)]
            states = [
                policy.start_episode(f"speed-{i}", r["instruction"]) for i, r in enumerate(batch)
            ]
            rgb = [r["rgb"] for r in batch]
            features = policy.navigation_features(rgb, states)
            states = [s.state for s in policy.step_batch(features, states, deterministic=True)]

            def body():
                return policy.step_batch(features, states, deterministic=True)

            def full():
                return policy.step_batch(
                    policy.navigation_features(rgb, states), states, deterministic=True
                )

            row = {
                "batch": count,
                "body_seconds": timed(body, args.iterations),
                "full_seconds": timed(full, args.iterations),
            }
            row["frames_per_second"] = count / row["full_seconds"]
            result["inference"].append(row)
            print(json.dumps(row), flush=True)
            if args.profile and count == 3:
                with torch.profiler.profile(
                    activities=[
                        torch.profiler.ProfilerActivity.CPU,
                        torch.profiler.ProfilerActivity.CUDA,
                    ]
                ) as profile:
                    full()
                result["profile"] = profile.key_averages().table(
                    sort_by="self_cpu_time_total", row_limit=30
                )
                print(result["profile"], flush=True)
        features = features[:3]
        state = states[0]
    policy.train()
    for checkpointing in (True, False):
        policy.settings["gradient_checkpointing"] = checkpointing
        tokens = [(torch.tensor(0, device=policy.device),) * 3] * 20

        def backward():
            policy.zero_grad(set_to_none=True)
            steps = policy.replay_fragment([features[i % 3] for i in range(20)], state, tokens)
            loss = sum(
                torch.nn.functional.cross_entropy(
                    x, torch.zeros(20, device=policy.device, dtype=torch.long)
                )
                for x in (steps.apos_logits, steps.opos_logits, steps.action_logits)
            )
            loss.backward()

        torch.cuda.empty_cache()
        backward()
        torch.cuda.reset_peak_memory_stats()
        row = {
            "checkpointing": checkpointing,
            "fragment_seconds": timed(backward, 3),
            "peak_vram_gib": torch.cuda.max_memory_allocated() / 1024**3,
        }
        result["backward"].append(row)
        print(json.dumps(row), flush=True)
    Path(args.output).write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    torch.set_num_threads(4)
    main()
