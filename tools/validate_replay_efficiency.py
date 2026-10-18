"""Compare cached input features and split teacher gradients on real RGB frames."""

import argparse
import gc
import json
import time
from pathlib import Path

import torch
import yaml
from PIL import Image

from streamnav.models.point_cot.policy import Qwen35PointCoTPolicy
from streamnav.training.pointcot_trainer import (
    cache_copy,
    load_adapters,
    replay_losses,
    replay_records,
    teacher_fragment_losses,
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--inputs", default="runtime/fixtures/navigation/rgb_inputs.pt")
    parser.add_argument("--output", default="runtime/reports/replay_efficiency_validation.json")
    args = parser.parse_args()
    config = yaml.safe_load(Path("configs/pointcot.yaml").read_text())
    config["device"] = "cuda:0"
    policy = Qwen35PointCoTPolicy(config).train()
    load_adapters(policy, args.checkpoint)
    rows = torch.load(args.inputs, weights_only=False)
    images = [Image.fromarray(row["rgb"].numpy()) for row in rows]
    images.append(Image.new("RGB", (128, 440), (120, 40, 70)))
    instructions = [row["instruction"] for row in rows] + ["Find a tall lamp beside a wall."]
    result = {"checkpoint": args.checkpoint, "feature_errors": [], "preprocessing": []}
    with torch.no_grad():
        for amp in (False, True):
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=amp):
                policy.settings["cache_frame_templates"] = False
                expected = policy.frame_features_batch(images, instructions)
                policy.settings["cache_frame_templates"] = True
                cold = policy.frame_features_batch(images, instructions)
                hot = policy.frame_features_batch(images, instructions)
                for original, first, cached in zip(expected, cold, hot, strict=True):
                    for candidate in (first, cached):
                        assert torch.equal(original.ids, candidate.ids)
                        assert torch.equal(original.positions, candidate.positions)
                        error = float((original.visual - candidate.visual).abs().max())
                        assert error < 1e-6, error
                        result["feature_errors"].append(error)
                for enabled in (False, True):
                    policy.settings["cache_frame_templates"] = enabled
                    torch.cuda.synchronize()
                    started = time.monotonic()
                    for _ in range(8):
                        policy.frame_features_batch(images, instructions)
                    torch.cuda.synchronize()
                    result["preprocessing"].append(
                        {
                            "autocast": amp,
                            "cached": enabled,
                            "seconds_per_batch": (time.monotonic() - started) / 8,
                        }
                    )
        states = [policy.start_episode(str(i), text) for i, text in enumerate(instructions)]
        raw = policy.navigation_features(
            [rows[i % len(rows)]["rgb"] for i in range(len(states))], states
        )
        starts = list(states)
        records = []
        with torch.autocast("cuda", dtype=torch.bfloat16):
            for _ in range(20):
                steps = policy.step_batch(raw, states, deterministic=True)
                s = steps[0]
                records.append(
                    {
                        "features": raw[0].cpu(),
                        "tokens": tuple(t.cpu() for t in (s.apos, s.opos, s.action)),
                        "old_log_prob": s.log_prob.cpu(),
                        "expert": 1,
                        "student_action": True,
                        "il_weight": 1.0,
                        "perception_label": {
                            "apos_valid": True,
                            "apos": 42,
                            "opos_valid": True,
                            "opos": 42,
                        },
                        "future_actions": [1, 1, 1, 1],
                    }
                )
                states = [s.state for s in steps]
    policy.inference_graphs.entries.clear()
    del states, steps, expected, cold, hot
    gc.collect()
    torch.cuda.empty_cache()
    state = cache_copy(starts[0], policy.device)
    cfg = config["trainer"]
    adv, returns = torch.ones(20), torch.zeros(20)

    def backward(split):
        policy.zero_grad(set_to_none=True)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            step = replay_records(policy, records, state)
            loss, _, _ = replay_losses(
                policy, state, {}, step, records, adv, returns, cfg, 0.25, include_teacher=not split
            )
            total = float(loss.detach().mean())
        loss.mean().backward()
        del step, loss
        if split:
            with torch.autocast("cuda", dtype=torch.bfloat16):
                loss, _ = teacher_fragment_losses(policy, state, {}, records, cfg)
            total += float(loss.detach().mean())
            loss.mean().backward()
        return total

    gradients, measurements = [], []
    for split in (False, True):
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
        started = time.monotonic()
        loss = backward(split)
        torch.cuda.synchronize()
        gradients.append(
            {
                name: p.grad.detach().cpu().clone()
                for name, p in policy.named_parameters()
                if p.requires_grad and p.grad is not None
            }
        )
        measurements.append(
            {
                "split": split,
                "loss": loss,
                "seconds": time.monotonic() - started,
                "peak_gib": torch.cuda.max_memory_allocated() / 1024**3,
            }
        )
    assert gradients[0].keys() == gradients[1].keys()
    grad_error = max(
        float((value - gradients[1][name]).abs().max()) for name, value in gradients[0].items()
    )
    assert grad_error < 2e-5, grad_error
    assert abs(measurements[0]["loss"] - measurements[1]["loss"]) < 1e-5
    assert measurements[1]["peak_gib"] < measurements[0]["peak_gib"] * 0.8, measurements
    # Preserve 12 GiB of external pressure while running the new gradient path.
    policy.zero_grad(set_to_none=True)
    gc.collect()
    torch.cuda.empty_cache()
    pressure = torch.empty(12 * 1024**3, dtype=torch.uint8, device=policy.device)
    pressure.fill_(0)
    pressure_loss = backward(True)
    del pressure
    result.update(
        passed=True,
        gradients_max_error=grad_error,
        backward=measurements,
        pressure_gib=12,
        pressure_loss=pressure_loss,
        template_hits=policy.frame_template_hits,
        template_misses=policy.frame_template_misses,
    )
    Path(args.output).write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    torch.set_num_threads(4)
    main()
