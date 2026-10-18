"""Validate original-batch probability replay and 20-step recurrent gradients."""

import argparse
import json
from dataclasses import replace
from pathlib import Path

import torch
import yaml

from streamnav.contracts.state import LayerState
from streamnav.models.point_cot.policy import Qwen35PointCoTPolicy
from streamnav.training.pointcot_trainer import (
    cache_copy,
    load_adapters,
    replay_batched_records,
    replay_probability_error,
)
from streamnav.utils.seed import seed_everything


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--inputs", default="runtime/fixtures/navigation/rgb_inputs.pt")
    parser.add_argument("--output", default="runtime/reports/behavior_replay_validation.json")
    args = parser.parse_args()
    config = yaml.safe_load(Path("configs/pointcot.yaml").read_text())
    config["device"] = "cuda:0"
    seed_everything(71)
    policy = Qwen35PointCoTPolicy(config).train()
    load_adapters(policy, args.checkpoint)
    rows = torch.load(args.inputs, weights_only=False)
    instructions = [
        rows[0]["instruction"],
        "Find a large cooker beside the wall.",
        rows[1]["instruction"],
    ]
    states = [policy.start_episode(str(i), text) for i, text in enumerate(instructions)]
    records, resets = [], {}
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        raw = policy.navigation_features([rows[i % len(rows)]["rgb"] for i in range(3)], states)
        for t in range(40):
            features = policy.condition_features(raw, [0, 100 + t, 0 if t % 2 else 700])
            states = [s.state for s in policy.step_batch(features, states, deterministic=True)]
        starts = [cache_copy(s, policy.device) for s in states]
        for t in range(20):
            if t in (7, 13):
                i = 1 if t == 7 else 2
                resets[t, i] = (f"reset-{i}", instructions[i])
                states[i] = policy.start_episode(*resets[t, i])
            features = policy.condition_features(raw, [0, 100 + t, 0 if t % 2 else 700])
            steps = policy.step_batch(features, states, deterministic=[False, True, False])
            records.append(
                [
                    {
                        "features": f.cpu(),
                        "tokens": tuple(x.cpu() for x in (s.apos, s.opos, s.action)),
                        "old_log_prob": s.log_prob.cpu(),
                    }
                    for f, s in zip(features, steps, strict=True)
                ]
            )
            states = [s.state for s in steps]
        replay = replay_batched_records(policy, records, starts, resets=resets)
        inference_errors = [
            replay_probability_error(step, [row[i] for row in records], policy.device)
            for i, step in enumerate(replay)
        ]
    policy.inference_graphs.entries.clear()
    del replay, steps, states
    torch.cuda.empty_cache()
    # The first environment never resets: its last decision must still have
    # a nonzero gradient to the cache at the beginning of this 20-step fragment.
    starts[0] = replace(
        starts[0],
        kda_cache=tuple(
            LayerState(None, s.recurrent.detach().clone().requires_grad_())
            if s.recurrent is not None
            else s
            for s in starts[0].kda_cache
        ),
    )
    with torch.autocast("cuda", dtype=torch.bfloat16):
        replay = replay_batched_records(policy, records, starts, resets=resets)
        gradient_errors = [
            replay_probability_error(step, [row[i] for row in records], policy.device)
            for i, step in enumerate(replay)
        ]
        initial = [s.recurrent for s in starts[0].kda_cache if s.recurrent is not None]
        cache_gradients = torch.autograd.grad(replay[0].log_prob[-1], initial, retain_graph=True)
        cache_gradient = float(
            torch.stack([g.square().sum() for g in cache_gradients]).sum().sqrt()
        )
        loss = -sum(step.log_prob.mean() for step in replay) / len(replay)
    loss.backward()
    norms = {
        name: float(getattr(policy, name).grad.norm())
        for name in ("apos_rows", "opos_rows", "action_rows")
    }
    assert max(inference_errors + gradient_errors) < 1e-4, (inference_errors, gradient_errors)
    assert cache_gradient > 0, cache_gradient
    assert all(torch.isfinite(p.grad).all() for p in policy._all_trainable if p.grad is not None)
    assert all(value > 0 for value in norms.values()), norms
    result = {
        "passed": True,
        "frames": 60,
        "history_steps": 40,
        "inference_log_prob_errors": inference_errors,
        "differentiable_log_prob_errors": gradient_errors,
        "initial_cache_gradient_norm": cache_gradient,
        "token_parameter_gradient_norms": norms,
        "peak_vram_gib": torch.cuda.max_memory_allocated() / 1024**3,
    }
    Path(args.output).write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    torch.set_num_threads(4)
    main()
