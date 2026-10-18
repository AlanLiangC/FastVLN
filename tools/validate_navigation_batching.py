"""Real-frame GPU acceptance and timings for independent streaming batches."""

import argparse
import json
import time
from pathlib import Path

import torch
import yaml
from torch.nn import functional as F

from streamnav.models.point_cot.policy import Qwen35PointCoTPolicy
from streamnav.training.point_supervision import supervised_fragment, targets_from_label
from streamnav.training.pointcot_trainer import load_adapters
from streamnav.utils.seed import seed_everything


def benchmark(function, repetitions=10):
    for _ in range(3):
        function()
    torch.cuda.synchronize()
    started = time.perf_counter()
    for _ in range(repetitions):
        function()
    torch.cuda.synchronize()
    return (time.perf_counter() - started) / repetitions


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--inputs", default="runtime/fixtures/navigation/rgb_inputs.pt")
    parser.add_argument("--output", default="runtime/reports/navigation_batching_validation.json")
    args = parser.parse_args()
    config = yaml.safe_load(Path("configs/pointcot.yaml").read_text())
    seed_everything(config["seed"])
    policy = Qwen35PointCoTPolicy(config).eval()
    load_adapters(policy, args.checkpoint, allow_new_plan=True)
    rows = torch.load(args.inputs, weights_only=False)
    states = [policy.start_episode(r["episode_id"], r["instruction"]) for r in rows]
    errors, replay_errors, token_matches = [], [], []
    with torch.no_grad():
        features = [policy.frame_features(r["rgb"], r["instruction"])[0] for r in rows]
        # Different prompt lengths and mixed fresh/persistent caches are essential:
        # fixed-length fresh-frame tests cannot detect padding or cache leakage.
        instructions = [
            rows[0]["instruction"],
            "Find a large cooker beside the white kitchen wall.",
        ]
        mixed_states = [policy.start_episode(f"batch-{i}", s) for i, s in enumerate(instructions)]
        mixed_features = policy.navigation_features([r["rgb"] for r in rows[:2]], mixed_states)
        assert mixed_features[0].ids.shape[-1] != mixed_features[1].ids.shape[-1]
        initial = list(mixed_states)
        fragments = [[], []]
        fragment_tokens = [[], []]
        old_log_probs = [[], []]
        for t in range(4):
            if t == 2:
                mixed_states[0] = policy.start_episode("batch-reset", instructions[0])
            serial = [
                policy.step(f, s, deterministic=True)
                for f, s in zip(mixed_features, mixed_states, strict=True)
            ]
            forced = [(s.apos, s.opos, s.action) for s in serial]
            batched = policy.step_batch(mixed_features, mixed_states, tokens=forced)
            greedy = policy.step_batch(mixed_features, mixed_states, deterministic=True)
            for i, (expected, actual, predicted) in enumerate(
                zip(serial, batched, greedy, strict=True)
            ):
                for name in ("apos_logits", "opos_logits", "action_logits", "value"):
                    errors.append(
                        float((getattr(expected, name) - getattr(actual, name)).abs().max())
                    )
                errors.append(float((expected.log_prob - actual.log_prob).abs()))
                token_matches.append(
                    all(
                        int(getattr(expected, name)) == int(getattr(predicted, name))
                        for name in ("apos", "opos", "action")
                    )
                )
                fragments[i].append(mixed_features[i])
                fragment_tokens[i].append(forced[i])
                old_log_probs[i].append(actual.log_prob)
            mixed_states = [s.state for s in batched]
        for i in range(2):
            replay = policy.replay_fragment(
                fragments[i],
                initial[i],
                fragment_tokens[i],
                resets={2: ("batch-reset", instructions[0])} if i == 0 else None,
            )
            replay_errors.append(
                float((replay.log_prob - torch.stack(old_log_probs[i])).abs().max())
            )
        assert max(errors) < config["trainer"]["replay_tolerance"], errors
        assert max(replay_errors) < config["trainer"]["replay_tolerance"], replay_errors
        assert all(token_matches), token_matches
        # Batch permutation may only permute results, including mixed reset states.
        ordered = policy.step_batch(mixed_features, mixed_states, deterministic=True)
        reversed_steps = policy.step_batch(
            mixed_features[::-1], mixed_states[::-1], deterministic=True
        )[::-1]
        permutation_error = max(
            float((a.opos_logits - b.opos_logits).abs().max())
            for a, b in zip(ordered, reversed_steps, strict=True)
        )
        assert permutation_error < 0.05
        serial_body = benchmark(
            lambda: [
                policy.step(f, s, deterministic=True)
                for f, s in zip(features[:2], states[:2], strict=True)
            ]
        )
        batch_body = benchmark(
            lambda: policy.step_batch(features[:2], states[:2], deterministic=True)
        )
        serial_full = benchmark(
            lambda: [
                policy.step(
                    policy.frame_features(r["rgb"], r["instruction"])[0], s, deterministic=True
                )
                for r, s in zip(rows[:2], states[:2], strict=True)
            ]
        )
        batch_full = benchmark(
            lambda: policy.step_batch(
                policy.navigation_features([r["rgb"] for r in rows[:2]], states[:2]),
                states[:2],
                deterministic=True,
            )
        )
    policy.train()
    zero = torch.tensor(0, device=policy.device)
    labels = [targets_from_label(r["perception"]) for r in rows]
    _, apos, _, _ = supervised_fragment(
        policy,
        features,
        states[0],
        [(zero, zero, zero)] * len(rows),
        labels,
        resets={i: (r["episode_id"], r["instruction"]) for i, r in enumerate(rows)},
    )
    point_index = next(
        i
        for i, r in enumerate(rows)
        if r["perception"]["apos_valid"] and 0 < r["perception"]["apos"] <= 1296
    )
    (point_grad,) = torch.autograd.grad(apos[point_index], policy.apos_rows, retain_graph=True)
    (all_grad,) = torch.autograd.grad(apos.mean(), policy.apos_rows)
    assert torch.isfinite(point_grad).all() and point_grad.norm() > 0
    assert torch.isfinite(all_grad).all() and all_grad.norm() > 0
    # PPO gradients must reach all three sampled token distributions.
    replay = policy.replay_fragment(fragments[1], initial[1], fragment_tokens[1])
    ratio = (replay.log_prob - torch.stack(old_log_probs[1]).detach()).exp()
    witness = torch.autograd.grad(
        -ratio.mean(), (replay.apos_logits, replay.opos_logits, replay.action_logits)
    )
    witness_norms = [float(g.norm()) for g in witness]
    assert all(torch.isfinite(g).all() and g.norm() > 0 for g in witness)
    grad_benchmarks = {}
    for checkpointing in (True, False):
        policy.settings["gradient_checkpointing"] = checkpointing

        def backward():
            policy.zero_grad(set_to_none=True)
            sample = policy.replay_fragment(
                [features[1]] * 20, states[1], [(zero, zero, zero)] * 20
            )
            teacher, apos_loss, _, action_loss = supervised_fragment(
                policy,
                [features[1]] * 20,
                states[1],
                [(zero, zero, zero)] * 20,
                [labels[1]] * 20,
                experts=torch.full((20,), rows[1]["expert"], device=policy.device),
            )
            loss = (
                apos_loss.mean()
                + action_loss.mean()
                + F.cross_entropy(
                    sample.action_logits, torch.full((20,), rows[1]["expert"], device=policy.device)
                )
            )
            loss.backward()
            assert all(
                torch.isfinite(p.grad).all() for p in policy._all_trainable if p.grad is not None
            )

        backward()
        torch.cuda.reset_peak_memory_stats()
        seconds = benchmark(backward, repetitions=3)
        grad_benchmarks[str(checkpointing)] = {
            "fragment_backward_seconds": seconds,
            "peak_vram_gib": torch.cuda.max_memory_allocated() / 1024**3,
        }
    result = {
        "passed": True,
        "checkpoint": str(Path(args.checkpoint).resolve()),
        "batch_size": 2,
        "different_feature_lengths": [f.ids.shape[-1] for f in mixed_features],
        "serial_batch_error_max": max(errors),
        "rollout_replay_error_max": max(replay_errors),
        "greedy_tokens_match": all(token_matches),
        "batch_permutation_error_max": permutation_error,
        "apos_only_gradient_norm": float(all_grad.norm()),
        "apos_pixel_only_gradient_norm": float(point_grad.norm()),
        "ppo_token_gradient_norms": witness_norms,
        "serial_pair_body_seconds": serial_body,
        "batch_pair_body_seconds": batch_body,
        "serial_pair_full_seconds": serial_full,
        "batch_pair_full_seconds": batch_full,
        "full_inference_speedup": serial_full / batch_full,
        "gradient_checkpointing_benchmarks": grad_benchmarks,
    }
    Path(args.output).write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    torch.set_num_threads(4)
    main()
