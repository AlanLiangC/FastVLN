"""Check real-frame CUDA graph outputs, resets, hints, weight updates and PPO replay."""

import argparse
import json
from pathlib import Path

import torch
import yaml

from streamnav.models.point_cot.policy import Qwen35PointCoTPolicy
from streamnav.training.pointcot_trainer import load_adapters


def difference(expected, actual):
    errors = []
    for a, b in zip(expected, actual, strict=True):
        for name in ("apos_logits", "opos_logits", "action_logits", "action_plan_logits", "value"):
            errors.append(float((getattr(a, name) - getattr(b, name)).abs().max()))
        errors.append(float((a.log_prob - b.log_prob).abs()))
        for x, y in zip(a.state.kda_cache, b.state.kda_cache, strict=True):
            if x.recurrent is not None:
                errors.append(float((x.recurrent - y.recurrent).abs().max()))
        assert all(
            int(getattr(a, name)) == int(getattr(b, name)) for name in ("apos", "opos", "action")
        )
    assert max(errors) < 1e-4, max(errors)
    return max(errors)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--inputs", default="runtime/fixtures/navigation/rgb_inputs.pt")
    parser.add_argument("--output", default="runtime/reports/inference_graphs_validation.json")
    args = parser.parse_args()
    config = yaml.safe_load((Path(args.checkpoint) / "resolved_config.yaml").read_text())
    config["device"] = "cuda:0"
    policy = Qwen35PointCoTPolicy(config).eval()
    load_adapters(policy, args.checkpoint)
    rows = torch.load(args.inputs, weights_only=False)
    instructions = [
        rows[0]["instruction"],
        "Find a large cooker beside the white kitchen wall.",
        rows[1]["instruction"],
    ]
    states = [
        policy.start_episode(f"graphs-{i}", instruction)
        for i, instruction in enumerate(instructions)
    ]
    starts = list(states)
    features, tokens, old = [], [], []
    errors = []
    with torch.no_grad():
        raw = policy.navigation_features([rows[i % len(rows)]["rgb"] for i in range(3)], states)
        for t in range(8):
            if t == 3:
                states[0] = policy.start_episode("graphs-reset", instructions[0])
                states[2] = policy.start_episode("graphs-reset-2", instructions[2])
            batch = policy.condition_features(raw, [0, 100 + 19 * t, 0 if t % 2 else 700])
            modes = [False, True, False] if t == 4 else True
            overrides = [None, None, (None, None, torch.tensor(1, device=policy.device))]
            random = torch.cuda.get_rng_state()
            policy.settings["inference_cuda_graphs"] = False
            eager = policy.step_batch(batch, states, tokens=overrides, deterministic=modes)
            torch.cuda.set_rng_state(random)
            policy.settings["inference_cuda_graphs"] = True
            graphed = policy.step_batch(batch, states, tokens=overrides, deterministic=modes)
            errors.append(difference(eager, graphed))
            features.append(batch[0])
            tokens.append((graphed[0].apos, graphed[0].opos, graphed[0].action))
            old.append(graphed[0].log_prob)
            states = [step.state for step in graphed]
            if t == 0:
                retained = [
                    (s.recurrent, s.recurrent.clone())
                    for s in states[0].kda_cache
                    if s.recurrent is not None
                ]
        retained_error = max(float((value - snapshot).abs().max()) for value, snapshot in retained)
        assert retained_error == 0
        # Interleave new sequence lengths with a hot graph, evict library
        # metadata caches and release unused allocations. A graph borrowing
        # FLA's warmup index tensor can otherwise survive short fixed-shape
        # checks and later read a freed address during a real rollout.
        for variant in range(12):
            varied_states = [
                policy.start_episode(
                    f"vary-{variant}-{i}", "Find a " + "bright " * (variant + i + 1) + "chair."
                )
                for i in range(3)
            ]
            varied = policy.navigation_features(
                [rows[i % len(rows)]["rgb"] for i in range(3)], varied_states
            )
            policy.settings["inference_cuda_graphs"] = False
            eager = policy.step_batch(varied, varied_states, deterministic=True)
            policy.settings["inference_cuda_graphs"] = True
            graphed = policy.step_batch(varied, varied_states, deterministic=True)
            errors.append(difference(eager, graphed))
            torch.cuda.empty_cache()
            policy.settings["inference_cuda_graphs"] = False
            eager = policy.step_batch(batch, states, deterministic=True)
            policy.settings["inference_cuda_graphs"] = True
            graphed = policy.step_batch(batch, states, deterministic=True)
            errors.append(difference(eager, graphed))
        parameter = policy.native.model.language_model.layers[0].linear_attn.in_proj_qkv.b
        snapshot = parameter.clone()
        before = policy.step_batch(batch, states, deterministic=True)
        parameter.add_(0.0001)
        after = policy.step_batch(batch, states, deterministic=True)
        policy.settings["inference_cuda_graphs"] = False
        changed_eager = policy.step_batch(batch, states, deterministic=True)
        errors.append(difference(changed_eager, after))
        weight_change = max(
            float((a.action_logits - b.action_logits).abs().max())
            for a, b in zip(before, after, strict=True)
        )
        assert weight_change > 1e-5, weight_change
        parameter.copy_(snapshot)
        for count in (2, 1):
            policy.settings["inference_cuda_graphs"] = False
            eager = policy.step_batch(batch[:count], states[:count], deterministic=True)
            policy.settings["inference_cuda_graphs"] = True
            graphed = policy.step_batch(batch[:count], states[:count], deterministic=True)
            errors.append(difference(eager, graphed))
        with policy.native_weights():
            policy.settings["inference_cuda_graphs"] = False
            eager = policy.step_batch(batch, states, deterministic=True)
            policy.settings["inference_cuda_graphs"] = True
            graphed = policy.step_batch(batch, states, deterministic=True)
            errors.append(difference(eager, graphed))
    policy.train()
    replay = policy.replay_fragment(
        features, starts[0], tokens, resets={3: ("graphs-reset", instructions[0])}
    )
    replay_error = float((replay.log_prob - torch.stack(old)).abs().max().detach())
    assert replay_error < config["trainer"]["replay_tolerance"], replay_error
    witness = torch.autograd.grad(
        -replay.log_prob.mean(), (replay.apos_logits, replay.opos_logits, replay.action_logits)
    )
    assert all(torch.isfinite(g).all() and g.norm() > 0 for g in witness)
    result = {
        "passed": True,
        "logits_and_state_error_max": max(errors),
        "retained_state_mutation": retained_error,
        "weight_update_output_change": weight_change,
        "replay_log_prob_error_max": replay_error,
        "ppo_token_gradient_norms": [float(g.norm()) for g in witness],
        "graph_hits": policy.inference_graphs.hits,
        "graph_misses": policy.inference_graphs.misses,
        "cached_graphs": len(policy.inference_graphs.entries),
    }
    Path(args.output).write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    torch.set_num_threads(4)
    main()
