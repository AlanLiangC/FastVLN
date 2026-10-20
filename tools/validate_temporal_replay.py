"""Check fused greedy actions, unchanged stochastic PPO logits, and recurrent replay."""

import argparse
import json
from pathlib import Path
from typing import Any

import torch
import yaml

from streamnav.models.point_cot.policy import FrameFeatures, Qwen35PointCoTPolicy
from streamnav.training.action_planning import attach_future_actions, future_action_loss
from streamnav.training.pointcot_trainer import load_adapters


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    config = yaml.safe_load((Path(args.checkpoint) / "resolved_config.yaml").read_text())
    config["device"] = "cuda:0"
    config["model"].update(
        temporal_action_weight=0.25, temporal_action_requires_hint=True, arrival_stop_weight=0.1
    )
    policy = Qwen35PointCoTPolicy(config).eval()
    load_adapters(policy, args.checkpoint)
    samples = torch.load("runtime/fixtures/navigation/rgb_inputs.pt", weights_only=False)
    starts = [
        policy.start_episode(str(i), samples[i % len(samples)]["instruction"]) for i in range(3)
    ]
    states = starts
    histories: list[list[torch.Tensor]] = [[], [], []]
    features_by_env: list[list[FrameFeatures]] = [[], [], []]
    tokens_by_env: list[list[tuple[torch.Tensor, torch.Tensor, torch.Tensor]]] = [[], [], []]
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        features = policy.navigation_features(
            [samples[i % len(samples)]["rgb"] for i in range(3)], starts, [42, 0, 42]
        )
        for _ in range(6):
            steps = policy.step_batch(features, states, deterministic=[True, True, False])
            assert torch.equal(steps[1].action_logits, steps[1].decision_logits)
            assert torch.equal(steps[2].action_logits, steps[2].decision_logits)
            assert steps[1].state.next_action_probabilities is None
            assert steps[0].action == steps[0].decision_logits.argmax()
            forced = policy.step_batch(
                features,
                states,
                tokens=[(s.apos, s.opos, s.action) for s in steps],
                deterministic=True,
            )
            for i, (step, reference) in enumerate(zip(steps, forced, strict=True)):
                torch.testing.assert_close(step.action_logits, reference.action_logits)
                for actual, expected in zip(
                    step.state.kda_cache, reference.state.kda_cache, strict=True
                ):
                    if actual.recurrent is not None:
                        torch.testing.assert_close(actual.recurrent, expected.recurrent)
                histories[i].append(step.log_prob)
                features_by_env[i].append(features[i])
                tokens_by_env[i].append((step.apos, step.opos, step.action))
            states = [s.state for s in steps]
    rows: list[list[dict[str, Any]]] = [
        [
            {
                "tokens": tokens_by_env[i][t],
                "expert": int(tokens_by_env[i][t][2]),
                "episode_id": str(i),
                "done": False,
                "expert_stream": False,
            }
            for i in range(3)
        ]
        for t in range(6)
    ]
    attach_future_actions(rows, mode="executed_prefix")
    errors, gradients, plan_gradients, future_targets = [], [], [], []
    for i in range(3):
        policy.zero_grad(set_to_none=True)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            replay = policy.replay_fragment(features_by_env[i], starts[i], tokens_by_env[i])
            error = float((replay.log_prob.detach() - torch.stack(histories[i])).abs().max())
            assert error <= 0.05, error
            records = [row[i] for row in rows]
            loss = (
                -replay.log_prob.mean()
                + 0.25 * future_action_loss(replay.action_plan_logits, records).mean()
            )
        loss.backward()
        plan_norm = float(policy.future_plan.up.weight.grad.norm())
        assert plan_norm > 0 and torch.isfinite(torch.tensor(plan_norm))
        norm = torch.nn.utils.clip_grad_norm_(
            [p for p in policy.parameters() if p.requires_grad], 1, error_if_nonfinite=True
        )
        errors.append(error)
        gradients.append(float(norm))
        plan_gradients.append(plan_norm)
        future_targets.append(
            [sum(r["future_actions"][k] != -100 for r in records) for k in range(4)]
        )
    result = {
        "passed": True,
        "scope": __doc__,
        "frames_per_stream": 6,
        "stream_modes": ["greedy_temporal_hint", "greedy_raw_no_hint", "stochastic_raw_hint"],
        "replay_log_probability_errors": errors,
        "gradient_norms_before_clipping": gradients,
        "planning_gradient_norms": plan_gradients,
        "future_target_counts": future_targets,
        "actual_action_cache_matches_forced_replay": True,
    }
    Path(args.output).write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result))


if __name__ == "__main__":
    torch.set_num_threads(2)
    main()
