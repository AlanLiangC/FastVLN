"""Real-frame acceptance for plan4, unchanged caches and joint PPO gradients."""

import argparse
import json
from pathlib import Path

import torch
import yaml

from streamnav.models.point_cot.policy import Qwen35PointCoTPolicy
from streamnav.training.action_planning import attach_future_actions, future_action_loss
from streamnav.training.pointcot_trainer import load_adapters, replay_records
from streamnav.utils.seed import seed_everything


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--inputs", default="runtime/fixtures/navigation/rgb_inputs.pt")
    parser.add_argument("--teacher-inputs", default="runtime/fixtures/navigation/expert_records.pt")
    parser.add_argument("--output", default="runtime/reports/plan4_validation.json")
    args = parser.parse_args()
    config = yaml.safe_load(Path("configs/pointcot.yaml").read_text())
    seed_everything(config["seed"])
    policy = Qwen35PointCoTPolicy(config).eval()
    load_adapters(policy, args.checkpoint, allow_new_plan=True)
    rows = torch.load(args.inputs, weights_only=False)
    features, sampled, records = [], [], []
    initial = policy.start_episode(rows[0]["episode_id"], rows[0]["instruction"])
    state = initial
    resets = {i: (rows[i]["episode_id"], rows[i]["instruction"]) for i in (2, 4)}
    with torch.no_grad():
        for i, row in enumerate(rows):
            if i in resets:
                state = policy.start_episode(*resets[i])
            feature = policy.frame_features(row["rgb"], row["instruction"])[0]
            if policy.supports_object_hints:
                feature = policy.condition_features([feature], [[0, 42, 0, 1296, 300][i]])[0]
            step = policy.step(feature, state)
            features.append(feature)
            sampled.append(step)
            records.append(
                {
                    "features": feature,
                    "tokens": (step.apos, step.opos, step.action),
                    "old_log_prob": step.log_prob.detach(),
                }
            )
            state = step.state
        states = [
            policy.start_episode(rows[i]["episode_id"], rows[i]["instruction"]) for i in range(3)
        ]
        before = policy.step_batch(features[:3], states, deterministic=True)
        head = policy.future_plan
        policy.future_plan = None
        without = policy.step_batch(features[:3], states, deterministic=True)
        policy.future_plan = head
        for a, b in zip(before, without, strict=True):
            assert a.action_plan_logits.shape == (4, 6)
            for name in ("apos_logits", "opos_logits", "action_logits", "value"):
                torch.testing.assert_close(getattr(a, name), getattr(b, name), rtol=0, atol=0)
            for left, right in zip(a.state.kda_cache, b.state.kda_cache, strict=True):
                if left.recurrent is not None:
                    torch.testing.assert_close(left.recurrent, right.recurrent, rtol=0, atol=0)
    policy.train()
    replay = replay_records(policy, records, initial, resets=resets)
    error = float(
        (replay.log_prob.detach() - torch.stack([s.log_prob for s in sampled])).abs().max()
    )
    assert error < config["trainer"]["replay_tolerance"]
    ratio = (replay.log_prob - torch.stack([s.log_prob for s in sampled]).detach()).exp()
    ppo_witness = torch.autograd.grad(
        -ratio.mean(), (replay.apos_logits, replay.opos_logits, replay.action_logits)
    )
    assert all(torch.isfinite(g).all() and g.norm() > 0 for g in ppo_witness)
    del replay, ratio
    samples = torch.load(args.teacher_inputs, weights_only=False)[:20]
    expert = []
    for i, row in enumerate(samples):
        done = i + 1 == len(samples) or samples[i + 1]["frame_id"] == 0
        expert.append(
            {
                "features": policy.condition_features([row["features"]], [42 if i % 2 else 0])[0]
                if policy.supports_object_hints
                else row["features"],
                "tokens": row["tokens"],
                "expert": row["expert"],
                "expert_stream": True,
                "episode_id": row["episode_id"],
                "done": done,
            }
        )
    attach_future_actions([[r] for r in expert], 4)
    resets = {
        i: (r["episode_id"], r["instruction"])
        for i, r in enumerate(samples)
        if i and r["frame_id"] == 0
    }
    initial = policy.start_episode(samples[0]["episode_id"], samples[0]["instruction"])
    step = replay_records(policy, expert, initial, resets=resets)
    future_loss = future_action_loss(step.action_plan_logits, expert).mean()
    witness = torch.autograd.grad(
        future_loss,
        (
            policy.future_plan.up.weight,
            policy.action_rows,
            policy.native.model.language_model.layers[0].linear_attn.in_proj_qkv.b,
        )
        + (
            (policy.object_hint.null, policy.object_hint.source)
            if policy.supports_object_hints
            else ()
        ),
    )
    assert all(torch.isfinite(g).all() and g.norm() > 0 for g in witness)
    horizon_norms = witness[0].view(3, 1024, -1).flatten(1).norm(dim=1).tolist()
    assert all(value > 0 for value in horizon_norms)
    assert all(p.grad is None for p in policy.parameters() if not p.requires_grad)
    result = {
        "passed": True,
        "parameters": policy.parameter_summary(),
        "replay_error": error,
        "future_token_writes_to_cache": 0,
        "planner_does_not_change_action_or_cache": True,
        "ppo_token_gradient_norms": [float(g.norm()) for g in ppo_witness],
        "future_horizon_gradient_norms": horizon_norms,
        "future_action_loss": float(future_loss.detach()),
        "future_action_table_gradient_norm": float(witness[1].norm()),
        "future_body_lora_gradient_norm": float(witness[2].norm()),
        "frozen_base_gradients_absent": True,
        "hint_gradient_norms": [float(g.norm()) for g in witness[3:]],
        "mixed_null_and_pixel_hint_replay": policy.supports_object_hints,
    }
    Path(args.output).write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    torch.set_num_threads(4)
    main()
