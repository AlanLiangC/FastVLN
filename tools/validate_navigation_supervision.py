"""Check the navigation objective on saved real frames without updating weights."""

import argparse
import json
from pathlib import Path

import torch
import yaml

from streamnav.models.point_cot.policy import Qwen35PointCoTPolicy
from streamnav.training.point_supervision import marginal_point_loss, targets_from_label
from streamnav.training.pointcot_trainer import load_adapters, replay_losses, replay_records
from streamnav.utils.seed import seed_everything


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--inputs", default="runtime/fixtures/navigation/rgb_inputs.pt")
    parser.add_argument(
        "--output", default="runtime/reports/navigation_supervision_validation.json"
    )
    args = parser.parse_args()
    config = yaml.safe_load(Path("configs/pointcot.yaml").read_text())
    seed_everything(config["seed"])
    policy = Qwen35PointCoTPolicy(config).train()
    load_adapters(policy, args.checkpoint, allow_new_plan=True)
    rows = torch.load(args.inputs, weights_only=False)
    initial = policy.start_episode(rows[0]["episode_id"], rows[0]["instruction"])
    state, records, resets = initial, [], {}
    with torch.no_grad():
        for i, row in enumerate(rows):
            if i in (2, 4):
                resets[i] = (row["episode_id"], row["instruction"])
                state = policy.start_episode(*resets[i])
            features = policy.frame_features(row["rgb"], row["instruction"])[0]
            sampled = policy.step(features, state)
            records.append(
                {
                    "features": features,
                    "tokens": (sampled.apos, sampled.opos, sampled.action),
                    "expert": row["expert"],
                    "perception_label": row["perception"],
                    "il_weight": 1.0,
                    "old_log_prob": sampled.log_prob.detach(),
                    "student_action": True,
                }
            )
            state = sampled.state
    replay = replay_records(policy, records, initial, resets=resets)
    loss, components, error = replay_losses(
        policy,
        initial,
        resets,
        replay,
        records,
        torch.ones(len(records)),
        torch.zeros(len(records)),
        config["trainer"],
        0.05,
    )
    targets = [targets_from_label(r["perception_label"])["apos_targets"] for r in records]
    expected = marginal_point_loss(replay.apos_logits, targets)
    assert torch.allclose(components["apos_loss"], expected)
    assert error < config["trainer"]["replay_tolerance"]
    gradients = {}
    for name in ("apos_loss", "apos_type_loss"):
        (gradient,) = torch.autograd.grad(
            components[name].mean(), policy.apos_rows, retain_graph=True
        )
        gradients[name] = float(gradient.norm())
        assert torch.isfinite(gradient).all() and gradient.norm() > 0
    witness = torch.autograd.grad(
        components["ppo_loss"].mean(),
        (replay.apos_logits, replay.opos_logits, replay.action_logits),
        retain_graph=True,
    )
    assert all(torch.isfinite(g).all() and g.norm() > 0 for g in witness)
    loss.mean().backward()
    assert all(torch.isfinite(p.grad).all() for p in policy._all_trainable if p.grad is not None)
    assert all(p.grad is None for p in policy.parameters() if not p.requires_grad)
    result = {
        "passed": True,
        "checkpoint": str(Path(args.checkpoint).resolve()),
        "frames": len(records),
        "point_supervision_prefix": config["trainer"]["point_supervision_prefix"],
        "apos_type_il_coef": config["trainer"]["apos_type_il_coef"],
        "replay_log_prob_error_max": float(error),
        "components": {name: float(value.detach().mean()) for name, value in components.items()},
        "apos_rows_gradient_norms": gradients,
        "ppo_token_gradient_norms": [float(g.norm()) for g in witness],
        "frozen_parameter_gradients_absent": True,
    }
    Path(args.output).write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    torch.set_num_threads(4)
    main()
