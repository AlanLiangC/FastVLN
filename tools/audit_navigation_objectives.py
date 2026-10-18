"""Measure real navigation loss gradients and conflicts without optimizer steps."""

import argparse
import json
from pathlib import Path

import torch
import yaml

from streamnav.models.point_cot.policy import Qwen35PointCoTPolicy
from streamnav.training.affordance_loss import imitation_weight
from streamnav.training.pointcot_trainer import load_adapters, replay_losses, replay_records


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--samples", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    config = yaml.safe_load(Path(args.checkpoint, "resolved_config.yaml").read_text())
    config["device"] = "cuda:0"
    config["model"]["gradient_checkpointing"] = True
    policy = Qwen35PointCoTPolicy(config).train()
    load_adapters(policy, args.checkpoint)
    samples = torch.load(args.samples, weights_only=False)[:20]
    state = policy.start_episode(samples[0]["episode_id"], samples[0]["instruction"])
    resets = {
        i: (row["episode_id"], row["instruction"])
        for i, row in enumerate(samples)
        if i > 0 and row["frame_id"] == 0
    }
    records = [
        {
            "features": row["features"],
            "tokens": row["tokens"],
            "perception_label": row["label"],
            "expert": row["expert"],
            "expert_stream": True,
            "student_action": False,
        }
        for row in samples
    ]
    weights = [imitation_weight(r, config["trainer"]) for r in records]
    for row, weight in zip(records, weights, strict=True):
        row["il_weight"] = weight / (sum(weights) / len(weights))
    with torch.no_grad():
        behavior = replay_records(policy, records, state, resets=resets)
        for i, row in enumerate(records):
            row["old_log_prob"] = behavior.log_prob[i].detach()
    del behavior
    replay = replay_records(policy, records, state, resets=resets)
    _, components, error = replay_losses(
        policy,
        state,
        resets,
        replay,
        records,
        torch.ones(len(records)),
        torch.zeros(len(records)),
        config["trainer"],
        0.5,
    )
    objectives = {
        "action_il": components["il_loss"].mean(),
        "apos_ce": components["apos_loss"].mean(),
        "apos_type_weighted": config["trainer"]["apos_type_il_coef"]
        * components["apos_type_loss"].mean(),
    }
    named = [(name, p) for name, p in policy.named_parameters() if p.requires_grad]
    norms, vectors = {}, {}
    for name, loss in objectives.items():
        gradients = torch.autograd.grad(
            loss, [p for _, p in named], retain_graph=True, allow_unused=True
        )
        groups = {"lora": [], "memory": [], "apos": [], "opos": [], "action": [], "critic": []}
        for (parameter_name, parameter), gradient in zip(named, gradients, strict=True):
            group = (
                "memory"
                if parameter_name.startswith("memories.")
                else "apos"
                if parameter_name == "apos_rows"
                else "opos"
                if parameter_name == "opos_rows"
                else "action"
                if parameter_name.startswith("action_")
                else "critic"
                if parameter_name.startswith("critic.")
                else "lora"
            )
            value = torch.zeros_like(parameter) if gradient is None else gradient
            assert torch.isfinite(value).all()
            groups[group].append(value.flatten().detach())
        vectors[name] = {group: torch.cat(values) for group, values in groups.items()}
        norms[name] = {group: float(vector.norm()) for group, vector in vectors[name].items()}
    combined = vectors["apos_ce"]["lora"] + vectors["apos_type_weighted"]["lora"]
    cosine = torch.nn.functional.cosine_similarity(vectors["action_il"]["lora"], combined, dim=0)
    result = {
        "checkpoint": str(Path(args.checkpoint).resolve()),
        "frames": len(records),
        "losses": {name: float(loss.detach()) for name, loss in objectives.items()},
        "gradient_norms": norms,
        "apos_total_to_action_lora_gradient_ratio": float(
            combined.norm() / vectors["action_il"]["lora"].norm()
        ),
        "apos_action_lora_gradient_cosine": float(cosine),
        "replay_error": float(error),
        "optimizer_steps": 0,
    }
    Path(args.output).write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    torch.set_num_threads(4)
    main()
