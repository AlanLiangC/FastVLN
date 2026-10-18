"""Train-scene-only OPOS gradient audit and isolated output-table comparison."""

import argparse
import json
from dataclasses import replace
from pathlib import Path

import torch
import yaml

from streamnav.contracts.action import NavigationAction
from streamnav.data.schema import HabitatEpisodeSource
from streamnav.envs.habitat_client import HabitatClient
from streamnav.models.point_cot.policy import Qwen35PointCoTPolicy
from streamnav.models.policy.perception import decode_point_logits
from streamnav.training.point_supervision import navigation_object_losses
from streamnav.training.pointcot_trainer import cache_copy, load_adapters
from streamnav.utils.seed import seed_everything


def metrics(logits, targets, hierarchical):
    tokens = decode_point_logits(logits, hierarchical=hierarchical).tolist()
    positive = [i for i, t in enumerate(targets) if t != [0]]
    negative = [i for i, t in enumerate(targets) if t == [0]]
    return {
        "visible_frames": len(positive),
        "absent_frames": len(negative),
        "visible_recall": sum(tokens[i] > 0 for i in positive) / max(1, len(positive)),
        "region_accuracy": sum(tokens[i] in targets[i] for i in positive) / max(1, len(positive)),
        "absence_accuracy": sum(tokens[i] == 0 for i in negative) / max(1, len(negative)),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", default="runtime/reports/opos_objective_audit")
    parser.add_argument("--steps", type=int, default=200)
    args = parser.parse_args()
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    config = yaml.safe_load((Path(args.checkpoint) / "resolved_config.yaml").read_text())
    config["device"], config["habitat"]["gpu_device_id"] = "cuda:0", 0
    config["habitat"]["perception_labels"] = False
    seed_everything(config["seed"])
    policy = Qwen35PointCoTPolicy(config).eval()
    load_adapters(policy, args.checkpoint)
    source = HabitatEpisodeSource(config["data"]["sources"][0]["manifest"])
    assert source.manifest["split"] == "train"
    env = HabitatClient(config["habitat"], "opos-gradient-audit")
    samples, witnesses = [], {}
    try:
        for scene_index, entry in enumerate(source.files[:6]):
            data = source._load(entry)
            episode = source.decode(data["episodes"][0], data)
            viewpoint = episode.goals[0]["view_points"][0]["agent_state"]
            episode = replace(
                episode, start_position=viewpoint["position"], start_rotation=viewpoint["rotation"]
            )
            observation = env.reset(episode)
            state = policy.start_episode(episode.uid, episode.goal_text)
            for frame in range(12):
                label = env.get_object_supervision()
                with torch.no_grad():
                    feature = policy.condition_features(
                        [policy.frame_features(observation["rgb"], episode.goal_text)[0]], [0]
                    )[0]
                    context = policy.begin(feature, state)
                    apos = decode_point_logits(
                        policy.point_logits(context, "apos"),
                        hierarchical=policy.apos_decoding == "hierarchical",
                    )
                    context = policy.append(context, policy.apos_rows[apos])
                    if label["opos_valid"]:
                        samples.append(
                            {
                                "hidden": context.hidden.detach().cpu(),
                                "targets": label["opos_targets"],
                                "scene": scene_index,
                                "frame": frame,
                            }
                        )
                        kind = "visible" if label["opos"] else "absent"
                        if kind not in witnesses:
                            witnesses[kind] = (feature.cpu(), cache_copy(state), label)
                    opos = policy.point_logits(context, "opos").argmax()
                    context = policy.append(context, policy.opos_rows[opos])
                    context = policy.append(context, policy.action_rows[2])
                    state = state.with_cache(context.memory)
                observation = env.step(NavigationAction.TURN_LEFT)
            print(json.dumps({"scene_complete": scene_index, "samples": len(samples)}), flush=True)
    finally:
        env.close()
    torch.save(samples, output / "train_scene_hidden_states.pt")
    gradients = {}
    for kind, (feature, state, label) in witnesses.items():
        policy.zero_grad(set_to_none=True)
        context = policy.begin(feature, cache_copy(state, policy.device))
        apos = decode_point_logits(
            policy.point_logits(context, "apos"),
            hierarchical=policy.apos_decoding == "hierarchical",
        )
        context = policy.append(context, policy.apos_rows[apos])
        logits = policy.point_logits(context, "opos")[None]
        loss = navigation_object_losses(logits, [label["opos_targets"]])[0].mean()
        logit_grad, rows_grad, lora_grad = torch.autograd.grad(
            loss,
            (
                logits,
                policy.opos_rows,
                policy.native.model.language_model.layers[0].linear_attn.in_proj_qkv.b,
            ),
        )
        gradients[kind] = {
            "none_logit_gradient": float(logit_grad[0, 0]),
            "spatial_logit_gradient_norm": float(logit_grad[0, 1:].norm()),
            "opos_rows_gradient_norm": float(rows_grad.norm()),
            "lora_gradient_norm": float(lora_grad.norm()),
        }
        assert all(torch.isfinite(g).all() and g.norm() > 0 for g in (rows_grad, lora_grad))
        assert (logit_grad[0, 0] > 0) == (kind == "visible")
        del context, logits, loss, logit_grad, rows_grad, lora_grad
    assert set(gradients) == {"visible", "absent"}
    assert all(p.grad is None for p in policy.parameters() if not p.requires_grad)
    print(json.dumps({"gradient_checks": gradients}), flush=True)

    hidden = torch.cat([s["hidden"] for s in samples]).to(policy.device)
    targets = [s["targets"] for s in samples]
    train_indices = [i for i, s in enumerate(samples) if s["scene"] < 4]
    holdout = [i for i, s in enumerate(samples) if s["scene"] >= 4]
    train_targets = [targets[i] for i in train_indices]
    positive = torch.tensor([t != [0] for t in train_targets], device=policy.device)
    negative = ~positive
    results = {}
    for objective in ("original", "balanced_anchor"):
        rows = torch.nn.Parameter(policy.opos_rows.detach().clone())
        optimizer = torch.optim.AdamW([rows], lr=5e-5, weight_decay=0)
        history = []
        for step in range(args.steps + 1):
            if step in (0, 25, 100, args.steps):
                with torch.no_grad():
                    logits = hidden @ rows.T
                    history.append(
                        {
                            "step": step,
                            **{
                                f"{split}_{mode}": metrics(
                                    logits[indices],
                                    [targets[i] for i in indices],
                                    mode == "hierarchical",
                                )
                                for split, indices in (("fit", train_indices), ("heldout", holdout))
                                for mode in ("flat", "hierarchical")
                            },
                        }
                    )
            if step == args.steps:
                break
            logits = hidden[train_indices] @ rows.T
            total, visibility, location = navigation_object_losses(
                logits, train_targets, anchor_weight=0.2 if objective == "balanced_anchor" else 0
            )
            loss = (
                0.5 * (visibility[positive].mean() + visibility[negative].mean())
                + location[positive].mean()
                if objective == "balanced_anchor"
                else 0.2 * total[positive].mean() + 0.8 * total[negative].mean()
            )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
        results[objective] = history
        print(json.dumps({"objective": objective, "final": history[-1]}), flush=True)
    report = {
        "checkpoint": args.checkpoint,
        "samples": len(samples),
        "training_only": True,
        "train_scenes": 4,
        "heldout_train_scenes": 2,
        "gradient_checks": gradients,
        "isolated_output_table_comparison": results,
        "limitations": "Frozen hidden states; no navigation rollout or validation-split weight update",
    }
    (output / "report.json").write_text(json.dumps(report, indent=2))


if __name__ == "__main__":
    torch.set_num_threads(4)
    main()
