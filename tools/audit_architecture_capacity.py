"""Controlled architecture diagnostics on fixed RGB/text teacher clips, never SR.

The clips deliberately include successful teacher endings to test fit capacity,
not exploration or navigation performance. No diagnostic weights enter training.
"""

import argparse
import json
import math
import time
from dataclasses import replace
from pathlib import Path

import torch
import yaml
from torch.nn import functional as F

from streamnav.contracts.action import NavigationAction
from streamnav.data.mixture import make_source
from streamnav.data.schema import HabitatEpisodeSource
from streamnav.envs.habitat_client import VectorHabitatEnvs
from streamnav.errors import StreamNavError
from streamnav.training.checkpoint import load_policy
from streamnav.utils.seed import seed_everything


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    temporary.replace(path)


def configuration(checkpoint, device):
    config = yaml.safe_load((Path(checkpoint) / "resolved_config.yaml").read_text())
    config.update(checkpoint=checkpoint, device=device)
    config["habitat"]["gpu_device_id"] = int(device.split(":")[-1])
    return config


@torch.no_grad()
def collect(args):
    config = configuration(args.checkpoint, args.device)
    source = make_source({**config["data"], "iterator_options": None, "scene_repeat": 1}, 8317)
    envs = VectorHabitatEnvs(config["habitat"], 1)
    clips, episodes, attempts = [], [], []
    try:
        for attempt in range(48):
            episode = source.sample_episode()
            if episode.goal_text in {e["goal"] for e in episodes}:
                continue
            observation = envs.reset([episode])[0]
            frames, actions = [], []
            for _ in range(500):
                expert = envs.get_oracle_actions()[0]
                if not isinstance(expert, NavigationAction):
                    break
                frames.append(observation["rgb"].clone())
                actions.append(int(expert))
                observation = envs.step([expert])[0]
                if observation["done"]:
                    break
            record = {
                "episode": episode.uid,
                "goal": episode.goal_text,
                "frames": len(frames),
                "success": observation.get("metrics", {}).get("success", 0),
                "final_distance": observation.get("geodesic_distance"),
                "collision_rate": observation.get("metrics", {}).get("collision_rate"),
                "action_counts": torch.bincount(torch.tensor(actions), minlength=6).tolist(),
            }
            attempts.append(record)
            print(json.dumps(record), flush=True)
            if not record["success"] or len(frames) < args.clip_steps:
                continue
            episodes.append(record)
            for name, start in (("start", 0), ("ending", len(frames) - args.clip_steps)):
                clips.append(
                    {
                        "episode": episode.uid,
                        "goal": episode.goal_text,
                        "selection": name,
                        "original_start_step": start,
                        "rgb": torch.stack(frames[start : start + args.clip_steps]),
                        "actions": torch.tensor(actions[start : start + args.clip_steps]),
                    }
                )
            if len(episodes) == args.episodes:
                break
        if len(episodes) < args.episodes:
            raise RuntimeError("Insufficient distinct successful teacher episodes for the audit")
    finally:
        envs.close()
    policy = load_policy(config, preserve_master_weights=True).eval()
    for clip in clips:
        with torch.autocast("cuda", dtype=torch.bfloat16):
            clip["visual"] = policy.backbone.encode_vision(clip["rgb"]).cpu()
    data = {"checkpoint": args.checkpoint, "seed": 8317, "clips": clips}
    destination = Path(args.data)
    destination.parent.mkdir(parents=True, exist_ok=True)
    torch.save(data, destination)
    write_json(
        destination.with_suffix(".json"),
        {
            "protocol": "Fixed teacher-executed RGB/text clips. Independent scene sampling, distinct goals, successful endings; short episodes can have overlapping start/end clips. Every clip resets memory. Capacity diagnostic only, never autonomous SR or a training-recipe change.",
            "checkpoint": args.checkpoint,
            "attempts": attempts,
            "clips": [
                {k: v for k, v in c.items() if k not in {"rgb", "actions", "visual"}} for c in clips
            ],
            "action_counts": torch.bincount(
                torch.cat([c["actions"] for c in clips]), minlength=6
            ).tolist(),
        },
    )


def set_memory_timescales(backbone, frames=(5, 20, 100)):
    """Change only bias offsets; keep learned input-dependent gate weights."""
    with torch.no_grad():
        for layer in backbone.layers:
            mixer = layer.mixer
            if not hasattr(mixer, "decay_proj"):
                continue
            bias = mixer.decay_proj.bias.view(mixer.num_heads, mixer.head_dim)
            for head, row in enumerate(bias):
                rate = math.log(2) / (138 * frames[head % len(frames)])
                target = math.log(math.expm1(rate))
                row.add_(target - row.mean())


@torch.no_grad()
def collect_pairs(args):
    config = configuration(args.checkpoint, args.device)
    manifest = args.manifest or config["data"]["sources"][0]["manifest"]
    source = HabitatEpisodeSource(manifest, args.collect_seed)
    envs = VectorHabitatEnvs(config["habitat"], 1)
    clips, holdout_clips, records = [], [], []
    entries = list(source.files)
    source.rng.shuffle(entries)
    try:
        for entry in entries[: args.max_scenes]:
            raw = source._load(entry)
            categories = {}
            for episode in raw["episodes"]:
                categories.setdefault(episode["object_category"], episode)
            category_episodes = list(categories.values())
            if args.shuffle_categories:
                source.rng.shuffle(category_episodes)
            choices = [source.decode(e, raw) for e in category_episodes[:32]]
            selected = None
            for first in choices:
                a = first.goals[0]["view_points"][0]["agent_state"]
                for second in choices:
                    if first.goal_text == second.goal_text:
                        continue
                    for goal in second.goals[: second.metadata["primary_goal_count"]]:
                        for viewpoint in goal["view_points"][:8]:
                            b = viewpoint["agent_state"]
                            distance = (
                                torch.tensor(a["position"])
                                .sub(torch.tensor(b["position"]))
                                .norm()
                                .item()
                            )
                            if 0.8 < distance < 3.0:
                                selected = (first, second, a, b)
                                break
                        if selected:
                            break
                    if selected:
                        break
                if selected:
                    break
            if not selected:
                continue
            first, second, a, b = selected
            candidate, candidate_records = [], []
            for pose_index, pose in enumerate((a, b)):
                pose_rgb, pose_actions = [], []
                for goal_index, episode in enumerate((first, second)):
                    pair = replace(
                        episode,
                        start_position=pose["position"],
                        start_rotation=pose["rotation"],
                        episode_id=episode.episode_id + f"-audit-pose{pose_index}",
                    )
                    try:
                        observation = envs.reset([pair])[0]
                    except StreamNavError as exc:
                        if "Unreachable goal" not in str(exc):
                            raise
                        print(json.dumps({"rejected_viewpoint": str(exc)}), flush=True)
                        break
                    if goal_index != pose_index and observation["distance"] < 0.5:
                        break
                    # Initial EXPLORE observation performs the native transition
                    # to BEELINE; query the now-initialized teacher without
                    # moving the simulator. This is diagnostic-only.
                    envs.get_oracle_actions()
                    expert = envs.get_oracle_actions()[0]
                    if not isinstance(expert, NavigationAction):
                        continue
                    pose_rgb.append(observation["rgb"])
                    pose_actions.append(int(expert))
                    candidate.append(
                        {
                            "episode": pair.uid,
                            "goal": pair.goal_text,
                            "scene": pair.scene_id,
                            "category": pair.object_category,
                            "selection": "same_rgb_different_goal",
                            "original_start_step": 0,
                            "pose_group": len(records) + pose_index,
                            "rgb": observation["rgb"].unsqueeze(0),
                            "actions": torch.tensor([int(expert)]),
                        }
                    )
                    candidate_records.append(
                        {
                            "goal": pair.goal_text,
                            "pose_index": pose_index,
                            "label": int(expert),
                            "distance": observation["distance"],
                        }
                    )
                if (
                    len(pose_actions) != 2
                    or pose_actions[pose_index] != 0
                    or pose_actions[1 - pose_index] == 0
                ):
                    print(json.dumps({"rejected_pair": candidate_records}), flush=True)
                    break
                if not torch.equal(pose_rgb[0], pose_rgb[1]):
                    raise AssertionError("Changing only the goal changed sensor pixels")
            else:
                clips.extend(candidate)
                for pose_index, pose in enumerate((a, b)):
                    x, y, z, w = pose["rotation"]
                    for angle in (-10, 10):
                        sine, cosine = (
                            math.sin(math.radians(angle) / 2),
                            math.cos(math.radians(angle) / 2),
                        )
                        rotation = [
                            cosine * x + sine * z,
                            cosine * y + sine * w,
                            cosine * z - sine * x,
                            cosine * w - sine * y,
                        ]
                        for episode in (first, second):
                            pair = replace(
                                episode,
                                start_position=pose["position"],
                                start_rotation=rotation,
                                episode_id=episode.episode_id
                                + f"-audit-holdout{pose_index}-{angle}",
                            )
                            observation = envs.reset([pair])[0]
                            envs.get_oracle_actions()
                            expert = envs.get_oracle_actions()[0]
                            if not isinstance(expert, NavigationAction):
                                raise RuntimeError(
                                    "Holdout teacher failed at an accepted viewpoint"
                                )
                            holdout_clips.append(
                                {
                                    "episode": pair.uid,
                                    "goal": pair.goal_text,
                                    "scene": pair.scene_id,
                                    "category": pair.object_category,
                                    "selection": "same_rgb_different_goal",
                                    "original_start_step": 0,
                                    "rgb": observation["rgb"].unsqueeze(0),
                                    "actions": torch.tensor([int(expert)]),
                                    "yaw_offset_degrees": angle,
                                }
                            )
                records.append(
                    {
                        "scene": first.scene_id,
                        "same_rgb_verified": True,
                        "records": candidate_records,
                    }
                )
                print(json.dumps(records[-1]), flush=True)
                if len(records) == args.episodes:
                    break
        if len(records) < args.episodes:
            raise RuntimeError("Insufficient valid counterfactual goal pairs")
    finally:
        envs.close()
    policy = load_policy(config, preserve_master_weights=True).eval()
    for clip in clips + holdout_clips:
        with torch.autocast("cuda", dtype=torch.bfloat16):
            clip["visual"] = policy.backbone.encode_vision(clip["rgb"]).cpu()
    destination = Path(args.data)
    destination.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {"checkpoint": args.checkpoint, "clips": clips, "holdout_clips": holdout_clips}, destination
    )
    write_json(
        destination.with_suffix(".json"),
        {
            "protocol": "Same exact RGB at each pose, two distinct goal texts with teacher STOP/non-STOP labels. Both categories serve as positive and negative at their respective official viewpoints. Diagnostic starts only; no changes to training starts, semantic inputs, rewards or evaluation.",
            "manifest": manifest,
            "collect_seed": args.collect_seed,
            "shuffle_categories": args.shuffle_categories,
            "groups": records,
            "holdout_protocol": "Same scenes/goals/positions, separately rendered camera/body yaw at +/-10 degrees with fresh native teacher labels. Never used for optimizer steps; this tests local view robustness, not unseen-scene/category generalization.",
            "holdout_frames": len(holdout_clips),
            "action_counts": torch.bincount(
                torch.cat([c["actions"] for c in clips]), minlength=6
            ).tolist(),
        },
    )


def clip_forward(policy, clips, visuals, *, wrong_goals=False):
    goals = [c["goal"] for c in clips]
    if wrong_goals:
        # Pair start/end clips consistently with another recorded category.
        if clips[0]["selection"] == "same_rgb_different_goal":
            goals = [goals[i ^ 1] for i in range(len(goals))]
        else:
            goals = goals[2:] + goals[:2]
    states = [policy.start_episode(str(i), goal) for i, goal in enumerate(goals)]
    logits = []
    with policy.backbone.reuse_token_embeddings():
        for step in range(visuals.shape[0]):
            frame = torch.stack([c["rgb"][step] for c in clips])
            out, _, states = policy.forward_batch(frame, states, visual_embeddings=visuals[step])
            logits.append(out)
    return torch.stack(logits)


@torch.no_grad()
def evaluate_fit(policy, clips, visuals, labels):
    policy.eval()
    with torch.autocast("cuda", dtype=torch.bfloat16):
        logits = clip_forward(policy, clips, visuals)
    predicted = logits.argmax(-1)
    return {
        "cross_entropy": F.cross_entropy(logits.flatten(0, 1), labels.flatten()).item(),
        "accuracy": (predicted == labels).float().mean().item(),
        "stop_recall": (predicted[labels == 0] == 0).float().mean().item(),
        "stop_binary_accuracy": ((predicted == 0) == (labels == 0)).float().mean().item(),
        "nonstop_recall": (predicted[labels != 0] != 0).float().mean().item(),
        "confusion": torch.bincount((labels * 6 + predicted).flatten(), minlength=36)
        .view(6, 6)
        .cpu()
        .tolist(),
        "class_recall": [
            (predicted[labels == i] == i).float().mean().item() if (labels == i).any() else None
            for i in range(6)
        ],
    }, logits


def fit(args):
    seed_everything(4971)
    config = configuration(args.checkpoint, args.device)
    policy = load_policy(config, training=True)
    if args.variant == "long_memory":
        set_memory_timescales(policy.backbone)
    data = torch.load(args.data, map_location="cpu", weights_only=False)
    clips = data["clips"]
    visuals = torch.stack([c["visual"] for c in clips], dim=1).to(args.device)
    labels = torch.stack([c["actions"] for c in clips], dim=1).to(args.device)
    backbone = [p for p in policy.backbone.parameters() if p.requires_grad]
    optimizer = torch.optim.Adam(
        [
            {"params": backbone, "lr": args.backbone_lr},
            {"params": policy.actor_critic.actor.parameters(), "lr": args.head_lr},
        ],
        eps=1e-5,
        fused=True,
    )
    history = []
    start = time.monotonic()
    initial, _ = evaluate_fit(policy, clips, visuals, labels)
    history.append({"step": 0, **initial})
    for step in range(1, args.fit_steps + 1):
        policy.train()
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            logits = clip_forward(policy, clips, visuals)
            loss = F.cross_entropy(logits.flatten(0, 1), labels.flatten())
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(
            backbone + list(policy.actor_critic.actor.parameters()), 0.2, error_if_nonfinite=True
        )
        optimizer.step()
        if step % 10 == 0 or step == args.fit_steps:
            metrics, _ = evaluate_fit(policy, clips, visuals, labels)
            row = {
                "step": step,
                **metrics,
                "grad_norm": grad_norm.item(),
                "elapsed_seconds": time.monotonic() - start,
            }
            history.append(row)
            print(json.dumps(row), flush=True)
            write_json(
                args.output,
                {
                    "variant": args.variant,
                    "backbone_lr": args.backbone_lr,
                    "head_lr": args.head_lr,
                    "history": history,
                },
            )
            if metrics["accuracy"] >= 0.99 and metrics["cross_entropy"] < 0.05:
                break
    final, reference = evaluate_fit(policy, clips, visuals, labels)
    holdout, heldout_logits = None, None
    if data.get("holdout_clips"):
        held = data["holdout_clips"]
        held_visuals = torch.stack([c["visual"] for c in held], dim=1).to(args.device)
        held_labels = torch.stack([c["actions"] for c in held], dim=1).to(args.device)
        holdout, heldout_logits = evaluate_fit(policy, held, held_visuals, held_labels)
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        wrong = clip_forward(policy, clips, visuals, wrong_goals=True)
        black_frame = torch.zeros_like(clips[0]["rgb"][0]).unsqueeze(0)
        black_visual = policy.backbone.encode_vision(black_frame)
        black = clip_forward(
            policy, clips, black_visual[None].expand(visuals.shape[0], len(clips), -1, -1)
        )
    torch.save(
        {
            "train_logits": reference.cpu(),
            "train_labels": labels.cpu(),
            "holdout_logits": heldout_logits.cpu() if heldout_logits is not None else None,
            "holdout_labels": held_labels.cpu() if heldout_logits is not None else None,
        },
        Path(args.output).with_suffix(".outputs.pt"),
    )
    write_json(
        args.output,
        {
            "protocol": "Whole-backbone fixed-clip pure IL fit, equal class weights, no PPO/entropy/exploration. Fresh optimizer, diagnostic learning rates; every clip resets memory. No navigation or generalization claim. Wrong-goal accuracy is a perturbation metric, not counterfactual ground truth.",
            "checkpoint": args.checkpoint,
            "variant": args.variant,
            "backbone_lr": args.backbone_lr,
            "head_lr": args.head_lr,
            "frames": labels.numel(),
            "history": history,
            "final": final,
            "heldout_view_perturbations": holdout,
            "wrong_goal_accuracy_against_original_labels": (wrong.argmax(-1) == labels)
            .float()
            .mean()
            .item(),
            "black_rgb_accuracy": (black.argmax(-1) == labels).float().mean().item(),
            "goal_probability_tv_mean": (
                (wrong.softmax(-1) - reference.softmax(-1)).abs().sum(-1) / 2
            )
            .mean()
            .item(),
            "black_rgb_probability_tv_mean": (
                (black.softmax(-1) - reference.softmax(-1)).abs().sum(-1) / 2
            )
            .mean()
            .item(),
            "elapsed_seconds": time.monotonic() - start,
        },
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=["collect", "collect_pairs", "fit"])
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--data", required=True)
    parser.add_argument("--output")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--episodes", type=int, default=4)
    parser.add_argument("--manifest")
    parser.add_argument("--collect-seed", type=int, default=9173)
    parser.add_argument("--max-scenes", type=int, default=24)
    parser.add_argument("--shuffle-categories", action="store_true")
    parser.add_argument("--clip-steps", type=int, default=8)
    parser.add_argument("--fit-steps", type=int, default=100)
    parser.add_argument("--variant", choices=["current", "long_memory"], default="current")
    parser.add_argument("--backbone-lr", type=float, default=1e-4)
    parser.add_argument("--head-lr", type=float, default=1e-3)
    args = parser.parse_args()
    torch.cuda.set_device(args.device)
    if args.mode == "collect":
        collect(args)
    elif args.mode == "collect_pairs":
        collect_pairs(args)
    else:
        fit(args)


if __name__ == "__main__":
    main()
