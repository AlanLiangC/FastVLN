"""Inspect policy and perception on fixed, physically replayed evaluation histories.

This is an open-loop readout diagnostic, not an autonomous SR comparison.
FP32 master weights with BF16 autocast match evaluation inside training.
"""

import argparse
import json
from pathlib import Path

import imageio.v2 as imageio
import numpy as np
import torch
import yaml

from streamnav.contracts.action import NavigationAction
from streamnav.contracts.episode import NavigationEpisode
from streamnav.contracts.perception import GRID_HEIGHT, GRID_WIDTH, POINT_CELLS
from streamnav.envs.habitat_client import VectorHabitatEnvs
from streamnav.evaluation.perception_records import perception_records
from streamnav.evaluation.runner import video_frame
from streamnav.training.checkpoint import load_policy


def closed_loop(policy, config, request, output, gpu):
    """Selected-case intervention; its success rate is not a benchmark score."""
    envs = VectorHabitatEnvs(
        {**config["habitat"], "gpu_device_id": gpu, "perception_labels": False}, 1
    )
    results = []
    try:
        for case in request["cases"]:
            episode = NavigationEpisode(**case["episode"])
            observation = envs.reset([episode])[0]
            with torch.autocast("cuda", dtype=torch.bfloat16):
                state = policy.start_episode(episode.uid, episode.goal_text)
            writer = imageio.get_writer(
                str(output / (case["name"] + ".mp4")),
                fps=6,
                codec="libx264",
                ffmpeg_params=["-preset", "veryfast", "-threads", "1"],
            )
            records, last_turn, streak, max_streak = [], None, 0, 0
            try:
                while True:
                    with torch.autocast("cuda", dtype=torch.bfloat16):
                        logits, _, states, predictions = policy.forward_batch(
                            observation["rgb"].to(policy.backbone.device).unsqueeze(0),
                            [state],
                            return_perception=True,
                        )
                    state = states[0]
                    action = NavigationAction(logits.argmax(-1).item())
                    point = perception_records(predictions, hierarchical=True)[0]
                    point["action_fusion_enabled"] = policy.perception.fusion_scale != 0
                    probability = logits.softmax(-1)[0, 0].item()
                    is_turn = action in (NavigationAction.TURN_LEFT, NavigationAction.TURN_RIGHT)
                    streak = (
                        streak + 1
                        if is_turn and last_turn is not None and action != last_turn
                        else int(is_turn)
                    )
                    max_streak = max(max_streak, streak)
                    last_turn = action if is_turn else None
                    writer.append_data(
                        video_frame(
                            observation["rgb"],
                            episode.goal_text,
                            action,
                            observation,
                            probability,
                            perception=point,
                        )
                    )
                    records.append(
                        {
                            "frame_id": observation["frame_id"],
                            "action": action.name,
                            "distance": observation.get(
                                "geodesic_distance", observation.get("distance")
                            ),
                            "stop_probability": probability,
                            "perception": point,
                        }
                    )
                    observation = envs.step([action])[0]
                    if observation["done"]:
                        writer.append_data(
                            video_frame(
                                observation["rgb"],
                                episode.goal_text,
                                action,
                                observation,
                                terminal=True,
                            )
                        )
                        break
            finally:
                writer.close()
            result = {
                "case": case["name"],
                "goal": episode.goal_text,
                **observation["metrics"],
                "max_alternating_turns": max_streak,
                "near_frames": sum(
                    x["distance"] < config["habitat"]["success_distance"] for x in records
                ),
                "false_stop": action == NavigationAction.STOP and not observation["success"],
            }
            (output / (case["name"] + ".json")).write_text(json.dumps(records))
            results.append(result)
            print(json.dumps(result), flush=True)
    finally:
        envs.close()
    (output / "closed_loop.json").write_text(json.dumps(results, indent=2) + "\n")


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("request", "replay", "config", "checkpoint", "output"):
        parser.add_argument("--" + name, required=True)
    parser.add_argument("--gpu", type=int, default=1)
    parser.add_argument("--closed-loop", action="store_true")
    parser.add_argument("--disable-fusion", action="store_true")
    args = parser.parse_args()
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    request = json.loads(Path(args.request).read_text())
    config = yaml.safe_load(Path(args.config).read_text())
    config.update(checkpoint=args.checkpoint, device=f"cuda:{args.gpu}")
    policy = load_policy(config, preserve_master_weights=True).eval()
    policy.batch_chat_body = config["trainer"].get("batch_chat_body", False)
    if args.disable_fusion:
        policy.perception.fusion_scale = 0.0
    if args.closed_loop:
        closed_loop(policy, config, request, output, args.gpu)
        return
    captured = {}

    def capture_actor(module, inputs, result):
        captured["base"] = result.detach().float()

    hook = policy.actor_critic.actor.register_forward_hook(capture_actor)
    device = policy.backbone.device
    cells = torch.arange(POINT_CELLS, device=device)
    pixels = torch.stack(
        (
            (cells % GRID_WIDTH + 0.5) * config["habitat"]["width"] / GRID_WIDTH,
            (cells // GRID_WIDTH + 0.5) * config["habitat"]["height"] / GRID_HEIGHT,
        ),
        -1,
    )
    summary = []
    try:
        for case in request["cases"]:
            frames = np.load(Path(args.replay) / (case["name"] + "_rgb.npy"), mmap_mode="r")
            original = [json.loads(x) for x in Path(case["trace"]).read_text().splitlines()]
            with torch.autocast("cuda", dtype=torch.bfloat16):
                state = policy.start_episode(
                    ":".join(
                        str(case["episode"][k])
                        for k in ["dataset_id", "split", "scene_id", "episode_id"]
                    ),
                    case["episode"]["goal_text"],
                )
            records = []
            for index, rgb in enumerate(frames):
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    logits, _, states, predictions = policy.forward_batch(
                        torch.from_numpy(np.array(rgb, copy=True)).to(device).unsqueeze(0),
                        [state],
                        return_perception=True,
                    )
                state = states[0]
                base = captured["base"]
                probabilities = logits.softmax(-1)[0]
                point = perception_records(predictions, hierarchical=True)[0]
                for name in ("apos", "opos"):
                    distribution = predictions[name][0, 1 : POINT_CELLS + 1].softmax(-1)
                    mean = distribution @ pixels
                    variance = (distribution * (pixels - mean).square().sum(-1)).sum()
                    point[name + "_conditional_location_mean_pixel"] = mean.tolist()
                    point[name + "_conditional_location_rms_pixels"] = variance.sqrt().item()
                records.append(
                    {
                        "frame_id": index,
                        "distance": original[index]["distance_before"],
                        "probabilities": probabilities.tolist(),
                        "base_probabilities": base.softmax(-1)[0].tolist(),
                        "base_logits": base[0].tolist(),
                        "fusion_residual_logits": (logits - base)[0].tolist(),
                        "action": logits.argmax(-1).item(),
                        "base_action": base.argmax(-1).item(),
                        "original_action": int(np.argmax(original[index]["action_probabilities"])),
                        "original_probability_error_max": (
                            probabilities
                            - torch.tensor(original[index]["action_probabilities"], device=device)
                        )
                        .abs()
                        .max()
                        .item(),
                        "perception": point,
                    }
                )
            (output / (case["name"] + ".json")).write_text(json.dumps(records))
            near = [x for x in records if x["distance"] < config["habitat"]["success_distance"]]
            result = {
                "case": case["name"],
                "frames": len(records),
                "original_action_agreement": np.mean(
                    [x["action"] == x["original_action"] for x in records]
                ).item(),
                "max_probability_difference": max(
                    x["original_probability_error_max"] for x in records
                ),
                "fusion_changes_action_fraction": np.mean(
                    [x["action"] != x["base_action"] for x in records]
                ).item(),
                "near_frames": len(near),
                "near_mean_stop_probability": np.mean([x["probabilities"][0] for x in near]).item()
                if near
                else None,
                "near_mean_base_stop_probability": np.mean(
                    [x["base_probabilities"][0] for x in near]
                ).item()
                if near
                else None,
            }
            summary.append(result)
            print(json.dumps(result), flush=True)
    finally:
        hook.remove()
    (output / "policy_audit.json").write_text(json.dumps(summary, indent=2) + "\n")


if __name__ == "__main__":
    torch.set_num_threads(1)
    main()
