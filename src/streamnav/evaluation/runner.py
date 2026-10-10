import json
import textwrap
import time
from itertools import islice
from pathlib import Path
from typing import Any, cast

import hydra
import imageio.v2 as imageio
import torch
from omegaconf import OmegaConf
from PIL import Image, ImageDraw

from streamnav.contracts.action import NavigationAction
from streamnav.contracts.perception import APOS_LEFT, APOS_RIGHT, APOS_STOP
from streamnav.data.schema import HabitatEpisodeSource
from streamnav.envs.habitat_client import VectorHabitatEnvs
from streamnav.evaluation.latency_metrics import latency_summary
from streamnav.evaluation.navigation_metrics import aggregate_metrics
from streamnav.evaluation.pointing_overlay import draw_pointing_overlay
from streamnav.evaluation.video_cases import (
    retain_video_cases,
    select_video_cases,
    write_video_index,
)
from streamnav.models.qwen35_kda.cache import state_bytes
from streamnav.training import distributed as parallel
from streamnav.training.checkpoint import load_policy
from streamnav.utils.logging import append_json
from streamnav.utils.seed import seed_everything


def video_frame(
    rgb, instruction, action, info, stop_probability=None, terminal=False, perception=None
):
    image = Image.fromarray(rgb.cpu().numpy())
    width, height = image.size
    canvas = Image.new("RGB", (width, ((height + 96 + 15) // 16) * 16), "#121827")
    canvas.paste(image, (0, 0))
    if perception is not None:
        draw_pointing_overlay(canvas, perception, (width, height))
    draw = ImageDraw.Draw(canvas)
    draw.multiline_text(
        (10, height + 4), "\n".join(textwrap.wrap(instruction, 65)[:2]), fill="white"
    )
    label = "TERMINAL" if terminal else action.name
    distance = info.get("geodesic_distance", info.get("distance", float("nan")))
    draw.text(
        (10, height + 38),
        f"{label} | step {info['frame_id']} | distance {distance:.2f} m",
        fill="white",
    )
    probability = f"P(STOP)={stop_probability:.4f}" if stop_probability is not None else ""
    draw.text(
        (10, height + 58),
        f"{probability} | previous collision={bool(info.get('collision', False))}",
        fill="white",
    )
    if terminal:
        draw.text(
            (10, height + 77),
            f"success={bool(info.get('success', False))} | episode ended",
            fill="white",
        )
    elif perception is not None:
        apos = {0: "none", APOS_LEFT: "left", APOS_RIGHT: "right", APOS_STOP: "stop"}.get(
            perception["apos"], "point"
        )
        draw.text(
            (10, height + 77),
            f"APOS={apos} | OPOS={'point' if perception['opos'] else 'out of view'} | "
            f"P(ready)={perception['arrival_probabilities'][2]:.3f}",
            fill="white",
        )
    import numpy as np

    return np.asarray(canvas)


@torch.no_grad()
def evaluate(policy, config, update=0, episodes=None, max_steps=None, video=False):
    was_training = policy.training
    policy.eval()
    output = Path(config["run_dir"]) / "evaluation" / f"update_{update:07d}"
    output.mkdir(parents=True, exist_ok=True)
    worker_output = output / f"rank_{parallel.rank():03d}" if parallel.world_size() > 1 else output
    worker_output.mkdir(parents=True, exist_ok=True)
    env_config = {
        **config["habitat"],
        "max_episode_steps": max_steps or config["eval"]["max_steps"],
        # Evaluation predictions never require a privileged training sensor.
        "perception_labels": False,
    }
    limit = episodes if episodes is not None else config["eval"]["episodes"]
    count = config["eval"].get("num_envs", 1)
    if count < 1 or (limit is not None and limit < 1):
        raise ValueError("Evaluation environment and episode counts must be positive")
    count = (
        min(count, max(1, (limit + parallel.world_size() - 1) // parallel.world_size()))
        if limit is not None
        else count
    )
    device = policy.backbone.device
    torch.cuda.empty_cache()
    results = {}
    cases_by_split = {}
    dtype = getattr(torch, config["model"]["dtype"])
    video_limit = config["eval"].get("video_episodes", 1)
    video_selection = config["eval"].get("video_selection", "first")
    if video_limit < 0 or video_selection not in ("first", "representative"):
        raise ValueError("Invalid evaluation video settings")
    representative = video_selection == "representative"
    envs = VectorHabitatEnvs(env_config, count)
    try:
        for manifest in config["eval"]["manifests"]:
            source = HabitatEpisodeSource(manifest, seed=config["seed"])
            split = source.manifest["dataset_id"] + "_" + source.manifest["split"]
            metrics, times, batch_times, video_records = [], [], [], []
            stratified = config["eval"].get("stratified", True)
            iterator = (
                (i, e)
                for i, e in enumerate(source.evaluation_episodes(limit, stratified, config["seed"]))
                if i % parallel.world_size() == parallel.rank()
            )
            scene_ids, categories = set(), set()
            action_counts = [0] * config["model"]["action_dim"]
            cache_bytes = 0
            while batch := list(islice(iterator, count)):
                for _, episode in batch:
                    scene_ids.add(episode.scene_id)
                    categories.add(episode.object_category)
                observations = list(
                    envs.executor.map(
                        lambda item: envs.clients[item[0]].reset(item[1][1]), enumerate(batch)
                    )
                )
                with torch.autocast(
                    device_type=device.type, dtype=dtype, enabled=dtype != torch.float32
                ):
                    states = [policy.start_episode(e.uid, e.goal_text) for _, e in batch]
                writers = []
                traces: list[list[dict[str, Any]]] = [[] for _ in batch]
                turn_streaks = [0 for _ in batch]
                alternating_turn_streaks = [0 for _ in batch]
                last_turn_actions: list[NavigationAction | None] = [None for _ in batch]
                active = list(range(len(batch)))
                stop_diagnostics: list[dict[str, Any]] = [
                    {
                        "near_goal_steps": 0,
                        "near_goal_stop_probability_sum": 0.0,
                        "near_goal_stop_probability_max": None,
                        "false_stop": False,
                        "max_consecutive_turns": 0,
                        "max_alternating_turns": 0,
                    }
                    for _ in batch
                ]
                try:
                    for index, _ in batch:
                        writers.append(
                            imageio.get_writer(
                                str(worker_output / f"{split}_{index:04d}.mp4"),
                                fps=6,
                                codec="libx264",
                                ffmpeg_params=["-preset", "veryfast", "-threads", "1"],
                            )
                            if video and video_limit and (representative or index < video_limit)
                            else None
                        )
                    while active:
                        torch.cuda.synchronize(device)
                        start = time.perf_counter()
                        with torch.autocast(
                            device_type=device.type, dtype=dtype, enabled=dtype != torch.float32
                        ):
                            pointing = getattr(policy, "perception", None) is not None
                            output_batch = policy.forward_batch(
                                torch.stack([observations[i]["rgb"] for i in active]),
                                [states[i] for i in active],
                                **({"return_perception": True} if pointing else {}),
                            )
                            logits, _, next_states = output_batch[:3]
                            actions = [NavigationAction(a) for a in logits.argmax(-1).tolist()]
                        perception_records: list[dict[str, Any] | None] = [None for _ in active]
                        if pointing:
                            predictions = output_batch[3]
                            ids = {
                                name: predictions[name].argmax(-1).tolist()
                                for name in ("apos", "opos")
                            }
                            ready = predictions["arrival"].softmax(-1).tolist()
                            perception_records = [
                                {
                                    "apos": ids["apos"][j],
                                    "opos": ids["opos"][j],
                                    "arrival_probabilities": ready[j],
                                }
                                for j in range(len(active))
                            ]
                        perception_by_slot = dict(zip(active, perception_records, strict=True))
                        for action in actions:
                            action_counts[int(action)] += 1
                        torch.cuda.synchronize(device)
                        duration = time.perf_counter() - start
                        times.extend([duration] * len(active))
                        batch_times.append(duration)
                        # Privileged distance is used only for diagnostics.
                        # Decisions above remain unconditional policy argmax.
                        action_probabilities = logits.softmax(-1).tolist()
                        stop_probabilities = [p[0] for p in action_probabilities]
                        for i, action, probability in zip(
                            active, actions, stop_probabilities, strict=True
                        ):
                            observation = observations[i]
                            distance = (
                                observation["geodesic_distance"]
                                if "geodesic_distance" in observation
                                else observation["distance"]
                            )
                            near = distance < env_config.get("success_distance", 0.1)
                            diagnostics = stop_diagnostics[i]
                            if near:
                                diagnostics["near_goal_steps"] += 1
                                diagnostics["near_goal_stop_probability_sum"] += probability
                                diagnostics["near_goal_stop_probability_max"] = max(
                                    diagnostics["near_goal_stop_probability_max"] or 0, probability
                                )
                            if action == NavigationAction.STOP and not near:
                                diagnostics["false_stop"] = True
                            turn_streaks[i] = (
                                turn_streaks[i] + 1
                                if action
                                in (NavigationAction.TURN_LEFT, NavigationAction.TURN_RIGHT)
                                else 0
                            )
                            diagnostics["max_consecutive_turns"] = max(
                                diagnostics["max_consecutive_turns"], turn_streaks[i]
                            )
                            is_turn = action in (
                                NavigationAction.TURN_LEFT,
                                NavigationAction.TURN_RIGHT,
                            )
                            alternating_turn_streaks[i] = (
                                alternating_turn_streaks[i] + 1
                                if is_turn
                                and last_turn_actions[i] is not None
                                and last_turn_actions[i] != action
                                else int(is_turn)
                            )
                            last_turn_actions[i] = action if is_turn else None
                            diagnostics["max_alternating_turns"] = max(
                                diagnostics["max_alternating_turns"], alternating_turn_streaks[i]
                            )
                            writer = writers[i]
                            if writer is not None:
                                writer.append_data(
                                    video_frame(
                                        observation["rgb"],
                                        batch[i][1].goal_text,
                                        action,
                                        observation,
                                        probability,
                                        perception=perception_by_slot[i],
                                    )
                                )
                        next_observations = list(
                            envs.executor.map(
                                lambda pair: envs.clients[pair[0]].step(pair[1]),
                                zip(active, actions, strict=True),
                            )
                        )
                        remaining = []
                        for i, action, probabilities, state, observation in zip(
                            active,
                            actions,
                            action_probabilities,
                            next_states,
                            next_observations,
                            strict=True,
                        ):
                            before = observations[i]
                            states[i], observations[i] = state, observation
                            cache_bytes = state_bytes(state)
                            episode = batch[i][1]
                            writer = writers[i]
                            if writer is not None:
                                traces[i].append(
                                    {
                                        "frame_id": before["frame_id"],
                                        "action": action.name,
                                        "action_probabilities": probabilities,
                                        "distance_before": before.get(
                                            "geodesic_distance", before.get("distance")
                                        ),
                                        "distance_after": observation["geodesic_distance"],
                                        "collision": observation.get("collision", False),
                                        "done": observation["done"],
                                        "success": observation.get("success", False),
                                        **(
                                            {"perception": perception_by_slot[i]}
                                            if pointing
                                            else {}
                                        ),
                                    }
                                )
                            if observation["done"]:
                                record = {
                                    "episode_id": episode.uid,
                                    "episode_index": batch[i][0],
                                    "scene_id": episode.scene_id,
                                    "goal": episode.goal_text,
                                    **observation["metrics"],
                                    **stop_diagnostics[i],
                                }
                                append_json(worker_output / f"{split}_episodes.jsonl", record)
                                metrics.append(record)
                                if writer is not None:
                                    writer.append_data(
                                        video_frame(
                                            observation["rgb"],
                                            episode.goal_text,
                                            action,
                                            observation,
                                            terminal=True,
                                        )
                                    )
                                    trace_path = worker_output / f"{split}_{batch[i][0]:04d}.jsonl"
                                    trace_path.write_text(
                                        "".join(json.dumps(t) + "\n" for t in traces[i])
                                    )
                                    video_records.append(
                                        {
                                            **record,
                                            "video_path": str(
                                                (
                                                    worker_output / f"{split}_{batch[i][0]:04d}.mp4"
                                                ).relative_to(output)
                                            ),
                                            "trace_path": str(trace_path.relative_to(output)),
                                        }
                                    )
                            else:
                                remaining.append(i)
                        active = remaining
                finally:
                    for writer in writers:
                        if writer is not None:
                            writer.close()
            payloads = parallel.gather(
                {
                    "metrics": metrics,
                    "times": times,
                    "batch_seconds": sum(batch_times),
                    "scenes": scene_ids,
                    "categories": categories,
                    "actions": action_counts,
                    "cache_bytes": cache_bytes,
                    "peak_vram": torch.cuda.max_memory_allocated(device),
                    "video_records": video_records,
                }
            )
            metrics = [m for p in payloads for m in p["metrics"]]
            times = [t for p in payloads for t in p["times"]]
            scene_ids = set().union(*(p["scenes"] for p in payloads))
            categories = set().union(*(p["categories"] for p in payloads))
            action_counts = [
                sum(p["actions"][i] for p in payloads) for i in range(len(action_counts))
            ]
            all_videos = [r for p in payloads for r in p["video_records"]]
            selected_videos = (
                select_video_cases(
                    all_videos, video_limit, config["eval"].get("video_anchor_episodes", 1)
                )
                if representative
                else [{**r, "selection_reasons": ["fixed_first"]} for r in all_videos]
            )
            if parallel.rank() == 0 and video:
                retain_video_cases(output, all_videos, selected_videos)
                cases_by_split[split] = selected_videos
                (output / f"{split}_video_cases.json").write_text(
                    json.dumps(selected_videos, indent=2) + "\n"
                )
            if parallel.rank() == 0 and parallel.world_size() > 1:
                for record in metrics:
                    append_json(output / f"{split}_episodes.jsonl", record)
            result = {
                "update": update,
                "split": split,
                "episodes": len(metrics),
                **aggregate_metrics(metrics),
                **latency_summary(times),
                "evaluation_batch_size": count * parallel.world_size(),
                "world_size": parallel.world_size(),
                "success_distance": env_config.get("success_distance", 0.1),
                "inference_mode": config["model"].get("inference_mode", "auto"),
                "compute_dtype": config["model"]["dtype"],
                "loaded_checkpoint": str(getattr(policy, "loaded_checkpoint", "unknown")),
                "evaluated_model_update": update,
                "scene_count": len(scene_ids),
                "category_count": len(categories),
                "selection": "scene_stratified" if stratified else "sequential",
                "greedy_action_histogram": [n / max(sum(action_counts), 1) for n in action_counts],
                "policy_decisions_per_second": len(times)
                / max(p["batch_seconds"] for p in payloads)
                if times
                else 0,
                "state_bytes": max(p["cache_bytes"] for p in payloads),
                "peak_vram_bytes": max(p["peak_vram"] for p in payloads),
                "saved_video_cases": len(selected_videos),
                "video_selection": video_selection if video else "disabled",
            }
            results[split] = result
            if parallel.rank() == 0:
                append_json(Path(config["run_dir"]) / "eval_metrics.jsonl", result)
                print(json.dumps({"evaluation": result}), flush=True)
        if parallel.rank() == 0:
            (output / "summary.json").write_text(json.dumps(results, indent=2))
            if video:
                write_video_index(output, cases_by_split, update)
        return results
    finally:
        envs.close()
        policy.train(was_training)
        torch.cuda.empty_cache()


@hydra.main(version_base=None, config_path="../../../configs", config_name="config")
def main(cfg):
    config = cast(dict[str, Any], OmegaConf.to_container(cfg, resolve=True))
    parallel.initialize(config)
    try:
        seed_everything(config["seed"])
        policy = load_policy(config, preserve_master_weights=True)
        manifest = Path(policy.loaded_checkpoint) / "manifest.json"
        update = json.loads(manifest.read_text()).get("update", 0) if manifest.exists() else 0
        evaluate(policy, config, update=update, video=config["eval"]["video"])
    finally:
        if torch.distributed.is_initialized():
            torch.distributed.destroy_process_group()


if __name__ == "__main__":
    main()
