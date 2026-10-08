import json
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
from streamnav.data.schema import HabitatEpisodeSource
from streamnav.envs.habitat_client import VectorHabitatEnvs
from streamnav.evaluation.latency_metrics import latency_summary
from streamnav.evaluation.navigation_metrics import aggregate_metrics
from streamnav.models.qwen35_kda.cache import state_bytes
from streamnav.training import distributed as parallel
from streamnav.training.checkpoint import load_policy
from streamnav.utils.logging import append_json
from streamnav.utils.seed import seed_everything


def video_frame(rgb, instruction, action, info):
    image = Image.fromarray(rgb.cpu().numpy())
    width, height = image.size
    canvas = Image.new("RGB", (width, ((height + 64 + 15) // 16) * 16), "#121827")
    canvas.paste(image, (0, 0))
    draw = ImageDraw.Draw(canvas)
    draw.text((10, height + 6), instruction, fill="white")
    draw.text(
        (10, height + 29),
        f"{action.name} | step {info['frame_id']} | distance {info['geodesic_distance']:.2f} m",
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
    envs = VectorHabitatEnvs(env_config, count)
    results = {}
    dtype = getattr(torch, config["model"]["dtype"])
    try:
        for manifest in config["eval"]["manifests"]:
            source = HabitatEpisodeSource(manifest, seed=config["seed"])
            split = source.manifest["dataset_id"] + "_" + source.manifest["split"]
            metrics, times, batch_times = [], [], []
            stratified = config["eval"].get("stratified", True)
            iterator = (
                (i, e)
                for i, e in enumerate(source.evaluation_episodes(limit, stratified, config["seed"]))
                if i % parallel.world_size() == parallel.rank()
            )
            scene_ids, categories = set(), set()
            action_counts = [0] * 4
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
                active = list(range(len(batch)))
                try:
                    for index, _ in batch:
                        writers.append(
                            imageio.get_writer(
                                str(worker_output / f"{split}_{index:04d}.mp4"), fps=6
                            )
                            if video and index < config["eval"].get("video_episodes", 1)
                            else None
                        )
                    while active:
                        torch.cuda.synchronize(device)
                        start = time.perf_counter()
                        with torch.autocast(
                            device_type=device.type, dtype=dtype, enabled=dtype != torch.float32
                        ):
                            logits, _, next_states = policy.forward_batch(
                                torch.stack([observations[i]["rgb"] for i in active]),
                                [states[i] for i in active],
                            )
                            actions = [NavigationAction(a) for a in logits.argmax(-1).tolist()]
                        for action in actions:
                            action_counts[int(action)] += 1
                        torch.cuda.synchronize(device)
                        duration = time.perf_counter() - start
                        times.extend([duration] * len(active))
                        batch_times.append(duration)
                        next_observations = list(
                            envs.executor.map(
                                lambda pair: envs.clients[pair[0]].step(pair[1]),
                                zip(active, actions, strict=True),
                            )
                        )
                        remaining = []
                        for i, action, state, observation in zip(
                            active, actions, next_states, next_observations, strict=True
                        ):
                            states[i], observations[i] = state, observation
                            cache_bytes = state_bytes(state)
                            episode = batch[i][1]
                            writer = writers[i]
                            if writer is not None:
                                writer.append_data(
                                    video_frame(
                                        observation["rgb"], episode.goal_text, action, observation
                                    )
                                )
                            if observation["done"]:
                                record = {
                                    "episode_id": episode.uid,
                                    "goal": episode.goal_text,
                                    **observation["metrics"],
                                }
                                append_json(worker_output / f"{split}_episodes.jsonl", record)
                                metrics.append(record)
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
                }
            )
            metrics = [m for p in payloads for m in p["metrics"]]
            times = [t for p in payloads for t in p["times"]]
            scene_ids = set().union(*(p["scenes"] for p in payloads))
            categories = set().union(*(p["categories"] for p in payloads))
            action_counts = [sum(p["actions"][i] for p in payloads) for i in range(4)]
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
            }
            results[split] = result
            if parallel.rank() == 0:
                append_json(Path(config["run_dir"]) / "eval_metrics.jsonl", result)
                print(json.dumps({"evaluation": result}), flush=True)
        if parallel.rank() == 0:
            (output / "summary.json").write_text(json.dumps(results, indent=2))
        return results
    finally:
        envs.close()
        policy.train(was_training)
        torch.cuda.empty_cache()


@hydra.main(version_base=None, config_path="../../../configs", config_name="config")
def main(cfg):
    config = cast(dict[str, Any], OmegaConf.to_container(cfg, resolve=True))
    seed_everything(config["seed"])
    policy = load_policy(config, preserve_master_weights=True)
    manifest = Path(policy.loaded_checkpoint) / "manifest.json"
    update = json.loads(manifest.read_text()).get("update", 0) if manifest.exists() else 0
    evaluate(policy, config, update=update, video=config["eval"]["video"])


if __name__ == "__main__":
    main()
