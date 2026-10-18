"""Read-only closed-loop decoding comparison on fixed original eval starts."""

import argparse
import json
import os
from pathlib import Path

import torch
import yaml

from streamnav.contracts.action import NavigationAction
from streamnav.data.schema import HabitatEpisodeSource
from streamnav.envs.habitat_client import HabitatClient
from streamnav.models.point_cot.policy import Qwen35PointCoTPolicy
from streamnav.training.diagnostics import observation_distance
from streamnav.training.pointcot_trainer import load_adapters
from streamnav.utils.seed import seed_everything


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--max-steps", type=int, default=100)
    args = parser.parse_args()
    rank = int(os.environ.get("LOCAL_RANK", 0))
    world = int(os.environ.get("WORLD_SIZE", 1))
    config = yaml.safe_load(Path(args.checkpoint, "resolved_config.yaml").read_text())
    config["device"] = f"cuda:{rank}"
    config["habitat"].update(
        gpu_device_id=rank, perception_labels=False, max_episode_steps=args.max_steps
    )
    torch.cuda.set_device(rank)
    seed_everything(config["seed"] + rank)
    policy = Qwen35PointCoTPolicy(config).eval()
    load_adapters(policy, args.checkpoint)
    source = HabitatEpisodeSource(config["eval"]["manifests"][0], seed=config["seed"])
    episodes = list(source.evaluation_episodes(config["eval"]["episodes"], True, config["seed"]))
    episode = episodes[rank]
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    results = []
    env = HabitatClient(config["habitat"], worker_id=rank)
    try:
        for name, decoding, deterministic in (
            ("flat_greedy", "flat", True),
            ("hierarchical_greedy", "hierarchical", True),
            ("flat_stochastic", "flat", False),
        ):
            seed_everything(config["seed"] + rank)
            policy.point_decoding = decoding
            policy.apos_decoding = policy.opos_decoding = decoding
            obs = env.reset(episode)
            state = policy.start_episode(episode.uid, episode.goal_text)
            initial_distance = observation_distance(obs)
            minimum, path, collisions, counts = initial_distance, 0.0, 0, [0] * 6
            trace = []
            for t in range(args.max_steps):
                features = policy.navigation_features([obs["rgb"]], [state])
                step = policy.step_batch(features, [state], deterministic=deterministic)[0]
                action = int(step.action)
                obs = env.step(NavigationAction(action))
                counts[action] += 1
                collisions += int(obs.get("collision", False))
                minimum = min(minimum, observation_distance(obs))
                path += obs["displacement"]
                trace.append(
                    {
                        "step": t,
                        "action": NavigationAction(action).name,
                        "apos": int(step.apos),
                        "opos": int(step.opos),
                        "distance": observation_distance(obs),
                        "forward_probability": float(step.action_logits.softmax(-1)[1]),
                    }
                )
                state = step.state
                if obs["done"]:
                    break
            result = {
                "mode": name,
                "episode_index": rank,
                "episode_id": episode.uid,
                "instruction": episode.goal_text,
                "initial_distance": initial_distance,
                "final_distance": observation_distance(obs),
                "minimum_distance": minimum,
                "progress": initial_distance - minimum,
                "path_length": path,
                "steps": len(trace),
                "success": bool(obs.get("success", False)),
                "collisions": collisions,
                "action_counts": counts,
            }
            results.append(result)
            (output / f"rank_{rank:03d}_{name}.json").write_text(
                json.dumps({"result": result, "trace": trace}, indent=2) + "\n"
            )
            print(json.dumps(result), flush=True)
    finally:
        env.close()
    (output / f"rank_{rank:03d}_summary.json").write_text(
        json.dumps(
            {
                "checkpoint": str(Path(args.checkpoint).resolve()),
                "fixed_original_eval_episodes": config["eval"]["episodes"],
                "diagnostic_episode_count": world,
                "max_steps": args.max_steps,
                "results": results,
            },
            indent=2,
        )
        + "\n"
    )


if __name__ == "__main__":
    torch.set_num_threads(4)
    main()
