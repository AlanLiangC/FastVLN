"""Measure RGB/goal sensitivity on identical real Habitat frames (not an SR score)."""

import argparse
import json
from pathlib import Path

import torch
import yaml
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

from streamnav.contracts.action import NavigationAction
from streamnav.data.mixture import make_source
from streamnav.envs.habitat_client import VectorHabitatEnvs
from streamnav.training.checkpoint import load_policy
from streamnav.utils.seed import seed_everything


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--steps", type=int, default=32)
    parser.add_argument("--goal-conditioning", choices=["episode", "nav_query", "chat_query"])
    parser.add_argument("--kda-output-norm", choices=["enabled", "disabled"])
    args = parser.parse_args()
    with initialize_config_dir(version_base=None, config_dir=str(Path("configs").resolve())):
        config = OmegaConf.to_container(compose(config_name="config"), resolve=True)
    saved = Path(args.checkpoint) / "resolved_config.yaml"
    if saved.exists():
        config = yaml.safe_load(saved.read_text())
    config.update(checkpoint=args.checkpoint, device=args.device)
    config["habitat"]["gpu_device_id"] = int(args.device.split(":")[-1])
    seed_everything(2025)
    policy = load_policy(config, preserve_master_weights=True).eval()
    if args.goal_conditioning:
        policy.backbone.goal_conditioning = args.goal_conditioning
    if args.kda_output_norm:
        enabled = args.kda_output_norm == "enabled"
        policy.backbone.kda_output_norm = enabled
        for layer in policy.backbone.layers:
            if hasattr(layer.mixer, "output_norm"):
                layer.mixer.output_norm = enabled
    envs = VectorHabitatEnvs(config["habitat"], 1)
    source = make_source(config["data"], 2025)
    records = []
    pooled = []
    # Chat pooling runs once per environment; capture the assembled actor input
    # so every RGB/goal control has its own row in both model paths.
    hook = policy.actor_critic.register_forward_pre_hook(
        lambda m, x: pooled.append(x[0].detach().float())
    )
    try:
        episode = source.sample_episode()
        observation = envs.reset([episode])[0]
        goals = [episode.goal_text, "Find a toilet.", "Find a lamp.", episode.goal_text]
        with torch.autocast("cuda", dtype=torch.bfloat16):
            states = [policy.start_episode(str(i), goal) for i, goal in enumerate(goals)]
        for step in range(args.steps):
            rgb = observation["rgb"]
            frames = torch.stack([rgb, rgb, rgb, torch.zeros_like(rgb)])
            pooled.clear()
            with torch.autocast("cuda", dtype=torch.bfloat16):
                logits, values, states = policy.forward_batch(frames, states)
            hidden = pooled[0]
            probabilities = logits.softmax(-1)
            records.append(
                {
                    "step": step + 1,
                    "probabilities": probabilities.cpu().tolist(),
                    "hidden_rms": hidden.square().mean(-1).sqrt().cpu().tolist(),
                    "hidden_max_abs": hidden.abs().amax(-1).cpu().tolist(),
                    "hidden_relative_change": (
                        (hidden[1:] - hidden[0]).norm(dim=-1) / hidden[0].norm()
                    )
                    .cpu()
                    .tolist(),
                    "probability_tv": ((probabilities[1:] - probabilities[0]).abs().sum(-1) / 2)
                    .cpu()
                    .tolist(),
                    "distance": observation.get("geodesic_distance", observation.get("distance")),
                }
            )
            expert = envs.get_oracle_actions()[0]
            if not isinstance(expert, NavigationAction):
                break
            observation = envs.step([expert])[0]
            if observation["done"]:
                break
        report = {
            "checkpoint": policy.loaded_checkpoint,
            "goal_conditioning": policy.backbone.goal_conditioning,
            "kda_output_norm": policy.backbone.kda_output_norm,
            "episode": episode.uid,
            "goals": goals,
            "controls": "Same expert trajectory RGB for columns 0/1/2; black RGB for column 3. No SR claim.",
            "records": records,
        }
        path = Path(args.output)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(report, indent=2))
        print(
            json.dumps({"output": str(path), "first": records[0], "last": records[-1]}), flush=True
        )
    finally:
        hook.remove()
        envs.close()


if __name__ == "__main__":
    main()
