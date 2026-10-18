"""Audit APOS/action learning on real expert trajectories without weight updates."""

import argparse
import json
import os
from pathlib import Path

import torch
import yaml
from torch.nn import functional as F

from streamnav.contracts.action import NavigationAction
from streamnav.data.mixed_starts import MixedStartEpisodeSource
from streamnav.data.schema import HabitatEpisodeSource
from streamnav.envs.habitat_client import HabitatClient
from streamnav.models.point_cot.policy import Qwen35PointCoTPolicy
from streamnav.models.policy.perception import decode_point_logits, point_type_logits
from streamnav.training.pointcot_trainer import load_adapters
from streamnav.utils.seed import seed_everything


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--steps", type=int, default=80)
    args = parser.parse_args()
    rank = int(os.environ.get("LOCAL_RANK", 0))
    config = yaml.safe_load(Path(args.checkpoint, "resolved_config.yaml").read_text())
    config["device"] = f"cuda:{rank}"
    config["habitat"]["gpu_device_id"] = rank
    torch.cuda.set_device(rank)
    seed_everything(config["seed"] + rank)
    policy = Qwen35PointCoTPolicy(config).eval()
    policy.point_decoding = "flat"
    load_adapters(policy, args.checkpoint)
    sampler = HabitatEpisodeSource(
        config["data"]["sources"][0]["manifest"],
        seed=config["seed"] + 1000 * rank,
        iterator_options=config["data"]["iterator_options"],
    )
    saved = torch.load(
        Path(args.checkpoint, "optimizer.pt"), map_location="cpu", weights_only=False
    )
    sampler.load_state_dict(saved["ranks"][rank]["source"])
    del saved
    source = MixedStartEpisodeSource(sampler, config["curriculum"])
    modes = ("flat", "apos_hierarchical", "teacher_apos", "flat_reset", "teacher_apos_reset")
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    env = HabitatClient(config["habitat"], worker_id=rank)
    rows, samples = [], []
    episode, obs, states = None, None, None
    try:
        for t in range(args.steps):
            if obs is None or obs.get("done", False):
                episode = source.sample_episode()
                obs = env.reset(episode)
                states = [policy.start_episode(episode.uid, episode.goal_text) for _ in modes]
            supervision = env.get_oracle_supervision()
            expert, label = supervision["action"], supervision["perception"]
            for i, mode in enumerate(modes):
                if mode.endswith("_reset"):
                    states[i] = policy.start_episode(episode.uid, episode.goal_text)
            feature = policy.frame_features(obs["rgb"], episode.goal_text)[0]
            context = policy.begin_batch([feature] * len(modes), states)
            logits = policy.output_projection(context.hidden, policy.apos_rows)
            apos = logits.argmax(-1)
            apos[1] = decode_point_logits(logits[1], hierarchical=True)
            if label["apos_valid"]:
                apos[2] = apos[4] = label["apos"]
            # APOS alone varies. OPOS stays flat; the appended action always
            # matches the expert action actually executed in this trajectory.
            steps = policy.step_batch(
                [feature] * len(modes),
                states,
                tokens=[(a, None, torch.tensor(expert, device=policy.device)) for a in apos],
                deterministic=True,
            )
            row = {
                "frame": t,
                "episode_id": episode.uid,
                "expert_action": expert,
                "label": label,
                "start_kind": obs.get("training_start", {}).get("kind", "original"),
                "modes": {},
            }
            for mode, step in zip(modes, steps, strict=True):
                probabilities = step.action_logits.softmax(-1)
                row["modes"][mode] = {
                    "apos": int(step.apos),
                    "greedy_action": int(step.action_logits.argmax()),
                    "expert_probability": float(probabilities[expert]),
                    "action_ce": float(
                        F.cross_entropy(
                            step.action_logits[None], torch.tensor([expert], device=policy.device)
                        )
                    ),
                    "point_mass": float(point_type_logits(step.apos_logits).softmax(-1)[1]),
                }
            samples.append(
                {
                    "features": feature.cpu(),
                    "tokens": tuple(
                        x.cpu() for x in (steps[0].apos, steps[0].opos, steps[0].action)
                    ),
                    "label": label,
                    "expert": expert,
                    "episode_id": episode.uid,
                    "instruction": episode.goal_text,
                    "frame_id": obs["frame_id"],
                }
            )
            rows.append(row)
            states = [step.state for step in steps]
            obs = env.step(NavigationAction(expert))
            source.step_taken()
    finally:
        env.close()
    result = {
        "checkpoint": str(Path(args.checkpoint).resolve()),
        "rank": rank,
        "frames": len(rows),
        "modes": {},
    }
    for mode in modes:
        result["modes"][mode] = {
            "action_accuracy": sum(
                r["modes"][mode]["greedy_action"] == r["expert_action"] for r in rows
            )
            / len(rows),
            "action_ce": sum(r["modes"][mode]["action_ce"] for r in rows) / len(rows),
            "per_action": {
                str(a): {
                    "count": sum(r["expert_action"] == a for r in rows),
                    "correct": sum(
                        r["expert_action"] == a and r["modes"][mode]["greedy_action"] == a
                        for r in rows
                    ),
                }
                for a in range(6)
            },
        }
    (output / f"rank_{rank:03d}.json").write_text(
        json.dumps({"summary": result, "rows": rows}, indent=2) + "\n"
    )
    torch.save(samples, output / f"rank_{rank:03d}_samples.pt")
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    torch.set_num_threads(4)
    main()
