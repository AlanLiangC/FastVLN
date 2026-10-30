"""Parent parity, real detector/teacher alignment, and feedback-aware GPU replay."""

import argparse
import json
from dataclasses import replace
from hashlib import sha256
from pathlib import Path

import torch
import yaml

from streamnav.contracts.action import NavigationAction
from streamnav.data.schema import HabitatEpisodeSource
from streamnav.envs.habitat_client import HabitatClient
from streamnav.models.point_cot.policy import Qwen35PointCoTPolicy
from streamnav.training.detector_hints import TrainingDetectorHints
from streamnav.training.diagnostics import observation_distance
from streamnav.training.object_hints import conditioned_hint_config
from streamnav.training.pointcot_trainer import fork_signature, load_adapters, replay_records
from streamnav.utils.seed import seed_everything


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--config", default="configs/pointcot_15300.yaml")
    parser.add_argument("--output", default="runtime/reports/realhint_gpu_validation.json")
    args = parser.parse_args()
    config = yaml.safe_load(Path(args.config).read_text())
    parent = yaml.safe_load((Path(args.checkpoint) / "resolved_config.yaml").read_text())
    for section in ("data", "grounding"):
        hashes = parent[section].get("manifest_sha256", {})
        for path, digest in hashes.items():
            assert sha256(Path(path).read_bytes()).hexdigest() == digest
        config[section]["manifest_sha256"] = hashes
    assert fork_signature(config) == fork_signature(parent)
    seed_everything(config["seed"])
    policy = Qwen35PointCoTPolicy(config).eval()
    load_adapters(policy, args.checkpoint, allow_new_plan=True)
    fixture = torch.load("runtime/fixtures/navigation/rgb_inputs.pt", weights_only=False)[0]
    state = policy.start_episode(fixture["episode_id"], fixture["instruction"])
    state = state.after_action(True, action=1, displacement=0)
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        features = policy.navigation_features([fixture["rgb"]], [state], [649])
        new = policy.step_batch(features, [state], deterministic=True)[0]
        adaptation = policy.adaptation
        policy.adaptation = None
        old = policy.step_batch(features, [state], deterministic=True)[0]
        policy.adaptation = adaptation
        for field in ("apos_logits", "opos_logits", "action_logits", "arrival_logits"):
            torch.testing.assert_close(getattr(new, field), getattr(old, field), rtol=0, atol=0)
        for a, b in zip(new.state.kda_cache, old.state.kda_cache, strict=True):
            if a.recurrent is not None:
                torch.testing.assert_close(a.recurrent, b.recurrent, rtol=0, atol=0)

    # A nonzero branch must also replay exactly; zero initialization alone cannot
    # reveal a feedback record silently lost in packed teacher forcing.
    with torch.no_grad():
        policy.adaptation.motion[-1].weight.normal_(std=1e-4)
        policy.adaptation.spatial[-1].weight.normal_(std=1e-4)
    source = HabitatEpisodeSource(config["data"]["sources"][0]["manifest"], seed=71)
    episode = source.sample_episode()
    scene_data = source._load(source.current)
    easy = [
        raw
        for raw in scene_data["episodes"]
        if raw["object_category"] in ("bed", "couch", "tv", "chair", "toilet")
    ]
    if easy:
        episode = source.decode(easy[0], scene_data)
    episode = replace(
        episode,
        metadata={
            **episode.metadata,
            "training_start": {
                "kind": "near",
                "distance_band": [1.0, 3.0],
                "goal_facing_probability": 1.0,
                "seed": 17,
            },
        },
    )
    env = HabitatClient(config["habitat"], "realhint-validation")
    forced = {
        **config,
        "trainer": {
            **config["trainer"],
            "object_hints": {
                **config["trainer"]["object_hints"],
                "episode_probability": 1.0,
                "real_probability": 1.0,
            },
        },
    }
    hints = TrainingDetectorHints(forced, Path(args.output).parent / "detector_validation")
    teacher_config = conditioned_hint_config(forced)
    records, resets, packets = [], {}, []
    initial = policy.start_episode(episode.uid, episode.goal_text)
    state = initial
    try:
        observation = env.reset(episode)
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            for frame in range(12):
                packet = hints.packets([observation], [state])[0]
                pure = env.get_object_supervision(teacher_config, packet)
                supervision = env.get_oracle_supervision(teacher_config, packet)
                assert pure["object_hint_token"] == supervision["hint"]["token"] == packet["token"]
                assert (supervision["action"] == 0) == (observation_distance(observation) < 0.25)
                feature = policy.navigation_features(
                    [observation["rgb"]], [state], [packet["token"]]
                )[0]
                step = policy.step_batch([feature], [state], deterministic=False)[0]
                records.append(
                    {
                        "features": feature.cpu(),
                        "tokens": tuple(t.cpu() for t in (step.apos, step.opos, step.action)),
                        "old_log_prob": step.log_prob.cpu(),
                    }
                )
                packets.append(
                    {
                        "frame": frame,
                        "source": packet["source"],
                        "token": packet["token"],
                        "quality_valid": supervision["hint"].get("quality_valid"),
                        "feedback": feature.feedback,
                    }
                )
                after = env.step(NavigationAction(int(step.action)))
                state = step.state.after_action(
                    after["collision"], action=int(step.action), displacement=after["displacement"]
                )
                observation = after
                if after["done"] and frame < 11:
                    episode = source.sample_episode()
                    observation = env.reset(episode)
                    state = policy.start_episode(episode.uid, episode.goal_text)
                    resets[frame + 1] = (episode.uid, episode.goal_text)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            replay = replay_records(policy, records, initial, resets=resets)
            old_probs = torch.stack([r["old_log_prob"] for r in records]).to(policy.device)
            error = float((replay.log_prob.detach() - old_probs).abs().max())
            assert error <= config["trainer"]["replay_tolerance"], error
            loss = -replay.log_prob.mean() + replay.hint_quality_logits.float().square().mean()
            loss.backward()
        assert all(torch.isfinite(p.grad).all() for p in policy.parameters() if p.grad is not None)
        assert policy.adaptation.motion[-1].weight.grad.norm() > 0
        # Exercise coordinate gradients independently of whether a small real
        # detector sample happens to contain a supported point.
        policy.zero_grad(set_to_none=True)
        policy.adaptation(
            torch.tensor([649], device=policy.device), torch.ones(1, 12, device=policy.device)
        ).sum().backward()
        assert policy.adaptation.spatial[-1].weight.grad.norm() > 0
        report = {
            "passed": True,
            "parent_zero_output_exact_parity": True,
            "nonzero_feedback_replay_error": error,
            "replay_tolerance": config["trainer"]["replay_tolerance"],
            "real_detector_teacher_frames": packets,
            "finite_gradients": True,
            "scope": "Integration and gradients only, not a navigation performance claim.",
        }
        Path(args.output).write_text(json.dumps(report, indent=2) + "\n")
        print(json.dumps(report))
    finally:
        env.close()
        hints.close()


if __name__ == "__main__":
    main()
