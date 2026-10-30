"""Execute real training-only hint routes and compare all-NULL teacher behavior.

These near-goal expert trajectories validate annotations and execution, not the
learned policy's success rate. No model weights or production jobs are modified.
"""

import argparse
import json
from dataclasses import replace
from pathlib import Path

import yaml

from streamnav.contracts.action import NavigationAction
from streamnav.data.schema import HabitatEpisodeSource
from streamnav.envs.habitat_client import HabitatClient
from streamnav.training.diagnostics import observation_distance
from streamnav.training.object_hints import conditioned_hint_config


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/pointcot_15300.yaml")
    parser.add_argument("--episodes", type=int, default=8)
    parser.add_argument("--max-steps", type=int, default=120)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--output", default="runtime/reports/hint_conditioned_routes.json")
    args = parser.parse_args()
    config = yaml.safe_load(Path(args.config).read_text())
    hints = conditioned_hint_config(config)
    assert hints is not None
    hints = {**hints, "episode_probability": 1.0}
    source = HabitatEpisodeSource(config["data"]["sources"][0]["manifest"])
    episodes = list(source.evaluation_episodes(args.episodes))
    env = HabitatClient({**config["habitat"], "gpu_device_id": args.gpu}, "hint-route-validation")
    results, null_checks = [], []
    try:
        for i, original in enumerate(episodes):
            episode = replace(
                original,
                metadata={
                    **original.metadata,
                    "training_start": {
                        "kind": "boundary",
                        "distance_band": [1.0, 4.0],
                        "goal_facing_probability": 1.0,
                        "seed": 317 + i,
                    },
                },
            )
            noise = {"dropout": 0.5 if i % 2 else 0.0, "jitter": 0, "false_positive": 0.0}
            hint_config = {**hints, "noise": noise}
            observation = env.reset(episode)
            start_result = observation.get("training_start")
            rows = []
            for t in range(args.max_steps):
                bootstrap = env.get_object_supervision(hint_config)
                result = env.get_oracle_supervision(hint_config)
                assert result == env.get_oracle_supervision(hint_config)
                hint = result["hint"]
                assert hint["token"] == bootstrap["object_hint_token"]
                label = result["perception"]
                assert label["frame_id"] == observation["frame_id"]
                ready = observation_distance(observation) < config["habitat"]["success_distance"]
                assert (result["action"] == int(NavigationAction.STOP)) == ready
                if hint["token"] and hint["active"] and label.get("opos_valid"):
                    assert hint["token"] in label["opos_targets"]
                after = env.step(NavigationAction(result["action"]))
                rows.append(
                    {
                        "frame": t,
                        "distance": observation_distance(observation),
                        "action": result["action"],
                        "hint": hint,
                        "apos": label["apos"],
                        "opos": label["opos"],
                        "collision": after["collision"],
                    }
                )
                observation = after
                if after["done"]:
                    break
            case = {
                "episode": episode.uid,
                "start": rows[0]["distance"],
                "final_distance": observation_distance(observation),
                "success": observation["success"],
                "sampled_start": start_result,
                "dropout": noise["dropout"],
                "steps": len(rows),
                "active_frames": sum(r["hint"]["active"] for r in rows),
                "active_null_frames": sum(
                    r["hint"]["active"] and not r["hint"]["token"] for r in rows
                ),
                "fallbacks": sum(bool(r["hint"]["fallback"]) for r in rows),
                "collisions": sum(r["collision"] for r in rows),
                "trace": rows,
            }
            results.append(case)
            print(json.dumps({k: v for k, v in case.items() if k != "trace"}), flush=True)
            # Separate resets ensure a stateful explorer is queried once per
            # state, rather than comparing its first and second advice calls.
            if i < 2:
                env.reset(episode)
                legacy = env.get_oracle_supervision()
                env.reset(episode)
                null = env.get_oracle_supervision({**hint_config, "episode_probability": 0.0})
                assert legacy["action"] == null["action"]
                assert legacy["perception"] == null["perception"]
                assert null["hint"]["token"] == 0 and not null["hint"]["active"]
                null_checks.append({"episode": episode.uid, "labels_and_action_equal": True})
    finally:
        env.close()
    assert any(r["active_frames"] for r in results), "No valid hint approach was exercised"
    if len(results) > 1:
        assert any(r["active_null_frames"] for r in results), (
            "No hint dropout persistence exercised"
        )
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(
            {"scope": __doc__, "passed": True, "cases": results, "null_checks": null_checks},
            indent=2,
        )
        + "\n"
    )


if __name__ == "__main__":
    main()
