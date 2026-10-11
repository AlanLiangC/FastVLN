"""Replay recorded evaluation actions and collect diagnostic-only weak labels.

Runs in the Habitat Python environment. No labels enter policy observations and
recorded actions are never changed by the teacher. Original validation starts
and episode identities are retained.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from env_factory import HabitatObjectNavEnv
from perception_labels import PerceptionLabeler, project_camera

from streamnav.contracts.action import NavigationAction as A
from streamnav.errors import OracleUnavailableError


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--labels-only", action="store_true")
    args = parser.parse_args()
    request = json.loads(Path(args.input).read_text())
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    config = {**request["habitat"], "perception_labels": True}
    env = HabitatObjectNavEnv(config)
    results = []
    try:
        for case in request["cases"]:
            episode = case["episode"]
            trace = [json.loads(s) for s in Path(case["trace"]).read_text().splitlines()]
            rgb, info = env.reset(episode)
            frames = (
                None
                if args.labels_only
                else np.lib.format.open_memmap(
                    output / (case["name"] + "_rgb.npy"),
                    mode="w+",
                    dtype=np.uint8,
                    shape=(len(trace), *rgb.shape),
                )
            )
            records, max_error = [], 0.0
            labeler = PerceptionLabeler(env)
            for index, saved in enumerate(trace):
                assert env.steps == saved["frame_id"] == index
                error = abs(env.previous_distance - saved["distance_before"])
                max_error = max(max_error, error)
                if error > 1e-4:
                    raise ValueError("Recorded and replayed camera poses diverged")
                if frames is not None:
                    frames[index] = rgb
                oracle_error = None
                try:
                    # Emulate the train-only success-aligned teacher solely for
                    # annotation diagnostics; validation actions remain fixed.
                    advice = (
                        A.STOP
                        if env.previous_distance < config["success_distance"]
                        else env.oracle()
                    )
                except OracleUnavailableError as exc:
                    advice, oracle_error = None, str(exc)
                label = labeler.labels(advice)
                if oracle_error is not None:
                    label.update(apos_valid=False, apos_confidence=0.0)
                state = env.sim.get_agent(0).get_state()
                camera = state.sensor_states["rgb"]
                anchors = []
                for goal in episode["goals"]:
                    pixel, z = project_camera(
                        goal["position"], camera, config["width"], config["height"], config["hfov"]
                    )
                    surface_depth = None
                    if pixel is not None:
                        u, v = pixel
                        if 1 <= u < config["width"] - 1 and 1 <= v < config["height"] - 1:
                            depth = env._sensor_observations()["perception_depth"]
                            patch = depth[int(v) - 1 : int(v) + 2, int(u) - 1 : int(u) + 2]
                            valid = patch[(patch > 0.05) & (patch < 19.8) & np.isfinite(patch)]
                            if valid.size >= 5:
                                surface_depth = float(np.median(valid))
                    anchors.append(
                        {
                            "pixel": pixel,
                            "z": z,
                            "position": goal["position"],
                            "object_category": goal.get("object_category"),
                            "surface_depth": surface_depth,
                            "depth_residual": surface_depth - z
                            if surface_depth is not None
                            else None,
                        }
                    )
                records.append(
                    {
                        "frame_id": index,
                        "distance": env.previous_distance,
                        "teacher_action": advice.name if advice is not None else None,
                        "teacher_error": oracle_error,
                        "label": label,
                        "anchors": anchors,
                    }
                )
                rgb, info = env.step(A[saved["action"]])
                assert abs(info["geodesic_distance"] - saved["distance_after"]) <= 1e-4
                assert info["done"] == saved["done"]
                assert info["success"] == saved["success"]
            if frames is not None:
                frames.flush()
            del frames
            (output / (case["name"] + "_labels.json")).write_text(json.dumps(records))
            result = {
                "name": case["name"],
                "frames": len(records),
                "distance_error_max": max_error,
                "object_point_labels": sum(
                    x["label"]["opos_valid"] and x["label"]["opos"] > 0 for x in records
                ),
                "object_none_labels": sum(
                    x["label"]["opos_valid"] and x["label"]["opos"] == 0 for x in records
                ),
                "oracle_errors": sum(x["teacher_error"] is not None for x in records),
            }
            results.append(result)
            print(json.dumps(result), flush=True)
    finally:
        env.close()
    (output / "simulator_replay.json").write_text(json.dumps(results, indent=2) + "\n")


if __name__ == "__main__":
    main()
