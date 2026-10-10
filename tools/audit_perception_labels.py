"""Render real training-only pointing labels at original and goal-viewpoint starts."""

import argparse
import json
from collections import Counter
from dataclasses import replace
from pathlib import Path

import imageio.v2 as imageio
import torch
import yaml
from PIL import Image, ImageDraw

from streamnav.contracts.action import NavigationAction
from streamnav.data.schema import HabitatEpisodeSource
from streamnav.envs.habitat_client import VectorHabitatEnvs
from streamnav.errors import OracleUnavailableError
from streamnav.evaluation.pointing_overlay import draw_pointing_overlay


def frame(rgb, label, goal, distance):
    image = Image.fromarray(rgb.numpy())
    width, height = image.size
    canvas = Image.new("RGB", (width, ((height + 80 + 15) // 16) * 16), "#121827")
    canvas.paste(image)
    draw_pointing_overlay(
        canvas,
        {name: label[name] if label[name + "_valid"] else 0 for name in ("apos", "opos")},
        (width, height),
    )
    draw = ImageDraw.Draw(canvas)
    draw.text(
        (10, height + 4),
        f"{goal} | distance={distance:.2f} m | frame={label['frame_id']}",
        fill="white",
    )
    draw.text(
        (10, height + 25),
        f"APOS={label['apos']} valid={label['apos_valid']} | "
        f"OPOS={label['opos']} valid={label['opos_valid']}",
        fill="white",
    )
    draw.text(
        (10, height + 46),
        f"arrival={label['arrival']} | object label: weak geometry + depth",
        fill="white",
    )
    import numpy as np

    return np.asarray(canvas)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--scenes", type=int, default=4)
    parser.add_argument("--steps", type=int, default=32)
    parser.add_argument("--gpu", type=int, default=0)
    args = parser.parse_args()
    config = yaml.safe_load(Path(args.config).read_text())
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    source = HabitatEpisodeSource(config["data"]["sources"][0]["manifest"])
    envs = VectorHabitatEnvs(
        {**config["habitat"], "perception_labels": True, "gpu_device_id": args.gpu}, 1
    )
    counts, cases = Counter(), []
    try:
        for scene, entry in enumerate(source.files[: args.scenes]):
            data = source._load(entry)
            preferred = {"refrigerator", "microwave", "glass", "couch", "sink", "lamp"}
            raw = next(
                (e for e in data["episodes"] if e["object_category"] in preferred),
                data["episodes"][0],
            )
            original = source.decode(raw, data)
            viewpoint = original.goals[0]["view_points"][0]["agent_state"]
            for kind in ("original", "goal_viewpoint"):
                episode = (
                    original
                    if kind == "original"
                    else replace(
                        original,
                        episode_id=original.episode_id + "-label-audit",
                        start_position=viewpoint["position"],
                        start_rotation=viewpoint["rotation"],
                    )
                )
                name = f"scene{scene:02d}_{kind}"
                records, error = [], None
                observation = envs.reset([episode])[0]
                writer = imageio.get_writer(
                    str(output / (name + ".mp4")),
                    fps=6,
                    codec="libx264",
                    ffmpeg_params=["-preset", "veryfast", "-threads", "1"],
                )
                try:
                    for step in range(args.steps):
                        try:
                            supervision = envs.clients[0].get_oracle_supervision()
                        except OracleUnavailableError as exc:
                            error = str(exc)
                            break
                        label = supervision["perception"]
                        assert label["frame_id"] == observation["frame_id"]
                        assert label["episode_id"] == episode.uid
                        counts["frames"] += 1
                        for channel in ("apos", "opos", "arrival"):
                            if label[channel + "_valid"]:
                                counts[channel + "_valid"] += 1
                                category = (
                                    ("point" if 0 < label[channel] <= 1296 else str(label[channel]))
                                    if channel != "arrival"
                                    else str(label[channel])
                                )
                                counts[channel + "_" + category] += 1
                        distance = observation.get("geodesic_distance", observation.get("distance"))
                        rendered = frame(observation["rgb"], label, episode.goal_text, distance)
                        writer.append_data(rendered)
                        if step == 0 or (
                            label["opos_valid"] and label["opos"] > 0 and counts["opos_point"] <= 10
                        ):
                            Image.fromarray(rendered).save(output / f"{name}_{step:03d}.png")
                        action = NavigationAction(supervision["action"])
                        records.append({"action": action.name, "distance": distance, **label})
                        observation = envs.step([action])[0]
                        if observation["done"]:
                            break
                finally:
                    writer.close()
                case = {
                    "name": name,
                    "episode": episode.uid,
                    "goal": episode.goal_text,
                    "records": records,
                    "oracle_error": error,
                }
                cases.append(case)
                print(
                    json.dumps(
                        {
                            "case": name,
                            "goal": episode.goal_text,
                            "frames": len(records),
                            "error": error,
                        }
                    ),
                    flush=True,
                )
    finally:
        envs.close()
    report = {
        "counts": dict(counts),
        "cases": cases,
        "scope": "training annotation diagnostic; goal-viewpoint starts are not navigation evaluation",
    }
    (output / "audit.json").write_text(json.dumps(report, indent=2) + "\n")
    cards = [
        f'<article><h3>{c["name"]}: {c["goal"]}</h3><video controls src="{c["name"]}.mp4"></video></article>'
        for c in cases
    ]
    (output / "index.html").write_text(
        '<!doctype html><meta charset="utf-8"><title>Pointing labels</title>'
        "<style>body{font:16px system-ui}video{width:480px}</style>"
        "<h1>Training label audit</h1><p>Green: local affordance; pink: weak object point.</p>"
        + "".join(cards)
    )
    print(json.dumps(dict(counts)), flush=True)


if __name__ == "__main__":
    torch.set_num_threads(1)
    main()
