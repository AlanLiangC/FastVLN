"""Render verified train target viewpoints, then turn away; never train on this audit."""

import argparse
import json
from dataclasses import replace
from pathlib import Path

import yaml
from PIL import Image, ImageDraw

from streamnav.contracts.action import NavigationAction
from streamnav.contracts.perception import decode_pixel
from streamnav.data.schema import HabitatEpisodeSource
from streamnav.envs.habitat_client import HabitatClient


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scenes", type=int, default=6)
    parser.add_argument("--output", default="runtime/reports/instance_label_audit")
    args = parser.parse_args()
    output = Path(args.output)
    output.mkdir(exist_ok=True)
    config = yaml.safe_load(Path("configs/pointcot.yaml").read_text())
    source = HabitatEpisodeSource(config["data"]["sources"][0]["manifest"])
    ready = [
        p
        for p in Path("runtime/data/scene_datasets/hm3d/train").glob("*/*.semantic.glb")
        if p.with_suffix(".txt").exists()
    ]
    names = {p.name.removesuffix(".semantic.glb") for p in ready}
    entries = [e for e in source.files if Path(e["path"]).name.removesuffix(".json.gz") in names][
        : args.scenes
    ]
    env = HabitatClient(config["habitat"], "label-audit")
    records = []
    try:
        for index, entry in enumerate(entries):
            data = source._load(entry)
            episode = source.decode(data["episodes"][0], data)
            viewpoint = episode.goals[0]["view_points"][0]["agent_state"]
            episode = replace(
                episode, start_position=viewpoint["position"], start_rotation=viewpoint["rotation"]
            )
            observation = env.reset(episode)
            for t in range(12):
                label = env.get_object_supervision()
                record = {"scene": episode.scene_id, "goal": episode.goal_text, "frame": t, **label}
                records.append(record)
                if t in (0, 4, 8):
                    image = Image.fromarray(observation["rgb"].numpy())
                    draw = ImageDraw.Draw(image)
                    point = (
                        decode_pixel(label["opos"], image.width, image.height)
                        if label["opos_valid"]
                        else None
                    )
                    if point:
                        x, y = point
                        draw.ellipse(
                            (x - 8, y - 8, x + 8, y + 8), fill="#39ff14", outline="white", width=3
                        )
                    draw.rectangle((0, 0, image.width, 26), fill="black")
                    draw.text(
                        (5, 5),
                        f"{episode.goal_text} | GT OPOS={label['opos']} valid={label['opos_valid']}",
                        fill="white",
                    )
                    image.save(output / f"{index:02d}_{t:02d}.jpg")
                observation = env.step(NavigationAction.TURN_LEFT)
    finally:
        env.close()
    summary = {
        "scenes": len(entries),
        "frames": len(records),
        "visible": sum(r["opos_valid"] and r["opos"] > 0 for r in records),
        "absent": sum(r["opos_valid"] and r["opos"] == 0 for r in records),
        "unknown": sum(not r["opos_valid"] for r in records),
        "records": records,
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps({k: v for k, v in summary.items() if k != "records"}))


if __name__ == "__main__":
    main()
