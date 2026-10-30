"""Replay saved actions to audit point coordinates without rerunning the policy."""

import argparse
import html
import json
from pathlib import Path

import imageio.v2 as imageio
import numpy as np
import yaml
from PIL import Image, ImageDraw

from streamnav.contracts.action import NavigationAction
from streamnav.contracts.perception import decode_pixel
from streamnav.data.schema import HabitatEpisodeSource
from streamnav.envs.habitat_client import HabitatClient
from streamnav.training.object_hints import evaluation_object_hint_packet


def save_diagnostic(rgb, label, prediction, packet, path):
    image = Image.fromarray(rgb.numpy())
    draw = ImageDraw.Draw(image)
    for name, token, color in (
        ("GT-O", label["opos"] if label["opos_valid"] else 0, "#00ffff"),
        ("HINT", packet["token"], "#ffb000"),
        ("PRED-O", prediction["opos"], "#ff30e8"),
    ):
        point = decode_pixel(token, image.width, image.height)
        if point is None:
            continue
        x, y = point
        draw.ellipse((x - 7, y - 7, x + 7, y + 7), outline="black", width=5)
        draw.ellipse((x - 7, y - 7, x + 7, y + 7), outline=color, width=3)
        draw.text((max(0, x - 12), max(0, y - 22)), name, fill=color, stroke_width=1)
    text = (
        f"GT-O={label['opos']} valid={label['opos_valid']} | "
        f"PRED-O={prediction['opos']} P(point)={prediction['opos_point_probability']:.3f}\n"
        f"HINT={packet['token']} source={packet['source']}"
    )
    canvas = Image.new("RGB", (image.width, image.height + 40), "black")
    canvas.paste(image)
    ImageDraw.Draw(canvas).multiline_text((5, image.height + 3), text, fill="white")
    canvas.save(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", default="runs/evaluation_15300")
    parser.add_argument("--update", type=int, required=True)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--output", default="runtime/reports/eval_pointing_audit")
    args = parser.parse_args()
    root, output = Path(args.run_dir), Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    recipe = root / "checkpoints" / f"update_{args.update:07d}" / "resolved_config.yaml"
    if not recipe.exists():
        recipe = root / "resolved_config.yaml"
    config = yaml.safe_load(recipe.read_text())
    base = root / "evaluation" / f"update_{args.update:07d}"
    environment = {
        **config["habitat"],
        "gpu_device_id": args.gpu,
        "max_episode_steps": config["eval"]["max_steps"],
        "perception_labels": False,
    }
    env = HabitatClient(environment, "point-video-audit")
    records, summaries = [], []
    try:
        for setting in ("no_hint", "simulated_hint"):
            cases = json.loads((base / setting / "video_cases.json").read_text())["splits"]
            evaluated = json.loads((base / setting / "summary.json").read_text())
            for manifest in config["eval"]["manifests"]:
                source = HabitatEpisodeSource(manifest, seed=config["seed"])
                split = source.manifest["dataset_id"] + "_" + source.manifest["split"]
                episodes = list(
                    source.evaluation_episodes(
                        config["eval"]["full_episodes"], True, config["seed"]
                    )
                )
                for case in cases[split]:
                    if setting == "no_hint" and not case["success"]:
                        continue
                    episode = episodes[case["episode_index"]]
                    assert episode.uid == case["episode_id"]
                    rows = [
                        json.loads(s)
                        for s in (base / setting / case["trace_path"]).read_text().splitlines()
                    ]
                    video = imageio.get_reader(str(base / setting / case["video_path"]))
                    observation = env.reset(episode)
                    trace_records, counts = [], {}
                    for index, row in enumerate(rows):
                        assert observation["frame_id"] == row["frame_id"]
                        distance = observation.get("geodesic_distance", observation.get("distance"))
                        assert abs(distance - row["distance_before"]) < 0.002
                        label = env.get_object_supervision()
                        packet = (
                            evaluation_object_hint_packet(
                                label,
                                episode.uid,
                                row["frame_id"],
                                config["seed"],
                                evaluated[split].get(
                                    "hint_noise",
                                    config["eval"].get(
                                        "hint_noise", config["trainer"]["object_hints"]["noise"]
                                    ),
                                ),
                            )
                            if setting == "simulated_hint"
                            else {"token": 0, "source": "no_hint"}
                        )
                        prediction = row["perception"]
                        assert packet["token"] == prediction.get("object_hint_token", 0)
                        visible = label["opos_valid"] and label["opos"] > 0
                        targets = label.get("opos_targets", [])
                        constrained = evaluation_object_hint_packet(
                            label,
                            episode.uid,
                            row["frame_id"],
                            config["seed"],
                            {
                                **config["trainer"]["object_hints"]["noise"],
                                "constrain_to_support": True,
                            },
                        )
                        hint = packet["token"]
                        on_target = hint in targets if hint and label["opos_valid"] else None
                        record = {
                            "setting": setting,
                            "split": split,
                            "episode_index": case["episode_index"],
                            "frame_id": row["frame_id"],
                            "gt_valid": label["opos_valid"],
                            "gt_token": label["opos"],
                            "gt_visible_pixels": label.get("opos_visible_pixels", 0),
                            "predicted_token": prediction["opos"],
                            "point_probability": prediction["opos_point_probability"],
                            "conditional_location_peak": prediction[
                                "opos_conditional_location_peak"
                            ],
                            "hint_token": hint,
                            "hint_source": packet["source"],
                            "hint_on_target": on_target,
                            "hierarchical_visible": prediction["opos_point_probability"] > 0.5,
                            "constrained_training_hint": constrained["token"],
                        }
                        conditions = {
                            "frames": True,
                            "visible": visible,
                            "absent": label["opos_valid"] and not visible,
                            "unknown": not label["opos_valid"],
                            "predicted_points": prediction["opos"] > 0,
                            "visible_predicted_none": visible and prediction["opos"] == 0,
                            "visible_hierarchical_point": visible
                            and record["hierarchical_visible"],
                            "absent_hierarchical_point": label["opos_valid"]
                            and not visible
                            and record["hierarchical_visible"],
                            "hint_frames": bool(hint),
                            "random_false_positive": packet["source"] == "false_positive",
                            "jitter_hint": packet["source"] == "instance_jitter",
                            "jitter_off_target": packet["source"] == "instance_jitter"
                            and on_target is False,
                            "constrained_jitter_off_target": constrained["source"]
                            == "instance_jitter"
                            and constrained["token"] not in targets,
                        }
                        sample = (
                            (visible and counts.get("visible", 0) < 2)
                            or (hint and counts.get(packet["source"], 0) < 2)
                            or index == len(rows) - 1
                        )
                        if sample:
                            name = f"{setting}_{split}_{case['episode_index']:04d}_{index:04d}.png"
                            save_diagnostic(
                                observation["rgb"], label, prediction, packet, output / name
                            )
                            record["image"] = name
                            rendered = observation["rgb"].numpy().astype(np.float32)
                            encoded = video.get_data(index)[: rendered.shape[0]].astype(np.float32)
                            record["video_rgb_median_absolute_error"] = float(
                                np.median(np.abs(rendered - encoded))
                            )
                        for key, condition in conditions.items():
                            counts[key] = counts.get(key, 0) + int(condition)
                        if hint:
                            counts[packet["source"]] = counts.get(packet["source"], 0) + 1
                        trace_records.append(record)
                        observation = env.step(NavigationAction[row["action"]])
                    video.close()
                    assert bool(observation["success"]) == bool(case["success"])
                    summaries.append(
                        {
                            "setting": setting,
                            "split": split,
                            "episode_index": case["episode_index"],
                            "goal": case["goal"],
                            "success": case["success"],
                            **counts,
                        }
                    )
                    records.extend(trace_records)
                    print(json.dumps(summaries[-1]), flush=True)
    finally:
        env.close()
    (output / "frames.jsonl").write_text("".join(json.dumps(r) + "\n" for r in records))
    (output / "summary.json").write_text(json.dumps(summaries, indent=2))
    cards = []
    for record in records:
        if "image" not in record:
            continue
        title = (
            f"{record['setting']} / {record['split']} / "
            f"case {record['episode_index']} / frame {record['frame_id']}"
        )
        detail = (
            f"GT valid={record['gt_valid']}, token={record['gt_token']}; "
            f"predicted={record['predicted_token']}; "
            f"hint={record['hint_token']}, source={record['hint_source']}, "
            f"on target={record['hint_on_target']}"
        )
        cards.append(
            f"<article><h3>{html.escape(title)}</h3>"
            f'<img loading="lazy" src="{html.escape(record["image"], quote=True)}">'
            f"<p>{html.escape(detail)}</p></article>"
        )
    (output / "index.html").write_text(
        '<meta charset="utf-8"><title>OPOS / HINT replay audit</title>'
        "<style>body{font:16px system-ui;margin:24px;background:#f4f6f8}"
        ".grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(480px,1fr));gap:16px}"
        "article{background:white;padding:12px}img{max-width:100%}h3,p{overflow-wrap:anywhere}</style>"
        "<h1>保存轨迹的点位重放核验</h1>"
        "<p>青色 GT-O：实例标注，仅供核验；橙色 HINT：实际输入点；洋红 PRED-O：模型输出。"
        "预测 NONE 时没有 PRED-O 点。没有重跑或更改模型决策。</p>"
        '<p><a href="summary.json">案例汇总</a> · <a href="frames.jsonl">逐帧审计</a></p>'
        + '<div class="grid">'
        + "".join(cards)
        + "</div>"
    )
    print(f"Saved coordinate and prediction audit to {output}", flush=True)


if __name__ == "__main__":
    main()
