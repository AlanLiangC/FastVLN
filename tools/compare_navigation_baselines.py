"""Paired case videos for frozen FastVLN and the author's OVSegDT checkpoint.

Each method keeps its own camera, robot, simulator and action selection. Only
episodes, start poses, task horizon and success metrics are shared. Training
continues; this is not an exclusive latency measurement.
"""

import argparse
import hashlib
import html
import json
import os
import shutil
import subprocess
import time
import uuid
from contextlib import nullcontext
from pathlib import Path
from typing import Any, cast

import imageio.v2 as imageio
import numpy as np
import torch
import yaml
import zmq
from PIL import Image, ImageDraw

from streamnav.contracts.action import NavigationAction
from streamnav.contracts.episode import NavigationEpisode
from streamnav.data.schema import HabitatEpisodeSource
from streamnav.envs.habitat_client import HabitatClient
from streamnav.evaluation.baseline_profiles import method_profiles
from streamnav.evaluation.detector_client import DetectorClient
from streamnav.evaluation.detector_points import detector_point
from streamnav.evaluation.perception_records import perception_records
from streamnav.evaluation.runner import video_frame
from streamnav.models.point_cot.policy import Qwen35PointCoTPolicy
from streamnav.training.diagnostics import observation_distance
from streamnav.training.object_hints import evaluation_object_hint_packet
from streamnav.training.pointcot_trainer import load_adapters
from streamnav.utils.seed import seed_everything

ROOT = Path(__file__).resolve().parents[1]
SETTINGS = [
    "ovsegdt_native_sampled",
    "fastvln_yoloe_point",
    "fastvln_no_hint",
    "fastvln_simulated_hint",
]
DEFAULT_SETTINGS = ["ovsegdt_native_sampled", "fastvln_no_hint", "fastvln_yoloe_point"]


class YOLOEPointClient:
    """Detector-only bridge; never loads or evaluates the OVSegDT policy."""

    def __init__(self, output, vocabulary=None):
        self.detector = DetectorClient(output / "detector_logs")
        self.vocabulary = vocabulary
        self.report = {"model": "yoloe-v8l-seg", "navigation_policy_loaded": False}

    def request(self, command, rgb=None, **fields):
        if command == "RESET":
            self.split = self.vocabulary or fields["split"]
            self.category = fields["category"]
            aliases = {"sofa": "couch", "tv_monitor": "tv"} if self.split == "hm3d_v2_val" else {}
            return {"detector_category": aliases.get(self.category, self.category)}, None
        if command == "DETECT":
            masks = self.detector.request(
                "DETECT", images=[rgb], categories=[self.category], split=self.split
            )
            return {}, masks[0].astype(bool)
        raise ValueError("Detector-only client cannot run navigation actions: " + command)

    def close(self):
        self.detector.close()


class OVSegDTClient:
    def __init__(self, output):
        self.context = zmq.Context()
        self.socket = self.context.socket(zmq.REQ)
        self.socket.linger = 0
        self.socket.rcvtimeo = self.socket.sndtimeo = 180_000
        endpoint = f"ipc://@streamnav-ovsegdt-{os.getpid()}-{uuid.uuid4().hex[:8]}"
        self.log = (output / "ovsegdt_worker.log").open("w")
        self.process = subprocess.Popen(
            [
                str(ROOT / "runtime/ovsegdt-env/bin/python"),
                str(ROOT / "tools/ovsegdt_worker.py"),
                "--endpoint",
                endpoint,
            ],
            cwd=ROOT,
            env={
                **os.environ,
                "PYTHONPATH": str(ROOT / "tools"),
                "YOLO_CONFIG_DIR": str(ROOT / "runtime/cache/ovsegdt/ultralytics"),
                "MPLCONFIGDIR": str(ROOT / "runtime/cache/ovsegdt/matplotlib"),
            },
            stdout=self.log,
            stderr=subprocess.STDOUT,
        )
        self.socket.connect(endpoint)
        try:
            self.report = self.request("PING")[0]
        except Exception:
            self.close(force=True)
            raise

    def request(self, command, rgb=None, **fields):
        if rgb is not None:
            rgb = np.asarray(rgb, dtype=np.uint8)
            fields["rgb_shape"] = list(rgb.shape)
        self.socket.send_multipart(
            [
                json.dumps({"command": command, **fields}).encode(),
                rgb.tobytes() if rgb is not None else b"",
            ]
        )
        header, pixels = self.socket.recv_multipart()
        result = json.loads(header)
        if not result.pop("ok"):
            raise RuntimeError("OVSegDT worker: " + result["error"])
        mask = (
            np.frombuffer(pixels, dtype=np.uint8).reshape(result["mask_shape"]).astype(bool)
            if "mask_shape" in result
            else None
        )
        return result, mask

    def close(self, force=False):
        if self.process.poll() is None:
            if not force:
                try:
                    self.request("CLOSE")
                except (zmq.ZMQError, RuntimeError):
                    pass
            self.process.terminate()
            try:
                self.process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                self.process.kill()
                self.process.wait()
        self.socket.close()
        self.context.term()
        self.log.close()


def pin_checkpoint(source, target):
    if (target / "manifest.json").exists():
        return
    target.mkdir(parents=True, exist_ok=True)
    source = source.resolve()
    # Pin the adapter rather than retaining a moving best symlink while training
    # may rotate checkpoints. Optimizer and RNG are unnecessary for inference.
    for name in ("resolved_config.yaml", "adapters.safetensors", "manifest.json"):
        shutil.copy2(source / name, target / name)
    (target / "source.json").write_text(json.dumps({"source": str(source)}, indent=2) + "\n")


def overlay_mask(rgb, mask):
    if mask is None:
        return rgb
    array = rgb.numpy().copy()
    pixels = np.asarray(
        Image.fromarray(mask).resize((array.shape[1], array.shape[0]), Image.Resampling.NEAREST)
    )
    array[pixels] = (0.55 * array[pixels] + 0.45 * np.array([0, 240, 255])).astype(np.uint8)
    return torch.from_numpy(array)


def write_summary(output, results, protocol):
    aggregates = {}
    for setting in protocol["settings"]:
        rows = [row for row in results if row["setting"] == setting]
        if rows:
            aggregates[setting] = {
                "episodes": len(rows),
                "successes": sum(row["success"] for row in rows),
                **{
                    name: float(np.mean([row[name] for row in rows]))
                    for name in (
                        "success",
                        "spl",
                        "episode_length",
                        "collision_rate",
                        "max_alternating_turns",
                        "false_stop",
                        "min_distance_to_goal",
                    )
                },
            }
            by_seed = {}
            for seed in sorted({row.get("evaluation_seed", 2025) for row in rows}):
                seeded = [row for row in rows if row.get("evaluation_seed", 2025) == seed]
                by_seed[str(seed)] = {
                    "episodes": len(seeded),
                    "success": float(np.mean([row["success"] for row in seeded])),
                    "spl": float(np.mean([row["spl"] for row in seeded])),
                }
            aggregates[setting]["by_seed"] = by_seed
            aggregates[setting]["success_std_across_seeds"] = float(
                np.std([row["success"] for row in by_seed.values()])
            )
    summary = {"protocol": protocol, "aggregate": aggregates, "results": results}
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    cells = "".join(
        f"<tr><td>{html.escape(name)}</td><td>{row['successes']:.0f}/{row['episodes']}</td>"
        f"<td>{100 * row['spl']:.1f}%</td><td>{row['episode_length']:.1f}</td>"
        f"<td>{100 * row['collision_rate']:.1f}%</td><td>{row['max_alternating_turns']:.1f}</td></tr>"
        for name, row in aggregates.items()
    )
    sections = []
    for case in protocol["case_keys"]:
        rows = [row for row in results if row["case"] == case]
        cards = "".join(
            f"<article><h3>{html.escape(row['setting'])} | seed={row.get('evaluation_seed', 2025)}</h3>"
            f"<p>Success={int(row['success'])} | SPL={row['spl']:.3f} | "
            f"steps={row['episode_length']} | collisions={row['collisions']} | "
            f"distance={row['distance_to_goal']:.2f}m</p>"
            + (
                f'<video controls preload="metadata" src="{row["video"]}"></video>'
                if row["video"]
                else "<p>Trace only</p>"
            )
            + f'<p><a href="{row["trace"]}">Trace</a> / '
            f'<a href="{row["result_file"]}">Result + robot parameters</a></p></article>'
            for row in rows
        )
        if rows:
            sections.append(
                f"<h2>{html.escape(case)} — {html.escape(rows[0]['goal'])}</h2>"
                f'<div class="grid">{cards}</div>'
            )
    page = (
        '<!doctype html><meta charset="utf-8"><title>FastVLN / OVSegDT paired cases</title>'
        "<style>body{font:15px system-ui;margin:28px;background:#f4f6fa;color:#172139}"
        "table{border-collapse:collapse}td,th{padding:8px;border:1px solid #aab6cc}"
        ".grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(340px,1fr));gap:18px}"
        "article{background:white;padding:12px;border:1px solid #ccd4e4}"
        "video{width:100%;max-height:650px;background:#101827}h3{overflow-wrap:anywhere}</style>"
        f"<h1>FastVLN update {protocol['fastvln_update']} / official OVSegDT</h1>"
        f"<p>{len(results)}/{protocol['expected_episodes']} episodes complete. "
        "Same goals and start poses; each method keeps its own settings. "
        "No exclusive latency measurement. Training continues.</p>"
        "<p>OVSegDT: original camera/robot, Habitat-Sim 0.2.3, no sliding, sampled actions. "
        "FastVLN: its own camera/robot, sliding enabled and greedy actions. "
        "Cyan: YOLOE mask. Orange: one input point. Green/magenta: FastVLN APOS/OPOS. "
        "simulated_hint uses privileged GT points and is reported separately.</p>"
        '<p><a href="summary.json">Summary and protocol</a> / <a href="progress.json">Progress</a></p>'
        "<table><tr><th>Setting</th><th>Success</th><th>SPL</th><th>Steps</th>"
        "<th>Collision rate</th><th>Max alternating turns</th></tr>"
        + cells
        + "</table>"
        + "".join(sections)
    )
    if (output / "audit.json").exists():
        audits = '<p><a href="audit.json">Trace / video / paired-start audit</a>'
        if (output / "stair_pointing_audit.json").exists():
            audits += (
                ' / <a href="stair_pointing_audit.json">Replayed stair HINT audit</a>'
                ' / <a href="stair_hint_frame100_semantic.png">Target instance overlay</a>'
            )
        page = page.replace("<table>", audits + "</p><table>", 1)
    (output / "index.html").write_text(page)


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=ROOT / "checkpoints/fastvln_15300")
    parser.add_argument("--shared-checkpoint", action="store_true", help="Read an immutable pinned checkpoint without duplicating its files")
    parser.add_argument(
        "--cases",
        type=Path,
        help="Optional fixed case JSON; otherwise use current validation manifests",
    )
    parser.add_argument("--episodes-per-split", type=int, default=12)
    parser.add_argument(
        "--case-keys",
        nargs="+",
        default=None,
    )
    parser.add_argument("--settings", nargs="+", choices=SETTINGS, default=DEFAULT_SETTINGS)
    parser.add_argument("--seeds", type=int, nargs="+", default=[100, 101, 102])
    parser.add_argument(
        "--video-case-keys",
        nargs="+",
        default=["val_seen_0000", "val_seen_synonyms_0000", "val_unseen_0000"],
    )
    parser.add_argument("--max-steps", type=int, default=500)
    parser.add_argument("--detector-vocabulary", choices=["hm3d_v2_val"], default=None)
    parser.add_argument("--navmesh-max-climb", type=float, default=None)
    parser.add_argument("--navmesh-cell-height", type=float, default=None)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(2)
    if not 1 <= args.max_steps <= 500:
        raise ValueError("max-steps must be 1..500")
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    snapshot = args.checkpoint.resolve() if args.shared_checkpoint else output / "checkpoint"
    if not args.shared_checkpoint:
        pin_checkpoint(args.checkpoint, snapshot)
    manifest = json.loads((snapshot / "manifest.json").read_text())
    config = yaml.safe_load((snapshot / "resolved_config.yaml").read_text())
    config["device"] = "cuda:0"
    official_path = (
        ROOT / "third_party/OVSegDT/config/experiments/transformer_rl_segm_loss-validation.yaml"
    )
    official = yaml.safe_load(official_path.read_text())
    saved_ovsegdt = json.loads((ROOT / "checkpoints/ovsegdt/resolved_config.json").read_text())
    profiles = method_profiles(
        config, saved_ovsegdt, official, seed=config["seed"], max_steps=args.max_steps
    )
    if args.navmesh_max_climb is not None:
        if not 0 < args.navmesh_max_climb <= 0.5:
            raise ValueError("navmesh-max-climb must be in (0, 0.5]")
        profiles["fastvln"]["navmesh_agent_max_climb"] = args.navmesh_max_climb
    if args.navmesh_cell_height is not None:
        if not 0 < args.navmesh_cell_height <= 0.3:
            raise ValueError("navmesh-cell-height must be in (0, 0.3]")
        profiles["fastvln"]["navmesh_cell_height"] = args.navmesh_cell_height
    if args.cases is not None:
        all_cases = {case["key"]: case for case in json.loads(args.cases.read_text())}
    else:
        all_cases = {}
        for path in config["eval"]["manifests"]:
            source = HabitatEpisodeSource(path, seed=config["seed"])
            for index, episode in enumerate(
                source.evaluation_episodes(args.episodes_per_split, True, config["seed"])
            ):
                key = f"{episode.split}_{index:04d}"
                all_cases[key] = {"key": key, "episode": episode.to_dict()}
    if args.case_keys is None:
        args.case_keys = list(all_cases)
    if len(set(args.seeds)) != len(args.seeds):
        raise ValueError("Repeated sampling seeds")
    cases = [all_cases[key] for key in args.case_keys]
    runs = [
        (setting, seed)
        for setting in args.settings
        for seed in (args.seeds if setting == "ovsegdt_native_sampled" else [config["seed"]])
    ]
    protocol = {
        "fastvln_update": manifest["update"],
        "fastvln_adapter_sha256": manifest["adapter_sha256"],
        "ovsegdt_checkpoint_sha256": "9615be13e1a8a873e4997faa7b5dcf38eb5f5e4080fb8e99c12e0a4b0ed631cb",
        "case_keys": args.case_keys,
        "settings": args.settings,
        "sampling_seeds": args.seeds,
        "expected_episodes": len(cases) * len(runs),
        "video_case_keys": args.video_case_keys,
        "comparison": "Each method uses its own settings; only the task is aligned.",
        "official_eval_config_sha256": hashlib.sha256(official_path.read_bytes()).hexdigest(),
        "official_eval_config": str(official_path),
        "cases_sha256": hashlib.sha256(json.dumps(cases, sort_keys=True).encode()).hexdigest(),
        "profiles": profiles,
        "max_steps": args.max_steps,
        "gpu": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "seed": config["seed"],
        "point_selection": "Largest supported component; deepest 48x27 grid-center point; no GT.",
        "decoding": "OVSegDT: sampled; FastVLN: greedy, following each original method.",
        "fastvln_graph_cache_entries": 8,
        "detector_vocabulary_override": args.detector_vocabulary,
        "scope": __doc__,
    }
    protocol_path = output / "protocol.json"
    if protocol_path.exists() and json.loads(protocol_path.read_text()) != protocol:
        raise ValueError("Existing output uses a different protocol; choose a new output directory")
    protocol_path.write_text(json.dumps(protocol, indent=2) + "\n")
    (output / "cases.json").write_text(json.dumps(cases, indent=2) + "\n")
    results = []
    write_summary(output, results, protocol)
    (output / "progress.json").write_text(
        json.dumps(
            {
                "phase": "loading",
                "completed": 0,
                "total": protocol["expected_episodes"],
                "time_utc": time.time(),
            },
            indent=2,
        )
        + "\n"
    )
    seed_everything(config["seed"])
    worker, env, profile = None, None, None
    try:
        has_ovsegdt = "ovsegdt_native_sampled" in args.settings
        worker = (
            OVSegDTClient(output)
            if has_ovsegdt
            else YOLOEPointClient(output, args.detector_vocabulary)
        )
        report_name = "ovsegdt_load.json" if has_ovsegdt else "detector_load.json"
        (output / report_name).write_text(json.dumps(worker.report, indent=2) + "\n")
        policy = None
        if any(setting.startswith("fastvln") for setting in args.settings):
            policy = Qwen35PointCoTPolicy(config).eval()
            policy.inference_graphs.max_entries = 8
            load_adapters(policy, snapshot)
        for case in cases:
            episode = NavigationEpisode(**case["episode"])
            hashes = {}
            for setting, evaluation_seed in runs:
                key = case["key"] + "_" + setting
                if setting == "ovsegdt_native_sampled":
                    key += f"_seed_{evaluation_seed}"
                result_path = output / (key + "_result.json")
                if result_path.exists():
                    saved = json.loads(result_path.read_text())
                    hashes[saved["profile"]] = saved["initial_rgb_sha256"]
                    results.append(saved)
                    write_summary(output, results, protocol)
                    continue
                desired_profile = "ovsegdt" if setting.startswith("ovsegdt") else "fastvln"
                if profile != desired_profile:
                    if env is not None:
                        env.close()
                    env = HabitatClient(profiles[desired_profile], "baseline-comparison")
                    profile = desired_profile
                assert env is not None
                observation = env.reset(episode)
                robot = observation["robot"]
                initial_distance = observation_distance(observation)
                digest = hashlib.sha256(observation["rgb"].numpy().tobytes()).hexdigest()
                if profile in hashes and hashes[profile] != digest:
                    raise ValueError("Paired initial RGB differs for case " + case["key"])
                hashes[profile] = digest
                case_seed = int(
                    hashlib.sha256((str(evaluation_seed) + episode.uid).encode()).hexdigest()[:8],
                    16,
                )
                reset, _ = worker.request(
                    "RESET", split=episode.split, category=episode.object_category, seed=case_seed
                )
                is_fast = setting.startswith("fastvln")
                state = policy.start_episode(episode.uid, episode.goal_text) if is_fast else None
                trace_path, video_path = output / (key + ".jsonl"), output / (key + ".mp4")
                print(
                    json.dumps(
                        {
                            "phase": "starting",
                            "case": case["key"],
                            "setting": setting,
                            "initial_distance": initial_distance,
                            "robot": robot,
                        }
                    ),
                    flush=True,
                )
                collisions, blocked, hint_frames, alternating, max_alternating, previous_turn = (
                    0,
                    0,
                    0,
                    0,
                    0,
                    None,
                )
                traces = []
                with (
                    (
                        imageio.get_writer(
                            str(video_path),
                            fps=6,
                            codec="libx264",
                            macro_block_size=1,
                            ffmpeg_params=["-preset", "veryfast", "-threads", "1"],
                        )
                        if case["key"] in args.video_case_keys
                        else nullcontext(None)
                    ) as writer,
                    trace_path.open("w") as trace,
                ):
                    for frame in range(args.max_steps):
                        perception, mask = None, None
                        packet = {"token": 0, "source": "no_hint"}
                        started = time.monotonic()
                        if is_fast:
                            if setting == "fastvln_yoloe_point":
                                _, mask = worker.request("DETECT", rgb=observation["rgb"])
                                packet = detector_point(mask)
                            elif setting == "fastvln_simulated_hint":
                                packet = evaluation_object_hint_packet(
                                    env.get_object_supervision(),
                                    episode.uid,
                                    frame,
                                    config["seed"],
                                    config["eval"]["hint_noise"],
                                )
                            with torch.autocast("cuda", dtype=torch.bfloat16):
                                features = policy.navigation_features(
                                    [observation["rgb"]], [state], [packet["token"]]
                                )
                                prediction = policy.step_batch(
                                    features, [state], deterministic=True
                                )[0]
                            action = int(prediction.action)
                            probabilities = prediction.action_logits.softmax(-1).tolist()
                            assert (
                                state.step_index == frame
                                and prediction.state.step_index == frame + 1
                            )
                            perception = perception_records(
                                {
                                    "point_cot": True,
                                    "apos": prediction.apos_logits[None],
                                    "opos": prediction.opos_logits[None],
                                    "apos_tokens": prediction.apos[None],
                                    "opos_tokens": prediction.opos[None],
                                    "arrival": prediction.arrival_logits[None],
                                    "action_plan_logits": prediction.action_plan_logits[None],
                                    "object_hint_tokens": [packet["token"]],
                                }
                            )[0]
                            perception["object_hint_source"] = packet["source"]
                        else:
                            response, mask = worker.request(
                                "STEP",
                                rgb=observation["rgb"],
                                frame_id=frame,
                                deterministic=not setting.endswith("sampled"),
                            )
                            action, probabilities = (
                                response["action"],
                                response["action_probabilities"],
                            )
                            assert response["cache_step"] == frame + 1
                        policy_ms = (time.monotonic() - started) * 1000
                        if writer is not None:
                            display = overlay_mask(observation["rgb"], mask)
                            rendered = Image.fromarray(
                                video_frame(
                                    display,
                                    episode.goal_text,
                                    NavigationAction(action),
                                    observation,
                                    probabilities[0],
                                    perception=perception,
                                )
                            )
                            draw = ImageDraw.Draw(rendered)
                            draw.rectangle((0, 0, rendered.width, 21), fill="#101827")
                            draw.text(
                                (5, 4),
                                setting + (" | cyan=YOLOE" if mask is not None else ""),
                                fill="#ffffff",
                            )
                            cast(Any, writer).append_data(np.asarray(rendered))
                        after = env.step(NavigationAction(action))
                        is_turn = action in (2, 3)
                        alternating = (
                            alternating + 1
                            if is_turn and previous_turn is not None and action != previous_turn
                            else int(is_turn)
                        )
                        previous_turn = action if is_turn else None
                        max_alternating = max(max_alternating, alternating)
                        collisions += bool(after["collision"])
                        blocked += bool(after.get("forward_blocked"))
                        hint_frames += bool(packet["token"])
                        row = {
                            "frame_id": frame,
                            "action": action,
                            "action_name": NavigationAction(action).name,
                            "action_probabilities": probabilities,
                            "distance_before": observation_distance(observation),
                            "distance_after": observation_distance(after),
                            "collision": after["collision"],
                            "forward_blocked": after.get("forward_blocked", False),
                            "displacement": after["displacement"],
                            "done": after["done"],
                            "success": after["success"],
                            "hint": packet,
                            "perception": perception,
                            "detector_mask_pixels": int(mask.sum()) if mask is not None else None,
                            "detector_category": reset["detector_category"]
                            if mask is not None
                            else None,
                            "policy_wall_ms_under_training_contention": policy_ms,
                        }
                        trace.write(json.dumps(row) + "\n")
                        traces.append(row)
                        if is_fast:
                            state = prediction.state.after_action(
                                after["collision"],
                                action=action,
                                displacement=after["displacement"],
                            )
                        observation = after
                        if frame % 20 == 0 or after["done"]:
                            trace.flush()
                            (output / "progress.json").write_text(
                                json.dumps(
                                    {
                                        "phase": "evaluating",
                                        "case": case["key"],
                                        "setting": setting,
                                        "frame": frame + 1,
                                        "completed": len(results),
                                        "total": protocol["expected_episodes"],
                                        "time_utc": time.time(),
                                    },
                                    indent=2,
                                )
                                + "\n"
                            )
                        if after["done"]:
                            if writer is not None:
                                cast(Any, writer).append_data(
                                    video_frame(
                                        observation["rgb"],
                                        episode.goal_text,
                                        NavigationAction(action),
                                        observation,
                                        probabilities[0],
                                        terminal=True,
                                    )
                                )
                            break
                result = {
                    **observation["metrics"],
                    "case": case["key"],
                    "goal": episode.goal_text,
                    "episode_id": episode.uid,
                    "setting": setting,
                    "profile": profile,
                    "robot": robot,
                    "initial_distance": initial_distance,
                    "initial_rgb_sha256": digest,
                    "collisions": collisions,
                    "blocked_forwards": blocked,
                    "hint_frames": hint_frames,
                    "max_alternating_turns": max_alternating,
                    "false_stop": traces[-1]["action"] == 0 and not observation["success"],
                    "forward_actions": sum(row["action"] == 1 for row in traces),
                    "decoding": "sampled" if setting.endswith("sampled") else "greedy",
                    "case_seed": case_seed,
                    "evaluation_seed": evaluation_seed,
                    "video": video_path.name if case["key"] in args.video_case_keys else None,
                    "trace": trace_path.name,
                    "result_file": result_path.name,
                }
                temporary_result = result_path.with_suffix(".json.tmp")
                temporary_result.write_text(json.dumps(result, indent=2) + "\n")
                temporary_result.replace(result_path)
                results.append(result)
                write_summary(output, results, protocol)
                print(
                    json.dumps(
                        {
                            key: result[key]
                            for key in (
                                "case",
                                "setting",
                                "success",
                                "spl",
                                "episode_length",
                                "distance_to_goal",
                                "max_alternating_turns",
                                "collisions",
                                "hint_frames",
                            )
                        }
                    ),
                    flush=True,
                )
        (output / "progress.json").write_text(
            json.dumps(
                {
                    "phase": "complete",
                    "completed": len(results),
                    "total": protocol["expected_episodes"],
                    "time_utc": time.time(),
                },
                indent=2,
            )
            + "\n"
        )
    except Exception as error:
        (output / "progress.json").write_text(
            json.dumps(
                {
                    "phase": "failed",
                    "error": str(error),
                    "completed": len(results),
                    "total": protocol["expected_episodes"],
                    "time_utc": time.time(),
                },
                indent=2,
            )
            + "\n"
        )
        raise
    finally:
        if env is not None:
            env.close()
        if worker is not None:
            worker.close()


if __name__ == "__main__":
    main()
