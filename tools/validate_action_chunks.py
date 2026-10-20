"""Paired closed-loop action commitment test with per-frame RGB/KDA updates.

The policy and training recipe are unchanged. Pending actions are forced through
step_batch BEFORE its action token is written to KDA. This isolates execution
horizon from visual/cache cadence, and is not a skipped-inference speed test.
"""

import argparse
import hashlib
import html
import json
import time
from pathlib import Path
from typing import Any, cast

import imageio.v2 as imageio
import numpy as np
import torch
import yaml
from PIL import Image, ImageDraw

from streamnav.contracts.action import NavigationAction
from streamnav.contracts.episode import NavigationEpisode
from streamnav.envs.habitat_client import HabitatClient
from streamnav.evaluation.action_chunks import ActionChunk
from streamnav.evaluation.perception_records import perception_records
from streamnav.evaluation.runner import video_frame
from streamnav.models.point_cot.policy import Qwen35PointCoTPolicy
from streamnav.training.diagnostics import observation_distance
from streamnav.training.object_hints import evaluation_object_hint_packet
from streamnav.training.pointcot_trainer import load_adapters
from streamnav.utils.seed import seed_everything


def write_summary(output, results, args, update):
    data = {
        "checkpoint": str(Path(args.checkpoint).resolve()),
        "update": update,
        "execution_horizon": args.horizon,
        "batch_size": 1,
        "scope": __doc__,
        "cases": results,
    }
    (output / "summary.json").write_text(json.dumps(data, indent=2) + "\n")
    rows = "".join(
        f"<tr><td>{html.escape(r['case'])}</td><td>{r['setting']}</td>"
        f"<td>{int(r['success'])}</td><td>{r['spl']:.3f}</td>"
        f"<td>{r['episode_length']}</td><td>{r['max_alternating_turns']}</td>"
        f'<td>{r["distance_to_goal"]:.3f}</td><td><a href="{r["video"]}">video</a>'
        f' / <a href="{r["trace"]}">trace</a></td></tr>'
        for r in results
    )
    (output / "index.html").write_text(
        '<meta charset="utf-8"><style>body{font-family:sans-serif;background:#f4f6fa;'
        "color:#152038}td,th{padding:8px;border:1px solid #bac4d3}table{border-collapse:collapse}</style>"
        f"<h1>Update {update}: execute {args.horizon} actions per plan</h1>"
        "<p>Fixed batch=1. Fresh RGB / KDA each physical step. Collision interrupts plan. "
        "This tests action commitment, not inference skipping.</p>"
        "<table><tr><th>Case</th><th>Setting</th><th>Success</th><th>SPL</th>"
        "<th>Steps</th><th>Max alternating turns</th><th>Final distance</th><th>Artifacts</th></tr>"
        + rows
        + "</table>"
    )


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--cases", required=True)
    parser.add_argument("--horizon", default=1, type=int, choices=(1, 2, 4))
    parser.add_argument("--temporal-weight", type=float, default=0.0)
    parser.add_argument("--arrival-weight", type=float, default=0.0)
    parser.add_argument("--temporal-requires-hint", action="store_true")
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--shards", type=int, default=1)
    parser.add_argument("--output", required=True)
    parser.add_argument("--settings", nargs="+", default=["simulated_hint", "no_hint"])
    parser.add_argument("--max-steps", type=int, default=500)
    args = parser.parse_args()
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    config = yaml.safe_load((Path(args.checkpoint) / "resolved_config.yaml").read_text())
    config["device"] = "cuda:0"
    config["habitat"]["gpu_device_id"] = 0
    config["habitat"]["max_episode_steps"] = args.max_steps
    config["habitat"]["perception_labels"] = True
    config["model"]["temporal_action_weight"] = args.temporal_weight
    config["model"]["temporal_action_requires_hint"] = args.temporal_requires_hint
    config["model"]["arrival_stop_weight"] = args.arrival_weight
    manifest = json.loads((Path(args.checkpoint) / "manifest.json").read_text())
    seed_everything(config["seed"])
    policy = Qwen35PointCoTPolicy(config).eval()
    policy.inference_graphs.max_entries = 8
    load_adapters(policy, args.checkpoint)
    cases = json.loads(Path(args.cases).read_text())
    env = HabitatClient(config["habitat"], "action-chunk-validation")
    results = []
    try:
        for case_index, case in enumerate(cases):
            if case_index % args.shards != args.shard_index:
                continue
            episode = NavigationEpisode(**case["episode"])
            for setting in args.settings:
                key = f"{case['key']}_{setting}"
                result_path = output / f"{key}_result.json"
                if result_path.exists():
                    results.append(json.loads(result_path.read_text()))
                    continue
                observation = env.reset(episode)
                initial_rgb_sha256 = hashlib.sha256(
                    observation["rgb"].numpy().tobytes()
                ).hexdigest()
                state = policy.start_episode(episode.uid, episode.goal_text)
                chunk = ActionChunk(args.horizon)
                trace_path = output / f"{key}.jsonl"
                video_path = output / f"{key}.mp4"
                previous_turn, alternating, consecutive_turns = None, 0, 0
                maximum_alternating, maximum_turns, forced_count, replans = 0, 0, 0, 0
                collision_count, hint_count = 0, 0
                policy_times, traces = [], []
                initial_logits = None
                initial_arrival = None
                with (
                    imageio.get_writer(
                        str(video_path),
                        fps=6,
                        codec="libx264",
                        ffmpeg_params=["-preset", "veryfast", "-threads", "1"],
                    ) as writer,
                    trace_path.open("w") as trace_file,
                ):
                    for t in range(args.max_steps):
                        label = env.get_object_supervision()
                        packet = (
                            evaluation_object_hint_packet(
                                label,
                                episode.uid,
                                observation["frame_id"],
                                config["seed"],
                                config["eval"]["hint_noise"],
                            )
                            if setting == "simulated_hint"
                            else {"token": 0, "source": "no_hint"}
                        )
                        forced = chunk.forced_action
                        reason = chunk.reason if forced is None else "committed_plan"
                        override = (
                            None
                            if forced is None
                            else (None, None, torch.tensor(forced, device=policy.device))
                        )
                        torch.cuda.synchronize()
                        started = time.perf_counter()
                        with torch.autocast("cuda", dtype=torch.bfloat16):
                            features = policy.navigation_features(
                                [observation["rgb"]], [state], [packet["token"]]
                            )
                            step = policy.step_batch(
                                features, [state], tokens=[override], deterministic=True
                            )[0]
                        torch.cuda.synchronize()
                        policy_times.append(time.perf_counter() - started)
                        action = int(step.action)
                        greedy = int(step.action_logits.argmax())
                        if initial_logits is None:
                            initial_logits = step.action_logits.tolist()
                            initial_arrival = step.arrival_logits.softmax(-1).tolist()
                        plan = step.action_plan_logits.argmax(-1).tolist()
                        plan[0] = action
                        if forced is None:
                            assert action == int(step.decision_logits.argmax()) == plan[0]
                            chunk.start(plan, observation["frame_id"])
                            replans += 1
                        else:
                            assert action == forced
                            forced_count += 1
                        assert state.step_index == t and step.state.step_index == t + 1
                        assert action == chunk.forced_action
                        predictions = {
                            "point_cot": True,
                            "apos": step.apos_logits[None],
                            "opos": step.opos_logits[None],
                            "apos_tokens": step.apos[None],
                            "opos_tokens": step.opos[None],
                            "action_plan_logits": torch.cat(
                                (step.decision_logits[None], step.action_plan_logits[1:]), 0
                            )[None],
                            "arrival": step.arrival_logits[None],
                            "object_hint_tokens": [packet["token"]],
                        }
                        perception = perception_records(predictions)[0]
                        perception.update(
                            object_hint_source=packet["source"],
                            opos_ground_truth_valid=label.get("opos_valid", False),
                            opos_ground_truth_token=label.get("opos"),
                        )
                        frame = Image.fromarray(
                            video_frame(
                                observation["rgb"],
                                episode.goal_text,
                                NavigationAction(action),
                                observation,
                                float(step.action_logits.softmax(-1)[0]),
                                perception=perception,
                            )
                        )
                        canvas = Image.new("RGB", (frame.width, frame.height + 48), "#121827")
                        canvas.paste(frame, (0, 0))
                        draw = ImageDraw.Draw(canvas)
                        short = ["S", "F", "L", "R", "U", "D"]
                        draw.text(
                            (10, frame.height + 4),
                            f"COMMITTED {''.join(short[a] for a in chunk.plan)} | "
                            f"execute {chunk.offset + 1}/{len(chunk.plan)} | {reason}",
                            fill="#7cfc00",
                        )
                        draw.text(
                            (10, frame.height + 24),
                            f"fresh head={short[greedy]} | executed={short[action]} | RGB/KDA every step",
                            fill="#ffd84d",
                        )
                        cast(Any, writer).append_data(np.asarray(canvas))
                        after = env.step(NavigationAction(action))
                        is_turn = action in (2, 3)
                        alternating = (
                            alternating + 1
                            if is_turn and previous_turn is not None and previous_turn != action
                            else int(is_turn)
                        )
                        consecutive_turns = consecutive_turns + 1 if is_turn else 0
                        previous_turn = action if is_turn else None
                        maximum_alternating = max(maximum_alternating, alternating)
                        maximum_turns = max(maximum_turns, consecutive_turns)
                        collision_count += bool(after["collision"])
                        hint_count += bool(packet["token"])
                        row = {
                            "frame_id": observation["frame_id"],
                            "action": NavigationAction(action).name,
                            "executed_action": action,
                            "fresh_greedy_action": greedy,
                            "temporal_action_changed": action != greedy and forced is None,
                            "action_probabilities": step.action_logits.softmax(-1).tolist(),
                            "fresh_plan": plan,
                            "committed_plan": chunk.plan,
                            "chunk_origin_frame": chunk.origin_frame,
                            "chunk_offset": chunk.offset,
                            "decision_source": reason,
                            "cache_action": int(step.action),
                            "cache_step_index": step.state.step_index,
                            "distance_before": observation_distance(observation),
                            "distance_after": observation_distance(after),
                            "collision": after["collision"],
                            "done": after["done"],
                            "success": after.get("success", False),
                            "perception": perception,
                        }
                        trace_file.write(json.dumps(row) + "\n")
                        if t % 10 == 0:
                            trace_file.flush()
                        traces.append(row)
                        state, observation = step.state.after_action(after["collision"]), after
                        chunk.advance(action, collision=after["collision"], done=after["done"])
                        if after["done"]:
                            break
                result = {
                    **observation["metrics"],
                    "case": case["key"],
                    "goal": episode.goal_text,
                    "episode_id": episode.uid,
                    "setting": setting,
                    "execution_horizon": args.horizon,
                    "initial_rgb_sha256": initial_rgb_sha256,
                    "initial_action_logits": initial_logits,
                    "initial_arrival_probabilities": initial_arrival,
                    "temporal_weight": args.temporal_weight,
                    "temporal_requires_hint": args.temporal_requires_hint,
                    "arrival_weight": args.arrival_weight,
                    "max_alternating_turns": maximum_alternating,
                    "max_consecutive_turns": maximum_turns,
                    "false_stop": traces[-1]["executed_action"] == 0 and not observation["success"],
                    "forward_actions": sum(r["executed_action"] == 1 for r in traces),
                    "replans": replans,
                    "committed_actions": forced_count,
                    "collisions": collision_count,
                    "hint_frames": hint_count,
                    "actual_action_cache_verified_frames": len(traces),
                    "temporal_action_changes": sum(r["temporal_action_changed"] for r in traces),
                    "policy_mean_ms": 1000 * sum(policy_times) / len(policy_times),
                    "peak_vram_bytes": torch.cuda.max_memory_allocated(),
                    "video": video_path.name,
                    "trace": trace_path.name,
                }
                result_path.write_text(json.dumps(result, indent=2) + "\n")
                results.append(result)
                write_summary(output, results, args, manifest["update"])
                print(
                    json.dumps(
                        {
                            k: result[k]
                            for k in (
                                "case",
                                "setting",
                                "execution_horizon",
                                "success",
                                "spl",
                                "episode_length",
                                "distance_to_goal",
                                "max_alternating_turns",
                                "forward_actions",
                                "collisions",
                            )
                        }
                    ),
                    flush=True,
                )
    finally:
        env.close()
    write_summary(output, results, args, manifest["update"])


if __name__ == "__main__":
    torch.set_num_threads(2)
    main()
