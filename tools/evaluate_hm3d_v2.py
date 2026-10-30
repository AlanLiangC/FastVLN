"""Resumable full V2 validation on independent GPUs, using frozen FastVLN only."""

import argparse
import fcntl
import hashlib
import html
import json
import os
import signal
import subprocess
import sys
import time
from collections import Counter
from dataclasses import replace
from pathlib import Path

from streamnav.data.schema import HabitatEpisodeSource

ROOT = Path(__file__).resolve().parents[1]
SETTING = "fastvln_yoloe_point"


def write_json(path, value, *, compact=False):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=None if compact else 2) + "\n")
    temporary.replace(path)


def build_cases(manifest):
    source = HabitatEpisodeSource(manifest, seed=2025)
    cases = []
    for index, episode in enumerate(source.evaluation_episodes(limit=None, stratified=False)):
        # V2 restarts episode IDs for every scene/category combination.
        # Preserve the original ID while making streaming episode UIDs unique.
        episode = replace(
            episode,
            episode_id=f"{episode.object_category}:{episode.episode_id}",
            goal_text="Find a tv." if episode.object_category == "tv_monitor" else episode.goal_text,
            metadata={**episode.metadata, "original_episode_id": episode.episode_id},
        )
        if not Path(episode.scene_id).is_file():
            raise FileNotFoundError(episode.scene_id)
        cases.append({"key": f"hm3d_v2_val_{index:04d}", "episode": episode.to_dict()})
    identities = {
        (c["episode"]["scene_id"], c["episode"]["episode_id"]) for c in cases
    }
    if len(cases) != 1000 or len(identities) != 1000:
        raise ValueError(f"Expected 1,000 distinct V2 validation episodes, got {len(cases)}/{len(identities)}")
    return cases


def shard_cases(cases, count):
    # Source order groups scenes: contiguous shards avoid excessive mesh loads.
    return [cases[len(cases) * i // count : len(cases) * (i + 1) // count] for i in range(count)]


def aggregate(rows):
    if not rows:
        return {"episodes": 0}
    names = (
        "success", "spl", "episode_length", "collision_rate", "oracle_success",
        "false_stop", "min_distance_to_goal", "max_alternating_turns",
    )
    return {
        "episodes": len(rows),
        "successes": sum(int(r["success"]) for r in rows),
        **{name: sum(float(r[name]) for r in rows) / len(rows) for name in names},
    }


def remaining_seconds(elapsed, completed, total, baseline_completed=0):
    fresh = completed - baseline_completed
    return elapsed * (total - completed) / fresh if fresh > 0 else None


def collect(output, cases, protocol, workers, started, phase, baseline_completed=0):
    by_key = {c["key"]: c for c in cases}
    results = []
    seen = set()
    for shard in sorted((output / "shards").glob("gpu_*")):
        for path in sorted(shard.rglob("*_result.json")):
            try:
                row = json.loads(path.read_text())
            except json.JSONDecodeError:
                continue  # Worker may be committing the current result.
            key = row["case"]
            if key not in by_key or key in seen or row["setting"] != SETTING:
                raise ValueError(f"Unexpected or duplicate result: {path}")
            seen.add(key)
            episode = by_key[key]["episode"]
            row["object_category"] = episode["object_category"]
            for name in ("video", "trace", "result_file"):
                if row.get(name):
                    row[name] = str(path.parent.relative_to(output) / row[name])
            results.append(row)
    categories = sorted({c["episode"]["object_category"] for c in cases})
    summary = {
        "protocol": protocol,
        "phase": phase,
        "aggregate": aggregate(results),
        "by_category": {
            category: aggregate([r for r in results if r["object_category"] == category])
            for category in categories
        },
        "results": results,
    }
    write_json(output / "summary.json", summary)
    elapsed = time.time() - started
    write_json(output / "progress.json", {
        "phase": phase, "completed": len(results), "total": len(cases),
        "elapsed_seconds": elapsed, "workers": workers, "time_utc": time.time(),
        "estimated_remaining_seconds": remaining_seconds(
            elapsed, len(results), len(cases), baseline_completed
        ) if phase == "evaluating" else None,
    })
    cells = []
    for name, metrics in [("All", summary["aggregate"]), *summary["by_category"].items()]:
        if metrics["episodes"]:
            cells.append(
                f"<tr><td>{name}</td><td>{metrics['successes']}/{metrics['episodes']}</td>"
                f"<td>{metrics['success']:.2%}</td><td>{metrics['spl']:.2%}</td>"
                f"<td>{metrics['episode_length']:.1f}</td></tr>"
            )
    videos = "".join(
        f"<article><h3>{r['case']} — {html.escape(r['goal'])}</h3>"
        f"<p>Success={int(r['success'])}, SPL={r['spl']:.3f}, steps={r['episode_length']}</p>"
        f'<video controls preload="none" src="{r["video"]}"></video>'
        f'<p><a href="{r["trace"]}">Trace</a> / <a href="{r["result_file"]}">Result</a></p></article>'
        for r in results if r.get("video")
    )
    (output / "index.html").write_text(
        '<!doctype html><html lang="zh"><meta charset="utf-8"><title>FastVLN full V2</title>'
        '<style>body{font:17px system-ui;max-width:1200px;margin:32px auto;padding:16px;'
        'background:#f4f6fa;color:#172139}td,th{border:1px solid #abb8ce;padding:8px}'
        'table{border-collapse:collapse}.grid{display:grid;grid-template-columns:repeat(3,1fr);gap:16px}'
        'article{background:white;padding:12px}video{width:100%}</style>'
        '<h1>FastVLN update 15300 · HM3D ObjectNav V2 val</h1>'
        f'<p>{phase}: {len(results)}/1000 episodes. YOLOE single point, greedy, max 500 actions, '
        'STOP within 0.25 m; native FastVLN camera and sliding. No OVSegDT policy evaluation.</p>'
        '<p>Fixed YOLOE Seen vocabulary + plant; sofa→couch, tv_monitor→tv. '
        'No GT HINT. Orange=HINT, green=APOS, magenta=OPOS.</p>'
        '<p><a href="summary.json">Summary</a> / <a href="protocol.json">Protocol</a> / '
        '<a href="progress.json">Progress</a></p>'
        '<table><tr><th>Category</th><th>Successes</th><th>SR</th><th>SPL</th><th>Steps</th></tr>'
        + "".join(cells) + '</table><h2>Preselected video cases</h2><div class="grid">'
        + videos + '</div></html>'
    )
    return len(results)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=ROOT / "runtime/data/manifests/hm3d_v2_val.json")
    parser.add_argument("--checkpoint", type=Path, default=ROOT / "checkpoints/fastvln_15300")
    parser.add_argument("--output", type=Path, default=ROOT / "runs/evaluation_v2_full_15300")
    parser.add_argument("--gpus", type=int, nargs="+", default=list(range(8)))
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--workers-per-gpu", type=int, default=4)
    parser.add_argument("--mps-pipe-directory", type=Path)
    args = parser.parse_args()
    if not args.gpus or len(set(args.gpus)) != len(args.gpus):
        parser.error("Expected distinct GPUs")
    if not 1 <= args.workers_per_gpu <= 4:
        parser.error("workers-per-gpu must be 1..4")
    if args.mps_pipe_directory is not None and not (args.mps_pipe_directory / "control").exists():
        parser.error("Start the private MPS daemon before using its pipe directory")
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    # flock is released by the kernel even after a crash; stale PID files cannot
    # admit a duplicate controller that races over the same result files.
    lock = (output / "controller.lock").open("a")
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError as error:
        raise RuntimeError("An evaluation controller already owns this output") from error
    cases = build_cases(args.manifest)
    shards = shard_cases(cases, len(args.gpus))
    manifest = json.loads((args.checkpoint / "manifest.json").read_text())
    if manifest["update"] != 15300:
        raise ValueError("This benchmark pins update 15300")
    video_keys = []
    for category in sorted({c["episode"]["object_category"] for c in cases}):
        selected = [c["key"] for c in cases if c["episode"]["object_category"] == category]
        video_keys.extend(selected[i] for i in (0, len(selected) // 2, len(selected) - 1))
    protocol = {
        "dataset": "hm3d_v2", "split": "val", "episodes": 1000, "settings": [SETTING],
        "update": 15300, "checkpoint": str(args.checkpoint.resolve()),
        "adapter_sha256": manifest["adapter_sha256"],
        "cases_sha256": hashlib.sha256(json.dumps(cases, sort_keys=True).encode()).hexdigest(),
        "manifest_sha256": hashlib.sha256(args.manifest.read_bytes()).hexdigest(),
        "category_counts": dict(Counter(c["episode"]["object_category"] for c in cases)),
        "scenes": len({c["episode"]["scene_id"] for c in cases}),
        "gpus": args.gpus, "seed": 2025, "max_steps": 500, "success_distance": 0.25,
        "decoding": "greedy; execute one primitive action and observe again",
        "detector": "YOLOE-v8l-seg, 224x224, original calibrated thresholds",
        "detector_vocabulary": "val_seen + plant (plant threshold from val_unseen)",
        "detector_aliases": {"sofa": "couch", "tv_monitor": "tv"},
        "goal_text_aliases": {"tv_monitor": "tv"},
        "episode_identity": "scene + category + original episode_id; no deduplication by raw ID",
        "video_case_keys": video_keys, "ovsegdt_policy_evaluated": False,
        "shards": {str(g): [c["key"] for c in s] for g, s in zip(args.gpus, shards, strict=True)},
    }
    protocol_path = output / "protocol.json"
    if protocol_path.exists() and json.loads(protocol_path.read_text()) != protocol:
        raise ValueError("Existing evaluation has a different protocol")
    write_json(protocol_path, protocol)
    if not (output / "cases.json").exists():
        write_json(output / "cases.json", cases, compact=True)
    for gpu, shard in zip(args.gpus, shards, strict=True):
        directory = output / "shards" / f"gpu_{gpu}"
        directory.mkdir(parents=True, exist_ok=True)
        if not (directory / "input_cases.json").exists():
            write_json(directory / "input_cases.json", shard, compact=True)
    if args.prepare_only:
        print(json.dumps({k: v for k, v in protocol.items() if k != "shards"}, indent=2))
        return
    # Only orchestration changes. Keep the scientific protocol and all completed
    # serial results. Lane assignments are persisted so retries use identical
    # worker protocols and never race with a different lane for an episode.
    execution_path = output / "execution.json"
    if execution_path.exists():
        execution = json.loads(execution_path.read_text())
        if execution["workers_per_gpu"] != args.workers_per_gpu:
            raise ValueError("Resume with the original workers-per-gpu")
    else:
        lanes = []
        for gpu, shard in zip(args.gpus, shards, strict=True):
            directory = output / "shards" / f"gpu_{gpu}"
            completed = {
                json.loads(p.read_text())["case"] for p in directory.rglob("*_result.json")
            }
            pending = [c for c in shard if c["key"] not in completed]
            for index, lane in enumerate(shard_cases(pending, args.workers_per_gpu)):
                if lane:
                    lanes.append({
                        "id": f"gpu_{gpu}_lane_{index}", "gpu": gpu,
                        "directory": f"shards/gpu_{gpu}/lane_{index}",
                        "case_keys": [c["key"] for c in lane],
                    })
        execution = {
            "workers_per_gpu": args.workers_per_gpu, "lanes": lanes,
            "preserved_completed": len(cases) - sum(len(x["case_keys"]) for x in lanes),
            "created_utc": time.time(), "inference_batch_size": 1,
            "mps_pipe_directory": str(args.mps_pipe_directory) if args.mps_pipe_directory else None,
            "reason": "Overlap independent detector/policy/simulator pipelines without batch roundoff changes",
        }
        write_json(execution_path, execution)
    by_key = {c["key"]: c for c in cases}
    for lane in execution["lanes"]:
        directory = output / lane["directory"]
        directory.mkdir(parents=True, exist_ok=True)
        if not (directory / "input_cases.json").exists():
            write_json(directory / "input_cases.json", [by_key[k] for k in lane["case_keys"]], compact=True)
    baseline_completed = sum(1 for _ in (output / "shards").rglob("*_result.json"))
    stopped = [False]
    for sig in (signal.SIGTERM, signal.SIGINT):
        signal.signal(sig, lambda *unused: stopped.__setitem__(0, True))
    jobs, logs, attempts = {}, {}, Counter()
    started = time.time()
    failed = False
    try:
        while not stopped[0]:
            states = {}
            for lane in execution["lanes"]:
                job_id, gpu = lane["id"], lane["gpu"]
                directory = output / lane["directory"]
                process = jobs.get(job_id)
                completed = len(list(directory.glob("*_result.json")))
                if process is not None and process.poll() is not None:
                    # Also reap RPC descendants left by an abnormal worker exit.
                    try:
                        os.killpg(process.pid, signal.SIGTERM)
                    except ProcessLookupError:
                        pass
                    logs.pop(job_id).close()
                    del jobs[job_id]
                    process = None
                if process is None and completed < len(lane["case_keys"]):
                    if attempts[job_id] >= 3:
                        failed = True
                    else:
                        log = (directory / "worker.log").open("a")
                        command = [
                            sys.executable, str(ROOT / "tools/compare_navigation_baselines.py"),
                            "--checkpoint", str(args.checkpoint.resolve()),
                            "--cases", str(directory / "input_cases.json"),
                            "--case-keys", *lane["case_keys"],
                            "--settings", SETTING, "--detector-vocabulary", "hm3d_v2_val",
                            "--video-case-keys", *video_keys, "--max-steps", "500",
                            "--output", str(directory),
                        ]
                        log.write(f"\nAttempt {attempts[job_id] + 1} at {time.time()}\n")
                        log.flush()
                        process = subprocess.Popen(
                            command, cwd=ROOT, stdin=subprocess.DEVNULL, stdout=log,
                            stderr=subprocess.STDOUT, start_new_session=True,
                            env={
                                **os.environ, "CUDA_VISIBLE_DEVICES": str(gpu), "OMP_NUM_THREADS": "1",
                                **({"CUDA_MPS_PIPE_DIRECTORY": str(args.mps_pipe_directory)}
                                   if args.mps_pipe_directory is not None else {}),
                            },
                        )
                        jobs[job_id], logs[job_id] = process, log
                        attempts[job_id] += 1
                        (directory / "worker.pid").write_text(str(process.pid) + "\n")
                states[job_id] = {
                    "pid": process.pid if process else None, "completed": completed,
                    "total": len(lane["case_keys"]), "attempts": attempts[job_id], "gpu": gpu,
                }
            phase = "failed" if failed else "evaluating"
            completed = collect(output, cases, protocol, states, started, phase, baseline_completed)
            if completed == len(cases):
                collect(output, cases, protocol, states, started, "complete")
                print(f"Complete: {completed}/1000. Results: {output}", flush=True)
                break
            if failed:
                raise RuntimeError("A shard failed three attempts; see shards/gpu_*/worker.log")
            time.sleep(20)
    finally:
        for process in jobs.values():
            try:
                os.killpg(process.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
        for process in jobs.values():
            try:
                process.wait(timeout=15)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
        for log in logs.values():
            log.close()
        if stopped[0]:
            collect(output, cases, protocol, {}, started, "paused")


if __name__ == "__main__":
    main()
