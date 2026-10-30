"""Full V1, V2 and OVON evaluation on one explicit robot navigation profile."""

import argparse
import fcntl
import hashlib
import html
import json
import os
import shutil
import signal
import subprocess
import sys
import time
from collections import Counter
from dataclasses import replace
from pathlib import Path

from evaluate_hm3d_v2 import SETTING, aggregate, remaining_seconds, write_json

from streamnav.data.schema import HabitatEpisodeSource

ROOT = Path(__file__).resolve().parents[1]
BENCHMARKS = {
    "hm3d_v1_val": 2000,
    "hm3d_v2_val": 1000,
    "hm3d_ovon_val_seen": 3000,
    "hm3d_ovon_val_seen_synonyms": 3000,
    "hm3d_ovon_val_unseen": 3000,
}


def episode_case(episode, index):
    original_id = episode.episode_id
    # V1 IDs can repeat even within the same scene/category. Never deduplicate
    # episodes by that field. Preserve OVON UIDs for the retained 300-case audit.
    if episode.dataset_id in ("hm3d_v1", "hm3d_v2"):
        episode = replace(
            episode, episode_id=f"{episode.object_category}:{original_id}"
            + (f":{index:06d}" if episode.dataset_id == "hm3d_v1" else ""),
            goal_text="Find a tv." if episode.object_category == "tv_monitor" else episode.goal_text,
            metadata={**episode.metadata, "original_episode_id": original_id},
        )
    key = f"{episode.dataset_id}_{episode.split}_{index:05d}"
    # Unlike dataclasses.asdict(), this shares immutable per-category goal
    # records until a chunk is serialized, avoiding thousands of deep copies.
    return {"key": key, "episode": vars(episode)}


def choose_videos(cases, limit=18):
    categories = sorted({c["episode"]["object_category"] for c in cases})
    if len(categories) <= 6:
        keys = []
        for category in categories:
            group = [c["key"] for c in cases if c["episode"]["object_category"] == category]
            keys.extend(group[i] for i in dict.fromkeys((0, len(group) // 2, len(group) - 1)))
        return keys[:limit]
    selected = [categories[len(categories) * i // limit] for i in range(limit)]
    keys = []
    for category in selected:
        group = [c["key"] for c in cases if c["episode"]["object_category"] == category]
        keys.append(group[len(group) // 2])
    return keys


def prepare(output, checkpoint, gpus, chunk_size, max_climb=0.2, cell_height=0.2):
    manifests = {
        name: ROOT / "runtime/data/manifests" / f"{name}.json" for name in BENCHMARKS
    }
    adapter_manifest = json.loads((checkpoint / "manifest.json").read_text())
    if adapter_manifest["update"] != 15300:
        raise ValueError("Full evaluation pins update 15300")
    inputs = {
        "manifest_sha256": {name: hashlib.sha256(p.read_bytes()).hexdigest() for name, p in manifests.items()},
        "adapter_file_sha256": hashlib.sha256((checkpoint / "adapters.safetensors").read_bytes()).hexdigest(),
        "resolved_config_sha256": hashlib.sha256((checkpoint / "resolved_config.yaml").read_bytes()).hexdigest(),
        "detector_checkpoint_sha256": hashlib.sha256((ROOT / "checkpoints/ovsegdt/yoloe-v8l-seg.pt").read_bytes()).hexdigest(),
        "detector_vocabulary_sha256": {
            split: hashlib.sha256((ROOT / "third_party/OVSegDT" / f"cats_{split}.json").read_bytes()).hexdigest()
            for split in ("val_seen", "val_seen_synonyms", "val_unseen")
        },
        "gpus": gpus, "chunk_size": chunk_size, "navmesh_agent_max_climb": max_climb,
        "navmesh_cell_height": cell_height,
    }
    protocol_path = output / "protocol.json"
    if protocol_path.exists():
        protocol = json.loads(protocol_path.read_text())
        if protocol["inputs"] != inputs:
            raise ValueError("Existing output has different inputs or chunk size")
        return protocol, json.loads((output / "cases_index.json").read_text())
    snapshot = output / "checkpoint"
    snapshot.mkdir(exist_ok=True)
    for name in ("manifest.json", "resolved_config.yaml", "adapters.safetensors"):
        shutil.copy2(checkpoint / name, snapshot / name)
    indices, jobs, details, video_keys = [], [], {}, []
    all_uids = set()
    for name, expected in BENCHMARKS.items():
        source = HabitatEpisodeSource(manifests[name], seed=2025)
        cases = [episode_case(e, i) for i, e in enumerate(source.evaluation_episodes(None, False))]
        identities = [
            f"{c['episode']['dataset_id']}:{c['episode']['split']}:{c['episode']['scene_id']}:{c['episode']['episode_id']}"
            for c in cases
        ]
        if len(cases) != expected or len(set(identities)) != expected or all_uids.intersection(identities):
            raise ValueError(f"Incomplete or duplicate evaluation cases: {name}")
        all_uids.update(identities)
        scenes = {c["episode"]["scene_id"] for c in cases}
        for scene in scenes:
            for asset in (Path(scene), Path(scene).with_name(Path(scene).name.replace(".basis.glb", ".semantic.glb"))):
                if not asset.is_file():
                    raise FileNotFoundError(asset)
        selected = choose_videos(cases)
        video_keys.extend(selected)
        for case, uid in zip(cases, identities, strict=True):
            e = case["episode"]
            indices.append({
                "key": case["key"], "benchmark": name, "uid": uid,
                "dataset_id": e["dataset_id"], "split": e["split"],
                "scene_id": e["scene_id"], "object_category": e["object_category"],
                "original_episode_id": e["metadata"].get("original_episode_id", e["episode_id"]),
                "start_position": e["start_position"], "start_rotation": e["start_rotation"],
            })
        for index, start in enumerate(range(0, len(cases), chunk_size)):
            chunk = cases[start:start + chunk_size]
            job_id = f"{name}_{index:04d}"
            directory = output / "jobs" / job_id
            directory.mkdir(parents=True, exist_ok=True)
            payload = json.dumps(chunk, sort_keys=True)
            (directory / "input_cases.json").write_text(payload + "\n")
            jobs.append({
                "id": job_id, "benchmark": name, "episodes": len(chunk),
                "case_keys": [c["key"] for c in chunk],
                "cases_sha256": hashlib.sha256(payload.encode()).hexdigest(),
                "detector_vocabulary": "hm3d_v2_val" if name in ("hm3d_v1_val", "hm3d_v2_val") else None,
                "video_case_keys": [c["key"] for c in chunk if c["key"] in selected],
            })
        details[name] = {
            "episodes": expected, "scenes": len(scenes),
            "category_counts": dict(Counter(c["episode"]["object_category"] for c in cases)),
        }
        print(f"Prepared {name}: {len(cases)} unique cases / {len(scenes)} scenes", flush=True)
    protocol = {
        "inputs": inputs, "benchmarks": details, "expected_episodes": sum(BENCHMARKS.values()),
        "checkpoint": str(snapshot), "fastvln_update": 15300,
        "adapter_sha256": adapter_manifest["adapter_sha256"], "settings": [SETTING],
        "max_steps": 500, "success_distance": 0.25, "seed": 2025,
        "evaluation_robot_override": {"navmesh_agent_max_climb": max_climb, "navmesh_cell_height": cell_height},
        "reason_for_robot_override": "Checkpoint climb=0.10/cell_height=0.05 disconnects V1 stairs and displaces starts. Five problem cases match original geodesic metadata with climb=0.20/cell_height=0.20; body/camera unchanged; all benchmarks rerun consistently",
        "decoding": "greedy, one primitive action per observed frame; batch=1",
        "detector": "YOLOE-v8l-seg, original 224x224 preprocessing and calibrated thresholds",
        "detector_vocabularies": {
            "V1": "Same as V2: val_seen + plant; sofa->couch, tv_monitor->tv",
            "OVON": "Each split's original calibrated vocabulary and L1-nearest SigLIP goal mapping",
        },
        "v1_identity": "scene + category + raw episode_id + source index; no episodes dropped",
        "ovon_identity": "Unchanged dataset/split/scene/episode ID and goal text",
        "goal_text_aliases_v1": {"tv_monitor": "tv"},
        "ovsegdt_policy_evaluated": False, "video_case_keys": video_keys, "jobs": jobs,
    }
    write_json(output / "cases_index.json", indices)
    write_json(protocol_path, protocol)
    return protocol, indices


def add_results(output, protocol, indices, results):
    index = {c["key"]: c for c in indices}
    counts = Counter()
    for job in protocol["jobs"]:
        directory = output / "jobs" / job["id"]
        for path in directory.glob("*_result.json"):
            key = path.name.removesuffix(f"_{SETTING}_result.json")
            if key not in index or key not in job["case_keys"]:
                raise ValueError(f"Result in the wrong evaluation chunk: {path}")
            if key in results:
                counts[job["id"]] += 1
                continue
            row = json.loads(path.read_text())
            if row["case"] != key or row["setting"] != SETTING:
                raise ValueError(f"Unexpected evaluation result: {path}")
            if row["episode_id"] != index[key]["uid"]:
                raise ValueError(f"Episode UID differs from fixed inputs: {path}")
            for name in ("video", "trace", "result_file"):
                if row.get(name):
                    row[name] = str(path.parent.relative_to(output) / row[name])
            results[key] = {**row, **{k: index[key][k] for k in ("benchmark", "dataset_id", "split", "object_category")}}
            counts[job["id"]] += 1
    return counts


def publish(output, protocol, results, states, phase, started, baseline, failed):
    rows = sorted(results.values(), key=lambda r: r["case"])
    total = protocol["expected_episodes"]
    groups = {
        name: aggregate([r for r in rows if r["benchmark"] == name]) for name in BENCHMARKS
    }
    by_category = {}
    for name in BENCHMARKS:
        selected = [r for r in rows if r["benchmark"] == name]
        by_category[name] = {
            c: aggregate([r for r in selected if r["object_category"] == c])
            for c in sorted({r["object_category"] for r in selected})
        }
    elapsed = time.time() - started
    summary = {"protocol": "protocol.json", "phase": phase, "aggregate": aggregate(rows),
               "by_split": groups, "by_category": by_category, "results": rows}
    write_json(output / "summary.json", summary)
    write_json(output / "progress.json", {
        "phase": phase, "completed": len(rows), "total": total, "by_split": groups,
        "elapsed_seconds": elapsed, "active_jobs": states, "failed_jobs": failed,
        "estimated_remaining_seconds": remaining_seconds(elapsed, len(rows), total, baseline),
        "time_utc": time.time(),
    })
    cells = []
    for name, metrics in groups.items():
        expected = BENCHMARKS[name]
        cells.append(f"<tr><td>{name}</td><td>{metrics['episodes']}/{expected}</td>"
                     + (f"<td>{metrics['success']:.2%}</td><td>{metrics['spl']:.2%}</td>"
                        f"<td>{metrics['episode_length']:.1f}</td>" if metrics["episodes"] else "<td>—</td><td>—</td><td>—</td>")
                     + "</tr>")
    videos = "".join(
        f"<article><h3>{html.escape(r['benchmark'])} · {html.escape(r['goal'])}</h3>"
        f"<p>{r['case']}: success={int(r['success'])}, SPL={r['spl']:.3f}, steps={r['episode_length']}</p>"
        f'<video controls preload="none" src="{r["video"]}"></video>'
        f'<p><a href="{r["trace"]}">Trace</a> / <a href="{r["result_file"]}">Result</a></p></article>'
        for r in rows if r.get("video")
    )
    (output / "index.html").write_text(
        '<!doctype html><html lang="zh"><meta charset="utf-8"><title>FastVLN full V1 / V2 / OVON</title>'
        '<style>body{font:17px system-ui;max-width:1250px;margin:32px auto;padding:16px;background:#f4f6fa;color:#172139}'
        'td,th{border:1px solid #aab6cc;padding:9px}table{border-collapse:collapse}'
        '.grid{display:grid;grid-template-columns:repeat(3,1fr);gap:16px}article{background:white;padding:12px}video{width:100%}</style>'
        '<h1>FastVLN 15300 · Full HM3D V1 / V2 / OVON</h1>'
        f'<p>{phase}: {len(rows)}/{total} episodes. YOLOE single point; greedy; max 500 actions; '
        'STOP within 0.25 m. FastVLN camera and body, sliding, max climb/cell height 0.20 m.</p>'
        '<p>8 GPUs × 4 independent workers, batch=1, CUDA MPS; shared chunk queue. '
        'Orange=HINT, green=APOS, magenta=OPOS. Metrics are partial until each split completes.</p>'
        '<p><a href="summary.json">Summary</a> / <a href="protocol.json">Protocol</a> / '
        '<a href="progress.json">Progress</a> / <a href="cases_index.json">All cases</a></p>'
        '<table><tr><th>Benchmark</th><th>Complete</th><th>SR</th><th>SPL</th><th>Steps</th></tr>'
        + "".join(cells) + '</table><h2>Preselected videos</h2><div class="grid">' + videos + '</div></html>'
    )


def stop_process(process):
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        return
    try:
        process.wait(timeout=10)
    except subprocess.TimeoutExpired:
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        process.wait(timeout=5)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=ROOT / "checkpoints/fastvln_15300")
    parser.add_argument("--output", type=Path, default=ROOT / "runs/evaluation_full_15300_hm3d")
    parser.add_argument("--gpus", type=int, nargs="+", default=list(range(8)))
    parser.add_argument("--workers-per-gpu", type=int, default=4)
    parser.add_argument("--chunk-size", type=int, default=32)
    parser.add_argument("--navmesh-max-climb", type=float, default=0.2)
    parser.add_argument("--navmesh-cell-height", type=float, default=0.2)
    parser.add_argument("--mps-pipe-directory", type=Path, default=Path("/tmp/fastvln-v2-mps"))
    parser.add_argument("--prepare-only", action="store_true")
    args = parser.parse_args()
    if not args.gpus or len(set(args.gpus)) != len(args.gpus) or not 1 <= args.workers_per_gpu <= 4 or args.chunk_size < 1:
        parser.error("Invalid GPU list, concurrency or chunk size")
    if not (args.mps_pipe_directory / "control").exists():
        parser.error("The private CUDA MPS daemon must be running")
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    lock = (output / "controller.lock").open("a")
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError as error:
        raise RuntimeError("An evaluation controller already owns this output") from error
    protocol, indices = prepare(output, args.checkpoint.resolve(), args.gpus, args.chunk_size, args.navmesh_max_climb, args.navmesh_cell_height)
    if args.prepare_only:
        print(f"Prepared {len(indices)} cases / {len(protocol['jobs'])} jobs", flush=True)
        return
    assignments_path = output / "assignments.json"
    assignments = json.loads(assignments_path.read_text()) if assignments_path.exists() else {}
    execution = {"gpus": args.gpus, "workers_per_gpu": args.workers_per_gpu,
                 "batch_size": 1, "mps_pipe_directory": str(args.mps_pipe_directory), "chunk_size": args.chunk_size}
    write_json(output / "execution.json", execution)
    results, jobs, logs, attempts, failed = {}, {}, {}, Counter(), {}
    counts = add_results(output, protocol, indices, results)
    baseline, started = len(results), time.time()
    stopped = [False]
    for sig in (signal.SIGTERM, signal.SIGINT):
        signal.signal(sig, lambda *unused: stopped.__setitem__(0, True))
    phase = "evaluating"
    try:
        while not stopped[0]:
            counts = add_results(output, protocol, indices, results)
            for job_id, (process, gpu) in list(jobs.items()):
                if process.poll() is not None:
                    stop_process(process)
                    logs.pop(job_id).close()
                    del jobs[job_id]
            for gpu in args.gpus:
                capacity = args.workers_per_gpu - sum(g == gpu for _, g in jobs.values())
                candidates = [j for j in protocol["jobs"] if j["id"] not in jobs and j["id"] not in failed
                              and counts[j["id"]] < j["episodes"] and assignments.get(j["id"], gpu) == gpu]
                # Retry assigned work before claiming a fresh global chunk.
                candidates.sort(key=lambda j: j["id"] not in assignments)
                for job in candidates[:capacity]:
                    job_id = job["id"]
                    if attempts[job_id] >= 3:
                        failed[job_id] = "Three worker exits before all episodes completed"
                        continue
                    assignments[job_id] = gpu
                    write_json(assignments_path, assignments)
                    directory = output / "jobs" / job_id
                    payload = (directory / "input_cases.json").read_bytes().removesuffix(b"\n")
                    if hashlib.sha256(payload).hexdigest() != job["cases_sha256"]:
                        raise ValueError(f"Fixed case inputs changed: {directory}")
                    command = [
                        sys.executable, str(ROOT / "tools/compare_navigation_baselines.py"),
                        "--checkpoint", protocol["checkpoint"], "--shared-checkpoint",
                        "--cases", str(directory / "input_cases.json"), "--settings", SETTING,
                        "--navmesh-max-climb", str(args.navmesh_max_climb),
                        "--navmesh-cell-height", str(args.navmesh_cell_height),
                        "--video-case-keys", *(job["video_case_keys"] or ["__no_video__"]),
                        "--max-steps", "500", "--output", str(directory),
                    ]
                    if job["detector_vocabulary"]:
                        command.extend(["--detector-vocabulary", job["detector_vocabulary"]])
                    log = (directory / "worker.log").open("a")
                    log.write(f"\nAttempt {attempts[job_id] + 1}; GPU {gpu}; UTC {time.time()}\n")
                    log.flush()
                    process = subprocess.Popen(
                        command, cwd=ROOT, stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT,
                        start_new_session=True, env={**os.environ, "CUDA_VISIBLE_DEVICES": str(gpu),
                                                    "CUDA_MPS_PIPE_DIRECTORY": str(args.mps_pipe_directory), "OMP_NUM_THREADS": "1"},
                    )
                    jobs[job_id], logs[job_id] = (process, gpu), log
                    attempts[job_id] += 1
                    (directory / "worker.pid").write_text(str(process.pid) + "\n")
            states = {
                key: {"pid": p.pid, "gpu": gpu, "attempts": attempts[key], "completed": counts[key]}
                for key, (p, gpu) in jobs.items()
            }
            if len(results) == len(indices):
                phase = "complete"
            elif not jobs:
                phase = "failed"
            publish(output, protocol, results, states, phase, started, baseline, failed)
            if phase in ("complete", "failed"):
                print(f"{phase}: {len(results)}/{len(indices)}; {output}", flush=True)
                break
            time.sleep(10)
    finally:
        for process, _ in jobs.values():
            stop_process(process)
        for log in logs.values():
            log.close()
        if stopped[0]:
            add_results(output, protocol, indices, results)
            publish(output, protocol, results, {}, "paused", started, baseline, failed)
    if phase == "failed":
        raise RuntimeError("Evaluation has incomplete chunks; inspect failed_jobs and worker logs")


if __name__ == "__main__":
    main()
