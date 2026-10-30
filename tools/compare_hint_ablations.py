"""Paired checkpoint x detector ablation on fixed outcome-independent cases."""

import argparse
import fcntl
import hashlib
import html
import json
import math
import os
import random
import shutil
import signal
import subprocess
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np
import yaml
from evaluate_hm3d_v2 import aggregate, write_json
from evaluate_navigation_full import BENCHMARKS, ROOT, stop_process

ARMS = ["u15300_yoloe", "u15300_no_hint", "u12200_yoloe", "u12200_no_hint"]
CONTRASTS = {
    "15300_yoloe_minus_no_hint": (ARMS[0], ARMS[1]),
    "12200_yoloe_minus_no_hint": (ARMS[2], ARMS[3]),
    "no_hint_15300_minus_12200": (ARMS[1], ARMS[3]),
    "yoloe_15300_minus_12200": (ARMS[0], ARMS[2]),
}


def paired_stats(a, b, seed=20261030):
    if not len(a) or len(a) != len(b):
        raise ValueError("Paired outcomes must be nonempty and equal length")
    d = np.asarray(a, dtype=float) - np.asarray(b, dtype=float)
    wins, losses = int((d > 0).sum()), int((d < 0).sum())
    n = wins + losses
    p = (
        min(1.0, 2 * sum(math.comb(n, k) for k in range(min(wins, losses) + 1)) / 2**n)
        if n
        else 1.0
    )
    rng = np.random.default_rng(seed)
    boot = d[rng.integers(len(d), size=(10000, len(d)))].mean(axis=1)
    return {
        "n": len(d),
        "wins": wins,
        "losses": losses,
        "delta_sr": float(d.mean()),
        "paired_bootstrap_95ci": np.quantile(boot, [0.025, 0.975]).tolist(),
        "mcnemar_exact_p": p,
    }


def select_keys(indices, count, seed):
    selection = {}
    for i, benchmark in enumerate(BENCHMARKS):
        candidates = sorted(c["key"] for c in indices if c["benchmark"] == benchmark)
        if not 1 <= count <= len(candidates):
            raise ValueError("Requested sample exceeds benchmark size")
        selection[benchmark] = sorted(random.Random(seed + i).sample(candidates, count))
    return selection


def prepare(args, output):
    source = args.source.resolve()
    full_protocol = json.loads((source / "protocol.json").read_text())
    full = json.loads((source / "summary.json").read_text())
    if full["phase"] != "complete" or len(full["results"]) != 12000:
        raise ValueError("Source full benchmark must be complete")
    expected_robot = {"navmesh_agent_max_climb": 0.2, "navmesh_cell_height": 0.2}
    if full_protocol["evaluation_robot_override"] != expected_robot:
        raise ValueError("Source uses another robot navigation profile")
    indices = json.loads((source / "cases_index.json").read_text())
    selection = select_keys(indices, args.cases_per_split, args.seed)
    chosen = {k for keys in selection.values() for k in keys}
    by_key = {c["key"]: c for c in indices if c["key"] in chosen}
    original_results = {r["case"]: r for r in full["results"] if r["case"] in chosen}
    sources = {
        "u15300": ROOT / "checkpoints/fastvln_15300",
        "u12200": ROOT / "checkpoints/reference_12200",
    }
    inputs = {
        "source_protocol_sha256": hashlib.sha256(
            (source / "protocol.json").read_bytes()
        ).hexdigest(),
        "seed": args.seed,
        "cases_per_split": args.cases_per_split,
        "gpus": args.gpus,
        "chunk_size": args.chunk_size,
        "selection": selection,
        "checkpoints": {
            label: {
                "path": str(path),
                "adapter_sha256": json.loads((path / "manifest.json").read_text())[
                    "adapter_sha256"
                ],
                "adapter_file_sha256": hashlib.sha256(
                    (path / "adapters.safetensors").read_bytes()
                ).hexdigest(),
                "config_sha256": hashlib.sha256(
                    (path / "resolved_config.yaml").read_bytes()
                ).hexdigest(),
            }
            for label, path in sources.items()
        },
    }
    if (
        inputs["checkpoints"]["u15300"]["adapter_sha256"] != full_protocol["adapter_sha256"]
        or inputs["checkpoints"]["u15300"]["adapter_file_sha256"]
        != full_protocol["inputs"]["adapter_file_sha256"]
    ):
        raise ValueError("Cannot reuse source results from another 15300 adapter")
    protocol_path = output / "protocol.json"
    if protocol_path.exists():
        protocol = json.loads(protocol_path.read_text())
        if protocol["inputs"] != inputs:
            raise ValueError("Existing ablation uses different cases, checkpoints or protocol")
        return protocol, by_key, json.loads((output / "baseline_results.json").read_text())
    # Cameras, body and controls were identical in these two checkpoints. Only
    # explicit navmesh overrides differ from their saved training settings.
    configs = [
        yaml.safe_load((path / "resolved_config.yaml").read_text()) for path in sources.values()
    ]
    if configs[0]["habitat"] != configs[1]["habitat"] or configs[0]["seed"] != configs[1]["seed"]:
        raise ValueError("Checkpoint simulator profiles differ; explicitly align before comparing")
    for label, path in sources.items():
        snapshot = output / "checkpoints" / label
        snapshot.mkdir(parents=True, exist_ok=True)
        for name in ("adapters.safetensors", "manifest.json", "resolved_config.yaml"):
            shutil.copy2(path / name, snapshot / name)
    cases = {}
    for job in full_protocol["jobs"]:
        if not chosen.intersection(job["case_keys"]):
            continue
        for c in json.loads((source / "jobs" / job["id"] / "input_cases.json").read_text()):
            if c["key"] in chosen:
                cases[c["key"]] = c
    if cases.keys() != chosen:
        raise ValueError("Missing fixed case inputs")
    baseline = []
    for row in original_results.values():
        row = dict(row)
        row["arm"] = ARMS[0]
        for key in ("video", "trace", "result_file"):
            if row.get(key):
                row[key] = os.path.relpath(source / row[key], output)
        baseline.append(row)
    jobs = []
    for benchmark, keys in selection.items():
        videos = [k for k in keys if original_results[k].get("video")]
        for start in range(0, len(keys), args.chunk_size):
            group = keys[start : start + args.chunk_size]
            name = f"{benchmark}_{start // args.chunk_size:03d}"
            file = output / "inputs" / f"{name}.json"
            file.parent.mkdir(exist_ok=True)
            payload = json.dumps([cases[k] for k in group], sort_keys=True)
            file.write_text(payload + "\n")
            for label, settings in [
                ("u15300", ["fastvln_no_hint"]),
                ("u12200", ["fastvln_yoloe_point", "fastvln_no_hint"]),
            ]:
                jobs.append(
                    {
                        "id": label + "_" + name,
                        "benchmark": benchmark,
                        "model": label,
                        "settings": settings,
                        "expected": len(group) * len(settings),
                        "case_keys": group,
                        "input": str(file.relative_to(output)),
                        "cases_sha256": hashlib.sha256(payload.encode()).hexdigest(),
                        "video_case_keys": [k for k in group if k in videos],
                        "detector_vocabulary": "hm3d_v2_val"
                        if benchmark in ("hm3d_v1_val", "hm3d_v2_val")
                        else None,
                    }
                )
    protocol = {
        "inputs": inputs,
        "jobs": jobs,
        "arms": ARMS,
        "selection": selection,
        "source_full_run": str(source),
        "reused": ARMS[0],
        "expected_trajectories": len(chosen) * 4,
        "new_trajectories": len(chosen) * 3,
        "paired_cases": len(chosen),
        "max_steps": 500,
        "robot_override": expected_robot,
        "decoding": "greedy, batch=1, one action per observed frame",
        "case_selection": "Uniform sample without replacement per split; independent of observed outcomes",
        "causal_scope": "Hint effect within each checkpoint; checkpoint difference also includes adaptation architecture and training changes",
        "video_selection": "All sampled cases whose source YOLOE trajectory already has video; outcome-independent",
    }
    write_json(output / "cases_index.json", list(by_key.values()))
    write_json(output / "baseline_results.json", baseline)
    write_json(protocol_path, protocol)
    return protocol, by_key, baseline


def add_results(output, protocol, indices, results):
    counts = Counter()
    for job in protocol["jobs"]:
        for p in (output / "jobs" / job["id"]).glob("*_result.json"):
            r = json.loads(p.read_text())
            if r["case"] not in job["case_keys"] or r["setting"] not in job["settings"]:
                raise ValueError("Unexpected result " + str(p))
            if r["episode_id"] != indices[r["case"]]["uid"]:
                raise ValueError("Episode UID mismatch " + str(p))
            baseline = results[(r["case"], ARMS[0])]
            if r["initial_rgb_sha256"] != baseline["initial_rgb_sha256"]:
                raise ValueError("Paired initial RGB mismatch " + str(p))
            arm = job["model"] + ("_no_hint" if r["setting"] == "fastvln_no_hint" else "_yoloe")
            key = (r["case"], arm)
            counts[job["id"]] += 1
            if key in results:
                continue
            for name in ("video", "trace", "result_file"):
                if r.get(name):
                    r[name] = str(p.parent.relative_to(output) / r[name])
            r.update(arm=arm, benchmark=job["benchmark"])
            results[key] = r
    return counts


def summarize(protocol, results):
    groups, contrasts = {}, {}
    for benchmark, keys in protocol["selection"].items():
        complete = [k for k in keys if all((k, arm) in results for arm in ARMS)]
        groups[benchmark] = {"paired_completed": len(complete), "expected": len(keys)}
        if len(complete) != len(keys):
            continue  # Completion order favors short successes: never compare partial arms.
        for arm in ARMS:
            groups[benchmark][arm] = aggregate([results[(k, arm)] for k in keys])
        for name, (a, b) in CONTRASTS.items():
            contrasts[benchmark + ":" + name] = paired_stats(
                [results[(k, a)]["success"] for k in keys],
                [results[(k, b)]["success"] for k in keys],
            )
    # Holm adjustment is only final when all planned split/contrast tests exist.
    if len(contrasts) == len(protocol["selection"]) * len(CONTRASTS):
        previous = 0.0
        ordered = sorted(contrasts, key=lambda k: contrasts[k]["mcnemar_exact_p"])
        for i, key in enumerate(ordered):
            previous = max(
                previous, min(1.0, (len(ordered) - i) * contrasts[key]["mcnemar_exact_p"])
            )
            contrasts[key]["holm_adjusted_p"] = previous
    return groups, contrasts


def publish(output, protocol, results, phase, states, failed, started):
    groups, contrasts = summarize(protocol, results)
    rows = sorted(results.values(), key=lambda r: (r["case"], r["arm"]))
    write_json(
        output / "summary.json",
        {
            "phase": phase,
            "protocol": "protocol.json",
            "by_split": groups,
            "paired_comparisons": contrasts,
            "results": rows,
        },
    )
    write_json(
        output / "progress.json",
        {
            "phase": phase,
            "completed": len(rows),
            "total": protocol["expected_trajectories"],
            "reused": protocol["paired_cases"],
            "new_completed": len(rows) - protocol["paired_cases"],
            "new_total": protocol["new_trajectories"],
            "by_split": groups,
            "active_jobs": states,
            "failed_jobs": failed,
            "elapsed_seconds": time.time() - started,
        },
    )
    table = []
    for benchmark, group in groups.items():
        table.append(
            f"<tr><td>{benchmark}<br>{group['paired_completed']}/{group['expected']} pairs</td>"
            + "".join(
                f"<td>{group[a]['success']:.2%} / {group[a]['spl']:.2%}</td>"
                if a in group
                else "<td>pending</td>"
                for a in ARMS
            )
            + "</tr>"
        )
    comparison = "".join(
        f"<tr><td>{html.escape(key)}</td><td>{100 * v['delta_sr']:+.2f} pp</td>"
        f"<td>[{100 * v['paired_bootstrap_95ci'][0]:+.2f}, {100 * v['paired_bootstrap_95ci'][1]:+.2f}] pp</td>"
        f"<td>{v['wins']} / {v['losses']}</td></tr>"
        for key, v in contrasts.items()
    )
    videos = []
    for key in sorted({r["case"] for r in rows if r.get("video")}):
        cards = "".join(
            f"<article><h3>{arm}</h3><p>success={int(r['success'])}, SPL={r['spl']:.3f}, steps={r['episode_length']}</p>"
            + (
                f'<video controls preload="none" src="{r["video"]}"></video>'
                if r.get("video")
                else "<p>Trace only</p>"
            )
            + f'<p><a href="{r["trace"]}">Trace</a></p></article>'
            for arm in ARMS
            if (r := results.get((key, arm))) is not None
        )
        videos.append(f'<h2>{key}</h2><div class="grid">{cards}</div>')
    (output / "index.html").write_text(
        '<!doctype html><html lang="zh"><meta charset="utf-8"><title>12200 / 15300 × YOLOE / no hint</title>'
        "<style>body{font:16px/1.6 system-ui;max-width:1450px;margin:28px auto;padding:16px;background:#f4f6fa;color:#172139}td,th{padding:9px;border:1px solid #9aabc5}table{border-collapse:collapse}.grid{display:grid;grid-template-columns:repeat(4,1fr);gap:12px}article{background:white;padding:10px}video{width:100%}</style>"
        "<h1>12200 / 15300 × YOLOE / 无提示：固定配对对照</h1>"
        f"<p>{phase}: {len(rows)}/{protocol['expected_trajectories']} trajectories; reused {protocol['paired_cases']}. "
        f"{protocol['inputs']['cases_per_split']} random fixed cases per split. "
        "Same cases, camera, body, climb/cell-height=0.20 m, sliding, max 500 steps, STOP within 0.25 m, greedy and batch=1.</p>"
        "<p>SR/SPL shown only after all four arms of a split complete. Fixed seed, no outcome-based selection. "
        "Paired bootstrap intervals are exploratory; final JSON also records Holm-adjusted tests across all planned contrasts.</p>"
        '<p><a href="protocol.json">Protocol</a> / <a href="summary.json">Summary</a> / <a href="progress.json">Progress</a></p>'
        "<table><tr><th>Split</th>"
        + "".join(f"<th>{a}<br>SR / SPL</th>" for a in ARMS)
        + "</tr>"
        + "".join(table)
        + "</table>"
        "<h2>Paired differences</h2><table><tr><th>Contrast</th><th>SR delta</th><th>95% paired CI</th><th>Wins / losses</th></tr>"
        + comparison
        + "</table>"
        + "".join(videos)
        + "</html>"
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=ROOT / "runs/evaluation_full_15300_hm3d")
    parser.add_argument("--output", type=Path, default=ROOT / "runs/hint_ablation_12200_15300")
    parser.add_argument("--cases-per-split", type=int, default=300)
    parser.add_argument("--seed", type=int, default=20261030)
    parser.add_argument("--chunk-size", type=int, default=32)
    parser.add_argument("--gpus", type=int, nargs="+", default=list(range(8)))
    parser.add_argument("--workers-per-gpu", type=int, default=4)
    parser.add_argument("--mps-pipe-directory", type=Path, default=Path("/tmp/fastvln-v2-mps"))
    args = parser.parse_args()
    if (
        not args.gpus
        or len(set(args.gpus)) != len(args.gpus)
        or not 1 <= args.workers_per_gpu <= 4
        or args.chunk_size < 1
    ):
        parser.error("Invalid devices, concurrency or chunk size")
    if not (args.mps_pipe_directory / "control").exists():
        parser.error("Private CUDA MPS daemon is not running")
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    lock = (output / "controller.lock").open("a")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    protocol, indices, baseline = prepare(args, output)
    results = {(r["case"], r["arm"]): r for r in baseline}
    assignments_path = output / "assignments.json"
    assignments = json.loads(assignments_path.read_text()) if assignments_path.exists() else {}
    jobs, logs, attempts, failed = {}, {}, Counter(), {}
    stopped = [False]
    for sig in (signal.SIGTERM, signal.SIGINT):
        signal.signal(sig, lambda *unused: stopped.__setitem__(0, True))
    started, phase = time.time(), "evaluating"
    try:
        while not stopped[0]:
            counts = add_results(output, protocol, indices, results)
            for key, (p, gpu) in list(jobs.items()):
                if p.poll() is not None:
                    stop_process(p)
                    logs.pop(key).close()
                    del jobs[key]
            for gpu in args.gpus:
                capacity = args.workers_per_gpu - sum(g == gpu for _, g in jobs.values())
                pending = [
                    j
                    for j in protocol["jobs"]
                    if j["id"] not in jobs
                    and j["id"] not in failed
                    and counts[j["id"]] < j["expected"]
                    and assignments.get(j["id"], gpu) == gpu
                ]
                pending.sort(key=lambda j: j["id"] not in assignments)
                for job in pending[:capacity]:
                    key = job["id"]
                    if attempts[key] >= 3:
                        failed[key] = "Three incomplete worker exits"
                        continue
                    assignments[key] = gpu
                    write_json(assignments_path, assignments)
                    input_path = output / job["input"]
                    if (
                        hashlib.sha256(input_path.read_bytes().removesuffix(b"\n")).hexdigest()
                        != job["cases_sha256"]
                    ):
                        raise ValueError("Fixed cases changed " + str(input_path))
                    directory = output / "jobs" / key
                    directory.mkdir(parents=True, exist_ok=True)
                    command = [
                        sys.executable,
                        str(ROOT / "tools/compare_navigation_baselines.py"),
                        "--checkpoint",
                        str(output / "checkpoints" / job["model"]),
                        "--shared-checkpoint",
                        "--cases",
                        str(input_path),
                        "--settings",
                        *job["settings"],
                        "--navmesh-max-climb",
                        "0.2",
                        "--navmesh-cell-height",
                        "0.2",
                        "--video-case-keys",
                        *(job["video_case_keys"] or ["__none__"]),
                        "--max-steps",
                        "500",
                        "--output",
                        str(directory),
                    ]
                    if job["detector_vocabulary"]:
                        command.extend(["--detector-vocabulary", job["detector_vocabulary"]])
                    log = (directory / "worker.log").open("a")
                    log.write(f"\nAttempt {attempts[key] + 1}; GPU {gpu}; UTC {time.time()}\n")
                    log.flush()
                    p = subprocess.Popen(
                        command,
                        cwd=ROOT,
                        stdin=subprocess.DEVNULL,
                        stdout=log,
                        stderr=subprocess.STDOUT,
                        start_new_session=True,
                        env={
                            **os.environ,
                            "CUDA_VISIBLE_DEVICES": str(gpu),
                            "OMP_NUM_THREADS": "1",
                            "CUDA_MPS_PIPE_DIRECTORY": str(args.mps_pipe_directory),
                        },
                    )
                    jobs[key], logs[key] = (p, gpu), log
                    attempts[key] += 1
                    (directory / "worker.pid").write_text(str(p.pid) + "\n")
            if len(results) == protocol["expected_trajectories"]:
                phase = "complete"
            elif not jobs:
                phase = "failed"
            states = {
                key: {"pid": p.pid, "gpu": gpu, "attempts": attempts[key], "completed": counts[key]}
                for key, (p, gpu) in jobs.items()
            }
            publish(output, protocol, results, phase, states, failed, started)
            if phase in ("complete", "failed"):
                print(phase, len(results), flush=True)
                break
            time.sleep(15)
    finally:
        for p, _ in jobs.values():
            stop_process(p)
        for log in logs.values():
            log.close()
        if stopped[0]:
            add_results(output, protocol, indices, results)
            publish(output, protocol, results, "paused", {}, failed, started)
    if phase == "failed":
        raise RuntimeError("Ablation has incomplete chunks; inspect progress and logs")


if __name__ == "__main__":
    main()
