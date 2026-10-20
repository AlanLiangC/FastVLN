"""Summarize the fixed-checkpoint action commitment pilot and link its videos."""

import argparse
import hashlib
import html
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default="runs/plan_chunk_validation_20261020")
    args = parser.parse_args()
    root = Path(args.root)
    experiment = json.loads((root / "experiment.json").read_text())
    cases = json.loads((root / "cases.json").read_text())
    expected = len(cases) * len(experiment["settings"])
    records = {
        h: [
            json.loads(p.read_text()) for p in sorted((root / f"horizon_{h}").glob("*_result.json"))
        ]
        for h in experiment["execution_horizons"]
    }
    complete = all(len(rows) == expected for rows in records.values())
    indexed = {h: {(r["case"], r["setting"]): r for r in rows} for h, rows in records.items()}
    shared = set.intersection(*(set(rows) for rows in indexed.values()))
    pairs = []
    for key in sorted(shared):
        baseline = indexed[1][key]
        image_match = all(
            indexed[h][key]["initial_rgb_sha256"] == baseline["initial_rgb_sha256"] for h in indexed
        )
        logit_error = max(
            abs(a - b)
            for h in indexed
            for a, b in zip(
                baseline["initial_action_logits"],
                indexed[h][key]["initial_action_logits"],
                strict=True,
            )
        )
        assert image_match, (key, "Starting RGB differs between horizons")
        assert logit_error <= 1e-4, (key, "Initial policy logits differ", logit_error)
        pairs.append(
            {
                "case": key[0],
                "setting": key[1],
                "initial_rgb_matches": image_match,
                "initial_logit_error_max": logit_error,
            }
        )
    aggregates: dict[int, dict[str, dict[str, float | int | None]]] = {}
    for h, rows in records.items():
        aggregates[h] = {}
        for setting in experiment["settings"]:
            selected = [r for r in rows if r["setting"] == setting]
            n = len(selected)
            aggregates[h][setting] = {
                "episodes": n,
                "successes": int(sum(r["success"] for r in selected)),
                "mean_spl": sum(r["spl"] for r in selected) / n if n else None,
                "mean_steps": sum(r["episode_length"] for r in selected) / n if n else None,
                "alternating_episodes_ge8": sum(r["max_alternating_turns"] >= 8 for r in selected),
                "long_alternating_episodes_ge100": sum(
                    r["max_alternating_turns"] >= 100 for r in selected
                ),
                "collisions": sum(r["collisions"] for r in selected),
                "collision_rate": sum(r["collisions"] for r in selected)
                / sum(r["episode_length"] for r in selected)
                if n
                else None,
                "false_stops": sum(r["false_stop"] for r in selected),
                "cache_verified_frames": sum(
                    r["actual_action_cache_verified_frames"] for r in selected
                ),
            }
    result = {
        "complete": complete,
        "expected_cases_per_horizon": expected,
        "checkpoint_update": experiment["checkpoint_update"],
        "adapter_sha256": experiment["adapter_sha256"],
        "aggregates": aggregates,
        "paired_start_checks": pairs,
        "records": records,
        "scope": "Pilot, six fixed original-start cases, two scenes, two hint settings, batch=1. Fresh RGB/KDA every step. Only action commitment changes. No inference speedup or population SR claim.",
    }
    (root / "comparison.json").write_text(json.dumps(result, indent=2) + "\n")
    table = []
    for case in cases:
        for setting in experiment["settings"]:
            cells = []
            for h in records:
                row = indexed[h].get((case["key"], setting))
                cells.append(
                    f"<td>{int(row['success'])} / {row['episode_length']} / "
                    f'{row["max_alternating_turns"]}<br><a href="horizon_{h}/{row["video"]}">Video</a>'
                    f' · <a href="horizon_{h}/{row["trace"]}">Trace</a></td>'
                    if row
                    else "<td>Running</td>"
                )
            table.append(
                f"<tr><td>{html.escape(case['episode']['goal_text'])}<br>{case['key']}</td><td>{setting}</td>"
                + "".join(cells)
                + "</tr>"
            )
    (root / "index.html").write_text(
        '<meta charset="utf-8"><style>body{font-family:sans-serif;background:#f4f6fa;color:#152038;max-width:1200px;margin:30px auto}'
        "td,th{padding:10px;border:1px solid #bac4d3}table{border-collapse:collapse}a{color:#0756af}</style>"
        f"<h1>Action commitment pilot — update {experiment['checkpoint_update']}</h1>"
        "<p>Each cell: success / physical steps / longest alternating-turn streak. "
        "Same weights and original starts. Fixed batch=1. Fresh RGB and KDA each step. "
        "Collision interrupts the committed plan. This is not a skipped-inference speed test.</p>"
        "<p>本实验验证计划执行长度；不能作为完整验证集成功率。视频亮绿色显示锁定计划，黄色显示当前动作头与实际执行动作。</p>"
        '<p><a href="comparison.json">Metrics and paired-start checks</a></p>'
        "<table><tr><th>Case</th><th>Setting</th><th>1 step</th><th>2 steps</th><th>4 steps</th></tr>"
        + "".join(table)
        + "</table>"
    )
    experiment["status"] = "complete" if complete else "running"
    experiment["completed_cases"] = {h: len(rows) for h, rows in records.items()}
    experiment["implementation_sha256"] = {
        str(p): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in (
            Path("tools/validate_action_chunks.py"),
            Path("src/streamnav/evaluation/action_chunks.py"),
        )
    }
    (root / "experiment.json").write_text(json.dumps(experiment, indent=2) + "\n")
    print(
        json.dumps(
            {
                "complete": complete,
                "completed": experiment["completed_cases"],
                "aggregates": aggregates,
            }
        )
    )


if __name__ == "__main__":
    main()
