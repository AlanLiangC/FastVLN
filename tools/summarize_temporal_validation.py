"""Aggregate paired temporal-selection validation without touching training metrics."""

import argparse
import html
import json
import math
from pathlib import Path
from typing import Any


def probabilities(logits):
    values = [math.exp(x - max(logits)) for x in logits]
    return [x / sum(values) for x in values]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default="runs/temporal_policy_validation_20261020")
    args = parser.parse_args()
    root = Path(args.root)
    experiment = json.loads((root / "experiment.json").read_text())
    variants = experiment["variants"]
    records = {}
    for variant in variants:
        rows = []
        for path in sorted((root / variant).rglob("*_result.json")):
            row = json.loads(path.read_text())
            row["artifact_dir"] = str(path.parent.relative_to(root))
            rows.append(row)
        records[variant] = rows
    indexed = {name: {(r["case"], r["setting"]): r for r in rows} for name, rows in records.items()}
    complete = all(
        len(rows) == experiment["expected_episodes_per_variant"] for rows in records.values()
    )
    checks = []
    for name, keyed in indexed.items():
        if name == "baseline":
            continue
        for key in set(keyed) & set(indexed["baseline"]):
            row, base = keyed[key], indexed["baseline"][key]
            assert row["initial_rgb_sha256"] == base["initial_rgb_sha256"], (name, key)
            expected = probabilities(base["initial_action_logits"])
            beta = variants[name]["arrival"]
            if beta:
                stop = (1 - beta) * expected[0] + beta * base["initial_arrival_probabilities"][2]
                expected = [stop] + [(1 - stop) * x / (1 - expected[0]) for x in expected[1:]]
            error = max(
                abs(a - b)
                for a, b in zip(probabilities(row["initial_action_logits"]), expected, strict=True)
            )
            assert error <= 2e-5, (name, key, error)
            trajectory_equal = None
            if variants[name].get("requires_hint") and key[1] == "no_hint":
                traces = []
                for result in (base, row):
                    path = root / result["artifact_dir"] / result["trace"]
                    traces.append([json.loads(line) for line in path.read_text().splitlines()])
                assert len(traces[0]) == len(traces[1]), (name, key, "trajectory length")
                for first, second in zip(*traces, strict=True):
                    for field in ("executed_action", "cache_action", "cache_step_index"):
                        assert first[field] == second[field], (name, key, field)
                    assert second["perception"]["object_hint_token"] == 0
                    for field in ("distance_before", "distance_after"):
                        assert abs(first[field] - second[field]) < 1e-6, (name, key, field)
                trajectory_equal = True
            checks.append(
                {
                    "variant": name,
                    "case": key[0],
                    "setting": key[1],
                    "initial_rgb_equal": True,
                    "initial_probability_error": error,
                    "no_hint_entire_trajectory_equal": trajectory_equal,
                }
            )
    aggregate: dict[str, dict[str, dict[str, Any]]] = {}
    for name, rows in records.items():
        aggregate[name] = {}
        for setting in experiment["settings"]:
            selected = [r for r in rows if r["setting"] == setting]
            n = len(selected)
            steps = sum(r["episode_length"] for r in selected)
            aggregate[name][setting] = {
                "episodes": n,
                "successes": int(sum(r["success"] for r in selected)),
                "spl": sum(r["spl"] for r in selected) / n if n else None,
                "mean_steps": steps / n if n else None,
                "long_alternating": sum(r["max_alternating_turns"] >= 100 for r in selected),
                "alternating_ge8": sum(r["max_alternating_turns"] >= 8 for r in selected),
                "false_stops": sum(r["false_stop"] for r in selected),
                "reached_without_stop": sum(
                    r["oracle_success"] > 0 and not r["success"] for r in selected
                ),
                "collisions": sum(r["collisions"] for r in selected),
                "collision_rate": sum(r["collisions"] for r in selected) / steps if steps else None,
                "temporal_action_changes": sum(r["temporal_action_changes"] for r in selected),
            }
    selection = None
    acceptance = {}
    if complete:
        base = aggregate["baseline"]
        for name in variants:
            if name == "baseline":
                continue
            a = aggregate[name]
            acceptance[name] = {
                "hint_success_not_lower": a["simulated_hint"]["successes"]
                >= base["simulated_hint"]["successes"],
                "no_hint_success_not_lower": a["no_hint"]["successes"]
                >= base["no_hint"]["successes"],
                "hint_spl_within_0_01": a["simulated_hint"]["spl"]
                >= base["simulated_hint"]["spl"] - 0.01,
                "hint_long_alternating_not_higher": a["simulated_hint"]["long_alternating"]
                <= base["simulated_hint"]["long_alternating"],
                "hint_false_stops_within_one": a["simulated_hint"]["false_stops"]
                <= base["simulated_hint"]["false_stops"] + 1,
                "hint_collision_rate_within_0_01": a["simulated_hint"]["collision_rate"]
                <= base["simulated_hint"]["collision_rate"] + 0.01,
            }
        eligible = [name for name, tests in acceptance.items() if all(tests.values())]
        if eligible:
            selection = max(
                eligible,
                key=lambda name: (
                    aggregate[name]["simulated_hint"]["successes"],
                    aggregate[name]["simulated_hint"]["spl"],
                    -aggregate[name]["simulated_hint"]["long_alternating"],
                ),
            )
        else:
            selection = "baseline"
    report = {
        "complete": complete,
        "checkpoint_update": experiment["checkpoint_update"],
        "aggregate": aggregate,
        "paired_start_checks": checks,
        "acceptance": acceptance,
        "selected_variant": selection,
        "records": records,
        "scope": "36 fixed original-start episodes across three splits, both hint settings, batch=1. All variants use identical adapter weights. This evaluates inference selection before the new training recipe; not post-training performance.",
    }
    (root / "comparison.json").write_text(json.dumps(report, indent=2) + "\n")
    totals = "".join(
        f"<tr><td>{name}</td><td>{setting}</td><td>{a['successes']}/{a['episodes']}</td><td>{a['spl'] if a['spl'] is not None else ''}</td><td>{a['long_alternating']}</td><td>{a['false_stops']}</td><td>{a['collisions']}</td></tr>"
        for name, settings in aggregate.items()
        for setting, a in settings.items()
    )
    cases = json.loads((root / "cases.json").read_text())
    rows_html = []
    for case in cases:
        for setting in experiment["settings"]:
            cells = []
            for name in variants:
                row = indexed[name].get((case["key"], setting))
                cells.append(
                    f'<td>{int(row["success"])} / {row["episode_length"]} / {row["max_alternating_turns"]}<br><a href="{row["artifact_dir"]}/{row["video"]}">Video</a> · <a href="{row["artifact_dir"]}/{row["trace"]}">Trace</a></td>'
                    if row
                    else "<td>Running</td>"
                )
            rows_html.append(
                f"<tr><td>{html.escape(case['episode']['goal_text'])}<br>{case['key']}</td><td>{setting}</td>"
                + "".join(cells)
                + "</tr>"
            )
    (root / "index.html").write_text(
        '<meta charset="utf-8"><style>body{font-family:sans-serif;background:#f4f6fa;color:#152038;max-width:1400px;margin:24px auto}td,th{padding:8px;border:1px solid #bac4d3}table{border-collapse:collapse}a{color:#0756af}</style>'
        f"<h1>Temporal plan validation — checkpoint {experiment['checkpoint_update']}</h1><p>同权重、同原始起点、固定 batch=1。逐帧观察并执行一步。每个案例单元：成功 / 步数 / 最长左右交替。</p>"
        '<p>baseline: 原策略；temporal: 0.25 短期融合；temporal_arrival: 0.25 融合 + 0.1 当前到达概率校准；hint_temporal: 仅连续非空 HINT 帧融合。<a href="comparison.json">完整指标与配对校验</a></p>'
        "<table><tr><th>Variant</th><th>Setting</th><th>Success</th><th>SPL</th><th>Long oscillation</th><th>False stop</th><th>Collisions</th></tr>"
        + totals
        + "</table><br>"
        "<table><tr><th>Case</th><th>Setting</th>"
        + "".join(f"<th>{name}</th>" for name in variants)
        + "</tr>"
        + "".join(rows_html)
        + "</table>"
    )
    print(
        json.dumps(
            {
                "complete": complete,
                "counts": {k: len(v) for k, v in records.items()},
                "aggregate": aggregate,
                "selected_variant": selection,
            }
        )
    )


if __name__ == "__main__":
    main()
