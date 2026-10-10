"""Choose diagnostic videos from completed autonomous evaluation episodes."""

import html
import json
from pathlib import Path
from typing import Any


def select_video_cases(records, limit, anchors=1):
    if limit < 0 or anchors < 0:
        raise ValueError("Video and anchor counts must be nonnegative")
    ordered = sorted(records, key=lambda r: r["episode_index"])
    selected: dict[int, dict[str, Any]] = {}

    def pick(candidates, reason):
        if not candidates:
            return
        record = candidates[0]
        key = record["episode_index"]
        if key in selected:
            selected[key]["selection_reasons"].append(reason)
        elif len(selected) < limit:
            selected[key] = {**record, "selection_reasons": [reason]}

    for record in ordered[: min(anchors, limit)]:
        pick([record], "fixed_anchor")
    pick(
        sorted(
            (r for r in ordered if r["success"] > 0),
            key=lambda r: (-r["spl"], r["episode_length"], r["episode_index"]),
        ),
        "success",
    )
    failures = [r for r in ordered if r["success"] == 0]
    pick(
        sorted(
            (r for r in failures if r.get("oracle_success", 0) > 0),
            key=lambda r: (-r.get("near_goal_steps", 0), r["episode_index"]),
        ),
        "goal_reached_without_stop",
    )
    pick(
        sorted(
            (r for r in failures if r.get("false_stop", False)),
            key=lambda r: (-r["distance_to_goal"], r["episode_index"]),
        ),
        "false_stop",
    )
    pick(
        sorted(
            (r for r in failures if r["collision_rate"] >= 0.25),
            key=lambda r: (-r["collision_rate"], r["episode_index"]),
        ),
        "high_collision",
    )
    pick(
        sorted(
            (r for r in failures if r.get("max_consecutive_turns", 0) >= 30),
            key=lambda r: (
                -r.get("max_alternating_turns", 0),
                -r["max_consecutive_turns"],
                r["episode_index"],
            ),
        ),
        "sustained_turning",
    )
    # Fill spare slots with different scenes/goals before repeating a pair.
    remaining = [r for r in ordered if r["episode_index"] not in selected]
    while remaining and len(selected) < limit:
        pairs = {(r["scene_id"], r["goal"]) for r in selected.values()}
        record = next(
            (r for r in remaining if (r["scene_id"], r["goal"]) not in pairs), remaining[0]
        )
        pick([record], "scene_goal_coverage")
        remaining.remove(record)
    return list(selected.values())


def retain_video_cases(output, records, selected):
    """All writers must be closed before the shared-filesystem retention pass."""
    output = Path(output).resolve()
    keep = {r["episode_index"] for r in selected}
    for record in records:
        for key in ("video_path", "trace_path"):
            path = (output / record[key]).resolve()
            if not path.is_relative_to(output):
                raise ValueError("Video case path escapes evaluation directory")
            if record["episode_index"] not in keep:
                path.unlink(missing_ok=True)


def write_video_index(output, cases_by_split, update):
    output = Path(output)
    labels = {
        "fixed_anchor": "固定对照",
        "fixed_first": "固定对照",
        "success": "成功",
        "goal_reached_without_stop": "到达目标未停车",
        "false_stop": "错误停车",
        "high_collision": "碰撞较多",
        "sustained_turning": "持续转向",
        "scene_goal_coverage": "补充场景／目标",
    }
    sections = []
    for split, cases in cases_by_split.items():
        cards = []
        for case in cases:
            reasons = " · ".join(labels[r] for r in case["selection_reasons"])
            cards.append(
                f"<article><h3>#{case['episode_index']} {html.escape(case['goal'])}</h3>"
                f'<p>{html.escape(reasons)}</p><video controls preload="none" '
                f'src="{html.escape(case["video_path"], quote=True)}"></video>'
                f"<p>成功：{bool(case['success'])} · 步数：{case['episode_length']} · "
                f"最终距离：{case['distance_to_goal']:.2f} m · "
                f"碰撞率：{case['collision_rate']:.1%}</p>"
                + (
                    f"<p>左右交替转向最长：{case['max_alternating_turns']} 步</p>"
                    if "max_alternating_turns" in case
                    else ""
                )
                + f'<a href="{html.escape(case["trace_path"], quote=True)}">逐步动作概率与诊断</a>'
                f"<details><summary>Episode / 场景</summary>"
                f"<p>{html.escape(case['episode_id'])}</p>"
                f"<p>{html.escape(case['scene_id'])}</p></details></article>"
            )
        sections.append(f'<h2>{html.escape(split)}</h2><div class="cases">{"".join(cards)}</div>')
    page = (
        '<!doctype html><html lang="zh-CN"><meta charset="utf-8">'
        '<meta name="viewport" content="width=device-width, initial-scale=1">'
        f"<title>训练评估 · update {update}</title><style>"
        "body{font:16px system-ui;background:#f5f6f8;color:#172033;margin:24px auto;max-width:1500px;padding:0 20px}"
        ".cases{display:grid;grid-template-columns:repeat(auto-fit,minmax(350px,1fr));gap:20px}"
        "article{background:white;border:1px solid #dde2eb;border-radius:12px;padding:16px;overflow-wrap:anywhere}"
        "video{width:100%;background:#121827}h3{margin-top:0}p{line-height:1.5}a{color:#245ba6}"
        f"</style><h1>训练评估 · update {update}</h1>"
        "<p>自主 argmax 轨迹。固定 case 用于跨轮比较，其余按结果挑选；本页案例不能替代全量评估指标。"
        "画面中的 P(STOP) 对应当前画面下的动作决策。</p>" + "".join(sections) + "</html>"
    )
    (output / "index.html").write_text(page)
    (output / "video_cases.json").write_text(
        json.dumps({"update": update, "splits": cases_by_split}, indent=2) + "\n"
    )
