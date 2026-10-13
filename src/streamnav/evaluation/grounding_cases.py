"""Grounding case galleries with fixed anchors, annotated predictions and raw images."""

import html
import json
import math
import textwrap
from collections import defaultdict
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from streamnav.contracts.perception import decode_pixel
from streamnav.data.grounding_support import support_targets, unpack_grid
from streamnav.evaluation.pointing_overlay import OPOS_COLOR

TARGET_COLOR = "#00bf9d"


def write_gallery_index(root):
    root = Path(root)
    items = []
    for directory in sorted(root.glob("update_*"), reverse=True):
        if not (directory / "index.html").exists():
            continue
        payload = json.loads((directory / "cases.json").read_text())
        metrics = payload["metrics"]
        score = (
            f" · 可见目标定位 {metrics.get('visible_grounding_accuracy', metrics['visible_point_accuracy_5pct']):.1%}"
            if "visible_point_accuracy_5pct" in metrics
            else ""
        )
        items.append(
            f'<li><a href="{directory.name}/index.html">update {payload["update"]}</a> · {len(payload["cases"])} 个案例{score} · <a href="{directory.name}/overview.jpg">总览图</a></li>'
        )
    page = (
        '<!doctype html><html lang="zh-CN"><meta charset="utf-8"><title>Grounding 可视化</title><style>body{font-family:system-ui;margin:32px;color:#172033}li{margin:16px 0}</style><h1>Grounding 可视化</h1><p>洋红菱形为预测，青绿圆圈为标注，绿轮廓为可见实例的有效网格，橙框为弱区域。实例内部命中与标注点距离分别报告。</p><ul>'
        + "\n".join(items)
        + "</ul></html>"
    )
    temporary = root / ".index-writing.html"
    temporary.write_text(page)
    temporary.replace(root / "index.html")


def point_outcome(token, targets):
    if not targets:
        return "unlabeled", None
    point = decode_pixel(token, 1, 1)
    target_points = [p for t in targets if (p := decode_pixel(t, 1, 1)) is not None]
    if targets == [0]:
        return ("correct_absence" if token == 0 else "false_presence"), None
    if point is None:
        return "false_absence", None
    distance = min((math.dist(point, p) for p in target_points), default=None)
    return (
        "hit" if token in targets or (distance is not None and distance <= 0.05) else "miss"
    ), distance


def case_record(row, step, index):
    token = int(step.opos)
    probabilities = step.opos_logits.float().reshape(-1).softmax(-1)
    outcome, distance = point_outcome(token, row.get("opos_targets", []))
    canonical_outcome = outcome
    instance_hit = None
    if row.get("opos_support"):
        instance_hit = token in support_targets(row)
        outcome = (
            "false_absence" if token == 0 else "instance_hit" if instance_hit else "instance_miss"
        )
    if row.get("supervision") == "region":
        outcome = "region_hit" if token in row.get("opos_targets", []) else "region_miss"
        distance = None
    return {
        "index": index,
        "image": row["image"],
        "image_id": row["image_id"],
        "instruction": row["instruction"],
        "source": row.get("source", "unknown"),
        "supervision": row.get("supervision", "point"),
        "opos_boxes": row.get("opos_boxes", []),
        "opos_support": row.get("opos_support"),
        "apos": int(step.apos),
        "apos_targets": row.get("apos_targets", []),
        "opos": token,
        "opos_targets": row.get("opos_targets", []),
        "opos_confidence": float(probabilities[token]),
        "opos_absence_probability": float(probabilities[0]),
        "outcome": outcome,
        "point_outcome": canonical_outcome,
        "instance_hit": instance_hit,
        "normalized_distance": distance,
    }


def fixed_anchor_indices(rows, count):
    """Balance sources and annotation types without using model predictions."""
    groups = defaultdict(list)
    for index, row in enumerate(rows):
        targets = row.get("opos_targets", [])
        kind = "absence" if targets == [0] else "multiple" if len(targets) > 1 else "visible"
        groups[(row.get("source", "unknown"), kind)].append(index)
    chosen: list[int] = []
    keys = sorted(groups)
    offset = 0
    while len(chosen) < count:
        additions = [groups[key][offset] for key in keys if offset < len(groups[key])]
        if not additions:
            break
        chosen.extend(additions[: count - len(chosen)])
        offset += 1
    return chosen


def select_cases(records, count=24, anchors=12):
    records = sorted(records, key=lambda r: r["index"])
    fixed = fixed_anchor_indices(records, min(count, anchors))
    selected = [{**records[i], "selection": "fixed_anchor"} for i in fixed]
    used = {r["index"] for r in selected}
    groups = defaultdict(list)
    for record in records:
        if record["index"] not in used:
            groups[record["outcome"]].append(record)
    offset = 0
    while len(selected) < count:
        additions = [groups[key][offset] for key in sorted(groups) if offset < len(groups[key])]
        if not additions:
            break
        selected.extend(
            {**r, "selection": "outcome_example"} for r in additions[: count - len(selected)]
        )
        offset += 1
    return selected


def draw_case(rgb, record):
    image = rgb.convert("RGB").copy()
    image.thumbnail((960, 720))
    width, height = image.size
    draw = ImageDraw.Draw(image)
    radius = max(7, round(width / 60))
    if record.get("opos_support"):
        grid = unpack_grid(record["opos_support"])
        for y, x in zip(*np.nonzero(grid), strict=True):
            x1, x2 = x / 48 * width, (x + 1) / 48 * width
            y1, y2 = y / 27 * height, (y + 1) / 27 * height
            for edge, boundary in (
                ((x1, y1, x2, y1), y == 0 or not grid[y - 1, x]),
                ((x1, y2, x2, y2), y == 26 or not grid[y + 1, x]),
                ((x1, y1, x1, y2), x == 0 or not grid[y, x - 1]),
                ((x2, y1, x2, y2), x == 47 or not grid[y, x + 1]),
            ):
                if boundary:
                    draw.line(edge, fill="white", width=3)
                    draw.line(edge, fill="#00e6a8", width=1)
    # Targets are circles; predictions are diamonds so coincident points remain visible.
    for box in record.get("opos_boxes", []):
        x1, y1, x2, y2 = box
        draw.rectangle(
            (x1 * width, y1 * height, x2 * width, y2 * height), outline="#ffb000", width=4
        )
    for token in [] if record.get("supervision") == "region" else record["opos_targets"]:
        point = decode_pixel(token, width, height)
        if point is None:
            continue
        x, y = point
        box = (x - radius, y - radius, x + radius, y + radius)
        for color, stroke in (("black", 7), ("white", 5), (TARGET_COLOR, 3)):
            draw.ellipse(box, outline=color, width=stroke)
        draw.line((x - 4, y, x + 4, y), fill=TARGET_COLOR, width=2)
        draw.line((x, y - 4, x, y + 4), fill=TARGET_COLOR, width=2)
    point = decode_pixel(record["opos"], width, height)
    if point is not None:
        x, y = point
        r = radius + 6
        vertices = [(x, y - r), (x + r, y), (x, y + r), (x - r, y), (x, y - r)]
        for color, stroke in (("black", 7), ("white", 5), (OPOS_COLOR, 3)):
            draw.line(vertices, fill=color, width=stroke, joint="curve")
        draw.ellipse((x - 3, y - 3, x + 3, y + 3), fill=OPOS_COLOR, outline="white")
    return image


def write_gallery(directory, records, update, metrics=None, count=24, anchors=12):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    cases = select_cases(records, count, anchors)
    cards = []
    for record in cases:
        stem = f"case_{record['index']:04d}"
        with Image.open(record["image"]) as original:
            rgb = original.convert("RGB")
            annotated = draw_case(rgb, record)
            raw = rgb.copy()
            raw.thumbnail((960, 720))
            raw.save(directory / f"{stem}_original.jpg", quality=95)
            annotated.save(directory / f"{stem}.jpg", quality=95)
            width, height = rgb.size
        record.update(
            {
                "original_size": [width, height],
                "predicted_point_normalized": decode_pixel(record["opos"], 1, 1),
                "target_points_normalized": [decode_pixel(t, 1, 1) for t in record["opos_targets"]],
                "visualization": f"{stem}.jpg",
                "original": f"{stem}_original.jpg",
            }
        )
        target = (
            "unlabeled"
            if not record["opos_targets"]
            else (
                "none"
                if record["opos_targets"] == [0]
                else f"{len(record['opos_targets'])} valid point(s)"
            )
        )
        if record.get("supervision") == "region":
            target = "weak bounding region (no precise point GT)"
        elif record.get("opos_support"):
            target = f"visible instance support; {len(record['opos_targets'])} canonical anchor(s)"
        prediction = "none" if record["opos"] == 0 else "point"
        detail = f"Prediction: {prediction} | Label: {target} | P(prediction)={record['opos_confidence']:.3f} | P(none)={record['opos_absence_probability']:.3f}"
        if record["normalized_distance"] is not None:
            detail += f" | distance={record['normalized_distance']:.3f}"
        if record.get("opos_support"):
            detail += f" | canonical point score={record.get('point_outcome')}"
        cards.append(f'''<article class="case" data-outcome="{html.escape(record["outcome"])}">
<h3>#{record["index"]} · {html.escape(record["outcome"])} · {html.escape(record["source"])}</h3>
<p class="query">{html.escape(record["instruction"])}</p>
<a href="{stem}.jpg" target="_blank"><img loading="lazy" src="{stem}.jpg" data-marked="{stem}.jpg" data-raw="{stem}_original.jpg"></a>
<p>{html.escape(detail)}</p><p>{html.escape(record["selection"])} · <a href="{stem}_original.jpg" target="_blank">原图</a></p></article>''')
    payload = {
        "format": "grounding-cases-v2",
        "update": update,
        "metrics": metrics or {},
        "scoring": "True mask membership for instances; point distance <= 0.05 for point-only labels; weak boxes separate",
        "cases": cases,
    }
    (directory / "cases.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n")
    metric_text = ""
    if metrics and "visible_point_accuracy_5pct" in metrics:
        metric_text = f"可见目标命中 {metrics['visible_point_accuracy_5pct']:.1%} · 缺失目标准确率 {metrics['absence_accuracy']:.1%} · 验证查询 {metrics['count']}"
        if "visible_grounding_accuracy" in metrics:
            metric_text = (
                f"可见目标定位 {metrics['visible_grounding_accuracy']:.1%} · 标注点距离命中 {metrics['visible_point_accuracy_5pct']:.1%}"
                f" · 可见目标误判缺失 {metrics['false_absence_rate']:.1%} · 验证 loss {metrics['validation_loss']:.3f} · "
            ) + metric_text
        if metrics.get("by_source"):
            metric_text += " | " + " | ".join(
                f"{name}: 点 {m['visible_point_accuracy_5pct']:.1%} / 区域 {m['region_accuracy']:.1%}"
                for name, m in metrics["by_source"].items()
            )
    page = (
        """<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Grounding update UPDATE</title><style>
body{font-family:system-ui,sans-serif;margin:24px;background:#f5f7fa;color:#172033}h1{margin-bottom:8px}.legend{padding:12px;background:white;border-radius:10px}
.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(320px,1fr));gap:18px;margin-top:18px}.case{background:white;padding:15px;border-radius:12px;border:1px solid #dde3ed}.case img{width:100%;height:auto}.case p{overflow-wrap:anywhere}.query{font-weight:600}.case h3{font-size:15px}.controls{margin-top:14px}select{padding:6px}
</style><h1>Grounding · update UPDATE</h1><p>METRICS</p><div class="legend">◆ 洋红：预测 OPOS　○ 青绿：有效标注点（所有等价目标）。白边确保点位清晰，原图亮度保持不变。<br>
none 只显示文字，无标注的 APOS 不绘制。绿色轮廓为真实实例的有效网格，预测落入即为 instance_hit，同时保留标注点距离评分。橙框为弱区域标注，不伪造中心点，不计入可见目标定位准确率。坐标为 48×27 网格中心。固定案例便于跨 update 比较，其他案例展示命中、偏移和目标缺失错误。<br><a href="cases.json">案例 JSON</a> · <a href="predictions.jsonl">全部验证预测</a> · <a href="overview.jpg">总览图</a></div>
<div class="controls"><label><input id="raw" type="checkbox">显示原图</label>　<label>筛选：<select id="filter"><option value="all">全部</option><option>instance_hit</option><option>instance_miss</option><option>hit</option><option>miss</option><option>false_absence</option><option>false_presence</option><option>correct_absence</option><option>region_hit</option><option>region_miss</option></select></label></div>
<main class="grid">CARDS</main><script>
document.querySelector('#raw').onchange=e=>document.querySelectorAll('.case img').forEach(i=>{i.src=e.target.checked?i.dataset.raw:i.dataset.marked;i.parentElement.href=i.src});
document.querySelector('#filter').onchange=e=>document.querySelectorAll('.case').forEach(c=>{c.hidden=e.target.value!=='all'&&c.dataset.outcome!==e.target.value});
</script></html>""".replace("UPDATE", str(update))
        .replace("METRICS", html.escape(metric_text))
        .replace("CARDS", "\n".join(cards))
    )
    if cases:
        columns, cell_w, cell_h = 4, 320, 290
        overview = Image.new(
            "RGB", (columns * cell_w, math.ceil(len(cases) / columns) * cell_h), "white"
        )
        draw = ImageDraw.Draw(overview)
        font = ImageFont.load_default(size=14)
        for index, record in enumerate(cases):
            with Image.open(directory / record["visualization"]) as image:
                image.thumbnail((cell_w - 12, cell_h - 80))
                x, y = index % columns * cell_w, index // columns * cell_h
                overview.paste(image, (x + (cell_w - image.width) // 2, y + 70))
                draw.text(
                    (x + 6, y + 6),
                    f"#{record['index']} {record['outcome']}",
                    fill="black",
                    font=font,
                )
                query = textwrap.fill(record["instruction"], width=40, max_lines=2)
                draw.multiline_text((x + 6, y + 26), query, fill="black", font=font)
        overview.save(directory / "overview.jpg", quality=95)
    temporary = directory / ".index-writing.html"
    temporary.write_text(page)
    temporary.replace(directory / "index.html")
    if directory.name.startswith("update_"):
        write_gallery_index(directory.parent)
    return payload
