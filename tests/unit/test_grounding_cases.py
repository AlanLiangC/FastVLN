import json
from types import SimpleNamespace

import numpy as np
import torch
from PIL import Image

from streamnav.contracts.perception import encode_pixel
from streamnav.evaluation.grounding_cases import (
    draw_case,
    point_outcome,
    select_cases,
    write_gallery,
)
from streamnav.training.grounding_trainer import validate_grounding


def record(index, target, prediction, source="pixmo-points"):
    outcome, distance = point_outcome(prediction, target)
    return {
        "index": index,
        "image_id": str(index),
        "image": "",
        "instruction": "Find a chair.",
        "source": source,
        "apos": 0,
        "apos_targets": [],
        "opos": prediction,
        "opos_targets": target,
        "opos_confidence": 0.8,
        "opos_absence_probability": 0.1,
        "outcome": outcome,
        "normalized_distance": distance,
    }


def test_outcomes_distinguish_absence_from_missed_points():
    assert point_outcome(0, [0])[0] == "correct_absence"
    assert point_outcome(1, [0])[0] == "false_presence"
    assert point_outcome(0, [1])[0] == "false_absence"
    assert point_outcome(1, [1, 100])[0] == "hit"
    assert point_outcome(1296, [1])[0] == "miss"
    assert point_outcome(1, [])[0] == "unlabeled"


def test_fixed_cases_are_stable_across_predictions():
    rows = [
        record(
            i,
            [0] if i % 3 == 0 else [1, 100] if i % 3 == 1 else [42],
            1,
            "CoSyn-point" if i % 2 else "pixmo-points",
        )
        for i in range(30)
    ]
    first = select_cases(rows, count=24)
    changed = select_cases(
        [{**r, "opos": 0, "outcome": "false_absence"} for r in reversed(rows)], count=24
    )
    assert [r["index"] for r in first if r["selection"] == "fixed_anchor"] == [
        r["index"] for r in changed if r["selection"] == "fixed_anchor"
    ]
    assert len(first) == 24 and len({r["index"] for r in first}) == 24


def test_overlay_preserves_image_and_uses_correct_frame_coordinates():
    original = Image.new("RGB", (960, 540), (90, 100, 110))
    token = encode_pixel(480, 270, 960, 540)
    image = draw_case(original, record(0, [token], token))
    assert image.getpixel((0, 0)) == original.getpixel((0, 0))
    # Markers must be on the image, with distinct GT and prediction colors.
    colors = {tuple(pixel) for pixel in np.asarray(image).reshape(-1, 3)}
    assert (255, 48, 232) in colors and (0, 191, 157) in colors
    assert image.getpixel((490, 280)) != original.getpixel((490, 280))
    absent = draw_case(original, record(1, [0], 0))
    assert absent.tobytes() == original.tobytes()


def test_gallery_writes_images_traces_and_escaped_queries(tmp_path):
    original = tmp_path / "original.png"
    Image.new("RGB", (480, 270), "white").save(original)
    row = record(0, [1, 1296], 0)
    row.update(image=str(original), instruction="<script>alert('x')</script>")
    payload = write_gallery(tmp_path / "gallery", [row], 100)
    page = (tmp_path / "gallery/index.html").read_text()
    assert "&lt;script&gt;alert" in page and "<script>alert" not in page
    assert (tmp_path / "gallery/overview.jpg").exists()
    assert payload["cases"][0]["predicted_point_normalized"] is None
    assert len(payload["cases"][0]["target_points_normalized"]) == 2
    assert json.loads((tmp_path / "gallery/cases.json").read_text())["update"] == 100


def test_validation_visualization_reuses_same_forward_pass(tmp_path, monkeypatch):
    calls = []

    class Policy:
        def start_episode(self, uid, instruction):
            return None

        def step(self, feature, state, deterministic):
            calls.append(feature)
            return SimpleNamespace(
                apos=torch.tensor(0), opos=torch.tensor(0), opos_logits=torch.zeros(1297)
            )

    monkeypatch.setattr(
        "streamnav.training.grounding_trainer.GroundingSource.features", lambda p, r: r["image_id"]
    )
    image = tmp_path / "x.png"
    Image.new("RGB", (48, 27), "white").save(image)
    rows = [{**record(i, [0], 0), "image": str(image)} for i in range(3)]
    source = SimpleNamespace(rows=rows, order=list(range(3)))
    result = validate_grounding(Policy(), source, 3, output_dir=tmp_path / "eval", update=100)
    assert result["absence_accuracy"] == 1 and calls == ["0", "1", "2"]
    assert (tmp_path / "eval/index.html").exists()
