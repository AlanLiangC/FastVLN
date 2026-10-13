import json
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from streamnav.data.grounding import GroundingSource, validate_record
from streamnav.data.grounding_support import mask_grid, pack_grid, support_targets, unpack_grid
from streamnav.evaluation.grounding_cases import case_record
from streamnav.training.grounding_trainer import promote_grounding_best, validate_grounding
from streamnav.training.point_supervision import grounding_opos_losses


def row(target, support=None):
    return {
        "image": "unused",
        "image_id": "x",
        "instruction": "Find a chair.",
        "source": "instances",
        "supervision": "point",
        "opos_targets": target,
        **({"opos_support": support} if support else {}),
    }


def test_support_uses_original_mask_centers_and_never_box_background():
    mask = np.zeros((270, 480), bool)
    mask[20:250, 20:80] = True
    mask[180:250, 20:400] = True
    grid = mask_grid(mask)
    encoded = pack_grid(grid)
    assert np.array_equal(unpack_grid(encoded), grid)
    assert not grid[13, 24] and grid[20, 24]
    first = int(np.flatnonzero(grid)[0]) + 1
    validate_record(row([first], encoded))
    with pytest.raises(ValueError, match="inside"):
        validate_record(row([13 * 48 + 25], encoded))
    with pytest.raises(ValueError):
        validate_record(row([0], encoded))
    with pytest.raises(ValueError):
        unpack_grid("AA==")


def test_instance_objective_accepts_interior_but_penalizes_background_and_keeps_anchor():
    grid = np.zeros((27, 48), bool)
    grid[10:15, 10:35] = True
    canonical, inside, outside = 12 * 48 + 13, 12 * 48 + 31, 1
    labels = [row([canonical], pack_grid(grid))]

    def losses(prediction):
        logits = torch.full((1, 1297), -8.0)
        logits[0, prediction] = 8
        return grounding_opos_losses(logits, labels, anchor_weight=0.2)

    loss, anchor, support = losses(inside)
    assert support.item() < 0.001 and anchor.item() > 10 and loss.item() > 1
    assert losses(outside)[0].item() > loss.item() + 5
    assert losses(canonical)[0].item() < 0.001


def test_absence_has_no_spatial_smoothing_and_geometry_gradients_are_finite():
    assert support_targets(row([0]), 0.025) == [0]
    positive = row([649])
    neighbors = support_targets(positive, 0.025)
    assert 649 in neighbors and 650 in neighbors and 0 not in neighbors and 1 not in neighbors
    logits = torch.zeros((2, 1297), requires_grad=True)
    losses, _, _ = grounding_opos_losses(
        logits, [positive, row([0])], anchor_weight=0.2, point_radius=0.025
    )
    losses.mean().backward()
    assert torch.isfinite(logits.grad).all()
    assert logits.grad[0, 649] < 0 and logits.grad[1, 0] < 0 and logits.grad[1, 649] > 0


def test_validation_reports_object_membership_false_absence_and_nll_without_second_forward(
    monkeypatch,
):
    grid = np.zeros((27, 48), bool)
    grid[12, 10:35] = True
    canonical, inside = 12 * 48 + 12, 12 * 48 + 31
    rows = [
        row([canonical], pack_grid(grid)),
        row([canonical]),
        row([0]),
        {**row([]), "source": "weak", "supervision": "region", "opos_boxes": [[0, 0, 1, 1]]},
    ]
    for i, r in enumerate(rows):
        r["image_id"] = str(i)
    predictions = [inside, 0, 1, 649]
    calls = []

    class Policy:
        def start_episode(self, *args):
            return None

        def step(self, feature, state, deterministic):
            calls.append(feature)
            return SimpleNamespace(
                apos=torch.tensor(0),
                opos=torch.tensor(predictions[int(feature)]),
                opos_logits=torch.zeros(1297),
            )

    monkeypatch.setattr(GroundingSource, "features", lambda p, r: r["image_id"])
    result = validate_grounding(
        Policy(),
        SimpleNamespace(rows=rows, order=list(range(4))),
        4,
        anchor_weight=0.2,
        point_radius=0.025,
    )
    assert len(calls) == 4
    assert result["visible_point_accuracy_5pct"] == 0
    assert result["visible_grounding_accuracy"] == 0.5
    assert result["instance_accuracy"] == 1
    assert result["false_absence_rate"] == 0.5
    assert result["false_presence_rate"] == 1
    assert result["region_accuracy"] == 1
    assert result["validation_loss"] > 0 and result["nll_count"] == 4
    step = SimpleNamespace(
        opos=torch.tensor(inside), apos=torch.tensor(0), opos_logits=torch.zeros(1297)
    )
    case = case_record(rows[0], step, 0)
    assert case["outcome"] == "instance_hit" and case["point_outcome"] == "miss"


def test_new_sampler_limits_small_source_repetition_and_resumes_exactly(tmp_path):
    rows = []
    for source in ["coco", "ref", "scan", "box", "pixmo"]:
        rows.extend(
            [{**row([5]), "source": source, "image_id": f"{source}:{i}"} for i in range(20)]
        )
    rows.extend([{**row([0]), "source": "pixmo", "image_id": f"negative:{i}"} for i in range(30)])
    path = tmp_path / "rows.jsonl"
    path.write_text("".join(json.dumps(r) + "\n" for r in rows))
    source = GroundingSource(
        path,
        seed=9,
        source_weights={"coco": 0.38, "ref": 0.48, "scan": 0.11, "box": 0.02, "pixmo": 0.01},
        absence_probability=0.025,
    )
    state = source.state_dict()
    sampled = source.sample(10000)
    negatives = sum(r["opos_targets"] == [0] for r in sampled)
    pixmo_positive = sum(r["source"] == "pixmo" and r["opos_targets"] != [0] for r in sampled)
    assert 180 < negatives < 320 and 50 < pixmo_positive < 150
    source.load_state_dict(state)
    assert source.sample(10000) == sampled


def test_best_retains_initialization_and_rejects_absence_collapse(tmp_path):
    (tmp_path / "checkpoints").mkdir()
    baseline = {
        "visible_point_accuracy_5pct": 0.33,
        "macro_visible_grounding_accuracy": 0.5,
        "selection_metric": "macro_visible_grounding_accuracy",
        "absence_accuracy": 0.9,
    }
    promote_grounding_best(tmp_path, 0, baseline, min_absence_accuracy=0.85)
    assert (tmp_path / "checkpoints/best").readlink().name == "update_0000000"
    promote_grounding_best(
        tmp_path,
        200,
        {**baseline, "macro_visible_grounding_accuracy": 0.6, "absence_accuracy": 0.6},
        min_absence_accuracy=0.85,
    )
    assert json.loads((tmp_path / "best_grounding.json").read_text())["update"] == 0
    promote_grounding_best(
        tmp_path,
        400,
        {**baseline, "macro_visible_grounding_accuracy": 0.55, "absence_accuracy": 0.89},
        min_absence_accuracy=0.85,
    )
    assert json.loads((tmp_path / "best_grounding.json").read_text())["update"] == 400
