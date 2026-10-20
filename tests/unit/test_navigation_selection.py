import json
from fractions import Fraction

import pytest

from streamnav.training.navigation_selection import (
    anchor_score,
    make_record,
    reselect_best,
    update_best,
    write_best,
)


def results(counts, setting="simulated_hint"):
    return {
        s: {
            "setting": setting,
            "episodes": 48,
            "success": 0.1,
            "selection_episodes": 12,
            "selection_success": n / 12,
        }
        for s, n in zip(["seen", "synonyms", "unseen"], counts, strict=True)
    }


def checkpoint(root, update):
    p = root / "checkpoints" / f"update_{update:07d}"
    p.mkdir(parents=True)
    (p / "manifest.json").write_text(json.dumps({"update": update}))
    for n in ["adapters.safetensors", "optimizer.pt", "resolved_config.yaml"]:
        (p / n).write_text("fixture")
    return p


def test_equal_integer_counts_do_not_replace_best_due_to_float_tail(tmp_path):
    a, b = results([6, 4, 6]), results([7, 3, 6])
    assert sum(r["selection_success"] for r in b.values()) > sum(
        r["selection_success"] for r in a.values()
    )
    assert anchor_score(a) == anchor_score(b) == Fraction(4, 9)
    checkpoint(tmp_path, 1)
    checkpoint(tmp_path, 2)
    assert update_best(tmp_path, 1, a, "simulated_hint")
    assert not update_best(tmp_path, 2, b, "simulated_hint")
    assert (tmp_path / "checkpoints/best").resolve().name == "update_0000001"
    assert (
        json.loads((tmp_path / "best_evaluation.json").read_text())["selection_setting"]
        == "simulated_hint"
    )


def test_setting_migration_excludes_deleted_weights_and_never_compares_against_old_setting(
    tmp_path,
):
    checkpoint(tmp_path, 1)
    checkpoint(tmp_path, 3)
    write_best(tmp_path, make_record(1, results([10, 10, 10], "no_hint"), "no_hint"))
    for update, hint_counts in [(1, [2, 2, 2]), (2, [11, 11, 11])]:
        p = tmp_path / "evaluation" / f"update_{update:07d}"
        p.mkdir(parents=True)
        (p / "settings_summary.json").write_text(
            json.dumps(
                {
                    "no_hint": results([10, 10, 10], "no_hint"),
                    "simulated_hint": results(hint_counts),
                }
            )
        )
    assert update_best(tmp_path, 3, results([3, 3, 3]), "simulated_hint")
    d = json.loads((tmp_path / "best_evaluation.json").read_text())
    assert d["update"] == 3 and d["success"] == 0.25
    event = json.loads((tmp_path / "best_selection_changes.jsonl").read_text().splitlines()[-1])
    assert event["selected_update"] == 1 and event["historical_best_update"] == 2
    assert event["historical_best_checkpoint_available"] is False
    assert event["previous_setting"] == "no_hint"


def test_selection_rejects_wrong_setting_inconsistent_counts_and_incomplete_checkpoint(tmp_path):
    with pytest.raises(ValueError, match="disagrees"):
        make_record(1, results([3, 3, 3], "no_hint"), "simulated_hint")
    bad = results([3, 3, 3])
    bad["seen"]["selection_success"] = 0.123
    with pytest.raises(ValueError, match="integer counts"):
        anchor_score(bad)
    with pytest.raises(RuntimeError, match="complete saved"):
        write_best(tmp_path, make_record(1, results([3, 3, 3]), "simulated_hint"))
    with pytest.raises(RuntimeError, match="No evaluated"):
        reselect_best(tmp_path, "simulated_hint")
