import copy
import json

import pytest

from tools.run_training_recipe_trials import choose_recipe, evaluation_scores


def scores(success=0.1, spl=0.06, false_stop=0.02):
    return [
        {
            "update": update,
            "episodes": 144,
            "success": success,
            "spl": spl,
            "oracle_success": 0.3,
            "false_stop_rate": false_stop,
        }
        for update in (1100, 1200)
    ]


def test_recipe_selection_requires_repeated_gain_without_false_stop_or_spl_regression():
    results = {
        "control": {"scores": scores()},
        "false_stops": {"scores": scores(0.3, false_stop=0.2)},
        "inefficient": {"scores": scores(0.3, spl=0.04)},
        "small_gain": {"scores": scores(0.11)},
    }
    assert choose_recipe(copy.deepcopy(results)) == "control"
    results["improved"] = {"scores": scores(0.15, spl=0.08)}
    assert choose_recipe(results) == "improved"
    results["improved"]["scores"][-1]["update"] = 1300
    with pytest.raises(RuntimeError, match="updates differ"):
        choose_recipe(results)


def test_recipe_evaluation_requires_all_splits_at_two_updates(tmp_path):
    path = tmp_path / "eval_metrics.jsonl"
    complete = [
        {**score, "split": split, "episodes": 48, "saved_video_cases": 6}
        for score in scores()
        for split in ("seen", "synonyms", "unseen")
    ]
    path.write_text("".join(json.dumps(row) + "\n" for row in complete[:-1]))
    with pytest.raises(RuntimeError, match="Two complete"):
        evaluation_scores(tmp_path)
    path.write_text("".join(json.dumps(row) + "\n" for row in complete))
    result = evaluation_scores(tmp_path)
    assert [row["episodes"] for row in result] == [144, 144]
    assert [row["saved_video_cases"] for row in result] == [18, 18]
    assert result[-1]["success"] == pytest.approx(0.1)
