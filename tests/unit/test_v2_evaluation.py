"""Keep preserved serial results and concurrent lanes scientifically consistent."""

import importlib.util
import json
import time
from pathlib import Path

import pytest

SPEC = importlib.util.spec_from_file_location(
    "v2_evaluation", Path(__file__).resolve().parents[2] / "tools/evaluate_hm3d_v2.py"
)
evaluation = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(evaluation)


def result(case, success):
    return {
        "case": case, "setting": evaluation.SETTING, "success": success,
        "spl": success * 0.5, "episode_length": 100, "collision_rate": 0.1,
        "oracle_success": success, "false_stop": False, "min_distance_to_goal": 0.1,
        "max_alternating_turns": 2, "video": None, "trace": case + ".jsonl",
        "result_file": case + "_result.json",
    }


def store(directory, row):
    directory.mkdir(parents=True, exist_ok=True)
    (directory / row["result_file"]).write_text(json.dumps(row))


def test_parallel_resume_combines_legacy_and_nested_results_without_changing_weights(tmp_path):
    cases = [
        {"key": "old", "episode": {"object_category": "chair"}},
        {"key": "new", "episode": {"object_category": "plant"}},
    ]
    store(tmp_path / "shards/gpu_0", result("old", 1))
    store(tmp_path / "shards/gpu_1/lane_0", result("new", 0))
    count = evaluation.collect(tmp_path, cases, {}, {}, time.time(), "complete")
    summary = json.loads((tmp_path / "summary.json").read_text())
    assert count == 2
    assert summary["aggregate"]["success"] == 0.5
    assert summary["aggregate"]["spl"] == 0.25
    assert summary["by_category"]["chair"]["success"] == 1
    assert summary["by_category"]["plant"]["success"] == 0
    assert summary["results"][1]["trace"] == "shards/gpu_1/lane_0/new.jsonl"


def test_parallel_resume_rejects_duplicate_cases_across_lanes(tmp_path):
    cases = [{"key": "a", "episode": {"object_category": "chair"}}]
    store(tmp_path / "shards/gpu_0", result("a", 1))
    store(tmp_path / "shards/gpu_0/lane_0", result("a", 1))
    with pytest.raises(ValueError, match="duplicate"):
        evaluation.collect(tmp_path, cases, {}, {}, time.time(), "evaluating")


def test_result_being_committed_is_not_counted(tmp_path):
    directory = tmp_path / "shards/gpu_0/lane_0"
    directory.mkdir(parents=True)
    (directory / "a_result.json").write_text('{"case":')
    cases = [{"key": "a", "episode": {"object_category": "chair"}}]
    assert evaluation.collect(tmp_path, cases, {}, {}, time.time(), "evaluating") == 0


def test_eta_measures_work_since_resume_instead_of_counting_old_results_as_new():
    assert evaluation.remaining_seconds(60, 283, 1000, 283) is None
    assert evaluation.remaining_seconds(60, 383, 1000, 283) == 370.2
    assert evaluation.remaining_seconds(60, 1000, 1000, 283) == 0


@pytest.mark.parametrize("size", [0, 1, 3, 125, 1000])
def test_partition_covers_every_case_once_even_with_fewer_cases_than_workers(size):
    cases = list(range(size))
    chunks = evaluation.shard_cases(cases, 4)
    assert [case for chunk in chunks for case in chunk] == cases
    assert max(map(len, chunks)) - min(map(len, chunks)) <= 1
