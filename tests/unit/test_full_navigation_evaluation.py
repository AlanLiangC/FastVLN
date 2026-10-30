import importlib.util
import json
import sys
from pathlib import Path

import pytest

from streamnav.contracts.episode import NavigationEpisode

TOOLS = Path(__file__).resolve().parents[2] / "tools"
sys.path.insert(0, str(TOOLS))
SPEC = importlib.util.spec_from_file_location("full_evaluation", TOOLS / "evaluate_navigation_full.py")
evaluation = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(evaluation)


def episode(dataset="hm3d_v1", category="chair"):
    return NavigationEpisode(
        dataset, "val" if dataset == "hm3d_v1" else "val_seen", "scene", "0",
        f"Find a {category}.", [0, 0, 0], [0, 0, 0, 1], [], category,
    )


def test_v1_retains_all_cases_even_when_scene_category_and_raw_ids_repeat():
    e = episode()
    cases = [evaluation.episode_case(e, i) for i in range(100)]
    assert len({c["key"] for c in cases}) == 100
    assert len({c["episode"]["episode_id"] for c in cases}) == 100
    assert all(c["episode"]["metadata"]["original_episode_id"] == "0" for c in cases)
    assert all(c["episode"]["start_position"] == e.start_position for c in cases)
    assert e.episode_id == "0" and not e.metadata


def test_ovon_keeps_exact_episode_for_comparison_with_retained_300_cases():
    e = episode("hm3d_ovon")
    assert evaluation.episode_case(e, 42)["episode"] == e.to_dict()
    assert evaluation.episode_case(e, 42)["episode"]["goals"] is e.goals


def test_only_v1_internal_tv_category_code_changes_in_natural_language_prompt():
    assert evaluation.episode_case(episode(category="tv_monitor"), 0)["episode"]["goal_text"] == "Find a tv."
    assert evaluation.episode_case(episode("hm3d_ovon", "tv"), 0)["episode"]["goal_text"] == "Find a tv."


def test_video_selection_is_fixed_bounded_and_does_not_duplicate_small_categories():
    cases = [{"key": str(i), "episode": {"object_category": str(i)}} for i in range(79)]
    selected = evaluation.choose_videos(cases)
    assert len(selected) == len(set(selected)) == 18
    assert set(selected) <= {c["key"] for c in cases}
    assert evaluation.choose_videos(cases) == selected
    assert evaluation.choose_videos(cases[:1]) == ["0"]


def test_incremental_collection_resumes_without_double_counting_and_checks_uid(tmp_path):
    directory = tmp_path / "jobs" / "job0"
    directory.mkdir(parents=True)
    row = {"case": "a", "setting": evaluation.SETTING, "episode_id": "uid", "video": None,
           "trace": "a.jsonl", "result_file": f"a_{evaluation.SETTING}_result.json"}
    (directory / row["result_file"]).write_text(json.dumps(row))
    protocol = {"jobs": [{"id": "job0", "case_keys": ["a"]}]}
    indices = [{"key": "a", "uid": "uid", "benchmark": "v1", "dataset_id": "hm3d_v1",
                "split": "val", "object_category": "chair"}]
    results = {}
    assert evaluation.add_results(tmp_path, protocol, indices, results)["job0"] == 1
    assert evaluation.add_results(tmp_path, protocol, indices, results)["job0"] == 1
    assert len(results) == 1
    assert results["a"]["trace"] == "jobs/job0/a.jsonl"
    indices[0]["uid"] = "wrong_uid"
    with pytest.raises(ValueError, match="UID"):
        evaluation.add_results(tmp_path, protocol, indices, {})


def test_wrong_chunk_cannot_claim_another_chunks_completed_case(tmp_path):
    directory = tmp_path / "jobs" / "wrong"
    directory.mkdir(parents=True)
    (directory / f"a_{evaluation.SETTING}_result.json").write_text('{}')
    with pytest.raises(ValueError, match="wrong evaluation chunk"):
        evaluation.add_results(tmp_path, {"jobs": [{"id": "wrong", "case_keys": ["b"]}]},
                               [{"key": "a"}], {"a": {}})
