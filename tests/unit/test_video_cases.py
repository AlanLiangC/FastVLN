import pytest

from streamnav.evaluation.video_cases import retain_video_cases, select_video_cases


def episode(index, **kwargs):
    return {
        "episode_index": index,
        "scene_id": f"scene-{index}",
        "goal": "Find a chair.",
        "success": 0,
        "spl": 0,
        "episode_length": 500,
        "distance_to_goal": 5,
        "collision_rate": 0,
        **kwargs,
    }


def test_representatives_cover_outcomes_and_keep_a_fixed_anchor_across_ranks():
    records = [
        episode(7, max_consecutive_turns=400),
        episode(5, collision_rate=0.7),
        episode(4, false_stop=True, distance_to_goal=9),
        episode(3, oracle_success=1, near_goal_steps=40),
        episode(2, success=1, spl=0.8, episode_length=50),
        episode(1, success=1, spl=0.5),
        episode(0),
    ]
    selected = select_video_cases(records, 6)
    assert [r["episode_index"] for r in selected] == [0, 2, 3, 4, 5, 7]
    assert {r["selection_reasons"][0] for r in selected} == {
        "fixed_anchor",
        "success",
        "goal_reached_without_stop",
        "false_stop",
        "high_collision",
        "sustained_turning",
    }
    assert selected == select_video_cases(list(reversed(records)), 6)


def test_overlapping_reasons_do_not_duplicate_videos_or_exceed_quota():
    records = [episode(0, oracle_success=1, near_goal_steps=20, collision_rate=0.8)]
    selected = select_video_cases(records, 6)
    assert len(selected) == 1
    assert selected[0]["selection_reasons"] == [
        "fixed_anchor",
        "goal_reached_without_stop",
        "high_collision",
    ]
    assert select_video_cases(records, 0) == []
    with pytest.raises(ValueError):
        select_video_cases(records, -1)


def test_unselected_artifacts_are_removed_but_selected_video_and_trace_survive(tmp_path):
    records = [
        episode(i, video_path=f"rank_{i}/{i}.mp4", trace_path=f"rank_{i}/{i}.jsonl")
        for i in range(2)
    ]
    for record in records:
        for key in ("video_path", "trace_path"):
            path = tmp_path / record[key]
            path.parent.mkdir(exist_ok=True)
            path.write_bytes(b"case")
    retain_video_cases(tmp_path, records, [records[0]])
    for key in ("video_path", "trace_path"):
        assert (tmp_path / records[0][key]).exists()
        assert not (tmp_path / records[1][key]).exists()
    with pytest.raises(ValueError, match="escapes"):
        retain_video_cases(tmp_path, [episode(3, video_path="../outside", trace_path="trace")], [])
