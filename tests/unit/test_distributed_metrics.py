import pytest

from streamnav.training.distributed import combine_metrics


def test_global_episode_and_class_metrics_use_sample_counts():
    rows = []
    for completed, success, stop_fraction, stop_recall in (
        (1, 1.0, 0.1, 1.0),
        (3, 0.0, 0.3, 0.0),
        (0, 0.0, 0.0, None),
    ):
        rows.append(
            {
                "update": 7,
                "episodes_completed": completed,
                "success": success,
                "spl": success * 0.8,
                "expert_action_histogram": [stop_fraction, 1 - stop_fraction, 0, 0],
                "oracle_class_recall": [stop_recall, 1.0, None, None],
                "curriculum_fallbacks": 1,
                "gpu_memory_bytes": 100 + completed,
            }
        )
    result = combine_metrics(rows)
    assert result["episodes_completed"] == 4
    assert result["success"] == 0.25
    assert result["spl"] == pytest.approx(0.2)
    assert result["oracle_class_recall"] == [0.25, 1.0, None, None]
    assert result["curriculum_fallbacks"] == 3
    assert result["gpu_memory_bytes"] == 103
