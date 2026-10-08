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
                "preupdate_replay_log_prob_error_max": completed / 100,
            }
        )
    result = combine_metrics(rows)
    assert result["episodes_completed"] == 4
    assert result["success"] == 0.25
    assert result["spl"] == pytest.approx(0.2)
    assert result["oracle_class_recall"] == [0.25, 1.0, None, None]
    assert result["curriculum_fallbacks"] == 3
    assert result["gpu_memory_bytes"] == 103
    assert result["preupdate_replay_log_prob_error_max"] == 0.03


@pytest.mark.parametrize("zero_positive_rank", [False, True])
def test_stop_metrics_and_prior_use_global_sample_counts(zero_positive_rank):
    rows = []
    for counts, confusion, positive_probability, negative_probability in (
        ([1, 9], [[1, 0], [2, 7]], 0.9, 0.1),
        ([3, 3], [[1, 2], [0, 3]], 0.3, 0.2),
    ):
        rows.append(
            {
                "update": 1,
                "episodes_completed": 0,
                "success": 0.0,
                "spl": 0.0,
                "expert_action_counts": counts,
                "expert_action_histogram": [n / sum(counts) for n in counts],
                "greedy_action_confusion": confusion,
                "oracle_class_recall": [confusion[a][a] / counts[a] for a in range(2)],
                "stop_probability_on_teacher_stop": positive_probability,
                "stop_probability_on_teacher_nonstop": negative_probability,
                "oracle_prior_cross_entropy": 0.0,
                "il_loss": 0.4,
                "il_gain_over_prior": 0.0,
            }
        )
    if zero_positive_rank:
        rows[1]["expert_action_counts"] = [0, 6]
        rows[1]["expert_action_histogram"] = [0.0, 1.0]
        rows[1]["greedy_action_confusion"] = [[0, 0], [0, 6]]
        rows[1]["oracle_class_recall"] = [None, 1.0]
        rows[1]["stop_probability_on_teacher_stop"] = None
    result = combine_metrics(rows)
    if zero_positive_rank:
        assert result["teacher_stop_count"] == 1
        assert result["stop_probability_on_teacher_stop"] == 0.9
        assert result["stop_probability_on_teacher_nonstop"] == pytest.approx(0.14)
        return
    assert result["expert_action_counts"] == [4, 12]
    assert result["greedy_action_confusion"] == [[2, 2], [2, 10]]
    assert result["oracle_class_recall"] == pytest.approx([0.5, 10 / 12])
    assert result["teacher_stop_count"] == result["greedy_stop_count"] == 4
    assert result["greedy_stop_precision"] == 0.5
    assert result["stop_probability_on_teacher_stop"] == pytest.approx(0.45)
    assert result["stop_probability_on_teacher_nonstop"] == pytest.approx(0.125)
    assert result["oracle_prior_cross_entropy"] == pytest.approx(0.5623351446)
    assert result["il_gain_over_prior"] == pytest.approx(0.1623351446)
