import pytest

from streamnav.evaluation.navigation_metrics import aggregate_metrics


def test_near_goal_stop_probabilities_are_weighted_by_decision_count():
    common = {
        "success": 0.0,
        "spl": 0.0,
        "soft_spl": 0.0,
        "distance_to_goal": 1.0,
        "episode_length": 10,
        "collision_rate": 0.0,
    }
    episodes = [
        {
            **common,
            "oracle_success": 1.0,
            "near_goal_steps": 1,
            "near_goal_stop_probability_sum": 0.8,
            "near_goal_stop_probability_max": 0.8,
            "false_stop": False,
        },
        {
            **common,
            "oracle_success": 1.0,
            "near_goal_steps": 3,
            "near_goal_stop_probability_sum": 0.6,
            "near_goal_stop_probability_max": 0.4,
            "false_stop": False,
        },
        {
            **common,
            "oracle_success": 0.0,
            "near_goal_steps": 0,
            "near_goal_stop_probability_sum": 0.0,
            "near_goal_stop_probability_max": None,
            "false_stop": True,
        },
    ]
    result = aggregate_metrics(episodes)
    assert result["near_goal_steps"] == 4
    assert result["near_goal_stop_probability_mean"] == pytest.approx(0.35)
    assert result["near_goal_stop_probability_max"] == 0.8
    assert result["goal_reached_without_stop_rate"] == pytest.approx(2 / 3)
    assert result["false_stop_rate"] == pytest.approx(1 / 3)
