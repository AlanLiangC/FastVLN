import pytest

from streamnav.evaluation.navigation_metrics import aggregate_metrics


@pytest.mark.parametrize("batch_body", [False, True])
def test_standalone_evaluation_restores_training_body_mode(tmp_path, monkeypatch, batch_body):
    import json
    from types import SimpleNamespace

    from omegaconf import OmegaConf

    from streamnav.evaluation import runner

    (tmp_path / "manifest.json").write_text(json.dumps({"update": 900}))
    policy = SimpleNamespace(loaded_checkpoint=str(tmp_path), batch_chat_body=not batch_body)
    calls = []
    monkeypatch.setattr(runner.parallel, "initialize", lambda config: None)
    monkeypatch.setattr(runner, "seed_everything", lambda seed: None)

    def load(config, **kwargs):
        assert kwargs["preserve_master_weights"] is True
        return policy

    def evaluate(actual, config, **kwargs):
        calls.append((actual.batch_chat_body, kwargs["update"], kwargs["video"]))

    monkeypatch.setattr(runner, "load_policy", load)
    monkeypatch.setattr(runner, "evaluate", evaluate)
    runner.main.__wrapped__(
        OmegaConf.create(
            {"seed": 2025, "trainer": {"batch_chat_body": batch_body}, "eval": {"video": True}}
        )
    )
    assert calls == [(batch_body, 900, True)]


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
