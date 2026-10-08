from streamnav.training.health import learning_health


def training_rows(greedy, gain=0.0):
    return [
        {
            "update": u,
            "total_loss": 1.0,
            "greedy_action_histogram": greedy,
            "policy_action_histogram": [0.25] * 4,
            "success": 1.0,  # Oracle-assisted successes must not hide failure.
            "il_gain_over_prior": gain,
        }
        for u in range(451, 501)
    ]


def evaluations(success=0.0):
    return [
        {"update": u, "split": s, "success": success}
        for u in (1, 100, 200, 300, 400)
        for s in ("seen", "synonyms", "unseen")
    ]


def test_deterministic_collapse_not_hidden_by_sampling_or_dagger():
    status = learning_health(training_rows([0, 0, 1, 0]), [])
    assert status["stop_recommended"]
    assert "deterministic_action_collapse" in status["warnings"]


def test_repeated_zero_autonomous_success_stops_balanced_policy():
    status = learning_health(training_rows([0.25] * 4), evaluations())
    assert status["stop_recommended"]
    assert "sustained_zero_success_requires_review" in status["warnings"]


def test_partial_validation_does_not_complete_an_evaluation():
    rows = evaluations(0.1) + [{"update": 500, "split": "seen", "success": 0.0}]
    status = learning_health(training_rows([0.25] * 4, gain=0.4), rows)
    assert not status["stop_recommended"]
    assert status["last_evaluation_update"] == 400
    assert status["complete_evaluations"] == 5


def test_nonfinite_loss_requires_immediate_stop():
    assert learning_health([{"update": 1, "total_loss": float("nan")}], [])["stop_recommended"]


def test_seen_success_does_not_hide_zero_unseen_performance():
    evaluation = [
        {"update": 100, "split": "seen", "success": 0.1},
        {"update": 100, "split": "synonyms", "success": 0.0},
        {"update": 100, "split": "unseen", "success": 0.0},
    ]
    status = learning_health(training_rows([0.25] * 4), evaluation)
    assert status["zero_success_splits"] == ["synonyms", "unseen"]
    assert status["status"] == "warning"
    assert not status["stop_recommended"]


def test_nonzero_but_stalled_learning_is_reported_without_automatic_stop():
    history = [
        {"update": u, "split": split, "success": 0.05}
        for u in (1000, 1500, 2000, 2500, 3000)
        for split in ("seen", "synonyms", "unseen")
    ]
    report = learning_health(training_rows([0.25] * 4), history)
    assert report["validation_updates_since_sr_improvement"] == 2000
    assert "no_sr_improvement_for_1000_updates" in report["warnings"]
    assert "persistently_low_autonomous_success" in report["warnings"]
    assert not report["stop_recommended"]
