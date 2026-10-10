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
        for u in (1, 100, 200, 300, 400, 500)
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
    rows = [r for r in evaluations(0.1) if r["update"] < 500] + [
        {"update": 500, "split": "seen", "success": 0.0}
    ]
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


def test_finite_loss_cannot_hide_inactive_ppo_or_unlearned_stop():
    rows = training_rows([0, 0.3, 0.3, 0.4], gain=0.1)
    for row in rows:
        row.update(
            ealm_alpha=1.0,
            teacher_stop_count=2,
            greedy_stop_count=0,
            oracle_class_recall=[0.0, 0.5, 0.5, 0.5],
        )
    validation = [
        {"update": 500, "split": s, "success": 0.0, "oracle_success": 0.1, "collision_rate": 0.7}
        for s in ["seen", "synonyms", "unseen"]
    ]
    report = learning_health(rows, validation, min_updates=1000)
    assert not report["stop_recommended"]
    assert report["ppo_policy_inactive_updates"] == 50
    assert report["ppo_policy_coefficient_mean"] == 0
    assert report["teacher_stop_labels_recent"] == 100
    assert report["teacher_stop_recall_recent"] == 0
    assert report["greedy_stops_recent"] == 0
    assert {"ppo_policy_inactive", "stop_not_learned", "high_validation_collision_rate"} <= set(
        report["warnings"]
    )


def test_ppo_and_stop_warnings_follow_recent_learning():
    rows = training_rows([0.25] * 4)
    for row in rows:
        row.update(
            ealm_alpha=1.0,
            teacher_stop_count=2,
            greedy_stop_count=0,
            oracle_class_recall=[0.0, 0.5, 0.5, 0.5],
        )
    rows[-1].update(ealm_alpha=0.5, greedy_stop_count=1, oracle_class_recall=[0.5] * 4)
    report = learning_health(rows, [])
    assert report["ppo_policy_inactive_updates"] == 0
    assert report["teacher_stop_recall_recent"] == 0.01
    assert "ppo_policy_inactive" not in report["warnings"]
    assert "stop_not_learned" not in report["warnings"]


def test_zero_sr_stop_waits_for_complete_validation_at_review_budget():
    rows = training_rows([0.25] * 4)
    prior = [r for r in evaluations() if r["update"] < 500]
    incomplete = prior + [{"update": 500, "split": "seen", "success": 0.0}]
    assert not learning_health(rows, incomplete)["stop_recommended"]
    assert learning_health(rows, evaluations())["stop_recommended"]
