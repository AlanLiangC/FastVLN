import pytest

from streamnav.training.health import grounding_learning_health, learning_health


def test_opos_collapse_is_detected_and_old_recipe_results_do_not_stop_a_repair():
    rows = [{"update": 1800, "loss": 1.0}]
    cases = [
        {
            "update": update,
            "split": split,
            "success": 0.3,
            "opos_visible_count": 100,
            "opos_false_absence_rate": 0.99,
        }
        for update in (1600, 1700, 1800)
        for split in ("seen", "synonyms", "unseen")
    ]
    health = learning_health(rows, cases)
    assert health["stop_recommended"]
    assert "opos_no_hint_visible_detection_collapse" in health["warnings"]
    assert not learning_health(rows, cases, perception_start_update=1800)["stop_recommended"]
    for case in cases:
        case["opos_visible_count"] = 2
    assert not learning_health(rows, cases)["stop_recommended"]


def test_turning_collapse_cannot_be_hidden_by_a_few_easy_successes():
    rows = [{"update": 1000, "schedule_update": 200, "loss": 1.0}]
    evaluations = [
        {
            "update": u,
            "split": s,
            "success": 2 / 48 if s == "seen" else 0.0,
            "episodes": 48,
            "stationary_episode_rate": 0.9,
            "greedy_action_histogram": [0.001, 0.001, 0.2, 0.798, 0.0, 0.0],
        }
        for u in (900, 1000)
        for s in ("seen", "synonyms", "unseen")
    ]
    health = learning_health(rows, evaluations, min_updates=200)
    assert "turning_without_progress_in_validation" in health["warnings"]
    assert health["stop_recommended"]
    rows[0]["schedule_update"] = 50
    assert not learning_health(rows, evaluations, min_updates=200)["stop_recommended"]
    for row in evaluations:
        row["episodes"] = 6
    rows[0]["schedule_update"] = 200
    assert not learning_health(rows, evaluations, min_updates=200)["stop_recommended"]


def test_grounding_monitor_tracks_points_and_never_invents_navigation_success():
    rows = [
        {
            "update": 200,
            "loss": 2.0,
            "grad_norm": 1.0,
            "seconds": 2.0,
            "source_counts": {"coco": 230, "pixmo": 26},
            "absence_samples": 26,
        }
    ]
    evaluations = [{"update": 200, "macro_visible_point_accuracy_5pct": 0.0}]
    status = grounding_learning_health(rows, evaluations)
    assert status["samples_per_second"] == 128
    assert status["absence_fraction"] == 26 / 256
    assert status["best_visible_point_update"] == 200
    assert "macro_autonomous_sr" not in status and not status["stop_recommended"]
    rows[0]["opos_loss"] = float("nan")
    assert grounding_learning_health(rows, evaluations)["stop_recommended"]


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


def test_lightnav_warmup_does_not_trigger_inactive_ppo_warning():
    rows = [
        {"update": i, "loss": 1, "ppo_coefficient": 0, "ppo_token_gradient_norm_max": 0}
        for i in range(1, 51)
    ]
    report = learning_health(rows, [])
    assert report["ppo_policy_inactive_updates"] == 0
    assert "ppo_policy_inactive" not in report["warnings"]


def test_active_lightnav_ppo_without_token_gradients_is_reported():
    rows = [
        {"update": i, "loss": 1, "ppo_coefficient": 0.1, "ppo_token_gradient_norm_max": 0}
        for i in range(51, 71)
    ]
    report = learning_health(rows, [])
    assert "ppo_policy_inactive" in report["warnings"]
    rows[-1]["ppo_token_gradient_norm_max"] = 0.1
    assert "ppo_policy_inactive" not in learning_health(rows, [])["warnings"]


def test_low_stochastic_training_sr_alert_does_not_stop_or_invent_formal_sr():
    rows = [
        {"update": i, "loss": 1, "expert_probability": 0, "episodes": 10, "successes": 0}
        for i in range(51, 71)
    ]
    report = learning_health(rows, [])
    assert report["sampled_training_sr"] == 0
    assert "low_sampled_training_success" in report["warnings"]
    assert "macro_autonomous_sr" not in report
    assert not report["stop_recommended"]


def test_finite_total_loss_cannot_hide_nonfinite_components():
    report = learning_health([{"update": 1, "loss": 1, "grad_norm": float("nan")}], [])
    assert report["stop_recommended"]
    assert "nonfinite_training_component" in report["warnings"]


def test_lightnav_conditional_stop_monitor_uses_counts_and_distinct_label():
    rows = [
        {
            "update": i,
            "loss": 1,
            "teacher_stop_count": 3,
            "conditional_greedy_stop_count": 1,
            "conditional_stop_true_positive": 0,
        }
        for i in range(81, 101)
    ]
    rows[-1].update(teacher_stop_count=10, conditional_stop_true_positive=1)
    report = learning_health(rows, [])
    assert report["conditional_stop_labels_recent"] == 67
    assert report["conditional_stop_recall_recent"] == 1 / 67
    assert "conditional_stop_not_learned" in report["warnings"]
    assert "teacher_stop_recall_recent" not in report
    assert not report["stop_recommended"]
    rows[-1]["conditional_stop_true_positive"] = 10
    assert "conditional_stop_not_learned" not in learning_health(rows, [])["warnings"]


def test_lightnav_conditional_stop_monitor_does_not_invent_missing_positives():
    rows = [
        {
            "update": i,
            "loss": 1,
            "teacher_stop_count": 0,
            "conditional_greedy_stop_count": 0,
            "conditional_stop_true_positive": 0,
        }
        for i in range(81, 101)
    ]
    report = learning_health(rows, [])
    assert report["conditional_stop_recall_recent"] is None
    assert "conditional_stop_not_learned" not in report["warnings"]


def test_expert_stream_supervisor_checks_execution_and_student_accounting():
    row = {
        "update": 105,
        "loss": 1,
        "expert_probability": 0,
        "transitions": 1600,
        "student_transitions": 800,
        "expert_stream_transitions": 800,
        "expert_stream_action_correct": 800,
    }
    assert not learning_health([row], [])["stop_recommended"]
    row["expert_stream_action_correct"] = 799
    assert "expert_stream_oracle_execution_mismatch" in learning_health([row], [])["warnings"]
    row["expert_stream_action_correct"] = 800
    row["student_transitions"] = 1600
    assert "expert_student_stream_accounting_mismatch" in learning_health([row], [])["warnings"]


@pytest.mark.parametrize("teacher_probability,student_actions", [(0, 800), (0.7, 240)])
def test_three_stream_supervisor_counts_dagger_and_warmup_actions(
    teacher_probability, student_actions
):
    row = {
        "update": 415,
        "loss": 1,
        "expert_probability": teacher_probability,
        "transitions": 2400,
        "student_transitions": student_actions,
        "student_stream_transitions": 800,
        "dagger_stream_transitions": 800,
        "expert_stream_transitions": 800,
        "expert_stream_action_correct": 800,
    }
    assert not learning_health([row], [])["stop_recommended"]
    row["dagger_stream_transitions"] = 799
    report = learning_health([row], [])
    assert report["stop_recommended"]
    assert "expert_student_stream_accounting_mismatch" in report["warnings"]
