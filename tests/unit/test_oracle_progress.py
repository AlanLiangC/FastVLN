from streamnav.training.oracle_progress import OracleProgressTracker


def test_policy_errors_do_not_count_as_teacher_failure():
    tracker = OracleProgressTracker(5)
    for _ in range(100):
        tracker.record_advice(1)
        tracker.record_step(2, 5)
    assert not tracker.stalled
    assert tracker.stagnant_teacher_steps == 0


def test_followed_but_stalled_teacher_is_still_detected():
    tracker = OracleProgressTracker(5)
    for _ in range(64):
        tracker.record_advice(2)
        tracker.record_step(2, 5)
    assert tracker.stalled
    tracker.record_advice(1)
    tracker.record_step(1, 4.9)
    assert not tracker.stalled


def test_old_advice_cannot_be_counted_repeatedly():
    tracker = OracleProgressTracker(5)
    tracker.record_advice(2)
    for _ in range(64):
        tracker.record_step(2, 5)
    assert not tracker.stalled


def test_small_progress_accumulates_and_policy_deviation_resets_reference():
    tracker = OracleProgressTracker(5)
    for i in range(1, 64):
        tracker.record_advice(1)
        tracker.record_step(1, 5 - i * 0.005)
    assert tracker.stagnant_teacher_steps < 5
    tracker.record_advice(1)
    tracker.record_step(2, 8)
    assert tracker.best_distance == 8
    assert tracker.stagnant_teacher_steps == 0
