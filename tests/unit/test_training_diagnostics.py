import pytest
import torch

from streamnav.contracts.perception import APOS_LEFT, APOS_RIGHT, APOS_STOP
from streamnav.training.diagnostics import (
    merge_diagnostics,
    observation_distance,
    summarize_rollout,
)


def record(action, *, expert=1, student=True, done=False, episode="one", apos=0, opos=0):
    return {
        "tokens": (apos, opos, action),
        "expert": expert,
        "conditional_greedy_action": action,
        "student_action": student,
        "done": done,
        "episode_id": episode,
        "collision": action == 1,
        "success": action == 0 and expert == 0,
        "truncated": False,
        "reward": -0.001,
        "distance_before": 3.0,
        "distance_after": 2.9,
        "teacher": None,
    }


@pytest.mark.parametrize(
    "observation",
    [{"distance": 2.0}, {"geodesic_distance": 2.0}, {"distance": 100, "geodesic_distance": 2.0}],
)
def test_distance_diagnostics_accept_actual_reset_and_step_schemas(observation):
    assert observation_distance(observation) == 2.0


def test_alternating_turns_do_not_cross_episode_resets_or_environment_slots():
    rows = [
        [record(2), record(3, episode="other")],
        [record(3, done=True), record(2, episode="new")],
        [record(2), record(3, episode="new")],
    ]
    metrics = summarize_rollout(rows)
    assert metrics["turn_pair_count"] == 2
    assert metrics["alternating_turn_pair_count"] == 2
    assert metrics["max_alternating_turns"] == 2
    assert metrics["alternating_turn_fraction"] == 1


def test_same_direction_turns_break_alternation_and_forward_breaks_turn_pairs():
    metrics = summarize_rollout([[record(a)] for a in (2, 3, 2, 2, 1, 3, 2)])
    assert metrics["max_alternating_turns"] == 3
    assert metrics["turn_pair_count"] == 4
    assert metrics["alternating_turn_pair_count"] == 3


def test_student_accuracy_excludes_forced_teacher_actions_and_stop_recall_is_conditional():
    metrics = summarize_rollout(
        [[record(1, student=False)], [record(2)], [record(0, expert=0)], [record(0)]]
    )
    assert metrics["student_transitions"] == 3
    assert metrics["student_oracle_agreement"] == pytest.approx(1 / 3)
    assert metrics["conditional_stop_recall"] == 1
    assert metrics["false_stop_fraction"] == 0.5
    assert metrics["mean_progress"] == pytest.approx(0.1)


def test_distributed_diagnostics_sum_counts_before_computing_rates():
    first = summarize_rollout([[record(1)]])
    second = summarize_rollout([[record(2)] for _ in range(9)])
    metrics = merge_diagnostics([first, second])
    assert metrics["student_oracle_agreement"] == pytest.approx(0.1)
    assert metrics["mean_reward"] == pytest.approx(-0.001)
    assert metrics["false_stop_fraction"] is None


def test_point_modes_and_teacher_agreement_keep_none_and_stop_separate():
    rows = [
        [record(1, apos=a, opos=o)]
        for a, o in ((0, 0), (1, 2), (APOS_LEFT, 0), (APOS_RIGHT, 0), (APOS_STOP, 1))
    ]
    rows[1][0]["teacher"] = (torch.tensor([0, 1.0]), torch.tensor([0, 0, 1.0]))
    metrics = summarize_rollout(rows)
    assert metrics["apos_type_counts"] == [1, 1, 1, 1, 1]
    assert metrics["opos_type_counts"] == [3, 2]
    assert metrics["point_teacher_frames"] == 1
    assert metrics["apos_teacher_sample_agreement"] == 1
    assert metrics["opos_teacher_sample_agreement"] == 1


def test_expert_stream_labels_and_execution_are_counted_separately():
    student = record(2, expert=0)
    expert = record(0, expert=0, student=False)
    student["expert_stream"], expert["expert_stream"] = False, True
    metrics = summarize_rollout([[student, expert]])
    assert metrics["student_oracle_agreement"] == 0
    assert metrics["expert_stream_transitions"] == 1
    assert metrics["expert_stream_oracle_agreement"] == 1
    assert metrics["teacher_stop_count_student"] == metrics["teacher_stop_count_expert"] == 1


def test_expert_progress_reward_and_turns_cannot_improve_student_diagnostics():
    rows = []
    for action in (2, 3):
        student, expert = record(action), record(1, student=False)
        student.update(distance_after=3.1, reward=-0.5, collision=True)
        expert.update(expert_stream=True, distance_after=1.0, reward=5, collision=False)
        rows.append([student, expert])
    metrics = summarize_rollout(rows)
    assert metrics["student_mean_progress"] == pytest.approx(-0.1)
    assert metrics["student_mean_reward"] == -0.5
    assert metrics["student_collision_rate"] == 1
    assert metrics["student_alternating_turn_fraction"] == 1
    assert metrics["mean_progress"] > 0 and metrics["mean_reward"] > 0


def test_greedy_dagger_collision_and_stop_do_not_leak_into_sampling_metrics():
    stochastic = record(2, expert=2)
    greedy = record(1, expert=3, student=False)
    greedy.update(dagger_stream=True, collision_before=True)
    metrics = summarize_rollout([[stochastic, greedy]])
    assert metrics["student_stream_transitions"] == metrics["dagger_stream_transitions"] == 1
    assert metrics["student_oracle_agreement"] == 1
    assert metrics["student_collision_rate"] == 0
    assert metrics["dagger_collision_rate"] == 1
    assert metrics["dagger_oracle_agreement"] == 0
    assert metrics["after_collision_forward_rate"] == 1


def test_blocked_forward_streaks_reset_at_episode_and_slot_boundaries():
    first = record(1, done=True)
    second = record(1, episode="new")
    metrics = summarize_rollout([[first], [second], [record(1, episode="new")]])
    assert metrics["max_forward_collision_streak"] == 2


def test_geometric_apos_supervision_is_visible_without_a_visual_teacher():
    point = record(1, apos=42)
    turn = record(3, apos=APOS_RIGHT)
    stop = record(0, expert=0, apos=0)
    for row, target, greedy in (
        (point, 42, 43),
        (turn, APOS_RIGHT, APOS_RIGHT),
        (stop, APOS_STOP, APOS_STOP),
    ):
        row.update(perception_label={"apos_valid": True, "apos": target}, apos_greedy_token=greedy)
    metrics = summarize_rollout([[point], [turn], [stop]])
    assert metrics["point_teacher_frames"] == 0
    assert metrics["apos_geometry_labels"] == 3
    assert metrics["apos_geometry_point_labels"] == 1
    assert metrics["apos_geometry_right_labels"] == metrics["apos_geometry_stop_labels"] == 1
    assert metrics["apos_geometry_sample_accuracy"] == pytest.approx(2 / 3)
    assert metrics["apos_geometry_greedy_accuracy"] == pytest.approx(2 / 3)
    assert metrics["apos_geometry_pixel_greedy_accuracy"] == 0
    assert metrics["apos_geometry_pixel_greedy_accuracy_5pct"] == 1
    assert metrics["apos_geometry_special_greedy_accuracy"] == 1
