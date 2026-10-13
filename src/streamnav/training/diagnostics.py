"""Rollout diagnostics only; privileged measurements never enter the policy."""

import math
from typing import Any

from streamnav.contracts.perception import (
    APOS_LEFT,
    APOS_RIGHT,
    APOS_STOP,
    POINT_CELLS,
    decode_pixel,
)

COUNTS = (
    "transitions",
    "student_transitions",
    "student_action_correct",
    "student_stream_transitions",
    "student_reward_sum",
    "student_progress_sum",
    "student_collision_count",
    "student_false_stop_count",
    "student_stop_count",
    "student_turn_pair_count",
    "student_alternating_turn_pair_count",
    "expert_stream_transitions",
    "expert_stream_action_correct",
    "teacher_stop_count_student",
    "teacher_stop_count_expert",
    "dagger_stream_transitions",
    "dagger_action_correct",
    "dagger_collision_count",
    "dagger_reward_sum",
    "dagger_progress_sum",
    "dagger_stop_count",
    "dagger_false_stop_count",
    "teacher_stop_count_dagger",
    "after_collision_steps",
    "after_collision_forward_steps",
    "apos_geometry_labels",
    "apos_geometry_point_labels",
    "apos_geometry_turn_point_labels",
    "apos_geometry_left_labels",
    "apos_geometry_right_labels",
    "apos_geometry_stop_labels",
    "apos_geometry_sample_hits",
    "apos_geometry_greedy_labelled",
    "apos_geometry_greedy_hits",
    "apos_geometry_pixel_greedy_labelled",
    "apos_geometry_pixel_greedy_hits",
    "apos_geometry_pixel_greedy_near_hits",
    "apos_geometry_special_greedy_labelled",
    "apos_geometry_special_greedy_hits",
    "conditional_action_correct",
    "teacher_stop_count",
    "conditional_greedy_stop_count",
    "conditional_stop_true_positive",
    "collision_count",
    "false_stop_count",
    "stop_count",
    "timeout_count",
    "turn_pair_count",
    "alternating_turn_pair_count",
    "point_teacher_frames",
    "apos_teacher_match_count",
    "opos_teacher_match_count",
    "reward_sum",
    "progress_sum",
)
HISTOGRAMS = {
    "apos_type_counts": 5,
    "opos_type_counts": 2,
    "student_action_counts": 6,
    "dagger_action_counts": 6,
}


def observation_distance(observation):
    # Habitat RESET and STEP use different response schemas.
    return (
        observation["geodesic_distance"]
        if "geodesic_distance" in observation
        else observation["distance"]
    )


def summarize_rollout(records):
    result: dict[str, Any] = dict.fromkeys(COUNTS, 0)
    result.update({name: [0] * length for name, length in HISTOGRAMS.items()})
    result["max_alternating_turns"] = 0
    result["max_forward_collision_streak"] = 0
    previous: list[dict[str, Any] | None] = [None] * len(records[0])
    streaks = [0] * len(previous)
    collision_streaks = [0] * len(previous)
    for row in records:
        for slot, record in enumerate(row):
            action = int(record["tokens"][2])
            expert, greedy = record["expert"], record["conditional_greedy_action"]
            student = record["student_action"]
            expert_stream = record.get("expert_stream", False)
            dagger_stream = record.get("dagger_stream", False)
            stochastic_stream = not expert_stream and not dagger_stream
            result["transitions"] += 1
            result["student_transitions"] += int(student)
            result["student_action_correct"] += int(student and action == expert)
            result["expert_stream_transitions"] += int(expert_stream)
            result["expert_stream_action_correct"] += int(expert_stream and action == expert)
            result["teacher_stop_count_student"] += int(stochastic_stream and expert == 0)
            result["teacher_stop_count_expert"] += int(expert_stream and expert == 0)
            result["teacher_stop_count_dagger"] += int(dagger_stream and expert == 0)
            if dagger_stream:
                result["dagger_stream_transitions"] += 1
                result["dagger_action_correct"] += int(action == expert)
                result["dagger_collision_count"] += int(record["collision"])
                result["dagger_reward_sum"] += float(record["reward"])
                result["dagger_progress_sum"] += (
                    record["distance_before"] - record["distance_after"]
                )
                result["dagger_stop_count"] += int(action == 0)
                result["dagger_false_stop_count"] += int(action == 0 and not record["success"])
                result["dagger_action_counts"][action] += 1
            if stochastic_stream:
                result["student_action_counts"][action] += 1
            label = record.get("perception_label")
            if label and label["apos_valid"]:
                point = 0 < label["apos"] <= POINT_CELLS
                result["apos_geometry_labels"] += 1
                result["apos_geometry_point_labels"] += int(point)
                result["apos_geometry_turn_point_labels"] += int(point and expert in (2, 3))
                result["apos_geometry_left_labels"] += int(label["apos"] == APOS_LEFT)
                result["apos_geometry_right_labels"] += int(label["apos"] == APOS_RIGHT)
                result["apos_geometry_stop_labels"] += int(label["apos"] == APOS_STOP)
                result["apos_geometry_sample_hits"] += int(
                    int(record["tokens"][0]) == label["apos"]
                )
                if "apos_greedy_token" in record:
                    result["apos_geometry_greedy_labelled"] += 1
                    result["apos_geometry_greedy_hits"] += int(
                        record["apos_greedy_token"] == label["apos"]
                    )
                    if point:
                        result["apos_geometry_pixel_greedy_labelled"] += 1
                        result["apos_geometry_pixel_greedy_hits"] += int(
                            record["apos_greedy_token"] == label["apos"]
                        )
                        predicted = decode_pixel(record["apos_greedy_token"], 1, 1)
                        target = decode_pixel(label["apos"], 1, 1)
                        result["apos_geometry_pixel_greedy_near_hits"] += int(
                            predicted is not None
                            and target is not None
                            and math.dist(predicted, target) <= 0.05
                        )
                    else:
                        result["apos_geometry_special_greedy_labelled"] += 1
                        result["apos_geometry_special_greedy_hits"] += int(
                            record["apos_greedy_token"] == label["apos"]
                        )
            result["conditional_action_correct"] += int(greedy == expert)
            result["teacher_stop_count"] += int(expert == 0)
            result["conditional_greedy_stop_count"] += int(greedy == 0)
            result["conditional_stop_true_positive"] += int(greedy == expert == 0)
            result["collision_count"] += int(record["collision"])
            result["false_stop_count"] += int(action == 0 and not record["success"])
            result["stop_count"] += int(action == 0)
            result["timeout_count"] += int(record["truncated"])
            result["reward_sum"] += float(record["reward"])
            result["progress_sum"] += record["distance_before"] - record["distance_after"]
            if stochastic_stream:
                result["student_stream_transitions"] += 1
                result["student_reward_sum"] += float(record["reward"])
                result["student_progress_sum"] += (
                    record["distance_before"] - record["distance_after"]
                )
                result["student_collision_count"] += int(record["collision"])
                result["student_false_stop_count"] += int(action == 0 and not record["success"])
                result["student_stop_count"] += int(action == 0)
            prior = previous[slot]
            continuous = (
                prior is not None
                and not prior["done"]
                and prior["episode_id"] == record["episode_id"]
            )
            collision_streaks[slot] = (
                (collision_streaks[slot] if continuous else 0) + 1
                if action == 1 and record["collision"]
                else 0
            )
            result["max_forward_collision_streak"] = max(
                result["max_forward_collision_streak"], collision_streaks[slot]
            )
            after_collision = bool(record.get("collision_before", False))
            result["after_collision_steps"] += int(after_collision)
            result["after_collision_forward_steps"] += int(after_collision and action == 1)
            turning = action in (2, 3)
            pair = (
                prior is not None and continuous and turning and int(prior["tokens"][2]) in (2, 3)
            )
            alternating = prior is not None and pair and action != int(prior["tokens"][2])
            result["turn_pair_count"] += int(pair)
            result["alternating_turn_pair_count"] += int(alternating)
            if stochastic_stream:
                result["student_turn_pair_count"] += int(pair)
                result["student_alternating_turn_pair_count"] += int(alternating)
            streaks[slot] = streaks[slot] + 1 if alternating else int(turning)
            result["max_alternating_turns"] = max(result["max_alternating_turns"], streaks[slot])
            previous[slot] = record
            apos, opos = (int(t) for t in record["tokens"][:2])
            kind = (
                0
                if apos == 0
                else (
                    1
                    if 1 <= apos <= POINT_CELLS
                    else {APOS_LEFT: 2, APOS_RIGHT: 3, APOS_STOP: 4}[apos]
                )
            )
            result["apos_type_counts"][kind] += 1
            result["opos_type_counts"][int(opos != 0)] += 1
            if record.get("teacher") is not None:
                result["point_teacher_frames"] += 1
                result["apos_teacher_match_count"] += int(
                    apos == int(record["teacher"][0].argmax())
                )
                result["opos_teacher_match_count"] += int(
                    opos == int(record["teacher"][1].argmax())
                )
    return merge_diagnostics([result])


def merge_diagnostics(workers):
    result = {key: sum(w[key] for w in workers) for key in COUNTS}
    result.update(
        {
            name: [sum(w[name][i] for w in workers) for i in range(length)]
            for name, length in HISTOGRAMS.items()
        }
    )
    result["max_alternating_turns"] = max(w["max_alternating_turns"] for w in workers)
    result["max_forward_collision_streak"] = max(w["max_forward_collision_streak"] for w in workers)
    denominators = {
        "mean_reward": ("reward_sum", "transitions"),
        "mean_progress": ("progress_sum", "transitions"),
        "student_mean_reward": ("student_reward_sum", "student_stream_transitions"),
        "dagger_collision_rate": ("dagger_collision_count", "dagger_stream_transitions"),
        "dagger_oracle_agreement": ("dagger_action_correct", "dagger_stream_transitions"),
        "dagger_mean_reward": ("dagger_reward_sum", "dagger_stream_transitions"),
        "dagger_mean_progress": ("dagger_progress_sum", "dagger_stream_transitions"),
        "dagger_false_stop_fraction": ("dagger_false_stop_count", "dagger_stop_count"),
        "after_collision_forward_rate": ("after_collision_forward_steps", "after_collision_steps"),
        "apos_geometry_label_fraction": ("apos_geometry_labels", "transitions"),
        "apos_geometry_sample_accuracy": ("apos_geometry_sample_hits", "apos_geometry_labels"),
        "apos_geometry_greedy_accuracy": (
            "apos_geometry_greedy_hits",
            "apos_geometry_greedy_labelled",
        ),
        "apos_geometry_pixel_greedy_accuracy": (
            "apos_geometry_pixel_greedy_hits",
            "apos_geometry_pixel_greedy_labelled",
        ),
        "apos_geometry_pixel_greedy_accuracy_5pct": (
            "apos_geometry_pixel_greedy_near_hits",
            "apos_geometry_pixel_greedy_labelled",
        ),
        "apos_geometry_special_greedy_accuracy": (
            "apos_geometry_special_greedy_hits",
            "apos_geometry_special_greedy_labelled",
        ),
        "student_mean_progress": ("student_progress_sum", "student_stream_transitions"),
        "student_collision_rate": ("student_collision_count", "student_stream_transitions"),
        "student_false_stop_fraction": ("student_false_stop_count", "student_stop_count"),
        "student_alternating_turn_fraction": (
            "student_alternating_turn_pair_count",
            "student_turn_pair_count",
        ),
        "training_collision_rate": ("collision_count", "transitions"),
        "student_oracle_agreement": ("student_action_correct", "student_transitions"),
        "expert_stream_oracle_agreement": (
            "expert_stream_action_correct",
            "expert_stream_transitions",
        ),
        "conditional_greedy_oracle_agreement": ("conditional_action_correct", "transitions"),
        "conditional_stop_recall": ("conditional_stop_true_positive", "teacher_stop_count"),
        "false_stop_fraction": ("false_stop_count", "stop_count"),
        "alternating_turn_fraction": ("alternating_turn_pair_count", "turn_pair_count"),
        "apos_teacher_sample_agreement": ("apos_teacher_match_count", "point_teacher_frames"),
        "opos_teacher_sample_agreement": ("opos_teacher_match_count", "point_teacher_frames"),
    }
    for name, (numerator, denominator) in denominators.items():
        result[name] = result[numerator] / result[denominator] if result[denominator] else None
    return result
