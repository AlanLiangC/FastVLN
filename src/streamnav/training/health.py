"""Learning-quality checks independent of GPU occupancy and oracle-assisted SR."""

import math
import statistics
from typing import Any


def learning_health(rows, evaluations, expected_splits=3, min_updates=500, patience=5):
    result: dict[str, Any] = {
        "status": "starting",
        "update": 0,
        "warnings": [],
        "stop_recommended": False,
    }
    if not rows:
        return result
    recent = rows[-50:]
    update = rows[-1]["update"]
    result.update(status="training", update=update)
    if any(not math.isfinite(r.get("total_loss", r.get("loss", 0))) for r in recent):
        result["warnings"].append("nonfinite_loss")
        result["stop_recommended"] = True
    finite_fields = (
        "il_loss",
        "affordance_loss",
        "apos_loss",
        "apos_type_loss",
        "apos_teacher_prefix_loss",
        "apos_parameter_gradient_norm",
        "opos_loss",
        "opos_anchor_loss",
        "opos_support_loss",
        "ppo_loss",
        "point_kl",
        "apos_kl",
        "opos_kl",
        "value_loss",
        "entropy",
        "grad_norm",
        "replay_log_prob_error_max",
        "ppo_token_gradient_norm_max",
    )
    if any(not math.isfinite(r[k]) for r in recent for k in finite_fields if k in r):
        result["warnings"].append("nonfinite_training_component")
        result["stop_recommended"] = True
    expert_rows = [r for r in recent if r.get("expert_stream_transitions", 0)]
    if any(
        r["expert_stream_action_correct"] != r["expert_stream_transitions"] for r in expert_rows
    ):
        result["warnings"].append("expert_stream_oracle_execution_mismatch")
        result["stop_recommended"] = True
    if any(
        ("student_stream_transitions" in r or r.get("expert_probability") == 0)
        and (
            r.get("student_stream_transitions", r["student_transitions"])
            + r.get("dagger_stream_transitions", 0)
            + r["expert_stream_transitions"]
        )
        != r["transitions"]
        for r in expert_rows
    ):
        result["warnings"].append("expert_student_stream_accounting_mismatch")
        result["stop_recommended"] = True
    alphas = [r["ealm_alpha"] for r in recent if "ealm_alpha" in r]
    if alphas:
        result["ppo_policy_coefficient_mean"] = statistics.mean(1 - a for a in alphas)
        inactive = 0
        for row in reversed(rows):
            if row.get("ealm_alpha", 0) < 0.999:
                break
            inactive += 1
        result["ppo_policy_inactive_updates"] = inactive
        if inactive >= 20:
            result["warnings"].append("ppo_policy_inactive")
    elif "ppo_coefficient" in rows[-1]:
        result["ppo_policy_coefficient_mean"] = statistics.mean(
            r["ppo_coefficient"] for r in recent
        )
        result["ppo_policy_coefficient_latest"] = rows[-1]["ppo_coefficient"]
        inactive = 0
        for row in reversed(rows):
            if row.get("ppo_coefficient", 0) <= 0 or "ppo_token_gradient_norm_max" not in row:
                break
            if row["ppo_token_gradient_norm_max"] > 0:
                break
            inactive += 1
        result["ppo_policy_inactive_updates"] = inactive
        if inactive >= 20:
            result["warnings"].append("ppo_policy_inactive")
    autonomous = [r for r in recent[-20:] if r.get("expert_probability") == 0 and "successes" in r]
    if autonomous:
        episodes = sum(r["episodes"] for r in autonomous)
        result["sampled_training_episodes"] = episodes
        result["sampled_training_sr"] = (
            sum(r["successes"] for r in autonomous) / episodes if episodes else None
        )
        if episodes >= 100 and result["sampled_training_sr"] < 0.05:
            result["warnings"].append("low_sampled_training_success")
    labelled = [r for r in recent if "teacher_stop_count" in r and "greedy_stop_count" in r]
    if labelled:
        positives = sum(r["teacher_stop_count"] for r in labelled)
        stops = sum(r["greedy_stop_count"] for r in labelled)
        result["teacher_stop_labels_recent"] = positives
        result["greedy_stops_recent"] = stops
        if positives:
            result["teacher_stop_recall_recent"] = (
                sum((r["oracle_class_recall"][0] or 0) * r["teacher_stop_count"] for r in labelled)
                / positives
            )
        if len(labelled) >= 20 and positives >= 50 and stops == 0:
            result["warnings"].append("stop_not_learned")
    conditional = [
        r
        for r in recent
        if all(
            k in r
            for k in (
                "teacher_stop_count",
                "conditional_greedy_stop_count",
                "conditional_stop_true_positive",
            )
        )
    ]
    if conditional:
        positives = sum(r["teacher_stop_count"] for r in conditional)
        true_positives = sum(r["conditional_stop_true_positive"] for r in conditional)
        result["conditional_stop_labels_recent"] = positives
        result["conditional_greedy_stops_recent"] = sum(
            r["conditional_greedy_stop_count"] for r in conditional
        )
        result["conditional_stop_recall_recent"] = true_positives / positives if positives else None
        # These action logits are conditioned on sampled point tokens. Keep the
        # label distinct from fully greedy navigation/STOP recall in evaluation.
        if len(conditional) >= 20 and positives >= 50 and true_positives / positives < 0.05:
            result["warnings"].append("conditional_stop_not_learned")
    greedy = [r["greedy_action_histogram"] for r in recent if "greedy_action_histogram" in r]
    if greedy:
        means = [statistics.mean(r[a] for r in greedy) for a in range(len(greedy[0]))]
        result["greedy_distribution"] = means
        if max(means) >= 0.95:
            result["warnings"].append("deterministic_action_collapse")
            gains = [r.get("il_gain_over_prior", 0) for r in recent]
            if update >= min_updates and len(greedy) >= 50 and statistics.mean(gains) < 0.05:
                result["stop_recommended"] = True
    groups: dict[int, dict[str, Any]] = {}
    for row in evaluations:
        groups.setdefault(row["update"], {})[row["split"]] = row
    complete = [groups[u] for u in sorted(groups) if len(groups[u]) == expected_splits]
    result["complete_evaluations"] = len(complete)
    if complete:
        macro_scores = [statistics.mean(r["success"] for r in group.values()) for group in complete]
        best_index = max(range(len(complete)), key=lambda i: macro_scores[i])
        best_update = next(iter(complete[best_index].values()))["update"]
        latest_update = next(iter(complete[-1].values()))["update"]
        result["best_macro_sr"] = macro_scores[best_index]
        result["best_sr_update"] = best_update
        result["validation_updates_since_sr_improvement"] = latest_update - best_update
        # A small fixed diagnostic set is noisy: alert, but do not kill a learner
        # merely because its most recent score is below the historical maximum.
        if latest_update - best_update >= 1000:
            result["warnings"].append("no_sr_improvement_for_1000_updates")
        if latest_update >= 1000 and len(complete) >= 5 and max(macro_scores[-5:]) < 0.1:
            result["warnings"].append("persistently_low_autonomous_success")
        result["last_autonomous_sr"] = {s: r["success"] for s, r in complete[-1].items()}
        result["zero_success_splits"] = [s for s, r in complete[-1].items() if r["success"] == 0]
        result["macro_autonomous_sr"] = statistics.mean(result["last_autonomous_sr"].values())
        result["last_evaluation_update"] = next(iter(complete[-1].values()))["update"]
        latest = list(complete[-1].values())
        if all("collision_rate" in r for r in latest):
            collisions = statistics.mean(r["collision_rate"] for r in latest)
            result["macro_validation_collision_rate"] = collisions
            if collisions > 0.5:
                result["warnings"].append("high_validation_collision_rate")
        if all("oracle_success" in r for r in latest):
            result["macro_oracle_success"] = statistics.mean(r["oracle_success"] for r in latest)
        for field, warning, threshold in (
            ("false_stop_rate", "frequent_false_stops_in_validation", 0.5),
            ("alternating_turn_episode_rate", "alternating_turns_in_validation", 0.5),
            ("goal_reached_without_stop_rate", "goal_reached_without_stop", 0.1),
        ):
            if all(field in r for r in latest):
                score = statistics.mean(r[field] for r in latest)
                result["macro_validation_" + field] = score
                if score >= threshold:
                    result["warnings"].append(warning)
        if all(r["success"] == 0 for r in complete[-1].values()):
            result["warnings"].append("zero_autonomous_success")
        elif result["zero_success_splits"]:
            result["warnings"].append("zero_success_on_some_splits")
        if (
            update >= min_updates
            and latest_update >= min_updates
            and len(complete) >= patience
            and all(r["success"] == 0 for group in complete[-patience:] for r in group.values())
        ):
            result["warnings"].append("sustained_zero_success_requires_review")
            result["stop_recommended"] = True
    if result["warnings"]:
        result["status"] = "needs_review" if result["stop_recommended"] else "warning"
    return result


def grounding_learning_health(rows, evaluations):
    """Track localization scores without applying navigation SR stopping rules."""
    result = learning_health(rows, [])
    result["stage"] = "grounding"
    if rows:
        recent = rows[-50:]
        result["recent_loss"] = statistics.mean(r["loss"] for r in recent)
        seconds = sum(r.get("seconds", 0) for r in recent)
        queries = sum(sum(r.get("source_counts", {}).values()) for r in recent)
        if seconds and queries:
            result["samples_per_second"] = queries / seconds
            result["absence_fraction"] = sum(r.get("absence_samples", 0) for r in recent) / queries
    if evaluations:
        result["complete_evaluations"] = len(evaluations)
        result["latest_grounding_evaluation"] = evaluations[-1]
        metric = evaluations[-1].get("selection_metric", "macro_visible_point_accuracy_5pct")
        minimum = evaluations[-1].get("selection_min_absence_accuracy", 0.0)
        eligible = [
            r for r in evaluations if r["update"] == 0 or r.get("absence_accuracy", 0.0) >= minimum
        ]
        best = max(
            eligible or evaluations[:1],
            key=lambda r: r.get(metric, r["macro_visible_point_accuracy_5pct"]),
        )
        result["selection_metric"] = metric
        point_best = max(evaluations, key=lambda r: r["macro_visible_point_accuracy_5pct"])
        result["best_visible_point_macro_accuracy"] = point_best[
            "macro_visible_point_accuracy_5pct"
        ]
        result["best_visible_point_update"] = point_best["update"]
        result["best_selected_grounding_accuracy"] = best.get(
            metric, best["macro_visible_point_accuracy_5pct"]
        )
        result["best_selected_grounding_update"] = best["update"]
        latest = evaluations[-1]
        if "validation_loss" in latest:
            result["validation_loss"] = latest["validation_loss"]
            result["false_absence_rate"] = latest["false_absence_rate"]
        if latest["update"] - best["update"] >= 1000:
            result["warnings"].append("no_grounding_improvement_for_1000_updates")
            if result["status"] == "training":
                result["status"] = "warning"
    return result
