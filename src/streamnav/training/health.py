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
    if any(not math.isfinite(r.get("total_loss", 0)) for r in recent):
        result["warnings"].append("nonfinite_loss")
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
