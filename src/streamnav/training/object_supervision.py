"""Global OPOS class normalization and per-mode rollout diagnostics."""

import math


def object_loss_weights(visible_fraction, absent_fraction, *, balanced=False, power=1.0):
    if (
        not all(math.isfinite(v) and v >= 0 for v in (visible_fraction, absent_fraction))
        or visible_fraction + absent_fraction > 1 + 1e-6
    ):
        raise ValueError("Invalid global OPOS class fractions")
    if not math.isfinite(power) or not 0 <= power <= 1:
        raise ValueError("OPOS balancing power must lie in [0, 1]")
    valid = visible_fraction + absent_fraction
    if not balanced:
        weight = 1 / max(valid, 1e-8)
        return weight, weight, weight
    normalizer = sum(v ** (1 - power) for v in (visible_fraction, absent_fraction) if v)
    positive = visible_fraction ** (-power) / normalizer if visible_fraction else 0.0
    negative = absent_fraction ** (-power) / normalizer if absent_fraction else 0.0
    location = 1 / visible_fraction if visible_fraction else 0.0
    return positive, negative, location


def summarize_object_predictions(records):
    result = {}
    for mode in ("no_hint", "hint"):
        eligible = [
            r
            for row in records
            for r in row
            if bool(r.get("object_hint_token", 0)) == (mode == "hint")
            and (r.get("perception_label") or {}).get("opos_valid")
        ]
        visible = [r for r in eligible if r["perception_label"]["opos"] > 0]
        absent = [r for r in eligible if r["perception_label"]["opos"] == 0]
        result.update(
            {
                f"opos_{mode}_visible_count": len(visible),
                f"opos_{mode}_absent_count": len(absent),
                f"opos_{mode}_visible_detection_hits": sum(
                    r["opos_greedy_token"] > 0 for r in visible
                ),
                f"opos_{mode}_visible_location_hits": sum(
                    r["opos_greedy_token"]
                    in r["perception_label"].get("opos_targets", [r["perception_label"]["opos"]])
                    for r in visible
                ),
                f"opos_{mode}_absence_hits": sum(r["opos_greedy_token"] == 0 for r in absent),
            }
        )
    return result
