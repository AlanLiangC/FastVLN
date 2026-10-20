"""Prediction-only pointing diagnostics shared by case videos and audits."""

from typing import Any

from streamnav.contracts.perception import POINT_CELLS
from streamnav.models.policy.perception import decode_point_logits, point_type_logits


def perception_records(predictions, *, hierarchical=False):
    records: list[dict[str, Any]]
    if predictions.get("point_cot", False):
        records = [{"point_cot": True} for _ in predictions["apos"]]
    else:
        arrival = predictions["arrival"].softmax(-1).tolist()
        records = [{"arrival_probabilities": probabilities} for probabilities in arrival]
    if predictions.get("point_cot", False) and "arrival" in predictions:
        for record, probabilities in zip(
            records, predictions["arrival"].softmax(-1).tolist(), strict=True
        ):
            record["arrival_probabilities"] = probabilities
    if "object_hint_tokens" in predictions:
        for record, token in zip(records, predictions["object_hint_tokens"], strict=True):
            record["object_hint_token"] = int(token)
    if "current_action_logits" in predictions:
        probabilities = predictions["current_action_logits"].softmax(-1).tolist()
        tokens = predictions["selected_action_tokens"].tolist()
        for record, probs, selected in zip(records, probabilities, tokens, strict=True):
            record["current_action_probabilities"] = probs
            record["temporal_action_changed"] = selected != max(
                range(len(probs)), key=probs.__getitem__
            )
    if "action_plan_logits" in predictions:
        plans = predictions["action_plan_logits"]
        for record, tokens, probabilities in zip(
            records, plans.argmax(-1).tolist(), plans.softmax(-1).tolist(), strict=True
        ):
            end = tokens.index(0) + 1 if 0 in tokens else len(tokens)
            tokens, probabilities = tokens[:end], probabilities[:end]
            record["action_plan"] = tokens
            record["action_plan_probabilities"] = probabilities
    for name in ("apos", "opos"):
        logits = predictions[name]
        tokens = decode_point_logits(logits, hierarchical=hierarchical).tolist()
        if predictions.get("point_cot", False) and name + "_tokens" in predictions:
            # Draw the token that actually conditioned the next autoregressive
            # prediction, even when an external caller requests another mode.
            tokens = predictions[name + "_tokens"].tolist()
        flat = logits.argmax(-1).tolist()
        types = point_type_logits(logits).softmax(-1).tolist()
        locations = logits[..., 1 : POINT_CELLS + 1].softmax(-1).amax(-1).tolist()
        for i, token in enumerate(tokens):
            records[i].update(
                {
                    name: token,
                    name + "_flat_argmax": flat[i],
                    name + "_type_probabilities": types[i],
                    name + "_point_probability": types[i][1],
                    name + "_conditional_location_peak": locations[i],
                }
            )
    return records
