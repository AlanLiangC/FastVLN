"""Prediction-only pointing diagnostics shared by case videos and audits."""

from streamnav.contracts.perception import POINT_CELLS
from streamnav.models.policy.perception import decode_point_logits, point_type_logits


def perception_records(predictions, *, hierarchical=False):
    if predictions.get("point_cot", False):
        records = [{"point_cot": True} for _ in predictions["apos"]]
    else:
        arrival = predictions["arrival"].softmax(-1).tolist()
        records = [{"arrival_probabilities": probabilities} for probabilities in arrival]
    for name in ("apos", "opos"):
        logits = predictions[name]
        tokens = decode_point_logits(logits, hierarchical=hierarchical).tolist()
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
