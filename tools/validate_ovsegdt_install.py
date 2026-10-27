"""Strict-load and recurrent inference checks for the original OVSegDT checkpoint.

Run with scripts/ovsegdt_env.sh and its OVSEGDT_PYTHON interpreter. This repeats a
real RGB fixture to check integration; it does not measure navigation success.
"""

import argparse
import json
import pickle
from pathlib import Path

import numpy as np
import torch
from ovsegdt_runtime import (
    ASSETS,
    ROOT,
    ObjectMaskDetector,
    load_policy,
    prepare_observation,
    start_state,
    step,
)
from PIL import Image


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input", type=Path, default=ROOT / "runtime/fixtures/navigation/ovsegdt_input.npz"
    )
    parser.add_argument(
        "--output", type=Path, default=ROOT / "runtime/reports/ovsegdt_install_validation.json"
    )
    parser.add_argument("--detector", action="store_true", help="Also test the YOLOE mask path")
    parser.add_argument(
        "--category", help="Override the target for a single-frame integration check"
    )
    args = parser.parse_args()
    torch.set_num_threads(2)
    fixture = np.load(args.input)
    rgb = fixture["rgb"]
    source_instruction = str(fixture["instruction"])
    instruction = "Find a " + args.category + "." if args.category else source_instruction
    category = instruction.removeprefix("Find a ").removesuffix(".")
    with (ASSETS / "siglip.pkl").open("rb") as handle:
        goals = pickle.load(handle)
    goal = goals[category]
    assert goal.shape == (768,) and np.isfinite(goal).all()
    policy, _, report = load_policy()
    mask = np.zeros(rgb.shape[:2], dtype=np.float32)
    cache, previous = start_state(policy)
    first, probabilities, next_cache, value = step(policy, rgb, mask, goal, cache, previous, 0)
    observation = prepare_observation(rgb, mask, goal, 0)
    original_value, original_action, _, original_cache, _ = policy.act(
        observation,
        cache,
        previous,
        torch.zeros(1, 128, device="cuda"),
        torch.zeros(1, 1, dtype=torch.bool, device="cuda"),
        deterministic=True,
    )
    torch.testing.assert_close(first, original_action, rtol=0, atol=0)
    torch.testing.assert_close(next_cache, original_cache, rtol=0, atol=0)
    torch.testing.assert_close(value, original_value, rtol=0, atol=0)
    first_probabilities = probabilities.cpu().tolist()[0]
    previous, cache = first, next_cache
    expected_shape = tuple(cache.shape)
    for frame_id in range(1, 105):
        action, probabilities, cache, value = step(
            policy, rgb, mask, goal, cache, previous, frame_id
        )
        assert tuple(cache.shape) == expected_shape
        assert probabilities.shape == (1, 6)
        assert 0 <= action.item() < 6
        assert torch.isfinite(probabilities).all() and torch.isfinite(cache).all()
        assert torch.isfinite(value).all()
        torch.testing.assert_close(probabilities.sum(-1), torch.ones(1, device="cuda"))
        previous = action
    cache, previous = start_state(policy)
    reset_action, reset_probs, reset_cache, reset_value = step(
        policy, rgb, mask, goal, cache, previous, 0
    )
    assert reset_action.item() == first.item()
    torch.testing.assert_close(reset_probs[0], torch.tensor(first_probabilities, device="cuda"))
    torch.testing.assert_close(reset_cache, next_cache, rtol=0, atol=0)
    torch.testing.assert_close(reset_value, original_value, rtol=0, atol=0)
    # Exercise a nonempty mask too. This is an artificial integration input,
    # not a ground-truth object annotation and not used to claim performance.
    mask[rgb.shape[0] // 3 : 2 * rgb.shape[0] // 3, rgb.shape[1] // 3 : 2 * rgb.shape[1] // 3] = 1
    action, mask_probs, _, mask_value = step(policy, rgb, mask, goal, cache, previous, 0)
    assert torch.isfinite(mask_probs).all() and torch.isfinite(mask_value).all()
    assert not torch.allclose(mask_probs, reset_probs)
    report.update(
        input=str(args.input.resolve()),
        instruction=instruction,
        fixture_source_instruction=source_instruction,
        original_act_equivalence=True,
        recurrent_steps=105,
        context_window=100,
        cache_shape=list(expected_shape),
        cache_bytes=cache.numel() * cache.element_size(),
        episode_reset_reproducible=True,
        empty_mask_first_action=int(first.item()),
        empty_mask_first_probabilities=first_probabilities,
        synthetic_nonempty_mask_action=int(action.item()),
        synthetic_nonempty_mask_probabilities=mask_probs.cpu().tolist()[0],
        text_categories=len(goals),
        scope="Integration check on repeated real RGB, not closed-loop navigation or latency.",
        passed=True,
    )
    if args.detector:
        detector = ObjectMaskDetector("train")
        detected_mask = detector.predict(rgb, category)
        assert detected_mask.shape == (224, 224)
        assert detected_mask.dtype == torch.bool
        external_action, _, external_cache, external_value = step(
            policy, rgb, detected_mask, goal, cache, previous, 0
        )
        # Compare external-mask routing to the author's own inline detector
        # branch, with the same class vocabulary, thresholds and checkpoint.
        policy.net.segmentation_source = "yolo_train"
        policy.net.segm_model[0] = detector.model
        policy.net.cats_text = detector.categories
        policy.net.conf_per_cat = detector.thresholds
        policy.net.cache_values = torch.stack(
            [torch.from_numpy(goals[name]) for name in detector.categories]
        ).cuda()
        original_value, original_action, _, original_cache, _ = policy.act(
            prepare_observation(rgb, np.zeros(rgb.shape[:2]), goal, 0),
            cache,
            previous,
            torch.zeros(1, 128, device="cuda"),
            torch.zeros(1, 1, dtype=torch.bool, device="cuda"),
            deterministic=True,
        )
        torch.testing.assert_close(external_action, original_action, rtol=0, atol=0)
        torch.testing.assert_close(external_value, original_value, rtol=0, atol=0)
        torch.testing.assert_close(external_cache, original_cache, rtol=0, atol=0)
        image_path = args.output.with_suffix(".png")
        image_path.parent.mkdir(parents=True, exist_ok=True)
        overlay = rgb.copy()
        pixels = np.asarray(
            Image.fromarray(detected_mask.cpu().numpy()).resize(
                (rgb.shape[1], rgb.shape[0]), Image.Resampling.NEAREST
            )
        )
        overlay[pixels] = (0.45 * overlay[pixels] + 0.55 * np.array([0, 255, 90])).astype(np.uint8)
        Image.fromarray(overlay).save(image_path)
        report["detector"] = {
            "weight": str(ASSETS / "yoloe-v8l-seg.pt"),
            "categories": len(detector.categories),
            "vocabulary": "train (matches saved checkpoint config)",
            "input_size": [224, 224],
            "target_mask_pixels": int(detected_mask.sum().item()),
            "action": int(external_action.item()),
            "original_inline_detector_equivalence": True,
            "visualization": str(image_path.resolve()),
        }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report))


if __name__ == "__main__":
    main()
