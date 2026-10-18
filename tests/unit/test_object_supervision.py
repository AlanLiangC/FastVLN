import pytest
import torch

from streamnav.training.object_supervision import object_loss_weights, summarize_object_predictions
from streamnav.training.point_supervision import marginal_point_loss, navigation_object_losses


def test_global_class_balancing_and_visible_location_are_fragment_invariant():
    logits = torch.randn(6, 1297, requires_grad=True)
    targets = [[42, 43], [0], [0], [], [0], [0]]
    _, visibility, location = navigation_object_losses(logits, targets, anchor_weight=0.2)
    positive, negative, spatial = object_loss_weights(1 / 6, 4 / 6, balanced=True)
    weights = torch.tensor([positive, negative, negative, 0, negative, negative])
    per_frame = weights * visibility + spatial * location
    expected = 0.5 * visibility[0] + 0.5 * visibility[[1, 2, 4, 5]].mean() + location[0]
    torch.testing.assert_close(per_frame.mean(), expected)
    torch.testing.assert_close(sum(per_frame[i : i + 2].sum() / 6 for i in (0, 2, 4)), expected)
    per_frame.mean().backward()
    assert logits.grad[0, 0] > 0 and logits.grad[1, 0] < 0
    assert logits.grad[3].count_nonzero() == 0


def test_anchor_region_mixture_counts_visibility_once_and_spatial_loss_has_no_none_gradient():
    logits = torch.randn(3, 1297, requires_grad=True)
    targets = [[42, 43, 44], [0], []]
    total, visibility, location = navigation_object_losses(logits, targets, anchor_weight=0.2)
    expected = 0.2 * marginal_point_loss(logits, [[42], [0], []]) + 0.8 * marginal_point_loss(
        logits, targets
    )
    torch.testing.assert_close(total, expected)
    torch.testing.assert_close(total, visibility + location)
    gradient = torch.autograd.grad(location.sum(), logits)[0]
    assert gradient[0, 0].abs() < 1e-6
    unanchored = navigation_object_losses(logits, targets)[2]
    original_gradient = torch.autograd.grad(unanchored.sum(), logits)[0]
    assert gradient[0, 42] < original_gradient[0, 42]
    assert gradient[0, 43] > original_gradient[0, 43]
    assert gradient[1:].count_nonzero() == 0


def test_missing_classes_are_finite_and_only_present_classes_receive_weight():
    assert object_loss_weights(0, 0, balanced=True) == (0, 0, 0)
    assert object_loss_weights(0, 0.5, balanced=True) == (0, 2, 0)
    assert object_loss_weights(0.25, 0, balanced=True) == (4, 0, 4)
    with pytest.raises(ValueError):
        object_loss_weights(float("nan"), 0.8)
    with pytest.raises(ValueError):
        object_loss_weights(0.8, 0.8)


def test_tempered_visibility_keeps_negative_prior_between_raw_and_equal_weighting():
    positive, negative, location = object_loss_weights(0.2, 0.8, balanced=True, power=0.5)
    assert 0.2 < positive * 0.2 < 0.5
    assert negative * 0.8 > 0.5
    assert positive * 0.2 + negative * 0.8 == pytest.approx(1)
    assert location == 5
    assert object_loss_weights(0, 0, balanced=True, power=0.5) == (0, 0, 0)
    assert object_loss_weights(0.2, 0.8, balanced=True, power=0)[:2] == (1, 1)


def test_mode_diagnostics_count_null_input_positive_labels_and_mask_unknowns():
    def record(hint, gt, pred, valid=True):
        return {
            "object_hint_token": hint,
            "opos_greedy_token": pred,
            "perception_label": {"opos": gt, "opos_valid": valid, "opos_targets": [gt]},
        }

    result = summarize_object_predictions(
        [[record(0, 42, 42), record(0, 43, 0), record(100, 0, 12), record(0, 0, 0, False)]]
    )
    assert result["opos_no_hint_visible_count"] == 2
    assert result["opos_no_hint_visible_detection_hits"] == 1
    assert result["opos_no_hint_visible_location_hits"] == 1
    assert result["opos_hint_absent_count"] == 1
    assert result["opos_hint_absence_hits"] == 0
