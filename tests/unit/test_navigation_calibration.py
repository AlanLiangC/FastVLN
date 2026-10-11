import math

import pytest
import torch

from streamnav.contracts.perception import APOS_STOP, POINT_CELLS, perception_config
from streamnav.evaluation.perception_records import perception_records
from streamnav.models.policy.perception import (
    NavigationPerception,
    decode_point_logits,
    point_type_logits,
)
from streamnav.training.perception_loss import perception_losses, prior_corrected_logits
from streamnav.training.stop_loss import stop_calibration_loss, stop_hard_negative_loss


def test_type_decoding_compares_total_point_mass_without_changing_distribution():
    logits = torch.full((1, POINT_CELLS + 1), -10.0)
    logits[:, 0] = math.log(0.40)
    logits[:, 1:11] = math.log(0.06)
    assert decode_point_logits(logits).item() == 0
    assert 1 <= decode_point_logits(logits, hierarchical=True).item() <= POINT_CELLS
    types = point_type_logits(logits).softmax(-1)
    torch.testing.assert_close(types[:, 1], logits.softmax(-1)[:, 1:].sum(-1))
    torch.testing.assert_close(types.sum(-1), torch.ones(1))


def test_hierarchical_decoder_keeps_stop_as_a_type_and_records_both_decodings():
    logits = torch.full((1, POINT_CELLS + 4), -20.0)
    logits[:, APOS_STOP] = 10.0
    assert decode_point_logits(logits, hierarchical=True).item() == APOS_STOP
    predictions = {
        "apos": logits,
        "opos": logits[:, : POINT_CELLS + 1],
        "arrival": torch.zeros(1, 3),
    }
    records = perception_records(predictions, hierarchical=True)
    assert records[0]["apos"] == APOS_STOP
    assert records[0]["apos_flat_argmax"] == APOS_STOP
    assert records[0]["apos_point_probability"] < 1e-8
    assert sum(records[0]["apos_type_probabilities"]) == pytest.approx(1)


@pytest.mark.parametrize("all_masked", [False, True])
def test_hierarchical_loss_preserves_masks_and_has_finite_gradients(all_masked):
    module = NavigationPerception(8, 6)
    predictions, _ = module(torch.randn(2, 3, 8))
    labels = {
        name: {
            "target": torch.ones(2, 3, dtype=torch.long),
            "valid": torch.zeros(2, 3, dtype=torch.bool),
            "confidence": torch.ones(2, 3),
        }
        for name in predictions
    }
    if not all_masked:
        for label in labels.values():
            label["valid"][0, 0] = True
    loss, metrics = perception_losses(predictions, labels, {"spatial_mode": "hierarchical"})
    assert torch.isfinite(loss)
    loss.backward()
    for head in (module.apos, module.opos, module.arrival):
        assert head.weight.grad is not None and torch.isfinite(head.weight.grad).all()
        assert bool(head.weight.grad.abs().sum() > 0) == (not all_masked)
    assert "perception_opos_location_error_cells" in metrics


def test_coordinate_loss_is_not_diluted_by_many_sentinel_examples():
    def gradients(nonpoints):
        logits = torch.zeros(1, nonpoints + 1, POINT_CELLS + 4, requires_grad=True)
        predictions = {
            "apos": logits,
            "opos": torch.zeros(1, nonpoints + 1, POINT_CELLS + 1),
            "arrival": torch.zeros(1, nonpoints + 1, 3),
        }
        labels = {
            name: {
                "target": torch.zeros(1, nonpoints + 1, dtype=torch.long),
                "valid": torch.ones(1, nonpoints + 1, dtype=torch.bool),
                "confidence": torch.ones(1, nonpoints + 1),
            }
            for name in predictions
        }
        labels["apos"]["target"][0, 0] = 600
        config = {"spatial_mode": "hierarchical", "apos_coef": 1, "opos_coef": 0, "arrival_coef": 0}
        loss, _ = perception_losses(predictions, labels, config)
        loss.backward()
        # The centered location-gradient contrast cancels the uniform type term.
        return logits.grad[0, 0, 600] - logits.grad[0, 0, 1000]

    torch.testing.assert_close(gradients(1), gradients(100))


def test_stop_calibration_balances_rare_stop_and_penalizes_false_stop():
    logits = torch.zeros(1, 100, 6, requires_grad=True)
    expert = torch.ones(1, 100, dtype=torch.long)
    expert[0, 0] = 0
    valid = torch.ones_like(expert, dtype=torch.bool)
    loss, _ = stop_calibration_loss(logits, expert, valid)
    loss.backward()
    assert logits.grad[0, 0, 0] < 0
    assert logits.grad[0, 1, 0] > 0
    expected = 0.5 * (-math.log(1 / 6) - math.log(5 / 6))
    assert loss.item() == pytest.approx(expected)
    valid.zero_()
    logits.grad = None
    masked, _ = stop_calibration_loss(logits, expert, valid)
    masked.backward()
    assert torch.isfinite(masked) and logits.grad.abs().sum() == 0


def test_balanced_stop_prior_correction_preserves_natural_probability():
    # With identical observations, a 1% STOP class must have a 1% prediction,
    # even though both classes contribute equally to supervision.
    base = torch.tensor([math.log(0.01)] + [math.log(0.99 / 5)] * 5, requires_grad=True)
    logits = base[None, None].expand(1, 100, 6)
    expert = torch.ones(1, 100, dtype=torch.long)
    expert[0, 0] = 0
    loss, _ = stop_calibration_loss(
        logits, expert, torch.ones_like(expert, dtype=torch.bool), prior_correction=True
    )
    loss.backward()
    assert base.grad.abs().max() < 1e-6


def test_balanced_arrival_prior_correction_preserves_natural_probability():
    base = torch.tensor([math.log(0.90), math.log(0.09), math.log(0.01)], requires_grad=True)
    predictions = {
        "apos": torch.zeros(1, 100, POINT_CELLS + 4),
        "opos": torch.zeros(1, 100, POINT_CELLS + 1),
        "arrival": base[None, None].expand(1, 100, 3),
    }
    labels = {
        name: {
            "target": torch.zeros(1, 100, dtype=torch.long),
            "valid": torch.zeros(1, 100, dtype=torch.bool),
            "confidence": torch.ones(1, 100),
        }
        for name in predictions
    }
    labels["arrival"]["valid"].fill_(True)
    labels["arrival"]["target"][0, 90:99] = 1
    labels["arrival"]["target"][0, 99] = 2
    loss, _ = perception_losses(
        predictions,
        labels,
        {"prior_correction": True, "apos_coef": 0, "opos_coef": 0, "arrival_coef": 1},
    )
    loss.backward()
    assert base.grad.abs().max() < 1e-6


def test_prior_correction_handles_missing_classes_and_empty_ranks(monkeypatch):
    from streamnav.training import distributed

    logits = torch.zeros(1, 4, 2)
    target = torch.zeros(1, 4, dtype=torch.long)
    valid = torch.zeros_like(target, dtype=torch.bool)
    assert torch.equal(prior_corrected_logits(logits, target, valid), logits)
    valid.fill_(True)
    assert torch.equal(prior_corrected_logits(logits, target, valid), logits)
    valid.fill_(False)
    # The collective returns average counts: one global label on eight ranks
    # gives 0.125, which must still normalize to a prior of one.
    monkeypatch.setattr(distributed, "average_scalar", lambda counts: torch.tensor([0.125, 0.0]))
    torch.testing.assert_close(prior_corrected_logits(logits, target, valid), logits)
    monkeypatch.setattr(distributed, "average_scalar", lambda counts: torch.tensor([90.0, 10.0]))
    corrected = prior_corrected_logits(logits, target, valid)
    assert torch.isfinite(corrected).all()
    torch.testing.assert_close(
        corrected[..., 1] - corrected[..., 0], torch.full((1, 4), math.log(9))
    )


def test_auxiliary_gradient_scaling_preserves_policy_and_head_learning():
    original = NavigationPerception(8, 6)
    with torch.no_grad():
        original.action_residual.weight.normal_()
    scaled = NavigationPerception(8, 6, supervision_backbone_scale=0.125)
    scaled.load_state_dict(original.state_dict())
    first = torch.randn(2, 8, requires_grad=True)
    second = first.detach().clone().requires_grad_()
    normal_logits, normal_residual = original(first)
    scaled_logits, scaled_residual = scaled(second)
    torch.testing.assert_close(normal_residual, scaled_residual, atol=0, rtol=0)
    for name in normal_logits:
        torch.testing.assert_close(normal_logits[name], scaled_logits[name], atol=0, rtol=0)
    normal_residual.sum().backward(retain_graph=True)
    scaled_residual.sum().backward(retain_graph=True)
    torch.testing.assert_close(first.grad, second.grad, atol=0, rtol=0)
    original.zero_grad()
    scaled.zero_grad()
    first.grad = None
    second.grad = None
    normal_logits["apos"].square().sum().backward()
    scaled_logits["apos"].square().sum().backward()
    torch.testing.assert_close(second.grad, first.grad * 0.125)
    torch.testing.assert_close(scaled.apos.weight.grad, original.apos.weight.grad, atol=0, rtol=0)
    for value in (-1, 2, float("nan")):
        with pytest.raises(ValueError):
            perception_config({"supervision_backbone_scale": value})


def test_stop_hard_negatives_include_sampled_errors_and_exclude_legal_stop():
    logits = torch.zeros(1, 3, 6, requires_grad=True)
    with torch.no_grad():
        logits[0, :2, 0] = 5
        logits[0, 2, 1] = 5
    expert = torch.tensor([[1, 0, 1]])
    actions = torch.tensor([[1, 0, 0]])
    valid = torch.ones_like(expert, dtype=torch.bool)
    loss, metrics = stop_hard_negative_loss(logits, expert, valid, actions)
    loss.backward()
    assert logits.grad[0, 0, 0] > 0 and logits.grad[0, 2, 0] > 0
    assert logits.grad[0, 1].abs().sum() == 0
    assert metrics["stop_hard_negative_fraction"] == pytest.approx(2 / 3)
    valid.zero_()
    logits.grad = None
    loss, _ = stop_hard_negative_loss(logits, expert, valid, actions)
    loss.backward()
    assert torch.isfinite(loss) and logits.grad.abs().sum() == 0


def test_hard_stop_supervision_rejects_conflicting_teacher_before_touching_run():
    from streamnav.training.trainer import EndToEndObjectNavTrainer

    with pytest.raises(ValueError, match="success-aligned teacher"):
        EndToEndObjectNavTrainer({"trainer": {"stop_hard_negative_coef": 0.02}, "habitat": {}})


def test_fusion_reparameterization_preserves_logits_and_transforms_adam_moments():
    original = NavigationPerception(8, 6)
    with torch.no_grad():
        original.action_residual.weight.normal_(std=0.1)
        original.action_residual.bias.normal_(std=0.1)
    scaled = NavigationPerception(8, 6, fusion_scale=8)
    scaled.load_scaled_weights(original.state_dict(), source_scale=1)
    hidden = torch.randn(3, 8)
    old_predictions, old_residual = original(hidden)
    predictions, residual = scaled(hidden)
    torch.testing.assert_close(residual, old_residual, atol=0, rtol=0)
    for name in predictions:
        torch.testing.assert_close(predictions[name], old_predictions[name], atol=0, rtol=0)
    old_residual.sum().backward()
    residual.sum().backward()
    torch.testing.assert_close(
        scaled.action_residual.weight.grad, original.action_residual.weight.grad * 8
    )
    torch.testing.assert_close(scaled.apos.weight.grad, original.apos.weight.grad)
    optimizer = torch.optim.Adam(scaled.parameters())
    for parameter in scaled.action_residual.parameters():
        optimizer.state[parameter] = {
            "exp_avg": torch.ones_like(parameter),
            "exp_avg_sq": torch.ones_like(parameter),
        }
    scaled.rescale_optimizer_moments(optimizer)
    for parameter in scaled.action_residual.parameters():
        assert optimizer.state[parameter]["exp_avg"].eq(8).all()
        assert optimizer.state[parameter]["exp_avg_sq"].eq(64).all()
    for value in [0, -1, float("nan")]:
        with pytest.raises(ValueError):
            perception_config({"fusion_scale": value})
