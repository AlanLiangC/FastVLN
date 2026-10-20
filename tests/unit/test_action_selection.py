from types import SimpleNamespace

import pytest
import torch

from streamnav.contracts.cache import clone_state, state_bytes
from streamnav.contracts.state import LayerState, StreamingState
from streamnav.models.point_cot.action_selection import (
    calibrated_action_logits,
    temporal_greedy_logits,
)
from streamnav.models.point_cot.policy import Qwen35PointCoTPolicy


def test_hint_gated_greedy_fusion_preserves_null_and_stochastic_rows():
    # Include a stale forecast in every row: a NULL input must still bypass it.
    policy = SimpleNamespace(
        temporal_action_weight=0.3,
        temporal_action_requires_hint=True,
        device=torch.device("cpu"),
    )
    logits = torch.tensor([[0.01, 0.35, 0.15, 0.48, 0.005, 0.005]]).log().repeat(3, 1)
    prior = torch.tensor([0.01, 0.9, 0.04, 0.04, 0.005, 0.005])
    states = [SimpleNamespace(next_action_probabilities=prior) for _ in range(3)]
    selected = Qwen35PointCoTPolicy.greedy_action_logits(
        policy, logits, states, [True, True, False], [42, 0, 42]
    )
    assert selected[0].argmax().item() == 1
    assert torch.equal(selected[1:], logits[1:])
    assert (
        Qwen35PointCoTPolicy.greedy_action_logits(policy, logits, states, [True] * 3, [0] * 3)
        is logits
    )


def test_temporal_forecast_can_support_forward_without_locking_actions():
    current = torch.tensor([[0.01, 0.35, 0.15, 0.48, 0.005, 0.005]]).log()
    previous = torch.tensor([[0.01, 0.9, 0.04, 0.04, 0.005, 0.005]])
    selected = temporal_greedy_logits(current, previous, 0.3)
    assert current.argmax(-1).item() == 3 and selected.argmax(-1).item() == 1
    torch.testing.assert_close(selected.softmax(-1)[:, 0], current.softmax(-1)[:, 0])


def test_fusion_preserves_stop_choice_and_stop_probability_on_random_inputs():
    torch.manual_seed(17)
    logits = torch.randn(1000, 6)
    prior = torch.randn(1000, 6).softmax(-1)
    fused = temporal_greedy_logits(logits, prior, 0.4)
    assert torch.equal(logits.argmax(-1) == 0, fused.argmax(-1) == 0)
    torch.testing.assert_close(fused.softmax(-1)[:, 0], logits.softmax(-1)[:, 0])


def test_empty_history_and_disabled_fusion_preserve_exact_logits():
    logits = torch.randn(5, 6)
    assert temporal_greedy_logits(logits, None, 0.3) is logits
    assert temporal_greedy_logits(logits, torch.ones_like(logits), 0) is logits
    torch.testing.assert_close(
        temporal_greedy_logits(logits, torch.zeros_like(logits), 0.3), logits, rtol=0, atol=0
    )


def test_arrival_calibration_is_current_frame_probability_mixture_and_differentiable():
    logits = torch.tensor([[0.0, 1.0, 0.0, 0.0, 0.0, 0.0]], requires_grad=True)
    arrival = torch.tensor([[0.0, 0.0, 2.0]], requires_grad=True)
    selected = calibrated_action_logits(logits, arrival, 0.25)
    expected = 0.75 * logits.softmax(-1)[:, 0] + 0.25 * arrival.softmax(-1)[:, 2]
    torch.testing.assert_close(selected.softmax(-1)[:, 0], expected)
    torch.testing.assert_close(selected.softmax(-1).sum(-1), torch.ones(1))
    (-selected[:, 0]).sum().backward()
    assert logits.grad.abs().sum() > 0 and arrival.grad.abs().sum() > 0
    assert calibrated_action_logits(logits, None, 0) is logits
    with pytest.raises(ValueError, match="arrival head"):
        calibrated_action_logits(logits, None, 0.1)


def test_collision_and_clone_keep_plan_state_isolated_without_erasing_kda():
    state = StreamingState(
        (LayerState(None, torch.ones(1, 2)),), "e", "h", 1, "chair", torch.ones(6)
    )
    cloned = clone_state(state)
    cloned.next_action_probabilities[0] = 5
    assert state.next_action_probabilities[0] == 1
    assert state_bytes(state) == 8 + 24
    cleared = state.after_action(True)
    assert cleared.next_action_probabilities is None
    assert cleared.kda_cache is state.kda_cache and cleared.step_index == 1
