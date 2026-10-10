import importlib.util
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from streamnav.contracts.action import NavigationAction as A
from streamnav.contracts.perception import APOS_STOP, POINT_CELLS, decode_pixel, encode_pixel
from streamnav.models.policy.perception import NavigationPerception
from streamnav.training.checkpoint import restore_optimizer_branches
from streamnav.training.perception_loss import perception_losses, spatial_cross_entropy
from streamnav.training.rollout import replay_sequences


def test_new_action_branch_preserves_actor_and_uses_predicted_points():
    module = NavigationPerception(8, 6)
    hidden = torch.randn(2, 8, requires_grad=True)
    predictions, residual = module(hidden)
    assert torch.equal(residual, torch.zeros_like(residual))
    with torch.no_grad():
        module.action_residual.weight.normal_()
    _, residual = module(hidden)
    residual.square().sum().backward()
    assert module.apos.weight.grad.abs().sum() > 0
    assert module.opos.weight.grad.abs().sum() > 0
    assert module.arrival.weight.grad.abs().sum() > 0
    assert hidden.grad.abs().sum() > 0
    assert predictions["apos"].shape == (2, POINT_CELLS + 4)


@pytest.mark.parametrize("all_masked", [True, False])
def test_uncertain_labels_are_masked_without_losing_connected_gradients(all_masked):
    module = NavigationPerception(8, 6)
    hidden = torch.randn(2, 2, 8, requires_grad=True)
    predictions, _ = module(hidden)
    labels = {
        name: {
            "target": torch.zeros(2, 2, dtype=torch.long),
            "valid": torch.zeros(2, 2, dtype=torch.bool),
            "confidence": torch.ones(2, 2),
        }
        for name in predictions
    }
    if not all_masked:
        for name in predictions:
            labels[name]["valid"][0, 0] = True
            labels[name]["target"][0, 0] = 1
    loss, _ = perception_losses(predictions, labels)
    loss.backward()
    assert torch.isfinite(loss)
    assert hidden.grad[1].abs().sum() == 0
    assert hidden.grad[0, 1].abs().sum() == 0
    assert (hidden.grad[0, 0].abs().sum() > 0).item() == (not all_masked)
    for head in (module.apos, module.opos, module.arrival):
        assert head.weight.grad is not None


def test_spatial_loss_never_smooths_stop_or_out_of_view_into_pixels():
    logits = torch.randn(1, 2, POINT_CELLS + 4, requires_grad=True)
    target = torch.tensor([[0, APOS_STOP]])
    actual = spatial_cross_entropy(logits, target)
    expected = -logits.log_softmax(-1).gather(-1, target[..., None]).squeeze(-1)
    torch.testing.assert_close(actual, expected)
    actual.sum().backward()
    assert logits.grad[0, 1, APOS_STOP] < 0


def test_confidence_scales_weak_supervision_without_cancelling_in_normalizer():
    predictions = {
        "apos": torch.zeros(1, 1, POINT_CELLS + 4),
        "opos": torch.zeros(1, 1, POINT_CELLS + 1),
        "arrival": torch.zeros(1, 1, 3),
    }
    targets = {
        name: {
            "target": torch.ones(1, 1, dtype=torch.long),
            "valid": torch.ones(1, 1, dtype=torch.bool),
            "confidence": torch.ones(1, 1),
        }
        for name in predictions
    }
    full, _ = perception_losses(predictions, targets)
    for labels in targets.values():
        labels["confidence"].fill_(0.25)
    weak, _ = perception_losses(predictions, targets)
    assert weak.item() == pytest.approx(full.item() / 4)


def test_rare_arrival_classes_receive_balanced_supervision():
    from streamnav.training.perception_loss import balanced_mean

    target = torch.tensor([[0, 0, 0, 0, 1, 2]])
    losses = torch.ones(1, 6, requires_grad=True)
    result = balanced_mean(
        losses,
        target,
        torch.ones_like(target, dtype=torch.bool),
        torch.ones_like(losses),
        [lambda t, k=k: t == k for k in range(3)],
    )
    result.backward()
    assert losses.grad[0, 5] == pytest.approx(losses.grad[0, 0].item() * 4)


def test_pointing_replay_preserves_resets_predictions_and_full_sequence_gradients():
    from test_packed_replay import make_rollout

    policy, buffer, sequences = make_rollout()
    policy.perception = NavigationPerception(4, 6).double()
    with torch.no_grad():
        policy.perception.action_residual.weight.normal_(std=0.1)
    serial = replay_sequences(policy, buffer, sequences)
    serial_loss = serial[0].square().sum() + sum(p.square().sum() for p in serial[2].values())
    serial_loss.backward()
    gradients = {
        name: p.grad.clone() for name, p in policy.named_parameters() if p.grad is not None
    }
    policy.zero_grad(set_to_none=True)
    buffer.visual_embeddings.grad = None
    packed = replay_sequences(policy, buffer, sequences, pack_frames=7)
    for index in (0, 1):
        torch.testing.assert_close(serial[index], packed[index], atol=0, rtol=0)
    for name in serial[2]:
        torch.testing.assert_close(serial[2][name], packed[2][name], atol=0, rtol=0)
    (packed[0].square().sum() + sum(p.square().sum() for p in packed[2].values())).backward()
    for name, parameter in policy.named_parameters():
        if name in gradients:
            torch.testing.assert_close(parameter.grad, gradients[name], atol=1e-9, rtol=1e-9)


def test_optimizer_initialization_keeps_old_moments_and_new_branch_empty():
    actor = torch.nn.Parameter(torch.tensor([1.0]))
    previous = torch.optim.Adam([{"params": [actor], "role": "actor"}], lr=0.01)
    actor.square().sum().backward()
    previous.step()
    saved = previous.state_dict()
    copied, new = torch.nn.Parameter(actor.detach().clone()), torch.nn.Parameter(torch.ones(2))
    optimizer = torch.optim.Adam(
        [{"params": [new], "role": "perception"}, {"params": [copied], "role": "actor"}], lr=0.002
    )
    restore_optimizer_branches(optimizer, saved)
    torch.testing.assert_close(optimizer.state[copied]["exp_avg"], previous.state[actor]["exp_avg"])
    assert new not in optimizer.state
    assert optimizer.param_groups[1]["lr"] == 0.002
    bad = torch.optim.Adam([{"params": [new], "role": "actor"}])
    with pytest.raises(ValueError, match="shape"):
        restore_optimizer_branches(bad, saved)


def test_point_encoding_rejects_offscreen_coordinates():
    with pytest.raises(ValueError, match="outside"):
        encode_pixel(-1, 10, 480, 270)
    assert decode_pixel(APOS_STOP, 480, 270) is None
    for x, y in ((0, 0), (240, 135), (479, 269)):
        decoded = decode_pixel(encode_pixel(x, y, 480, 270), 480, 270)
        assert abs(decoded[0] - x) <= 5 and abs(decoded[1] - y) <= 5


@pytest.mark.parametrize("stale", [False, True])
def test_collector_pairs_labels_with_current_rgb_and_rejects_stale_frames(stale):
    from test_rollout_boundary_prefill import TinyPolicy

    from streamnav.training.rollout import RolloutCollector

    class Envs:
        config = {}

        def __init__(self):
            self.frame = 0

        def observation(self):
            return {
                "rgb": torch.full((2, 2, 3), self.frame, dtype=torch.uint8),
                "frame_id": self.frame,
            }

        def reset(self, episodes):
            return [self.observation()]

        def get_oracle_supervisions(self):
            labels = {"episode_id": "A", "frame_id": self.frame + int(stale)}
            for name in ("apos", "opos", "arrival"):
                labels.update(
                    {
                        name: 100 + self.frame if name != "arrival" else 0,
                        name + "_valid": True,
                        name + "_confidence": 1.0,
                    }
                )
            return [{"action": 1, "perception": labels}]

        def step(self, actions):
            self.frame += 1
            return [
                {
                    **self.observation(),
                    "reward": 0,
                    "done": False,
                    "collision": False,
                    "truncated": False,
                    "metrics": {},
                }
            ]

    policy = TinyPolicy()
    policy.perception = torch.nn.Identity()
    source = SimpleNamespace(sample_episode=lambda: SimpleNamespace(uid="A", goal_text="chair"))
    collector = RolloutCollector(
        policy, Envs(), [source], {"rollout_steps": 2, "sequence_length": 2}
    )
    if stale:
        with pytest.raises(ValueError, match="pre-action RGB"):
            collector.collect(1.0)
    else:
        buffer = collector.collect(1.0)
        assert buffer.perception_targets["opos"]["target"].flatten().tolist() == [100, 101]
        assert buffer.observations[:, 0, 0, 0, 0].tolist() == [0, 1]


@pytest.fixture
def label_module(monkeypatch):
    import sys

    fake = SimpleNamespace()
    monkeypatch.setitem(sys.modules, "habitat_sim", fake)
    monkeypatch.setitem(sys.modules, "habitat_sim.utils", SimpleNamespace())
    monkeypatch.setitem(
        sys.modules,
        "habitat_sim.utils.common",
        SimpleNamespace(quat_rotate_vector=lambda rotation, vector: vector),
    )
    spec = importlib.util.spec_from_file_location(
        "test_perception_labels", Path("services/habitat_server/perception_labels.py")
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize(
    "position,depth,expected_valid,expected_point",
    [
        ([0, 0, -2], 2.0, True, True),
        ([0, 0, -2], 0.5, False, False),  # Occlusion must not become a negative label.
        ([0, 0, 2], 2.0, True, False),
        ([0, 0, -2], float("nan"), False, False),
    ],
)
def test_depth_visibility_masks_uncertainty_and_keeps_preaction_pose(
    label_module, position, depth, expected_valid, expected_point
):
    camera = SimpleNamespace(position=np.zeros(3), rotation=SimpleNamespace(inverse=lambda: None))
    state = SimpleNamespace(sensor_states={"rgb": camera})
    env = SimpleNamespace(
        sim=SimpleNamespace(get_agent=lambda _: SimpleNamespace(get_state=lambda: state)),
        _sensor_observations=lambda: {"perception_depth": np.full((27, 48), depth)},
        episode={"goals": [{"position": position}]},
        uid="episode",
        steps=7,
        previous_distance=0.1,
        config={"success_distance": 0.25},
        explorer=None,
    )
    result = label_module.PerceptionLabeler(env).labels(A.STOP)
    assert result["frame_id"] == 7 and result["episode_id"] == "episode"
    assert result["opos_valid"] == expected_valid
    assert (result["opos"] > 0) == expected_point
    assert result["arrival"] == 2 and result["apos"] == APOS_STOP
