from types import SimpleNamespace

import pytest
import torch

from streamnav.contracts.action import NavigationAction as A
from streamnav.training.auxiliary_il import AuxiliaryILController
from streamnav.training.distributed import normalize_advantages
from streamnav.training.ealm import EntropyAdaptiveLossMixer
from streamnav.training.rollout import RolloutCollector
from streamnav.training.rollout_buffer import RecurrentRolloutBuffer, SequenceIndex
from streamnav.training.trainer import EndToEndObjectNavTrainer


def test_expert_episode_then_bounded_greedy_loop_recovery():
    controller = AuxiliaryILController({"num_envs": 1, "recovery_steps": 3}, 4)
    controller.reset(0)
    assert controller.select(0, A.TURN_LEFT, A.STOP) == (A.STOP, True)
    controller.reset(0)
    for index in range(8):
        action = A.TURN_LEFT if index % 2 == 0 else A.TURN_RIGHT
        selected, expert = controller.select(0, action, A.MOVE_FORWARD)
        assert (selected, expert) == (action, False)
        assert controller.observe(0, selected, expert) == (index == 7)
    for _ in range(3):
        action, followed = controller.select(0, A.TURN_LEFT, A.MOVE_FORWARD)
        assert (action, followed) == (A.MOVE_FORWARD, True)
        assert not controller.observe(0, action, followed)
    assert controller.select(0, A.TURN_RIGHT, A.STOP) == (A.TURN_RIGHT, False)
    controller.reset(0)
    assert controller.select(0, A.TURN_RIGHT, A.STOP) == (A.STOP, True)


def test_same_direction_spin_recovery_and_resume_episode_schedule():
    controller = AuxiliaryILController({"num_envs": 1}, 4)
    controller.reset(0)
    controller.reset(0)
    for index in range(16):
        assert controller.observe(0, A.TURN_LEFT, False) == (index == 15)
    resumed = AuxiliaryILController({"num_envs": 1}, 4)
    resumed.load_state_dict(controller.state_dict())
    resumed.reset(0)
    assert resumed.recovery_remaining == [0]
    assert resumed.select(0, A.TURN_RIGHT, A.STOP) == (A.STOP, True)
    with pytest.raises(ValueError, match="Only auxiliary"):
        resumed.select(1, A.TURN_LEFT, A.STOP)


@pytest.mark.parametrize("config", [{"num_envs": 4}, {"num_envs": -1}, {"recovery_steps": 0}])
def test_auxiliary_configuration_keeps_sampled_policy_slots(config):
    with pytest.raises(ValueError):
        AuxiliaryILController(config, 4)


def test_ppo_value_and_entropy_have_no_auxiliary_gradient():
    trainer = object.__new__(EndToEndObjectNavTrainer)
    trainer.device = torch.device("cpu")
    trainer.cfg = {
        "auxiliary_il": {"num_envs": 1},
        "ppo": {"clip_eps": 0.2, "use_clipped_value_loss": False},
        "loss": {"value_coef": 0.5, "entropy_coef": 0.01},
        "log_policy_gradient_terms": True,
    }
    trainer.policy = SimpleNamespace(
        distribution=SimpleNamespace(build=lambda x: torch.distributions.Categorical(logits=x))
    )
    trainer.mixer = EntropyAdaptiveLossMixer(enabled=False, fixed_alpha=0)
    buffer = RecurrentRolloutBuffer(2, 2, (1, 1, 3), 2, 0)
    buffer.ppo_mask[:, 0] = False
    buffer.executed_actions.zero_()
    buffer.expert_actions.fill_(2)
    buffer.old_log_probs.fill_(-torch.log(torch.tensor(6.0)))
    buffer.old_log_probs[:, 0] = 0  # deterministic behavior, excluded from PPO
    buffer.replay_log_probs = torch.full_like(buffer.old_log_probs, -torch.log(torch.tensor(6.0)))
    buffer.advantages.fill_(1)
    buffer.advantages[:, 0] = 1e30
    buffer.old_values.zero_()
    buffer.returns.fill_(2)
    buffer.returns[:, 0] = 1e30
    logits = torch.zeros(2, 2, 6, requires_grad=True)
    values = torch.ones(2, 2, requires_grad=True)
    sequences = [SequenceIndex(i, 0, 2) for i in range(2)]
    loss, metrics = trainer.compute_losses(logits, values, buffer, sequences)
    loss.backward()
    assert torch.isfinite(loss)
    assert torch.equal(logits.grad[:, 0], torch.zeros(2, 6))
    assert torch.equal(values.grad[:, 0], torch.zeros(2))
    assert logits.grad[:, 1].abs().sum() > 0
    assert values.grad[:, 1].abs().sum() > 0
    assert metrics["ppo_eligible_fraction"] == 0.5
    assert metrics["replay_log_prob_error_max"] == 0
    assert metrics["policy_logit_grad_norm_ppo"] > 0
    # Gradient diagnostics must not accumulate parameter/activation gradients.
    assert metrics["policy_logit_grad_norm_il"] == 0


def test_advantage_normalization_excludes_auxiliary_rewards():
    values = torch.tensor([[1e30, 1.0], [-1e30, 3.0]])
    mask = torch.tensor([[False, True], [False, True]])
    normalized = normalize_advantages(values, "cpu", mask)
    torch.testing.assert_close(normalized, torch.tensor([[0.0, -1.0], [0.0, 1.0]]))


def test_collector_records_actual_auxiliary_behavior_and_separate_success(monkeypatch):
    from test_rollout_boundary_prefill import BoundaryEnvs, TinyPolicy

    policy = TinyPolicy()
    monkeypatch.setattr(
        torch.distributions.Categorical,
        "sample",
        lambda self, sample_shape=torch.Size(): torch.zeros(self.batch_shape, dtype=torch.long),
    )
    source = SimpleNamespace(sample_episode=lambda: SimpleNamespace(uid="train", goal_text="chair"))
    collector = RolloutCollector(
        policy,
        BoundaryEnvs(),
        [source, source],
        {"rollout_steps": 1, "sequence_length": 1, "auxiliary_il": {"num_envs": 1}},
    )
    buffer = collector.collect(0)
    assert buffer.executed_actions[0, 0] == A.MOVE_FORWARD  # expert, not argmax STOP
    assert buffer.actions[0, 0] == A.STOP  # original sampled action retained
    assert buffer.ppo_mask.tolist() == [[False, True]]
    assert buffer.old_log_probs[0, 0] == 0
    assert buffer.replay_log_probs[0, 0] < 0
    assert len(buffer.episode_metrics) == 0
    assert len(buffer.auxiliary_episode_metrics) == 1
    assert buffer.auxiliary_expert_steps == 1
