from types import SimpleNamespace

import pytest
import torch

from streamnav.contracts.action import NavigationAction as A
from streamnav.training.ealm import EntropyAdaptiveLossMixer
from streamnav.training.rollout import RolloutCollector
from streamnav.training.rollout_buffer import RecurrentRolloutBuffer, SequenceIndex
from streamnav.training.trainer import EndToEndObjectNavTrainer


@pytest.mark.parametrize("valid", [True, False])
def test_invalid_label_has_no_il_gradient_but_keeps_policy_gradient(valid):
    trainer = object.__new__(EndToEndObjectNavTrainer)
    trainer.device = torch.device("cpu")
    trainer.cfg = {"ppo": {"clip_eps": 0.2}, "loss": {"value_coef": 0, "entropy_coef": 0}}
    trainer.policy = SimpleNamespace(
        distribution=SimpleNamespace(build=lambda x: torch.distributions.Categorical(logits=x))
    )
    buffer = RecurrentRolloutBuffer(1, 2, (1, 1, 3), 1, 0)
    buffer.il_mask[0, 0] = valid
    buffer.executed_actions.fill_(int(A.MOVE_FORWARD))
    buffer.expert_actions.fill_(int(A.MOVE_FORWARD))
    buffer.old_log_probs.fill_(-torch.log(torch.tensor(6.0)))
    buffer.advantages.fill_(1)
    buffer.old_values.zero_()
    buffer.returns.zero_()
    sequences = [SequenceIndex(i, 0, 1) for i in range(2)]
    logits = torch.zeros(1, 2, 6, requires_grad=True)
    trainer.mixer = EntropyAdaptiveLossMixer(enabled=False, fixed_alpha=1)
    loss, metrics = trainer.compute_losses(logits, torch.zeros(1, 2), buffer, sequences)
    loss.backward()
    assert metrics["il_eligible_fraction"] == (1 if valid else 0.5)
    assert metrics["ppo_eligible_fraction"] == 1
    assert metrics["il_loss"] == pytest.approx(torch.log(torch.tensor(6.0)).item())
    assert bool(logits.grad[0, 0].abs().sum() > 0) == valid
    assert logits.grad[0, 1, int(A.MOVE_FORWARD)] == pytest.approx(-5 / (12 if valid else 6))
    logits = torch.zeros(1, 2, 6, requires_grad=True)
    trainer.mixer = EntropyAdaptiveLossMixer(enabled=False, fixed_alpha=0)
    loss, _ = trainer.compute_losses(logits, torch.zeros(1, 2), buffer, sequences)
    loss.backward()
    assert logits.grad[0, 0].abs().sum() > 0


@pytest.mark.parametrize(
    "executed,collision,moved,expected",
    [
        (A.MOVE_FORWARD, True, 0.0, False),
        (A.MOVE_FORWARD, True, 0.2, True),
        (A.TURN_LEFT, True, 0.0, True),
        (A.MOVE_FORWARD, False, 0.0, True),
    ],
)
def test_collector_masks_only_observed_blocked_teacher_forward(
    monkeypatch, executed, collision, moved, expected
):
    from test_rollout_boundary_prefill import BoundaryEnvs, TinyPolicy

    envs = BoundaryEnvs()
    envs.config = {"forward_step": 0.25}
    step = envs.step

    def result(actions):
        values = step(actions)
        for value in values:
            value.update(collision=collision, displacement=moved)
        return values

    envs.step = result
    monkeypatch.setattr(
        torch.distributions.Categorical,
        "sample",
        lambda self: torch.full(self.batch_shape, int(executed)),
    )
    source = SimpleNamespace(sample_episode=lambda: SimpleNamespace(uid="train", goal_text="chair"))
    collector = RolloutCollector(
        TinyPolicy(),
        envs,
        [source, source],
        {
            "rollout_steps": 1,
            "sequence_length": 1,
            "filter_blocked_forward_labels": True,
        },
    )
    buffer = collector.collect(0)
    assert buffer.il_mask.tolist() == [[expected, expected]]
    assert buffer.ppo_mask.all()
    assert buffer.executed_actions.eq(int(executed)).all()
    assert buffer.expert_actions.eq(int(A.MOVE_FORWARD)).all()
