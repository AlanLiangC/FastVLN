from types import SimpleNamespace

import pytest
import torch
from torch import nn

from streamnav.errors import ReplayConsistencyError
from streamnav.training.ealm import EntropyAdaptiveLossMixer
from streamnav.training.ppo import replay_consistency_metrics
from streamnav.training.rollout_buffer import RecurrentRolloutBuffer
from streamnav.training.trainer import EndToEndObjectNavTrainer


class ProbeReplay(nn.Module):
    def __init__(self, policy, mismatch):
        super().__init__()
        self.policy = policy
        self.pack_frames = 64
        self.mismatch = mismatch
        self.calls = []

    def forward(self, buffer, sequences):
        self.calls.append((self.pack_frames, self.policy.batch_chat_body))
        count = len(sequences)
        hidden = self.policy.backbone.layers[0].mixer.beta_proj(torch.ones(2, count, 1))
        logits = self.policy.actor_critic.actor(hidden)
        if self.mismatch(self.pack_frames, self.policy.batch_chat_body):
            logits = logits + logits.new_tensor([0.7, 0.0])
        return logits, self.policy.actor_critic.critic(hidden).squeeze(-1)


class GuardedDDP(nn.Module):
    def __init__(self, module):
        super().__init__()
        self.module = module
        self.busy = False
        self.forward_calls = 0

    def forward(self, *args):
        assert not self.busy, "DDP forward retried without completing backward"
        self.busy = True
        self.forward_calls += 1
        logits, values = self.module(*args)

        def completed(gradient):
            self.busy = False
            return gradient

        logits.register_hook(completed)
        return logits, values


class CountingSGD(torch.optim.SGD):
    def __init__(self, parameters):
        super().__init__(parameters, lr=0.01)
        self.steps = 0

    def step(self, closure=None):
        self.steps += 1
        return super().step(closure)


def trainer_fixture(tmp_path, mismatch):
    policy = nn.Module()
    policy.batch_chat_body = True
    policy.backbone = nn.Module()
    policy.backbone.vision = nn.Linear(1, 1).requires_grad_(False)
    layer = nn.Module()
    layer.mixer = nn.Module()
    layer.mixer.beta_proj = nn.Linear(1, 1)
    policy.backbone.layers = nn.ModuleList([layer])
    policy.actor_critic = nn.Module()
    policy.actor_critic.actor = nn.Linear(1, 2)
    policy.actor_critic.critic = nn.Linear(1, 1)
    with torch.no_grad():
        for parameter in policy.parameters():
            parameter.fill_(0.1)
    trainer = EndToEndObjectNavTrainer.__new__(EndToEndObjectNavTrainer)
    trainer.policy = policy
    trainer.replay = GuardedDDP(ProbeReplay(policy, mismatch))
    trainer.ddp = True
    trainer.device = torch.device("cpu")
    trainer.dtype = torch.float32
    trainer.run_dir = tmp_path
    trainer.update_index = 7
    trainer.cfg = {
        "replay_preflight": True,
        "ppo": {"gamma": 0.99, "gae_lambda": 0.95},
        "update_epochs": 1,
        "sequence_batch_size": 1,
        "max_grad_norm": 0.2,
        "num_updates": 10,
    }
    trainer.optimizer = CountingSGD(policy.parameters())
    trainer.mixer = EntropyAdaptiveLossMixer()
    trainer.mixer.entropy_ema.fill_(0.4)
    buffer = RecurrentRolloutBuffer(2, 1, (1, 1, 3), 2, 0)
    buffer.old_log_probs.fill_(-torch.log(torch.tensor(2.0)))

    def losses(logits, values, buffer, sequences):
        labels = torch.zeros(logits.shape[:2], dtype=torch.long)
        loss = (
            torch.nn.functional.cross_entropy(logits.flatten(0, 1), labels.flatten())
            + values.square().mean()
        )
        metrics = replay_consistency_metrics(logits.log_softmax(-1)[..., 0], buffer.old_log_probs)
        metrics.update(total_loss=loss.item(), entropy=0.7, clip_fraction=0.0)
        return loss, metrics

    trainer.compute_losses = losses
    return trainer, buffer


def test_retry_never_prepares_ddp_or_updates_weights_until_valid(tmp_path):
    trainer, buffer = trainer_fixture(tmp_path, lambda pack, batch: pack == 64)
    before = [p.detach().clone() for p in trainer.policy.parameters()]
    module = trainer.replay.module
    observed = []

    def check_probe(_, inputs):
        if trainer.replay.forward_calls == 0:
            observed.append(trainer.optimizer.steps)
            assert all(
                torch.equal(a, b) for a, b in zip(before, trainer.policy.parameters(), strict=True)
            )
            assert trainer.mixer.entropy_ema.item() == 0.4

    handle = module.register_forward_pre_hook(check_probe)
    metrics = trainer.update(buffer)
    handle.remove()
    assert observed == [0, 0]
    assert trainer.replay.forward_calls == trainer.optimizer.steps == 1
    assert not trainer.replay.busy
    assert metrics["replay_pack_frames_used"] == 32
    assert metrics["replay_preflight_attempts"] == 2
    assert metrics["preupdate_replay_initial_error_max"] > 0.05
    assert metrics["preupdate_replay_log_prob_error_max"] == 0
    assert module.pack_frames == 64 and trainer.policy.batch_chat_body
    assert (tmp_path / "replay_fallbacks.jsonl").is_file()


def test_remote_rank_rejection_forces_local_retry(tmp_path, monkeypatch):
    trainer, buffer = trainer_fixture(tmp_path, lambda pack, batch: False)
    from streamnav.training import distributed as parallel

    calls = []

    def any_rank(flag, device):
        calls.append(flag)
        return True if len(calls) == 1 else flag

    monkeypatch.setattr(parallel, "any_rank", any_rank)
    metrics = trainer.update(buffer)
    assert metrics["preupdate_replay_initial_error_max"] == 0
    assert metrics["replay_pack_frames_used"] == 32
    assert trainer.replay.forward_calls == trainer.optimizer.steps == 1


def test_serial_fallback_remains_local_to_the_current_update(tmp_path):
    trainer, buffer = trainer_fixture(tmp_path, lambda pack, batch: batch)
    metrics = trainer.update(buffer)
    assert metrics["replay_pack_frames_used"] == 1
    assert metrics["replay_used_serial_chat_body"]
    assert trainer.optimizer.steps == 1
    assert trainer.policy.batch_chat_body and trainer.replay.module.pack_frames == 64


def test_all_replays_rejected_without_optimizer_or_ddp_side_effects(tmp_path):
    trainer, buffer = trainer_fixture(tmp_path, lambda pack, batch: True)
    before = [p.detach().clone() for p in trainer.policy.parameters()]
    with pytest.raises(ReplayConsistencyError, match="before any optimizer step"):
        trainer.update(buffer)
    assert trainer.optimizer.steps == trainer.replay.forward_calls == 0
    assert trainer.mixer.entropy_ema.item() == 0.4
    assert all(torch.equal(a, b) for a, b in zip(before, trainer.policy.parameters(), strict=True))
    assert trainer.policy.batch_chat_body and trainer.replay.module.pack_frames == 64


def test_run_checkpoints_last_completed_update_on_collective_replay_rejection(tmp_path):
    trainer, buffer = trainer_fixture(tmp_path, lambda pack, batch: True)
    trainer.stop_requested = False
    trainer.collect_rollout = lambda: buffer
    saved = []
    closed = []
    trainer.save_checkpoint = lambda: saved.append(trainer.update_index)
    trainer.envs = SimpleNamespace(close=lambda: closed.append(True))
    with pytest.raises(ReplayConsistencyError):
        trainer.run()
    assert saved == [7]
    assert closed == [True]
    assert trainer.optimizer.steps == 0
