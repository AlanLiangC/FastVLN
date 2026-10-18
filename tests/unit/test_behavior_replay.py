import json
from contextlib import nullcontext
from types import SimpleNamespace

import pytest
import torch

from streamnav.contracts.state import LayerState, StreamingState
from streamnav.models.point_cot.policy import TokenStep
from streamnav.training.pointcot_trainer import (
    ReplayMismatch,
    archive_uncommitted_metrics,
    optimize_rollout,
    replay_batched_records,
    replay_probability_error,
)


class RecurrentPolicy:
    device = torch.device("cpu")

    def __init__(self, trainable_feedback=False):
        self.weight = torch.tensor(0.3, requires_grad=True)
        self.batches = []
        self.trainable_feedback = trainable_feedback

    def start_episode(self, uid, instruction):
        return StreamingState((LayerState(None, None),), uid, "", 0, instruction)

    def step_batch(self, features, states, *, tokens, deterministic):
        assert deterministic
        self.batches.append([s.episode_id for s in states])
        steps = []
        for feature, state, token in zip(features, states, tokens, strict=True):
            cache = state.kda_cache[0].recurrent
            feedback = self.weight if self.trainable_feedback else 1.0
            x = self.weight * feature + (0 if cache is None else cache * feedback)
            logits = torch.stack([x, -x])
            state = state.with_cache((LayerState(None, x),))
            steps.append(TokenStep(*token, logits, logits + 0.2, logits - 0.2, x, state))
        return steps


def test_original_batch_replay_preserves_resets_rng_and_temporal_gradients():
    policy = RecurrentPolicy()
    states = [policy.start_episode(str(i), "goal") for i in range(3)]
    features = torch.arange(1.0, 13.0).view(4, 3).requires_grad_()
    records = [
        [{"features": feature, "tokens": (torch.tensor(0),) * 3} for feature in row]
        for row in features
    ]
    random_before = torch.get_rng_state()
    result = replay_batched_records(policy, records, states, resets={(2, 1): ("reset", "goal")})
    assert torch.equal(random_before, torch.get_rng_state())
    assert policy.batches == [["0", "1", "2"]] * 2 + [["0", "reset", "2"]] * 2
    assert result[1].state.step_index == 2
    assert result[0].state.step_index == 4
    torch.testing.assert_close(result[1].value, torch.tensor([0.6, 2.1, 2.4, 5.7]))
    gradient = torch.autograd.grad(result[0].value[-1], features, retain_graph=True)[0]
    torch.testing.assert_close(gradient[:, 0], torch.full((4,), 0.3))
    assert torch.count_nonzero(gradient[:, 1:]) == 0
    gradient = torch.autograd.grad(result[1].value[-1], features)[0]
    assert torch.count_nonzero(gradient[:2]) == 0
    torch.testing.assert_close(gradient[2:, 1], torch.full((2,), 0.3))


def test_checkpoint_recomputation_uses_each_frames_original_input_cache():
    policy = RecurrentPolicy(trainable_feedback=True)
    states = [policy.start_episode(str(i), "goal") for i in range(3)]
    features = torch.arange(1.0, 13.0).view(4, 3)
    records = [
        [{"features": feature, "tokens": (torch.tensor(0),) * 3} for feature in row]
        for row in features
    ]
    result = replay_batched_records(policy, records, states)
    reference_weight = policy.weight.detach().clone().requires_grad_()
    cache = torch.zeros(3)
    for row in features:
        cache = reference_weight * (row + cache)
    actual = torch.stack([step.value[-1] for step in result])
    torch.testing.assert_close(actual, cache)
    expected_gradient = torch.autograd.grad(cache.square().sum(), reference_weight)[0]
    actual_gradient = torch.autograd.grad(actual.square().sum(), policy.weight)[0]
    torch.testing.assert_close(actual_gradient, expected_gradient)


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), -float("inf")])
def test_nonfinite_probabilities_are_always_rejected(bad):
    assert replay_probability_error(
        SimpleNamespace(log_prob=torch.tensor([0.0, bad])),
        [{"old_log_prob": torch.tensor(0.0)}] * 2,
        "cpu",
    ) == float("inf")
    assert replay_probability_error(
        SimpleNamespace(log_prob=torch.zeros(2)),
        [{"old_log_prob": torch.tensor(0.0)}, {"old_log_prob": torch.tensor(bad)}],
        "cpu",
    ) == float("inf")


def test_unreplayable_batch_is_dumped_before_any_optimizer_mutation(monkeypatch, tmp_path):
    from streamnav.training import pointcot_trainer as trainer

    monkeypatch.setattr(trainer.torch, "autocast", lambda *args, **kwargs: nullcontext())
    monkeypatch.setattr(
        trainer,
        "guarded_replay_records",
        lambda *args, **kwargs: (
            SimpleNamespace(log_prob=torch.ones(2)),
            1.0,
            True,
        ),
    )
    policy = RecurrentPolicy()
    states = [policy.start_episode(str(i), "goal") for i in range(3)]
    records = [
        [
            {"features": 1.0, "tokens": (torch.tensor(0),) * 3, "old_log_prob": torch.tensor(0.0)}
            for _ in states
        ]
        for _ in range(2)
    ]
    # No optimizer is supplied: a rejection must happen before optimizer access.
    with pytest.raises(ReplayMismatch, match="rank 0"):
        optimize_rollout(
            policy,
            None,
            None,
            records,
            {(0, i): s for i, s in enumerate(states)},
            {},
            None,
            None,
            {"sequence_length": 2, "replay_tolerance": 0.05},
            {},
            0.25,
            tmp_path,
            17,
            0,
        )
    folder = tmp_path / "replay_failures" / "update_0000017"
    reports = json.loads((folder / "accepted_errors.json").read_text())
    assert reports[0]["max_error"] > 0.05
    sample = torch.load(folder / "rank_000_block_000.pt", weights_only=False)
    assert len(sample["states"]) == 3 and len(sample["records"]) == 2
    assert policy.weight.item() == pytest.approx(0.3)


def test_resume_archives_uncommitted_metrics_without_duplicate_update_history(tmp_path):
    committed = json.dumps({"update": 1850}) + "\n"
    tail = json.dumps({"update": 1851}) + "\n" + '{"update":1852'
    for name in ("train_metrics.jsonl", "rank_000_metrics.jsonl"):
        (tmp_path / name).write_text(committed + tail)
    archive_uncommitted_metrics(tmp_path, 1850)
    event = json.loads((tmp_path / "resume_events.jsonl").read_text())
    assert event["uncommitted_rows"] == {"train_metrics.jsonl": 2, "rank_000_metrics.jsonl": 2}
    from pathlib import Path

    for name in event["uncommitted_rows"]:
        assert (tmp_path / name).read_text() == committed
        assert (Path(event["archive"]) / name).read_text() == tail
    archive_uncommitted_metrics(tmp_path, 1850)
    assert len((tmp_path / "resume_events.jsonl").read_text().splitlines()) == 1
