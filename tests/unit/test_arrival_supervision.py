from types import SimpleNamespace

import pytest
import torch
import yaml

from streamnav.data.mixed_starts import MixedStartEpisodeSource, validate_curriculum
from streamnav.evaluation.perception_records import perception_records
from streamnav.training.arrival_supervision import arrival_losses, prepare_arrival_supervision
from streamnav.training.pointcot_trainer import fork_signature, resume_recipe


def record(distance, *, valid=True, proposed=0):
    target = 2 if distance < 0.25 else (1 if distance < 0.75 else 0)
    return {
        "distance_before": distance,
        "expert": 0 if target == 2 else 1,
        "tokens": (torch.tensor(0), torch.tensor(0), torch.tensor(proposed)),
        "conditional_greedy_action": proposed,
        "perception_label": {"arrival": target, "arrival_valid": valid},
    }


def test_both_sides_of_success_boundary_have_opposite_stop_gradients():
    records = [record(d) for d in (0.2499, 0.25, 0.5, 3.0, 0.1)]
    records[-1]["perception_label"]["arrival_valid"] = False
    prepare_arrival_supervision([records], "cpu")
    step = SimpleNamespace(
        arrival_logits=torch.zeros(5, 3, requires_grad=True),
        apos_logits=torch.zeros(5, 1300, requires_grad=True),
        action_logits=torch.zeros(5, 6, requires_grad=True),
    )
    cfg = dict(
        arrival_loss_coef=0.2,
        stop_calibration_coef=0.1,
        apos_stop_calibration_coef=0.05,
        stop_hard_negative_coef=0.2,
    )
    loss, metrics = arrival_losses(step, records, cfg)
    loss.mean().backward()
    assert step.action_logits.grad[0, 0] < 0
    assert (step.action_logits.grad[1:4, 0] > 0).all()
    assert step.apos_logits.grad[0, 1299] < 0
    assert (step.apos_logits.grad[1:4, 1299] > 0).all()
    assert step.arrival_logits.grad[0, 2] < 0
    for t in (step.arrival_logits, step.apos_logits, step.action_logits):
        assert torch.isfinite(t.grad).all() and t.grad[-1].count_nonzero() == 0
    assert metrics["stop_hard_negative_fraction"].tolist() == [0, 1, 1, 1, 0]


def test_fragmentation_does_not_rebalance_positive_or_negative_labels():
    records = [record(0.1), *[record(0.4) for _ in range(3)], *[record(3.0) for _ in range(16)]]
    prepare_arrival_supervision([records], "cpu")
    weights = torch.tensor([r["arrival_weight"] for r in records])
    torch.testing.assert_close(weights.mean(), torch.tensor(1.0))
    cfg = dict(
        arrival_loss_coef=0.2,
        stop_calibration_coef=0.1,
        apos_stop_calibration_coef=0.05,
        stop_hard_negative_coef=0.2,
    )
    generator = torch.Generator().manual_seed(3)
    tensors = [torch.randn(20, n, generator=generator) for n in (3, 1300, 6)]

    def losses(start, end):
        s = SimpleNamespace(
            **{
                k: v[start:end]
                for k, v in zip(
                    ["arrival_logits", "apos_logits", "action_logits"], tensors, strict=True
                )
            }
        )
        return arrival_losses(s, records[start:end], cfg)[0]

    torch.testing.assert_close(
        losses(0, 20).sum(), losses(0, 1).sum() + losses(1, 4).sum() + losses(4, 20).sum()
    )


def test_wrong_teacher_stop_label_is_rejected_instead_of_learned():
    r = record(0.4)
    r["expert"] = 0
    with pytest.raises(ValueError, match="Teacher STOP"):
        prepare_arrival_supervision([[r]], "cpu")
    r = record(0.25)
    r["perception_label"]["arrival"] = 2
    r["expert"] = 0
    with pytest.raises(ValueError, match="success distance"):
        prepare_arrival_supervision([[r]], "cpu")


def test_unknown_labels_produce_connected_zero_and_do_not_shift_stop_prior():
    records = [record(0.1, valid=False), record(0.5, valid=False)]
    prepare_arrival_supervision([records], "cpu")
    step = SimpleNamespace(
        arrival_logits=torch.zeros(2, 3, requires_grad=True),
        apos_logits=torch.zeros(2, 1300, requires_grad=True),
        action_logits=torch.zeros(2, 6, requires_grad=True),
    )
    loss, _ = arrival_losses(
        step, records, {"arrival_loss_coef": 0.2, "stop_calibration_coef": 0.1}
    )
    loss.sum().backward()
    for v in (step.arrival_logits, step.apos_logits, step.action_logits):
        assert v.grad.count_nonzero() == 0


def test_boundary_sampler_preserves_original_starts_and_roundtrips_rng():
    import random

    from streamnav.contracts.episode import NavigationEpisode

    config = yaml.safe_load(open("configs/pointcot.yaml"))["curriculum"]
    validate_curriculum(config)
    episode = NavigationEpisode(
        "x", "train", "s", "1", "Find a chair.", [0, 0, 0], [0, 0, 0, 1], [], "chair"
    )

    class Source:
        rng = random.Random(71)

        def sample_episode(self):
            return episode

    source = Source()
    sampler = MixedStartEpisodeSource(source, config)
    before = source.rng.getstate()
    samples = [sampler.sample_episode() for _ in range(1000)]
    kinds = [e.metadata.get("training_start", {}).get("kind", "original") for e in samples]
    assert 450 < kinds.count("original") < 550
    assert 150 < kinds.count("boundary") < 250
    for e in samples:
        spec = e.metadata.get("training_start", {})
        if spec.get("kind") == "boundary":
            assert spec["distance_band"] == [0.25, 0.75]
            assert spec["goal_facing_probability"] == 1
    source.rng.setstate(before)
    assert [sampler.sample_episode() for _ in range(1000)] == samples


def test_arrival_and_curriculum_changes_require_explicit_fork():
    import copy

    config = yaml.safe_load(open("configs/pointcot.yaml"))
    old = copy.deepcopy(config)
    old["model"].pop("arrival_head")
    old["curriculum"].update(
        original_probability=0.5,
        near_probability=0.35,
        ready_probability=0.15,
        boundary_probability=0,
    )
    assert resume_recipe(old) != resume_recipe(config)
    assert fork_signature(old) == fork_signature(config)


def test_arrival_probabilities_reach_case_records_without_changing_point_tokens():
    p = dict(
        point_cot=True,
        apos=torch.zeros(1, 1300),
        opos=torch.zeros(1, 1297),
        apos_tokens=torch.tensor([42]),
        opos_tokens=torch.tensor([11]),
        arrival=torch.tensor([[0.0, 1.0, 2.0]]),
    )
    r = perception_records(p)[0]
    assert r["arrival_probabilities"][2] > 0.6 and r["apos"] == 42 and r["opos"] == 11
