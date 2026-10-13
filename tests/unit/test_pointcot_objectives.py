import json
from dataclasses import replace

import pytest
import torch

from streamnav.contracts.episode import NavigationEpisode
from streamnav.data.grounding import GroundingSource, check_grounding_split, validate_record
from streamnav.data.mixed_starts import MixedStartEpisodeSource, validate_curriculum
from streamnav.models.point_cot.policy import TokenStep
from streamnav.training.point_supervision import marginal_point_loss, targets_from_label
from streamnav.training.pointcot_trainer import replay_losses


def test_mask_is_distinct_from_annotated_absence():
    logits = torch.tensor([[2.0, 0.0, -1.0], [2.0, 0.0, -1.0]], requires_grad=True)
    losses = marginal_point_loss(logits, [[], [0]])
    losses.sum().backward()
    assert losses[0] == 0 and losses[1] > 0
    assert logits.grad[0].count_nonzero() == 0
    assert logits.grad[1].count_nonzero() > 0
    assert targets_from_label({"opos": 0, "opos_valid": False})["opos_targets"] == []


def test_equivalent_objects_use_probability_mass():
    logits = torch.tensor([[1.0, 2.0, 3.0]])
    loss = marginal_point_loss(logits, [[1, 2, 2]])
    assert torch.allclose(loss, -logits.softmax(-1)[:, 1:].sum(-1).log())


def test_grounding_splits_are_by_image(tmp_path):
    row = {
        "image_id": "same-image",
        "image": "x.jpg",
        "instruction": "Find a chair.",
        "apos_targets": [],
        "opos_targets": [42],
    }
    manifests = []
    for split in ("train", "val"):
        path = tmp_path / (split + ".jsonl")
        path.write_text(json.dumps(row) + "\n")
        manifests.append(GroundingSource(path))
    with pytest.raises(ValueError, match="share images"):
        check_grounding_split(*manifests)
    with pytest.raises(ValueError):
        validate_record({**row, "opos_targets": [1297]})
    state = manifests[0].state_dict()
    first = manifests[0].sample(3)
    manifests[0].load_state_dict(state)
    assert first == manifests[0].sample(3)


def test_curriculum_rejects_eval_and_keeps_original_starts():
    import random

    class Source:
        rng = random.Random(10)

        def sample_episode(self):
            return episode

    episode = NavigationEpisode(
        "x", "train", "scene", "1", "Find a chair.", [0, 0, 0], [0, 0, 0, 1], [], "chair"
    )
    config = {
        "original_probability": 0.5,
        "near_probability": 0.35,
        "ready_probability": 0.15,
        "near_distance": [0.5, 3.0],
        "ready_distance": [0.0, 0.2],
        "goal_facing_probability": 0.5,
    }
    sampler = MixedStartEpisodeSource(Source(), config)
    outputs = [sampler.sample_episode() for _ in range(100)]
    assert any("training_start" not in e.metadata for e in outputs)
    assert any(e.metadata.get("training_start", {}).get("kind") == "ready" for e in outputs)
    assert all(e.start_position == [0, 0, 0] for e in outputs)
    episode = replace(episode, split="val_seen")
    with pytest.raises(ValueError, match="training-only"):
        sampler.sample_episode()
    with pytest.raises(ValueError):
        validate_curriculum({**config, "original_probability": 0})


def test_weighted_grounding_resume_and_real_image_priority(tmp_path):
    path = tmp_path / "train.jsonl"
    rows = [
        {
            "image_id": str(i),
            "image": "x.jpg",
            "instruction": "Find a chair.",
            "source": "real" if i < 4 else "synthetic",
            "opos_targets": [42],
        }
        for i in range(8)
    ]
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))
    source = GroundingSource(path, seed=2, source_weights={"real": 0.8, "synthetic": 0.2})
    state = source.state_dict()
    samples = source.sample(200)
    assert sum(r["source"] == "real" for r in samples) > 140
    source.load_state_dict(state)
    assert samples == source.sample(200)


def test_dagger_trains_actual_points_and_never_replaces_ppo_prefix():
    from types import SimpleNamespace

    zero = torch.tensor(0)
    apos = torch.zeros(1, 1300, requires_grad=True)
    opos = torch.zeros(1, 1297, requires_grad=True)
    actual = torch.tensor([[0.0, 2.0, 0.0, 0.0, 0.0, 0.0]], requires_grad=True)
    teacher = torch.tensor([[2.0, 0.0, 0.0, 0.0, 0.0, 0.0]], requires_grad=True)
    sampled = TokenStep(
        zero[None],
        zero[None],
        zero[None],
        apos,
        opos,
        actual,
        torch.zeros(1, requires_grad=True),
        None,
    )

    class Policy:
        device = torch.device("cpu")

        def replay_fragment(self, features, state, tokens, resets=None):
            assert tokens[0][0] == 1299
            return SimpleNamespace(apos_logits=apos, opos_logits=opos, action_logits=teacher)

    record = {
        "features": None,
        "tokens": (zero, zero, zero),
        "expert": 0,
        "perception_label": {"apos_valid": True, "apos": 1299},
        "il_weight": 1,
        "old_log_prob": sampled.log_prob[0].detach(),
        "student_action": False,
    }
    cfg = {
        "teacher_action_mix": 0.5,
        "num_envs": 1,
        "ppo_clip": 0.2,
        "point_il_coef": 1,
        "value_loss_coef": 0.5,
        "entropy_coef": 0.01,
    }
    loss, pieces, error = replay_losses(
        Policy(), None, {}, sampled, [record], torch.ones(1), torch.zeros(1), cfg, 0.5
    )
    loss.sum().backward()
    assert actual.grad[0, 0] < 0 and teacher.grad[0, 0] < 0
    assert sampled.apos[0] == 0 and error == 0
    assert pieces["ppo_loss"].item() == 0


def test_actual_history_point_supervision_and_masked_type_gradient():
    from types import SimpleNamespace

    from streamnav.models.policy.perception import point_type_logits, point_type_targets

    actual_apos = torch.zeros(3, 1300, requires_grad=True)
    actual_opos = torch.zeros(3, 1297, requires_grad=True)
    teacher_apos = torch.full((3, 1300), 0.5, requires_grad=True)
    teacher_opos = torch.full((3, 1297), 0.5, requires_grad=True)
    actual_action = torch.zeros(3, 6, requires_grad=True)
    teacher_action = torch.zeros(3, 6, requires_grad=True)
    tokens = torch.zeros(3, dtype=torch.long)
    step = TokenStep(
        tokens, tokens, tokens, actual_apos, actual_opos, actual_action, torch.zeros(3), None
    )

    class Policy:
        device = torch.device("cpu")

        def replay_fragment(self, features, state, forced, resets=None):
            assert [int(t[0]) for t in forced] == [42, 1299, 0]
            return SimpleNamespace(
                apos_logits=teacher_apos,
                opos_logits=teacher_opos,
                action_logits=teacher_action,
            )

    labels = [
        {"apos_valid": True, "apos": 42},
        {"apos_valid": True, "apos": 1299},
        {"apos_valid": False, "apos": 0},
    ]
    records = [
        {
            "features": None,
            "tokens": (tokens[i], tokens[i], tokens[i]),
            "expert": 0,
            "perception_label": label,
            "il_weight": 1,
            "old_log_prob": step.log_prob[i].detach(),
            "student_action": False,
        }
        for i, label in enumerate(labels)
    ]
    cfg = {
        "teacher_action_mix": 0.5,
        "num_envs": 1,
        "ppo_clip": 0.2,
        "point_il_coef": 1,
        "point_supervision_prefix": "actual",
        "apos_type_il_coef": 2,
        "value_loss_coef": 0.5,
        "entropy_coef": 0.01,
    }
    loss, pieces, error = replay_losses(
        Policy(), None, {}, step, records, torch.ones(3), torch.zeros(3), cfg, 0
    )
    assert torch.allclose(pieces["apos_loss"], marginal_point_loss(actual_apos, [[42], [1299], []]))
    expected_type = torch.nn.functional.cross_entropy(
        point_type_logits(actual_apos[:2]),
        point_type_targets(torch.tensor([42, 1299])),
        reduction="none",
    )
    assert torch.allclose(pieces["apos_type_loss"][:2], expected_type)
    assert pieces["apos_type_loss"][2] == 0 and error == 0
    loss.sum().backward()
    assert actual_apos.grad[:2].count_nonzero() > 0
    assert actual_apos.grad[0, 42] < 0 and actual_apos.grad[1, 1299] < 0
    assert actual_apos.grad[2].count_nonzero() == 0
    assert actual_opos.grad.count_nonzero() == 0
    # Teacher points condition action IL but no longer contribute point CE.
    assert teacher_apos.grad is None and teacher_opos.grad is None
    assert actual_action.grad.count_nonzero() > 0 and teacher_action.grad.count_nonzero() > 0
