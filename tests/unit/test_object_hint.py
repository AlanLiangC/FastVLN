import random
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from streamnav.contracts.object_hint import object_hint_token
from streamnav.contracts.object_labels import instance_object_label
from streamnav.models.point_cot.object_hint import ObjectHintToken
from streamnav.models.point_cot.policy import FrameFeatures, Qwen35PointCoTPolicy
from streamnav.training.object_hints import (
    episode_hint_enabled,
    evaluation_object_hint,
    evaluation_object_hint_packet,
    simulated_object_hint,
    simulated_object_hint_packet,
)
from streamnav.training.point_supervision import marginal_point_loss, navigation_object_losses


def test_pixel_hint_is_current_frame_optional_information():
    packet = {"frame_id": 3, "episode_id": "a", "xy_px": [240, 135]}
    assert object_hint_token(packet, 480, 270, frame_id=3, episode_id="a") == 649
    assert object_hint_token(packet, 480, 270, frame_id=4, episode_id="a") == 0
    assert object_hint_token(packet, 480, 270, frame_id=3, episode_id="b") == 0
    assert object_hint_token(None, 480, 270) == 0
    for value in (None, [], [480, 1], [1, float("nan")], [1, -1], 7):
        assert object_hint_token({"xy_px": value}, 480, 270) == 0
    with pytest.raises(ValueError):
        object_hint_token(1297, 480, 270)


def test_input_null_has_independent_semantics_and_gradients():
    rows = torch.nn.Parameter(torch.randn(1297, 8))
    module = ObjectHintToken(torch.ones(8))
    result = module(torch.tensor([0, 42]), rows)
    torch.testing.assert_close(result[0], module.null)
    torch.testing.assert_close(result[1], rows[42])
    result.sum().backward()
    assert module.null.grad.sum() == 8 and module.source.grad.sum() == 8
    assert rows.grad[0].count_nonzero() == 0
    assert rows.grad[42].sum() == 8


def test_exactly_one_hint_slot_survives_cpu_and_reconditioning():
    policy = Qwen35PointCoTPolicy.__new__(Qwen35PointCoTPolicy)
    torch.nn.Module.__init__(policy)
    policy.native = SimpleNamespace(device=torch.device("cpu"))
    policy.object_hint = ObjectHintToken(torch.ones(8))
    policy.opos_rows = torch.nn.Parameter(torch.randn(1297, 8))
    feature = FrameFeatures(
        torch.tensor([[1, 2, 3]]),
        torch.arange(3).view(1, 1, 3).expand(3, 1, 3),
        torch.zeros(0, 8),
        (),
    )
    conditioned = policy.condition_features([feature], [42])[0]
    assert conditioned.ids.shape[-1] == 4 and conditioned.hint_token == 42
    assert conditioned.positions[:, :, -1].eq(3).all()
    changed = policy.condition_features([conditioned.cpu()], [0])[0]
    assert changed.ids.shape[-1] == 4 and changed.hint_token == 0
    x = torch.zeros(2, 6, 8)
    result = policy.embed_hints(x, [conditioned, changed])
    torch.testing.assert_close(result[0, 3], policy.opos_rows[42])
    torch.testing.assert_close(result[1, 3], policy.object_hint.null)
    assert result[:, 4:].count_nonzero() == 0
    result.sum().backward()
    assert policy.object_hint.null.grad.norm() > 0


def test_instance_labels_distinguish_absence_from_incomplete_mapping_and_tiny_targets():
    semantic = np.zeros((270, 480), dtype=np.int32)
    assert instance_object_label(semantic, {7}, {0, 7})["opos_targets"] == [0]
    assert not instance_object_label(semantic, {7, 8}, {0, 7})["opos_valid"]
    semantic[100:150, 200:250] = 7
    label = instance_object_label(semantic, {7, 8}, {0, 7})
    assert label["opos_valid"] and label["opos"] > 0
    assert label["opos"] in label["opos_targets"]
    assert all(t > 0 for t in label["opos_targets"])
    semantic[:] = 0
    semantic[0, 0] = 7
    assert not instance_object_label(semantic, {7}, {0, 7})["opos_valid"]


def test_factorized_visibility_location_equals_token_ce_without_double_type_loss():
    logits = torch.randn(4, 1297, requires_grad=True)
    targets = [[42], [0], [], [51, 52, 53]]
    total, visibility, location = navigation_object_losses(logits, targets)
    torch.testing.assert_close(total, marginal_point_loss(logits, targets))
    torch.testing.assert_close(total, visibility + location)
    assert location[1] == 0 and total[2] == 0
    total.sum().backward()
    assert logits.grad[2].count_nonzero() == 0
    assert logits.grad[1, 0] < 0


def test_hint_simulator_covers_missing_false_positive_and_whole_episode_dropout():
    label = {"opos_valid": True, "opos": 42}
    assert simulated_object_hint(label, random.Random(1), dropout=1) == 0
    assert (
        simulated_object_hint(label, random.Random(1), dropout=0, jitter=0, false_positive=0) == 42
    )
    assert (
        simulated_object_hint(
            {"opos_valid": True, "opos": 0}, random.Random(1), dropout=0, false_positive=0
        )
        == 0
    )
    assert (
        simulated_object_hint(
            {"opos_valid": True, "opos": 0}, random.Random(1), dropout=0, false_positive=1
        )
        > 0
    )
    enabled = [episode_hint_enabled(str(i), 4) for i in range(100)]
    assert 30 < sum(enabled) < 70
    assert episode_hint_enabled("x", 4) == episode_hint_enabled("x", 4)


def test_hint_provenance_distinguishes_deliberate_noise_and_preserves_rng_consumption():
    label = {"opos_valid": True, "opos": 649}
    options = {"dropout": 0.2, "jitter": 1, "false_positive": 0.01}
    token_rng, packet_rng = random.Random(21), random.Random(21)
    for _ in range(40):
        token = simulated_object_hint(label, token_rng, **options)
        packet = simulated_object_hint_packet(label, packet_rng, **options)
        assert token == packet["token"]
    assert token_rng.getstate() == packet_rng.getstate()
    assert simulated_object_hint_packet(label, random.Random(1), dropout=1) == {
        "token": 0,
        "source": "dropout",
    }
    packet = simulated_object_hint_packet(
        {"opos_valid": True, "opos": 0}, random.Random(1), dropout=0, false_positive=1
    )
    assert packet["token"] > 0 and packet["source"] == "false_positive"
    packet = evaluation_object_hint_packet(label, "episode", 5, 2025, options)
    assert packet["token"] == evaluation_object_hint(label, "episode", 5, 2025, options)


def test_thin_instance_jitter_stays_inside_support_and_keeps_explicit_outlier_mode():
    label = {"opos_valid": True, "opos": 649, "opos_targets": [601, 649, 697]}
    unconstrained_misses = 0
    for seed in range(100):
        a, b = random.Random(seed), random.Random(seed)
        options = {"dropout": 0, "jitter": 1, "false_positive": 0}
        original = simulated_object_hint(label, a, **options)
        fixed = simulated_object_hint(label, b, **options, constrain_to_support=True)
        unconstrained_misses += original not in label["opos_targets"]
        assert fixed in label["opos_targets"]
        assert a.getstate() == b.getstate()
    assert unconstrained_misses > 50
    packet = simulated_object_hint_packet(
        label, random.Random(5), dropout=0, false_positive=1, constrain_to_support=True
    )
    assert packet["source"] == "false_positive"
    assert packet["token"] not in label["opos_targets"]
