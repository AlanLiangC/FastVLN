import random
from types import SimpleNamespace

import torch

from streamnav.contracts.state import StreamingState
from streamnav.evaluation.detector_client import DetectorClient
from streamnav.models.point_cot.generalization import GeneralizationAdapters
from streamnav.models.point_cot.policy import FrameFeatures
from streamnav.training.affordance_loss import imitation_weight
from streamnav.training.detector_hints import TrainingDetectorHints
from streamnav.training.object_hints import episode_hint_mode


def test_zero_output_preserves_parent_and_spatial_motion_branches_get_gradients():
    module = GeneralizationAdapters(8)
    tokens = torch.tensor([42, 0])
    feedback = torch.randn(2, 12)
    torch.testing.assert_close(module(tokens, feedback), torch.zeros(2, 8), rtol=0, atol=0)
    module(tokens, feedback).sum().backward()
    assert module.spatial[-1].weight.grad.norm() > 0
    assert module.motion[-1].weight.grad.norm() > 0
    assert module.spatial[0].weight.grad.count_nonzero() == 0


def test_motion_feedback_is_causal_optional_and_resets_between_episodes():
    state = StreamingState((), "a", "h", 0, "Find a chair.")
    assert state.motion_feedback() == (0.0,) * 12
    for i in range(8):
        state = state.after_action(False, action=2 + i % 2, displacement=0.0)
    assert state.alternating_steps == state.stationary_steps == 8
    assert state.motion_feedback()[-1] == 0.25
    state = state.after_action(True, action=1, displacement=0.0)
    assert state.previous_collision and state.alternating_steps == 0
    state = state.after_action(False, action=1, displacement=0.25)
    assert state.stationary_steps == 0 and state.motion_feedback(0.1)[8:10] == (1.0, 1.0)
    # Privileged simulator measurements cannot alter the input representation.
    other = state.after_action(True, action=1, displacement=0.0)
    assert state.motion_feedback(0.1) == other.motion_feedback(0.1)
    assert StreamingState((), "b", "h", 0, "Find a chair.").motion_feedback() == (0.0,) * 12


def test_feedback_is_preserved_in_rollout_cpu_storage():
    feature = FrameFeatures(
        torch.ones(1, 2, dtype=torch.long),
        torch.ones(3, 1, 2),
        torch.ones(1, 4),
        (),
        feedback=tuple(range(12)),
    )
    assert feature.cpu().feedback == feature.feedback


def test_episode_modes_are_stable_and_do_not_change_sampling_rng():
    rng = random.Random(1)
    before = rng.getstate()
    config = {"episode_probability": 0.75, "real_probability": 2 / 3}
    modes = [episode_hint_mode(str(i), 2025, config) for i in range(1000)]
    assert 430 < modes.count("real") < 570
    assert 180 < modes.count("null") < 320
    assert modes == [episode_hint_mode(str(i), 2025, config) for i in range(1000)]
    assert rng.getstate() == before


def test_real_detector_uses_only_rgb_and_query_and_bootstrap_reuses_packet(monkeypatch):
    calls = []

    class Detector:
        def __init__(self, directory):
            pass

        def predict(self, images, categories, split):
            calls.append((images, categories, split))
            return [{"token": 42, "source": "yoloe_point"} for _ in images]

        def close(self):
            pass

    monkeypatch.setattr("streamnav.training.detector_hints.DetectorClient", Detector)
    config = {
        "seed": 2025,
        "trainer": {"object_hints": {"episode_probability": 1, "real_probability": 1}},
    }
    hints = TrainingDetectorHints(config, "unused")
    observation = {"rgb": torch.zeros(2, 2, 3), "frame_id": 3, "hidden_truth": "never read"}
    state = SimpleNamespace(episode_id="a", instruction="Find a chair.")
    first = hints.packets([observation], [state])
    assert first == hints.packets([observation], [state])
    assert len(calls) == 1 and calls[0][1:] == (["chair"], "train")
    assert first[0]["episode_id"] == "a" and first[0]["frame_id"] == 3
    hints.packets([{**observation, "frame_id": 4}], [state])
    assert len(calls) == 2


def test_detector_worker_is_bound_to_rank_device_not_default_gpu_zero(monkeypatch, tmp_path):
    calls = []
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "1,3,5")
    monkeypatch.setattr(torch.cuda, "current_device", lambda: 2)

    def start(command, **kwargs):
        calls.append(kwargs["env"])
        return SimpleNamespace(poll=lambda: 0)

    monkeypatch.setattr("streamnav.evaluation.detector_client.subprocess.Popen", start)
    monkeypatch.setattr(DetectorClient, "request", lambda self, command: {"ready": True})
    client = DetectorClient(tmp_path)
    try:
        assert calls[0]["CUDA_VISIBLE_DEVICES"] == "5"
    finally:
        client.close()


def test_recovery_weights_target_stagnation_and_wrong_hints_without_blanket_turn_penalty():
    cfg = {"stagnation_il_weight": 2, "wrong_hint_il_weight": 1.5}
    assert imitation_weight({"expert": 2, "alternating_steps": 1}, cfg) == 1
    assert imitation_weight({"expert": 1, "stationary_steps": 8}, cfg) == 2
    assert (
        imitation_weight({"expert": 2, "hint_quality_valid": True, "hint_quality_target": 0}, cfg)
        == 1.5
    )
