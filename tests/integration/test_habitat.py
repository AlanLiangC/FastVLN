import os
from dataclasses import replace

import pytest

from streamnav.contracts.action import NavigationAction
from streamnav.data.schema import HabitatEpisodeSource
from streamnav.envs.habitat_client import HabitatClient

pytestmark = [
    pytest.mark.habitat,
    pytest.mark.skipif(
        os.environ.get("STREAMNAV_INTEGRATION") != "1",
        reason="Set STREAMNAV_INTEGRATION=1 for real Habitat tests",
    ),
]


def test_oracle_reaches_real_object_goal():
    episode = next(
        HabitatEpisodeSource("runtime/data/manifests/hm3d_v1_val.json").evaluation_episodes(1)
    )
    env = HabitatClient(
        {
            "gpu_device_id": 1,
            "width": 480,
            "height": 270,
            "hfov": 120,
            "sensor_height": 0.88,
            "sensor_pitch_deg": 0.0,
            "agent_height": 0.88,
            "agent_radius": 0.18,
        }
    )
    try:
        first = env.reset(episode)
        assert first["rgb"].shape == (270, 480, 3)
        assert first["robot"] == pytest.approx(
            {
                "width": 480,
                "height": 270,
                "hfov": 120,
                "sensor_height": 0.88,
                "sensor_pitch_deg": 0.0,
                "agent_height": 0.88,
                "agent_radius": 0.18,
                "navmesh_height": 0.88,
                "navmesh_radius": 0.18,
                "navmesh_agent_max_climb": 0.10,
                "navmesh_cell_height": 0.05,
            }
        )
        for _ in range(100):
            result = env.step(env.get_oracle_action())
            if result["done"]:
                break
        assert result["success"] and result["metrics"]["spl"] > 0.8
    finally:
        env.close()


def test_real_training_curriculum_preserves_validation_starts():
    train = next(
        HabitatEpisodeSource("runtime/data/manifests/hm3d_v1_train.json").evaluation_episodes(1)
    )
    env = HabitatClient({"gpu_device_id": 1})
    try:
        original = env.reset(train)
        warm = replace(train, metadata={**train.metadata, "training_warm_start_distance": 1.5})
        easier = env.reset(warm)
        assert easier["distance"] <= 1.5
        assert easier["curriculum_warmup_steps"] > 0
        assert original["distance"] > easier["distance"]
        assert env.reset(train)["distance"] == pytest.approx(original["distance"])
        with pytest.raises(Exception, match="Warm starts are allowed only for training"):
            env.reset(replace(warm, split="val"))
    finally:
        env.close()


@pytest.mark.parametrize("failure", ["oracle_failure", "step_limit", "early_stop"])
def test_curriculum_failed_follower_restores_original_start(tmp_path, failure):
    import json
    import subprocess
    from pathlib import Path

    episode = next(
        HabitatEpisodeSource("runtime/data/manifests/hm3d_v2_train.json").evaluation_episodes(1)
    )
    path = tmp_path / "episode.json"
    path.write_text(json.dumps(episode.to_dict()))
    code = """
import json, sys
import numpy as np
sys.path.insert(0, 'services/habitat_server')
from env_factory import HabitatObjectNavEnv
from streamnav.contracts.action import NavigationAction
from streamnav.errors import OracleUnavailableError
sim = HabitatObjectNavEnv({'gpu_device_id': 0})
try:
    episode = json.load(open(sys.argv[1]))
    original_rgb, original = sim.reset(episode)
    real_oracle = sim.oracle
    calls = 0
    def fail_after_one_step():
        global calls
        calls += 1
        if calls == 2:
            raise OracleUnavailableError('injected warmup follower failure')
        return real_oracle()
    mode = sys.argv[2]
    sim.oracle = (
        fail_after_one_step if mode == 'oracle_failure'
        else lambda: NavigationAction.TURN_LEFT if mode == 'step_limit' else NavigationAction.STOP
    )
    episode['metadata']['training_warm_start_distance'] = 0.3
    rgb, observation = sim.reset(episode)
    assert 'curriculum_fallback' in observation
    assert observation['distance'] == original['distance']
    assert observation['frame_id'] == 0
    assert observation['curriculum_warmup_steps'] == {
        'oracle_failure': 1, 'step_limit': 250, 'early_stop': 0
    }[mode]
    assert np.array_equal(rgb, original_rgb)
finally:
    sim.close()
"""
    result = subprocess.run(
        [str(Path("runtime/habitat-env/bin/python").resolve()), "-c", code, str(path), failure],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stdout + result.stderr


def test_stationary_policy_does_not_invalidate_a_working_teacher():
    episode = next(
        HabitatEpisodeSource("runtime/data/manifests/hm3d_v1_val.json").evaluation_episodes(1)
    )
    env = HabitatClient({"gpu_device_id": 0, "success_distance": 0.25})
    try:
        initial = env.reset(episode)
        assert initial["distance"] > 0.25
        env.get_oracle_action()
        for _ in range(64):
            env.step(NavigationAction.TURN_LEFT)
        assert isinstance(env.get_oracle_action(), NavigationAction)
        env.reset(episode)
        assert isinstance(env.get_oracle_action(), NavigationAction)
    finally:
        env.close()


def test_upstream_explorer_and_six_action_camera_reset():
    import torch

    episode = next(
        HabitatEpisodeSource("runtime/data/manifests/hm3d_ovon_val_seen.json").evaluation_episodes(
            1
        )
    )
    env = HabitatClient({"gpu_device_id": 0, "oracle": "objnav_explorer", "success_distance": 0.25})
    try:
        first = env.reset(episode)
        up = env.step(NavigationAction.LOOK_UP)
        down = env.step(NavigationAction.LOOK_DOWN)
        assert (up["rgb"].float() - first["rgb"].float()).abs().mean() > 1
        # Inverse rotations introduce tiny rasterization rounding differences.
        assert (down["rgb"].float() - first["rgb"].float()).abs().mean() < 0.01
        assert torch.equal(env.reset(episode)["rgb"], first["rgb"])
        for _ in range(100):
            result = env.step(env.get_oracle_action())
            if result["done"]:
                break
        assert result["success"] and result["metrics"]["spl"] > 0.9
    finally:
        env.close()
