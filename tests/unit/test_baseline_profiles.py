import json
from copy import deepcopy
from pathlib import Path

import yaml

from streamnav.evaluation.baseline_profiles import method_profiles

ROOT = Path(__file__).resolve().parents[2]


def profiles(fast):
    saved = {"habitat": {"simulator": {"forward_step_size": 0.25, "tilt_angle": 30}}}
    official = yaml.safe_load(
        (
            ROOT / "third_party/OVSegDT/config/experiments/transformer_rl_segm_loss-validation.yaml"
        ).read_text()
    )
    return method_profiles(fast, saved, official, seed=2025, max_steps=500)


def test_ovsegdt_never_inherits_fastvln_sensor_or_body_settings():
    fast = {
        "habitat": {
            "width": 480,
            "height": 270,
            "hfov": 120,
            "allow_sliding": True,
            "agent_radius": 0.18,
            "python": "runtime/habitat-env/bin/python",
        }
    }
    baseline = profiles(fast)
    changed = deepcopy(fast)
    changed["habitat"].update(width=1, height=2, hfov=10, agent_radius=3)
    assert baseline["ovsegdt"] == profiles(changed)["ovsegdt"]
    assert baseline["fastvln"]["hfov"] == 120
    assert baseline["fastvln"]["allow_sliding"] is True
    assert baseline["ovsegdt"]["width"] == 360
    assert baseline["ovsegdt"]["height"] == 640
    assert baseline["ovsegdt"]["hfov"] == 42
    assert baseline["ovsegdt"]["agent_radius"] == 0.17
    assert baseline["ovsegdt"]["allow_sliding"] is False
    assert baseline["ovsegdt"]["backend"] == "ovsegdt_original"
    assert baseline["ovsegdt"]["python"] == "runtime/ovsegdt-sim-env/bin/python"


def test_only_task_horizon_and_success_definition_are_shared():
    result = profiles({"habitat": {"success_distance": 10, "max_episode_steps": 5}})
    for profile in result.values():
        assert profile["success_distance"] == 0.25
        assert profile["max_episode_steps"] == 500
    assert not result["ovsegdt"]["perception_labels"]
    assert json.dumps(result)
