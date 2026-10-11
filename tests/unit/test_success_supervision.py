import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from streamnav.contracts.action import NavigationAction as A


@pytest.fixture
def env_module(monkeypatch):
    monkeypatch.setitem(sys.modules, "habitat_sim", SimpleNamespace())
    monkeypatch.setitem(sys.modules, "habitat_sim.utils", SimpleNamespace())
    monkeypatch.setitem(
        sys.modules, "habitat_sim.utils.common", SimpleNamespace(quat_from_coeffs=lambda x: x)
    )
    spec = importlib.util.spec_from_file_location(
        "test_success_env", Path("services/habitat_server/env_factory.py")
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize(
    "enabled,split,distance,expected",
    [
        (True, "train", 0.249, A.STOP),
        (True, "train", 0.25, A.TURN_LEFT),
        (True, "val", 0.249, A.TURN_LEFT),
        (False, "train", 0.249, A.TURN_LEFT),
    ],
)
def test_stop_advice_is_training_only_and_matches_success_boundary(
    env_module, enabled, split, distance, expected
):
    calls = []
    env = env_module.HabitatObjectNavEnv({"oracle_success_stop": enabled, "success_distance": 0.25})
    env.episode, env.ended, env.previous_distance = {"split": split}, False, distance
    env.explorer = SimpleNamespace(action=lambda: calls.append("teacher") or A.TURN_LEFT)
    assert env.oracle() == expected
    assert len(calls) == (0 if expected == A.STOP else 1)


@pytest.mark.parametrize("distance", [0.25, float("nan"), float("inf")])
def test_invalid_arrival_band_is_rejected(env_module, distance):
    with pytest.raises(ValueError, match="arrival_near_distance"):
        env_module.HabitatObjectNavEnv(
            {
                "oracle_success_stop": True,
                "success_distance": 0.25,
                "arrival_near_distance": distance,
            }
        )
