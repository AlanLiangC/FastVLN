import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from streamnav.contracts.action import NavigationAction as A
from streamnav.training.object_hints import conditioned_hint_config


@pytest.fixture
def rig(monkeypatch):
    class FollowerError(Exception):
        pass

    class Navigator:
        def __init__(self, env):
            self.env = env
            self.calls = 0
            self.follower = self

        def reset(self):
            pass

        def next_action_along(self, target):
            self.calls += 1
            return self.env.forced_action or (A.TURN_LEFT if target[0] < 0 else A.TURN_RIGHT)

        def forward_executable(self):
            return self.env.executable

    class Labeler:
        def __init__(self, env):
            self.env = env

        def object_labels(self):
            return {
                "opos": 582,
                "opos_valid": True,
                "opos_targets": [582, 622],
                "episode_id": self.env.uid,
                "frame_id": self.env.steps,
            }

        def labels(self, action, *, navigation_target=None, goal_ids=None):
            self.env.annotations.append((action, navigation_target, goal_ids))
            return {"episode_id": self.env.uid, "frame_id": self.env.steps}

    monkeypatch.setitem(
        sys.modules, "habitat_sim", SimpleNamespace(MultiGoalShortestPath=SimpleNamespace)
    )
    monkeypatch.setitem(
        sys.modules, "habitat_sim.errors", SimpleNamespace(GreedyFollowerError=FollowerError)
    )
    monkeypatch.setitem(
        sys.modules, "executable_oracle", SimpleNamespace(ExecutableNavigation=Navigator)
    )
    monkeypatch.setitem(
        sys.modules, "perception_labels", SimpleNamespace(PerceptionLabeler=Labeler)
    )
    spec = importlib.util.spec_from_file_location(
        "test_conditioned_teacher", Path("services/habitat_server/hint_conditioning.py")
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    semantic = np.zeros((27, 48), dtype=int)
    semantic[12, 5], semantic[12, 45] = 11, 22

    def goal(i, x):
        return {
            "object_id": str(i),
            "view_points": [{"agent_state": {"position": [x, 0, 2]}}],
        }

    env = SimpleNamespace(
        config={"object_label_source": "semantic_instance", "success_distance": 0.25},
        episode={"goals": [goal(11, -1), goal(22, 1)]},
        semantic_known_ids={11, 22},
        steps=0,
        uid="train:sample",
        previous_distance=2.0,
        annotations=[],
        oracle_calls=[],
        forced_action=None,
        executable=True,
        reachable=True,
        _sensor_observations=lambda: {"perception_semantic": semantic},
    )
    env.oracle = lambda: env.oracle_calls.append(env.steps) or A.TURN_RIGHT

    def pathfind(path):
        path.closest_end_point_index = 0
        path.geodesic_distance = 2.0
        return env.reachable

    env.sim = SimpleNamespace(
        get_agent=lambda _: SimpleNamespace(get_state=lambda: SimpleNamespace(position=[0, 0, 0])),
        pathfinder=SimpleNamespace(find_path=pathfind),
    )
    packet = {"token": 582, "source": "instance_point"}
    monkeypatch.setattr(module, "simulated_object_hint_packet", lambda *a, **kw: dict(packet))
    config = {
        "seed": 2025,
        "episode_probability": 1.0,
        "conditioned_supervision": {"enabled": True, "memory_steps": 2},
    }
    teacher = module.HintConditionedSupervision(env, config)
    return SimpleNamespace(
        env=env, teacher=teacher, packet=packet, semantic=semantic, module=module
    )


def test_actual_delivered_point_binds_action_and_both_point_labels(rig):
    result = rig.teacher.supervision()
    assert result["action"] == A.TURN_LEFT
    assert result["hint"]["bound_goal_id"] == 11
    action, target, identities = rig.env.annotations[-1]
    assert action == A.TURN_LEFT and identities == {11}
    np.testing.assert_array_equal(target, [-1, 0, 2])
    assert rig.env.oracle_calls == []
    # Repeated annotation requests must not advance a stateful teacher twice.
    assert rig.teacher.supervision() == result
    assert rig.teacher.navigator.calls == 1


def test_null_input_does_not_bind_using_unseen_ground_truth(rig):
    rig.packet["token"] = 0
    result = rig.teacher.supervision()
    assert not result["hint"]["active"] and result["hint"]["goal_key"] == "global"
    assert result["action"] == A.TURN_RIGHT
    assert rig.env.annotations[-1][1:] == (None, None)


def test_missing_hint_keeps_only_previously_delivered_candidate_then_expires(rig):
    first = rig.teacher.supervision()
    rig.packet["token"] = 0
    for frame in (1, 2):
        rig.env.steps = frame
        result = rig.teacher.supervision()
        assert result["hint"]["token"] == 0 and result["action"] == A.TURN_LEFT
        assert result["hint"]["goal_key"] == first["hint"]["goal_key"]
        assert rig.env.annotations[-1][2] == {11}
    rig.env.steps = 3
    assert rig.teacher.supervision()["hint"]["goal_key"] == "global"


def test_bootstrap_hint_query_is_pure_and_matches_next_supervision(rig):
    first = rig.teacher.object_supervision()
    assert first["object_hint_token"] == 582 and rig.teacher.goal_id is None
    assert rig.teacher.navigator is None and rig.env.oracle_calls == []
    assert rig.teacher.object_supervision() == first
    assert rig.teacher.supervision()["hint"]["token"] == first["object_hint_token"]


def test_final_pixel_ownership_determines_binding_not_noise_provenance(rig):
    rig.packet.update(token=622, source="false_positive")
    result = rig.teacher.supervision()
    assert result["hint"]["bound_goal_id"] == 22
    assert rig.env.annotations[-1][2] == {22}
    rig.env.steps = 1
    rig.packet["token"] = 1  # Background, never a valid goal instance.
    assert rig.teacher.supervision()["hint"]["bound_goal_id"] == 22


@pytest.mark.parametrize("mode", ["unreachable", "blocked", "local_stop"])
def test_invalid_routes_fall_back_without_false_stop_or_wrong_apos(rig, mode):
    if mode == "unreachable":
        rig.env.reachable = False
    elif mode == "blocked":
        rig.env.forced_action, rig.env.executable = A.MOVE_FORWARD, False
    else:
        # STOP is enum zero; use a function override instead of truthiness.
        rig.teacher.supervision()
        rig.teacher.navigator.next_action_along = lambda _: A.STOP
        rig.env.steps = 1
    result = rig.teacher.supervision()
    assert result["action"] == A.TURN_RIGHT
    assert not result["hint"]["active"] and result["hint"]["fallback"]
    assert rig.env.annotations[-1][1:] == (None, None)


def test_task_success_overrides_hint_route_and_does_not_call_follower(rig):
    rig.env.previous_distance = 0.249
    assert rig.teacher.supervision()["action"] == A.STOP
    assert rig.teacher.navigator.calls == 0


def test_candidate_change_creates_new_future_label_boundary(rig):
    first = rig.teacher.supervision()
    rig.env.steps, rig.packet["token"] = 1, 622
    second = rig.teacher.supervision()
    assert first["hint"]["goal_key"] != second["hint"]["goal_key"]
    assert second["hint"]["bound_goal_id"] == 22


def test_whole_null_episode_ignores_visible_ground_truth(rig):
    config = {**rig.teacher.config, "episode_probability": 0}
    teacher = rig.module.HintConditionedSupervision(rig.env, config)
    assert teacher.supervision()["hint"]["token"] == 0
    assert teacher.goal_id is None


def test_conditioning_is_opt_in_and_requires_compatible_inputs_and_stop_labels():
    config = {
        "seed": 1,
        "trainer": {"object_hints": {}},
        "model": {"object_hint_token": True},
        "habitat": {"oracle_success_stop": True, "object_label_source": "semantic_instance"},
    }
    assert conditioned_hint_config(config) is None
    hints = config["trainer"]["object_hints"]
    hints["conditioned_supervision"] = {"enabled": True, "memory_steps": 40}
    assert conditioned_hint_config(config)["seed"] == 1
    config["model"]["object_hint_token"] = False
    with pytest.raises(ValueError, match="hint input token"):
        conditioned_hint_config(config)
