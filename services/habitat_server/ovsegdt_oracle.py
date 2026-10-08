"""Run OVSegDT's pinned ObjNavExplorer unchanged behind the simulator RPC.

Only the Habitat-Lab simulator/episode interfaces are adapted. The actual
frontier selection, EXPLORE/BEELINE/PIVOT state machine and action logic come
from the upstream source at a8890d68cfa0d10254238abe9266a76856cb1f17.
"""

from __future__ import annotations

import importlib
import importlib.machinery
import importlib.util
import sys
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace

import habitat_sim
import numpy as np

from streamnav.contracts.action import NavigationAction
from streamnav.errors import OracleUnavailableError


def load_explorer():
    root = Path(__file__).resolve().parents[2]
    lab = root / "third_party/OVSegDT/habitat-lab/habitat-lab"
    source = root / "runtime/vendor/frontier_exploration/frontier_exploration"
    if not (source / "objnav_explorer.py").is_file():
        raise RuntimeError("Install the pinned oracle with scripts/setup_ovsegdt_oracle.sh")
    if str(lab) not in sys.path:
        sys.path.insert(0, str(lab))
    # Its __init__ imports an unrelated torch-based training policy. Load the
    # real sensor modules without registering that unused learner stack.
    if "frontier_exploration" not in sys.modules:
        spec = importlib.machinery.ModuleSpec("frontier_exploration", loader=None, is_package=True)
        module = importlib.util.module_from_spec(spec)
        module.__path__ = [str(source)]
        sys.modules["frontier_exploration"] = module
    return importlib.import_module("frontier_exploration.objnav_explorer")


class SimulatorBridge:
    def __init__(self, env):
        self.env = env
        self.pathfinder = PathfinderBridge(env.sim.pathfinder)

    def __getattr__(self, name):
        return getattr(self.env.sim, name)

    def get_agent_state(self):
        return self.env.sim.get_agent(0).get_state()


class PathfinderBridge:
    """Habitat 0.3.3 requires Vector3 where the pinned teacher passes lists."""

    def __init__(self, pathfinder):
        self.pathfinder = pathfinder

    def __getattr__(self, name):
        return getattr(self.pathfinder, name)

    def snap_point(self, point, island_index=-1):
        import magnum as mn

        return self.pathfinder.snap_point(mn.Vector3(*point), island_index)


class ViewpointDistance:
    """The VIEW_POINTS metric, including the MultiGoalShortestPath cache.

    The decoder has already expanded OVON child categories. Cache and position
    tolerance follow OVONDistanceToGoal, without importing its semantic metrics.
    """

    def __init__(self, env):
        self.env = env
        self.previous = None
        self.metric = None

    def reset_metric(self, episode, task):
        self.previous = None
        self.update_metric(episode, task)

    def update_metric(self, episode, task):
        position = self.env.sim.get_agent(0).get_state().position
        if self.previous is None or not np.allclose(self.previous, position, atol=1e-4):
            path = habitat_sim.MultiGoalShortestPath()
            path.requested_start = position
            path.requested_ends = self.env.goal_positions
            self.env.sim.pathfinder.find_path(path)
            episode._shortest_path_cache = path
            self.metric = float(path.geodesic_distance)
            self.previous = position.copy()

    def get_metric(self):
        return self.metric


class OVSegDTOracle:
    def __init__(self, env):
        upstream = load_explorer()
        from habitat.core.simulator import AgentState
        from habitat.tasks.nav.object_nav_task import ObjectGoal, ObjectViewLocation
        from omegaconf import OmegaConf

        options = asdict(upstream.ObjNavExplorerSensorConfig())
        options.update(env.config.get("oracle_config", {}))
        options.update(
            fov=env.config.get("hfov", 120),
            success_distance=env.config.get("success_distance", 0.25),
            turn_angle=env.config.get("turn_angle", 30),
            forward_step_size=env.config.get("forward_step", 0.25),
        )
        measure = ViewpointDistance(env)
        self.task = SimpleNamespace(
            measurements=SimpleNamespace(measures={"distance_to_goal": measure}),
            is_stop_called=False,
        )
        raw_goals = env.episode["goals"]
        primary_count = env.episode.get("metadata", {}).get("primary_goal_count", len(raw_goals))
        goals = [
            ObjectGoal(
                position=goal["position"],
                object_id=str(goal["object_id"]),
                view_points=[
                    ObjectViewLocation(AgentState(**v["agent_state"]), v.get("iou"))
                    for v in goal["view_points"]
                ],
            )
            for goal in raw_goals[:primary_count]
        ]
        self.episode = SimpleNamespace(episode_id=env.uid, goals=goals, _shortest_path_cache=None)
        self.explorer = upstream.ObjNavExplorer(
            sim=SimulatorBridge(env), config=OmegaConf.create(options), task=self.task
        )

    def action(self):
        try:
            action = self.explorer.get_observation(task=self.task, episode=self.episode)
            if self.task.is_stop_called:
                raise OracleUnavailableError(
                    "ObjNavExplorer exhausted its frontiers (invalid episode)"
                )
            return NavigationAction(int(np.asarray(action).item()))
        except (AssertionError, IndexError, ValueError) as exc:
            raise OracleUnavailableError(f"ObjNavExplorer failed: {exc}") from exc
