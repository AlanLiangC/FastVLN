"""Real Habitat-Sim ObjectNav with official episode viewpoints and a geodesic oracle.

Runs under Python 3.9, independently of the learner's torch environment.
"""

from __future__ import annotations

import fcntl
import hashlib
import json
import os
import time
from pathlib import Path

import habitat_sim
import numpy as np
from habitat_sim.utils.common import quat_from_coeffs

from streamnav.contracts.action import NavigationAction
from streamnav.errors import OracleUnavailableError
from streamnav.training.rewards import ObjectNavReward, RewardConfig


class HabitatObjectNavEnv:
    def __init__(self, config):
        self.config = config
        self.sim = None
        self.scene = None
        self.reward = ObjectNavReward(RewardConfig(**config.get("reward", {})))

    def _make_sim(self, scene):
        if self.sim is not None:
            self.sim.close()
        sim_cfg = habitat_sim.SimulatorConfiguration()
        sim_cfg.scene_id = scene
        sim_cfg.gpu_device_id = self.config.get("gpu_device_id", 1)
        sim_cfg.enable_physics = False
        sim_cfg.allow_sliding = False
        sensor = habitat_sim.CameraSensorSpec()
        sensor.uuid = "rgb"
        sensor.sensor_type = habitat_sim.SensorType.COLOR
        sensor.resolution = [self.config.get("height", 270), self.config.get("width", 480)]
        sensor.position = [0.0, self.config.get("sensor_height", 0.88), 0.0]
        sensor.orientation = [np.deg2rad(self.config.get("sensor_pitch_deg", 0.0)), 0.0, 0.0]
        sensor.hfov = self.config.get("hfov", 120)
        agent = habitat_sim.agent.AgentConfiguration()
        agent.height = self.config.get("agent_height", 0.88)
        agent.radius = self.config.get("agent_radius", 0.18)
        agent.sensor_specifications = [sensor]
        agent.action_space = {
            "move_forward": habitat_sim.agent.ActionSpec(
                "move_forward",
                habitat_sim.agent.ActuationSpec(amount=self.config.get("forward_step", 0.25)),
            ),
            "turn_left": habitat_sim.agent.ActionSpec(
                "turn_left",
                habitat_sim.agent.ActuationSpec(amount=self.config.get("turn_angle", 30)),
            ),
            "turn_right": habitat_sim.agent.ActionSpec(
                "turn_right",
                habitat_sim.agent.ActuationSpec(amount=self.config.get("turn_angle", 30)),
            ),
        }
        self.sim = habitat_sim.Simulator(habitat_sim.Configuration(sim_cfg, [agent]))
        if not self.sim.pathfinder.is_loaded:
            raise RuntimeError(f"Scene has no navigable mesh: {scene}")
        self._configure_navmesh(scene, agent)
        self.scene = scene
        self.follower = self.sim.make_greedy_follower(
            agent_id=0,
            goal_radius=self.config.get("success_distance", 0.1),
            stop_key=NavigationAction.STOP,
            forward_key="move_forward",
            left_key="turn_left",
            right_key="turn_right",
        )

    def _configure_navmesh(self, scene, agent):
        # AgentConfiguration alone does NOT change pathfinder collision geometry.
        # The supplied meshes were baked for height=1.5/radius=0.1; rebuild and
        # share a cached robot-specific navmesh across workers without races.
        settings = self.sim.pathfinder.nav_mesh_settings
        if np.isclose(settings.agent_height, agent.height) and np.isclose(
            settings.agent_radius, agent.radius
        ):
            return
        source = Path(scene).with_suffix(".navmesh")
        signature = hashlib.sha256(source.read_bytes())
        signature.update(f"{agent.height}:{agent.radius}:habitat-0.3.3".encode())
        root = Path(self.config.get("navmesh_cache", "runtime/cache/navmesh"))
        root.mkdir(parents=True, exist_ok=True)
        cache = root / f"{signature.hexdigest()}.navmesh"
        with (root / f"{signature.hexdigest()}.lock").open("w") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            if cache.exists():
                if not self.sim.pathfinder.load_nav_mesh(str(cache)):
                    raise RuntimeError(f"Invalid cached navmesh: {cache}")
            else:
                settings.agent_height, settings.agent_radius = agent.height, agent.radius
                if not self.sim.recompute_navmesh(self.sim.pathfinder, settings):
                    raise RuntimeError(f"Cannot build robot-specific navmesh: {scene}")
                temporary = cache.with_suffix(f".{os.getpid()}.navmesh")
                self.sim.pathfinder.save_nav_mesh(str(temporary))
                temporary.replace(cache)
        actual = self.sim.pathfinder.nav_mesh_settings
        if not (
            np.isclose(actual.agent_height, agent.height)
            and np.isclose(actual.agent_radius, agent.radius)
        ):
            raise RuntimeError("Navmesh collision geometry differs from robot configuration")

    def robot_configuration(self):
        agent = self.sim.get_agent(0).agent_config
        sensor = agent.sensor_specifications[0]
        navmesh = self.sim.pathfinder.nav_mesh_settings
        return {
            "width": int(sensor.resolution[1]),
            "height": int(sensor.resolution[0]),
            "hfov": float(sensor.hfov),
            "sensor_height": float(sensor.position[1]),
            "sensor_pitch_deg": float(np.rad2deg(sensor.orientation[0])),
            "agent_height": float(agent.height),
            "agent_radius": float(agent.radius),
            "navmesh_height": float(navmesh.agent_height),
            "navmesh_radius": float(navmesh.agent_radius),
        }

    def distance(self):
        path = habitat_sim.MultiGoalShortestPath()
        path.requested_start = self.sim.get_agent(0).get_state().position
        path.requested_ends = self.goal_positions
        if not self.sim.pathfinder.find_path(path):
            raise RuntimeError(
                f"Unreachable goal in {self.episode['scene_id']} episode {self.episode['episode_id']}"
            )
        self.closest_goal = self.goal_positions[path.closest_end_point_index]
        return float(path.geodesic_distance)

    def reset(self, episode):
        if episode["scene_id"] != self.scene:
            self._make_sim(episode["scene_id"])
        self.episode = episode
        self.sim.reset()
        state = habitat_sim.AgentState()
        state.position = episode["start_position"]
        state.rotation = quat_from_coeffs(episode["start_rotation"])
        self.sim.get_agent(0).set_state(state)
        self.follower.reset()
        self.goal_positions = np.asarray(
            [v["agent_state"]["position"] for g in episode["goals"] for v in g["view_points"]],
            dtype=np.float32,
        )
        if len(self.goal_positions) == 0:
            raise RuntimeError("ObjectNav episode has no goal viewpoints")
        self.steps, self.path_length, self.collisions = 0, 0.0, 0
        self.ended = False
        self.initial_distance = self.previous_distance = self.distance()
        self.min_distance = self.initial_distance
        self.oracle_goal = self.closest_goal.copy()
        self.oracle_recoveries = 0
        self.oracle_best_distance = self.initial_distance
        self.oracle_progress_step = 0
        warmup_steps = 0
        warm_distance = episode.get("metadata", {}).get("training_warm_start_distance")
        if warm_distance is not None:
            if episode["split"] != "train" or not 0.1 < warm_distance < 20:
                raise ValueError("Warm starts are allowed only for training, at 0.1–20 m")
            # Physical oracle steps move to an easier start, before any learner
            # observation/reward. Evaluation episodes never carry this metadata.
            while self.previous_distance > warm_distance and warmup_steps < 250:
                try:
                    action = self.oracle()
                except OracleUnavailableError as exc:
                    # Curriculum is optional. Restore the original legal start
                    # rather than letting a failed warmup terminate the learner.
                    metadata = dict(episode.get("metadata", {}))
                    metadata.pop("training_warm_start_distance", None)
                    image, info = self.reset({**episode, "metadata": metadata})
                    info["curriculum_warmup_steps"] = warmup_steps
                    info["curriculum_fallback"] = str(exc)
                    print(
                        json.dumps({"curriculum_fallback": str(exc), "episode": self.uid}),
                        flush=True,
                    )
                    return image, info
                if action == NavigationAction.STOP:
                    break
                self.step(action)
                warmup_steps += 1
                if self.ended:
                    break
            self.steps, self.path_length, self.collisions = 0, 0.0, 0
            self.ended = False
            self.initial_distance = self.previous_distance = self.distance()
            self.min_distance = self.initial_distance
            self.follower.reset()
            self.oracle_goal = self.closest_goal.copy()
            self.oracle_best_distance = self.initial_distance
            self.oracle_progress_step = 0
        return self._observation(), {
            "episode_id": self.uid,
            "goal_text": episode["goal_text"],
            "distance": self.initial_distance,
            "frame_id": 0,
            "timestamp_s": time.monotonic(),
            "robot": self.robot_configuration(),
            "curriculum_warmup_steps": warmup_steps,
        }

    @property
    def uid(self):
        e = self.episode
        return f"{e['dataset_id']}:{e['split']}:{e['scene_id']}:{e['episode_id']}"

    def _observation(self):
        return np.ascontiguousarray(self.sim.get_sensor_observations()["rgb"][:, :, :3])

    def oracle(self):
        if self.ended:
            raise RuntimeError("Reset required after episode termination")
        # reset()/step() already computed distance for this exact position.
        # Oracle queries never move the agent; avoid a duplicate multi-goal search.
        distance = self.previous_distance
        if distance < self.config.get("success_distance", 0.1):
            return NavigationAction.STOP
        if distance < self.oracle_best_distance - 0.02:
            self.oracle_best_distance = distance
            self.oracle_progress_step = self.steps
        elif self.steps - self.oracle_progress_step >= 64:
            raise OracleUnavailableError(
                f"Oracle made no geodesic progress for 64 steps; episode={self.uid}, distance={distance:.4f}"
            )
        try:
            # A stable target preserves the follower's anti-thrashing state.
            action = self.follower.next_action_along(self.oracle_goal)
        except habitat_sim.errors.GreedyFollowerError:
            action = None
            position = self.sim.get_agent(0).get_state().position
            nearest = np.argsort(np.linalg.norm(self.goal_positions - position, axis=1))[:64]
            for index in nearest:
                candidate = self.goal_positions[index]
                self.follower.reset()
                try:
                    proposed = self.follower.next_action_along(candidate)
                except habitat_sim.errors.GreedyFollowerError:
                    continue
                if proposed == NavigationAction.STOP:
                    continue
                self.oracle_goal = candidate.copy()
                action = proposed
                self.oracle_recoveries += 1
                print(
                    json.dumps(
                        {
                            "oracle_recovery": "alternate_goal_viewpoint",
                            "episode": self.uid,
                            "step": self.steps,
                            "distance": distance,
                        }
                    ),
                    flush=True,
                )
                break
            if action is None:
                raise OracleUnavailableError(
                    f"No valid follower action after 64 goal candidates; episode={self.uid}, step={self.steps}, distance={distance:.4f}"
                )
        mapping = {
            NavigationAction.STOP: NavigationAction.STOP,
            "move_forward": NavigationAction.MOVE_FORWARD,
            "turn_left": NavigationAction.TURN_LEFT,
            "turn_right": NavigationAction.TURN_RIGHT,
        }
        return mapping[action]

    def step(self, action):
        if self.ended:
            raise RuntimeError("Reset required after episode termination")
        action = NavigationAction(action)
        before = self.sim.get_agent(0).get_state().position.copy()
        stopped = action == NavigationAction.STOP
        sim_observations = {}
        if not stopped:
            sim_observations = self.sim.step(
                {
                    NavigationAction.MOVE_FORWARD: "move_forward",
                    NavigationAction.TURN_LEFT: "turn_left",
                    NavigationAction.TURN_RIGHT: "turn_right",
                }[action]
            )
        after = self.sim.get_agent(0).get_state().position
        moved = float(np.linalg.norm(after - before))
        collision = bool(sim_observations.get("collided", False))
        self.path_length += moved
        self.steps += 1
        self.collisions += int(collision)
        # Turning and fully blocked motion preserve geodesic distance.
        distance = self.distance() if moved > 0 else self.previous_distance
        success = stopped and distance < self.config.get("success_distance", 0.1)
        self.min_distance = min(self.min_distance, distance)
        truncated = self.steps >= self.config.get("max_episode_steps", 500) and not stopped
        self.ended = stopped or truncated
        reward = self.reward.compute(self.previous_distance, distance, success, collision, stopped)
        self.previous_distance = distance
        denom = max(self.initial_distance, self.path_length, 1e-8)
        efficiency = self.initial_distance / denom
        soft_success = max(0.0, 1.0 - distance / max(self.initial_distance, 1e-8))
        metrics = {
            "success": float(success),
            "success_strict_0_1": float(stopped and distance < 0.1),
            "oracle_success": float(self.min_distance < self.config.get("success_distance", 0.1)),
            "min_distance_to_goal": self.min_distance,
            "success_distance": self.config.get("success_distance", 0.1),
            "spl": success * efficiency,
            "soft_spl": soft_success * efficiency,
            "distance_to_goal": distance,
            "episode_length": self.steps,
            "collision_rate": self.collisions / self.steps,
            "path_length": self.path_length,
            "oracle_recoveries": self.oracle_recoveries,
        }
        # Simulator.step already rendered the post-action RGB. Rendering it a
        # second time adds cost to every learner and curriculum transition.
        rgb = (
            self._observation()
            if stopped
            else np.ascontiguousarray(sim_observations["rgb"][:, :, :3])
        )
        return rgb, {
            "episode_id": self.uid,
            "reward": reward,
            "done": self.ended,
            "terminated": stopped,
            "truncated": truncated,
            "collision": collision,
            "success": success,
            "geodesic_distance": distance,
            "metrics": metrics,
            "frame_id": self.steps,
            "timestamp_s": time.monotonic(),
        }

    def close(self):
        if self.sim is not None:
            self.sim.close()
