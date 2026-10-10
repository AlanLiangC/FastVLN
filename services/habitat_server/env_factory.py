"""Real Habitat-Sim ObjectNav with official episode viewpoints and a geodesic oracle.

Runs under Python 3.9, independently of the learner's torch environment.
"""

from __future__ import annotations

import fcntl
import hashlib
import json
import os
import random
import time
from pathlib import Path

import habitat_sim
import numpy as np
from habitat_sim.utils.common import quat_from_coeffs

from streamnav.contracts.action import NavigationAction
from streamnav.errors import OracleUnavailableError
from streamnav.training.oracle_progress import OracleProgressTracker
from streamnav.training.rewards import ObjectNavReward, RewardConfig


class HabitatObjectNavEnv:
    def __init__(self, config):
        self.config = config
        random.seed(config.get("seed", 2025))
        np.random.seed(config.get("seed", 2025))
        self.sim = None
        self.scene = None
        self.reward = ObjectNavReward(RewardConfig(**config.get("reward", {})))
        self.explorer = None
        self._cached_observations = None

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
        if self.config.get("perception_labels", False):
            depth = habitat_sim.CameraSensorSpec()
            depth.uuid = "perception_depth"
            depth.sensor_type = habitat_sim.SensorType.DEPTH
            depth.resolution = list(sensor.resolution)
            depth.position = list(sensor.position)
            depth.orientation = list(sensor.orientation)
            depth.hfov = sensor.hfov
            depth.near, depth.far = 0.05, 20.0
            agent.sensor_specifications.append(depth)
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
            "look_up": habitat_sim.agent.ActionSpec(
                "look_up",
                habitat_sim.agent.ActuationSpec(amount=self.config.get("tilt_angle", 30)),
            ),
            "look_down": habitat_sim.agent.ActionSpec(
                "look_down",
                habitat_sim.agent.ActuationSpec(amount=self.config.get("tilt_angle", 30)),
            ),
        }
        self.sim = habitat_sim.Simulator(habitat_sim.Configuration(sim_cfg, [agent]))
        self.sim.seed(self.config.get("seed", 2025))
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
        expected = {
            "agent_height": agent.height,
            "agent_radius": agent.radius,
            "agent_max_climb": self.config.get("navmesh_agent_max_climb", 0.10),
            "cell_height": self.config.get("navmesh_cell_height", 0.05),
            "include_static_objects": False,
        }
        if all(np.isclose(getattr(settings, key), value) for key, value in expected.items()):
            return
        source = Path(scene).with_suffix(".navmesh")
        signature = hashlib.sha256(source.read_bytes())
        signature.update(
            json.dumps({**expected, "sim_version": "habitat-0.3.3"}, sort_keys=True).encode()
        )
        root = Path(self.config.get("navmesh_cache", "runtime/cache/navmesh"))
        root.mkdir(parents=True, exist_ok=True)
        cache = root / f"{signature.hexdigest()}.navmesh"
        with (root / f"{signature.hexdigest()}.lock").open("w") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            if cache.exists():
                if not self.sim.pathfinder.load_nav_mesh(str(cache)):
                    raise RuntimeError(f"Invalid cached navmesh: {cache}")
            else:
                settings = habitat_sim.NavMeshSettings()
                settings.set_defaults()
                for key, value in expected.items():
                    setattr(settings, key, value)
                if not self.sim.recompute_navmesh(self.sim.pathfinder, settings):
                    raise RuntimeError(f"Cannot build robot-specific navmesh: {scene}")
                temporary = cache.with_suffix(f".{os.getpid()}.navmesh")
                self.sim.pathfinder.save_nav_mesh(str(temporary))
                temporary.replace(cache)
        actual = self.sim.pathfinder.nav_mesh_settings
        if not all(np.isclose(getattr(actual, key), value) for key, value in expected.items()):
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
            "navmesh_agent_max_climb": float(navmesh.agent_max_climb),
            "navmesh_cell_height": float(navmesh.cell_height),
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
        self._cached_observations = None
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
        self.oracle_progress = OracleProgressTracker(self.initial_distance)
        if self.config.get("oracle", "greedy") == "objnav_explorer":
            from ovsegdt_oracle import OVSegDTOracle

            self.explorer = OVSegDTOracle(self)
        warmup_steps = 0
        warm_distance = episode.get("metadata", {}).get("training_warm_start_distance")
        if warm_distance is not None:
            if episode["split"] != "train" or not 0.1 < warm_distance < 20:
                raise ValueError("Warm starts are allowed only for training, at 0.1–20 m")
            # Physical oracle steps move to an easier start, before any learner
            # observation/reward. Evaluation episodes never carry this metadata.
            try:
                while self.previous_distance > warm_distance and warmup_steps < 250:
                    action = self.oracle()
                    if action == NavigationAction.STOP:
                        break
                    self.step(action)
                    warmup_steps += 1
                    if self.ended:
                        break
                if self.distance() > warm_distance:
                    raise OracleUnavailableError(
                        f"Warmup did not reach {warm_distance:.3f} m after {warmup_steps} steps"
                    )
            except OracleUnavailableError as exc:
                # Failed or exhausted warmups must not silently publish an
                # intermediate pose as a successful near-goal training start.
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
            self.steps, self.path_length, self.collisions = 0, 0.0, 0
            self.ended = False
            self.initial_distance = self.previous_distance = self.distance()
            self.min_distance = self.initial_distance
            self.follower.reset()
            self.oracle_goal = self.closest_goal.copy()
            self.oracle_progress = OracleProgressTracker(self.initial_distance)
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

    def _sensor_observations(self):
        if self._cached_observations is None:
            self._cached_observations = self.sim.get_sensor_observations()
        return self._cached_observations

    def _observation(self):
        return np.ascontiguousarray(self._sensor_observations()["rgb"][:, :, :3])

    def oracle_supervision(self):
        if not self.config.get("perception_labels", False) or self.episode["split"] != "train":
            raise ValueError("Perception labels require opt-in training episodes")
        from perception_labels import PerceptionLabeler

        action = self.oracle()
        return {"action": int(action), "perception": PerceptionLabeler(self).labels(action)}

    def oracle(self):
        if self.ended:
            raise RuntimeError("Reset required after episode termination")
        if self.explorer is not None:
            # Exploration can legitimately move away from the object. Do not
            # apply the old greedy follower's distance-progress heuristic.
            return self.explorer.action()
        # reset()/step() already computed distance for this exact position.
        # Oracle queries never move the agent; avoid a duplicate multi-goal search.
        distance = self.previous_distance
        if distance < self.config.get("success_distance", 0.1):
            return NavigationAction.STOP
        if self.oracle_progress.stalled:
            raise OracleUnavailableError(
                f"Oracle made no geodesic progress for 64 followed advice steps; episode={self.uid}, distance={distance:.4f}"
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
        advice = mapping[action]
        self.oracle_progress.record_advice(advice)
        return advice

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
                    NavigationAction.LOOK_UP: "look_up",
                    NavigationAction.LOOK_DOWN: "look_down",
                }[action]
            )
            self._cached_observations = sim_observations
        after = self.sim.get_agent(0).get_state().position
        moved = float(np.linalg.norm(after - before))
        collision = bool(sim_observations.get("collided", False))
        self.path_length += moved
        self.steps += 1
        self.collisions += int(collision)
        # Turning and fully blocked motion preserve geodesic distance.
        distance = self.distance() if moved > 0 else self.previous_distance
        self.oracle_progress.record_step(action, distance)
        success = stopped and distance < self.config.get("success_distance", 0.1)
        self.min_distance = min(self.min_distance, distance)
        truncated = self.steps >= self.config.get("max_episode_steps", 500) and not stopped
        self.ended = stopped or truncated
        reward = self.reward.compute(self.previous_distance, distance, success, collision, stopped)
        self.previous_distance = distance
        denom = max(self.initial_distance, self.path_length, 1e-8)
        efficiency = self.initial_distance / denom
        soft_success = max(0.0, 1.0 - distance / max(self.initial_distance, 1e-8))
        navigation = getattr(self.explorer, "navigation", None)
        repairs = navigation.repairs if navigation is not None else 0
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
            "oracle_navigation_repairs": repairs,
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
            "displacement": moved,
            "oracle_navigation_repairs": repairs,
            "success": success,
            "geodesic_distance": distance,
            "metrics": metrics,
            "frame_id": self.steps,
            "timestamp_s": time.monotonic(),
        }

    def close(self):
        if self.sim is not None:
            self.sim.close()
