"""Training-only candidate routing. The actor receives only the sampled pixel token.

Binding is caused exclusively by the point actually delivered to the actor, never
by unseen detections on NULL frames. Annotation queries do not execute actions.
"""

from __future__ import annotations

import hashlib
import json
import random

import habitat_sim
import numpy as np
from executable_oracle import ExecutableNavigation
from habitat_sim.errors import GreedyFollowerError
from perception_labels import PerceptionLabeler

from streamnav.contracts.action import NavigationAction as A
from streamnav.contracts.perception import GRID_HEIGHT, GRID_WIDTH
from streamnav.training.object_hints import episode_hint_enabled, simulated_object_hint_packet


class HintConditionedSupervision:
    def __init__(self, env, config):
        self.env, self.config = env, config
        self.signature = json.dumps(config, sort_keys=True)
        self.memory_steps = config["conditioned_supervision"].get("memory_steps", 40)
        if type(self.memory_steps) is not int or self.memory_steps < 0:
            raise ValueError("Hint memory_steps must be a nonnegative integer")
        if not 0 <= config.get("episode_probability", 0.5) <= 1:
            raise ValueError("Hint episode_probability must be in [0, 1]")
        if env.config.get("object_label_source") != "semantic_instance":
            raise ValueError("Hint-conditioned supervision requires semantic instance labels")
        self.goal_id, self.target, self.navigator = None, None, None
        self.generation, self.last_hint_frame = 0, -1
        self.packet_frame, self.packet_cache = -1, None
        self.result_frame, self.result_cache = -1, None
        self.goals = {}
        for goal in env.episode["goals"]:
            identity = int(str(goal["object_id"]).rsplit("_", 1)[-1])
            self.goals.setdefault(identity, []).extend(
                view["agent_state"]["position"] for view in goal["view_points"]
            )

    def object_supervision(self):
        """Pure, repeatable hint query for value bootstrap; it never binds a goal."""
        if self.packet_frame != self.env.steps:
            label = PerceptionLabeler(self.env).object_labels()
            seed = self.config.get("seed", 2025)
            if episode_hint_enabled(
                self.env.uid, seed, self.config.get("episode_probability", 0.5)
            ):
                key = f"{seed}:{self.env.uid}:{self.env.steps}:training_hint"
                rng = random.Random(int(hashlib.sha256(key.encode()).hexdigest()[:16], 16))
                packet = simulated_object_hint_packet(label, rng, **self.config.get("noise", {}))
            else:
                packet = {"token": 0, "source": "episode_null"}
            self.packet_cache = {**label, "object_hint_token": packet["token"], "hint": packet}
            self.packet_frame = self.env.steps
        return self.packet_cache

    def _candidate(self, token):
        if not token:
            return None
        semantic = self.env._sensor_observations()["perception_semantic"]
        row, column = divmod(token - 1, GRID_WIDTH)
        height, width = semantic.shape
        x = min(int((column + 0.5) / GRID_WIDTH * width), width - 1)
        y = min(int((row + 0.5) / GRID_HEIGHT * height), height - 1)
        identity = int(semantic[y, x])
        return (
            identity if identity in self.goals and identity in self.env.semantic_known_ids else None
        )

    def _choose_target(self, identity):
        path = habitat_sim.MultiGoalShortestPath()
        path.requested_start = self.env.sim.get_agent(0).get_state().position
        path.requested_ends = np.asarray(self.goals[identity], dtype=np.float32)
        if not self.env.sim.pathfinder.find_path(path) or not np.isfinite(path.geodesic_distance):
            return None
        return np.array(path.requested_ends[path.closest_end_point_index], copy=True)

    def _release(self):
        self.goal_id, self.target = None, None
        if self.navigator is not None:
            self.navigator.follower.reset()

    def supervision(self):
        env = self.env
        if self.result_frame == env.steps:
            return self.result_cache
        packet = self.object_supervision()["hint"]
        candidate = self._candidate(packet["token"])
        fallback = None
        if self.goal_id is not None and env.steps - self.last_hint_frame > self.memory_steps:
            self._release()
        if candidate is not None:
            if candidate != self.goal_id:
                target = self._choose_target(candidate)
                if target is not None:
                    if self.navigator is None:
                        self.navigator = ExecutableNavigation(env)
                    self.navigator.follower.reset()
                    self.goal_id, self.target = candidate, target
                    self.generation += 1
                else:
                    fallback = "hint_candidate_unreachable"
            if candidate == self.goal_id:
                self.last_hint_frame = env.steps
        # The existing task success rule has priority over a local approach goal.
        ready = env.previous_distance < env.config.get("success_distance", 0.25)
        if ready:
            action = A.STOP
        elif self.goal_id is not None:
            try:
                action = A(self.navigator.follower.next_action_along(self.target))
                if action == A.STOP:
                    raise ValueError("Local follower STOP outside task success")
                if action == A.MOVE_FORWARD and not self.navigator.forward_executable():
                    raise ValueError("Hint follower requests blocked forward")
            except (GreedyFollowerError, ValueError) as exc:
                fallback = f"hint_route_unavailable: {exc}"
                self._release()
                action = env.oracle()
        else:
            action = env.oracle()
        labels = PerceptionLabeler(env).labels(
            action,
            navigation_target=self.target,
            goal_ids=None if self.goal_id is None else {self.goal_id},
        )
        self.result_cache = {
            "action": int(action),
            "perception": labels,
            "hint": {
                **packet,
                "bound_goal_id": self.goal_id,
                "active": self.goal_id is not None,
                "goal_key": (
                    f"hint:{self.goal_id}:{self.generation}"
                    if self.goal_id is not None
                    else "global"
                ),
                "fallback": fallback,
            },
        }
        self.result_frame = env.steps
        return self.result_cache
