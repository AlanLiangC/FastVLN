"""Opt-in discrete navigation repair for the training-only OVSegDT teacher."""

from __future__ import annotations

import numpy as np
from habitat_sim.errors import GreedyFollowerError
from habitat_sim.utils.common import quat_rotate_vector

from streamnav.contracts.action import NavigationAction
from streamnav.errors import OracleUnavailableError


class ExecutableNavigation:
    """Keep upstream targets/PIVOT, repair a blocked path with actual robot actions.

    A recovery remains active for the same target so the upstream heading rule
    cannot undo the follower's corrective turn on the next query. Path queries
    and the follower do not step the simulator or consume policy observations.
    """

    def __init__(self, env):
        self.env = env
        self.target = None
        self.repairs = 0
        self.follower = env.sim.make_greedy_follower(
            agent_id=0,
            goal_radius=min(
                env.config.get("success_distance", 0.25), env.config.get("forward_step", 0.25) / 2
            ),
            stop_key=NavigationAction.STOP,
            forward_key=NavigationAction.MOVE_FORWARD,
            left_key=NavigationAction.TURN_LEFT,
            right_key=NavigationAction.TURN_RIGHT,
        )

    def forward_executable(self):
        state = self.env.sim.get_agent(0).get_state()
        step = self.env.config.get("forward_step", 0.25)
        endpoint = state.position + quat_rotate_vector(state.rotation, np.array([0.0, 0.0, -step]))
        actual = np.asarray(self.env.sim.pathfinder.try_step_no_sliding(state.position, endpoint))
        # Match allow_sliding=False and the robot-specific baked navmesh.
        remaining = np.linalg.norm((actual - endpoint)[[0, 2]])
        return bool(np.isfinite(actual).all() and remaining <= step * 0.05)

    def action(self, target, upstream_action):
        upstream_action = NavigationAction(int(np.asarray(upstream_action).item()))
        if target is None:
            self.target = None
            return upstream_action
        if self.target is not None and not np.allclose(target, self.target, atol=1e-4):
            self.target = None
            self.follower.reset()
        blocked = upstream_action == NavigationAction.MOVE_FORWARD and not self.forward_executable()
        if blocked and self.target is None:
            self.target = np.array(target, copy=True)
            self.repairs += 1
        if self.target is None:
            return upstream_action
        try:
            proposed = self.follower.next_action_along(self.target)
        except GreedyFollowerError as exc:
            raise OracleUnavailableError(
                "Selected frontier has no executable discrete path"
            ) from exc
        if proposed == NavigationAction.STOP:
            # Reaching a frontier is not object success. Only upstream PIVOT
            # may label STOP. Give frontier selection back to the explorer.
            self.target = None
            if blocked:
                raise OracleUnavailableError(
                    "Reached frontier but upstream still requests blocked forward"
                )
            return upstream_action
        if proposed == NavigationAction.MOVE_FORWARD and not self.forward_executable():
            raise OracleUnavailableError("Discrete follower also requests blocked forward")
        return NavigationAction(proposed)
