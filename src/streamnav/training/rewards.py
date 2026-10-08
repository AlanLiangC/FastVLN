from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass
class RewardConfig:
    success_reward: float = 5.0
    slack_penalty: float = 0.001
    progress_scale: float = 0.0
    collision_penalty: float = 0.003
    false_stop_penalty: float = 0.0


class ObjectNavReward:
    def __init__(self, config=RewardConfig()):
        self.config = config

    def compute(self, previous, current, success, collision, stopped=False):
        if not math.isfinite(previous) or not math.isfinite(current):
            raise ValueError("Unreachable navigation goal: nonfinite geodesic distance")
        c = self.config
        return (
            c.progress_scale * (previous - current)
            + c.success_reward * success
            - c.slack_penalty
            - c.collision_penalty * collision
            - c.false_stop_penalty * (stopped and not success)
        )
