from dataclasses import dataclass
from typing import Protocol

from streamnav.contracts.action import NavigationAction
from streamnav.contracts.observation import Observation


@dataclass
class EnvReset:
    observation: Observation
    episode_id: str
    goal_text: str


@dataclass
class EnvStep:
    observation: Observation
    reward: float
    done: bool
    success: bool
    geodesic_distance: float | None
    collision: bool | None
    terminated: bool = False
    truncated: bool = False


class ObjectNavEnv(Protocol):
    def reset(self) -> EnvReset: ...
    def step(self, action: NavigationAction) -> EnvStep: ...
    def get_oracle_action(self) -> NavigationAction: ...
