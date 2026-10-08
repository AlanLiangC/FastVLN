from types import SimpleNamespace

import pytest
import torch

from streamnav.contracts.action import NavigationAction
from streamnav.contracts.state import LayerState, PolicyOutput, StreamingState
from streamnav.errors import OracleUnavailableError
from streamnav.models.policy.action_distribution import ObjectNavActionDistribution
from streamnav.training.rollout import RolloutCollector


class Policy:
    distribution = ObjectNavActionDistribution()

    def eval(self):
        return self

    def start_episode(self, uid, instruction):
        return StreamingState(
            (LayerState(None, torch.zeros(1, 2, 2)),), uid, instruction, 0, instruction
        )

    def forward_batch(self, rgb, states):
        return (
            torch.tensor([[0.0, 1.0, 0.0, 0.0]]),
            torch.tensor([7.0]),
            [s.with_cache(s.kda_cache) for s in states],
        )

    def forward_step(self, rgb, state):
        return PolicyOutput(torch.zeros(4), torch.tensor(7.0), state.with_cache(state.kda_cache))


class Source:
    def __init__(self):
        self.index = 0

    def sample_episode(self):
        self.index += 1
        return SimpleNamespace(uid=str(self.index), goal_text="Find a chair.")


class Envs:
    def __init__(self, first_terminal):
        self.clients = [self]
        self.calls, self.steps = 0, 0
        self.first_terminal = first_terminal

    def reset(self, episode):
        observation = {"rgb": torch.zeros(8, 8, 3, dtype=torch.uint8)}
        return [observation] if isinstance(episode, list) else observation

    def get_oracle_actions(self):
        self.calls += 1
        return (
            [OracleUnavailableError("fixture failure")]
            if self.calls == 2
            else [NavigationAction.MOVE_FORWARD]
        )

    def get_oracle_action(self):
        return NavigationAction.MOVE_FORWARD

    def reset_at(self, episodes):
        return {i: self.reset(episode) for i, episode in episodes.items()}

    def step(self, actions):
        self.steps += 1
        return [
            {
                "rgb": torch.zeros(8, 8, 3, dtype=torch.uint8),
                "reward": 1.0,
                "collision": False,
                "truncated": False,
                "done": self.first_terminal and self.steps == 1,
                "metrics": {},
            }
        ]


@pytest.mark.parametrize("first_terminal", [False, True])
def test_oracle_failure_bootstraps_only_its_own_episode(first_terminal):
    collector = RolloutCollector(
        Policy(), Envs(first_terminal), [Source()], {"rollout_steps": 2, "sequence_length": 2}
    )
    buffer = collector.collect(beta=0.8)
    assert buffer.dones[0, 0]
    assert buffer.timeout_bootstrap[0, 0].item() == (0.0 if first_terminal else 7.0)
    assert len(buffer.oracle_failures) == 1
    assert (0, 1) in buffer.resets
    assert buffer.expert_actions.eq(int(NavigationAction.MOVE_FORWARD)).all()
