from types import SimpleNamespace

import torch

from streamnav.contracts.state import LayerState, StreamingState
from streamnav.models.policy.action_distribution import ObjectNavActionDistribution
from streamnav.training.dagger import behavior_log_prob
from streamnav.training.rollout import RolloutCollector, replay_sequences


class TinyPolicy(torch.nn.Module):
    distribution = ObjectNavActionDistribution()

    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.tensor(1.0))

    def start_episode(self, uid, instruction):
        cache = (LayerState(None, self.weight.reshape(1, 1).clone()),)
        return StreamingState(cache, uid, instruction, 0, instruction)

    def forward_batch(self, rgb, states):
        values = torch.stack([s.kda_cache[0].recurrent.squeeze() for s in states])
        logits = torch.stack([values, values * 0, values * 0, values * 0], dim=-1)
        return logits, values, [s.with_cache(s.kda_cache) for s in states]


class BoundaryEnvs:
    def __init__(self):
        self.step_index = 0

    def observation(self):
        return {"rgb": torch.zeros(2, 2, 3, dtype=torch.uint8)}

    def reset(self, episodes):
        return [self.observation() for _ in episodes]

    def reset_at(self, episodes):
        return {i: self.observation() for i in episodes}

    def get_oracle_actions(self):
        return [1, 1]

    def step(self, actions):
        self.step_index += 1
        return [
            {
                **self.observation(),
                "reward": 0.0,
                "done": self.step_index == 1 and i == 0,
                "truncated": False,
                "collision": False,
                "metrics": {},
            }
            for i in range(2)
        ]


def test_reset_on_final_rollout_step_replays_under_current_prefill_weights():
    policy = TinyPolicy()
    source = SimpleNamespace(sample_episode=lambda: SimpleNamespace(uid="A", goal_text="chair"))
    collector = RolloutCollector(
        policy, BoundaryEnvs(), [source, source], {"rollout_steps": 1, "sequence_length": 1}
    )
    collector.collect(beta=0.0)
    assert [s.step_index for s in collector.states] == [0, 1]
    # The optimizer changes prefill weights after an episode reset on the last step.
    with torch.no_grad():
        policy.weight.add_(1.0)
    buffer = collector.collect(beta=0.0)
    sequences = next(buffer.sequence_batches(2, shuffle=False))
    logits, _ = replay_sequences(policy, buffer, sequences)
    replay_log_probs = behavior_log_prob(
        logits, buffer.executed_actions, buffer.expert_actions, buffer.beta
    )
    torch.testing.assert_close(replay_log_probs, buffer.old_log_probs, atol=1e-7, rtol=0)
    # The ongoing episode retains its memory; only the unused new prefix is refreshed.
    assert buffer.initial_states[(0, 0)].kda_cache[0].recurrent.item() == 2.0
    assert buffer.initial_states[(1, 0)].kda_cache[0].recurrent.item() == 1.0
