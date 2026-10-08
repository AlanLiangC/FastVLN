import os
from dataclasses import replace

import pytest
import torch

from streamnav.contracts.state import LayerState, StreamingState
from streamnav.data.schema import HabitatEpisodeSource
from streamnav.envs.habitat_client import VectorHabitatEnvs
from streamnav.models.policy.action_distribution import ObjectNavActionDistribution
from streamnav.training.rollout import RolloutCollector

pytestmark = pytest.mark.skipif(
    os.environ.get("STREAMNAV_INTEGRATION") != "1", reason="Real Habitat required"
)


def test_parallel_episode_resets_preserve_real_rollout():
    episode = next(
        HabitatEpisodeSource("runtime/data/manifests/hm3d_v1_train.json").evaluation_episodes(1)
    )
    episode = replace(episode, metadata={"training_warm_start_distance": 0.3})

    class Source:
        def sample_episode(self):
            return episode

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
                torch.tensor([[0.0, 1.0, 0.0, 0.0]]).repeat(len(states), 1),
                torch.zeros(len(states)),
                [s.with_cache(s.kda_cache) for s in states],
            )

    envs = VectorHabitatEnvs({"gpu_device_id": 0, "success_distance": 0.25}, 2)
    try:

        def collect():
            torch.manual_seed(42)
            collector = RolloutCollector(
                Policy(), envs, [Source(), Source()], {"rollout_steps": 8, "sequence_length": 4}
            )
            return collector.collect(beta=1.0)

        parallel = collect()
        envs.reset_at = lambda episodes: {
            i: envs.clients[i].reset(episode) for i, episode in episodes.items()
        }
        serial = collect()
        assert parallel.episode_metrics and parallel.episode_metrics == serial.episode_metrics
        assert parallel.resets == serial.resets
        for key in (
            "observations",
            "actions",
            "expert_actions",
            "rewards",
            "dones",
            "old_log_probs",
        ):
            torch.testing.assert_close(getattr(parallel, key), getattr(serial, key), rtol=0, atol=0)
    finally:
        envs.close()
