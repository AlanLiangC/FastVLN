import os
from dataclasses import replace
from pathlib import Path

import pytest
import torch
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

from streamnav.training.trainer import EndToEndObjectNavTrainer

pytestmark = [
    pytest.mark.gpu,
    pytest.mark.habitat,
    pytest.mark.skipif(
        os.environ.get("STREAMNAV_INTEGRATION") != "1" or not torch.cuda.is_available(),
        reason="Real Habitat and CUDA required",
    ),
]


def test_auxiliary_expert_stop_and_real_ppo_replay_are_separate(tmp_path):
    with initialize_config_dir(version_base=None, config_dir=str(Path("configs").resolve())):
        cfg = compose(
            config_name="config",
            overrides=[
                "model=qwen35_0p8b_kda_chat_temporal_contrast",
                "trainer=ovsegdt_kda_chat_ppo_auxiliary_il",
                "data=hm3d_v1",
                "eval=hm3d_v1",
                f"run_dir={tmp_path}",
                "trainer.num_envs=2",
                "trainer.rollout_steps=12",
                "trainer.sequence_length=12",
                "trainer.sequence_batch_size=2",
                "trainer.actor_warmup_updates=0",
            ],
        )
    trainer = EndToEndObjectNavTrainer(OmegaConf.to_container(cfg, resolve=True))
    sample = trainer.sources[0].sample_episode
    trainer.sources[0].sample_episode = lambda: replace(
        sample(), metadata={"training_warm_start_distance": 0.3}
    )
    try:
        buffer = trainer.collect_rollout()
        assert not buffer.ppo_mask[:, 0].any() and buffer.ppo_mask[:, 1].all()
        assert torch.count_nonzero(buffer.old_log_probs[:, 0]) == 0
        assert buffer.replay_log_probs is not None
        assert (buffer.expert_actions[:, 0] == 0).any()
        assert buffer.auxiliary_episode_metrics
        metrics = trainer.update(buffer)
        assert metrics["preupdate_replay_log_prob_error_max"] <= 0.05
        assert metrics["preupdate_clip_fraction"] == 0
        assert metrics["ppo_policy_coefficient"] == pytest.approx(0.2)
        assert metrics["ppo_eligible_fraction"] == 0.5
        assert metrics["policy_logit_grad_norm_ppo"] > 0
        assert metrics["actor_grad_norm"] > 0 and metrics["kda_grad_norm"] > 0
        assert metrics["vision_grad_norm"] == 0
    finally:
        trainer.envs.close()
