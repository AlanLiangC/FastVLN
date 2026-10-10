import os
from pathlib import Path

import pytest
import torch
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

from streamnav.training.trainer import EndToEndObjectNavTrainer

pytestmark = [
    pytest.mark.habitat,
    pytest.mark.gpu,
    pytest.mark.skipif(
        os.environ.get("STREAMNAV_INTEGRATION") != "1" or not torch.cuda.is_available(),
        reason="Set STREAMNAV_INTEGRATION=1 with CUDA and real data",
    ),
]


@pytest.mark.parametrize("batch_chat_body", [False, True])
def test_packed_real_rollout_keeps_probabilities_and_all_trainable_branches(
    tmp_path, batch_chat_body
):
    with initialize_config_dir(version_base=None, config_dir=str(Path("configs").resolve())):
        cfg = compose(
            config_name="config",
            overrides=[
                "model=qwen35_0p8b_kda_chat_temporal_contrast",
                "trainer=ovsegdt_kda_chat_temporal_fast",
                "data=hm3d_v1",
                "eval=hm3d_v1",
                f"run_dir={tmp_path}",
                "trainer.num_envs=2",
                "trainer.rollout_steps=12",
                "trainer.sequence_length=12",
                "trainer.sequence_batch_size=2",
                "trainer.actor_warmup_updates=0",
                f"trainer.batch_chat_body={str(batch_chat_body).lower()}",
            ],
        )
    trainer = EndToEndObjectNavTrainer(OmegaConf.to_container(cfg, resolve=True))
    try:
        metrics = trainer.update(trainer.collect_rollout())
        assert metrics["preupdate_replay_log_prob_error_max"] <= 0.05
        assert metrics["preupdate_clip_fraction"] == 0
        assert metrics["vision_grad_norm"] == 0
        assert metrics["kda_grad_norm"] > 0
        assert metrics["actor_grad_norm"] > 0
        assert metrics["critic_grad_norm"] > 0
    finally:
        trainer.envs.close()
