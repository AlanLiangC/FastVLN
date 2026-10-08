import os
from pathlib import Path

import pytest
import torch
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

from streamnav.training.rollout import replay_sequences
from streamnav.training.trainer import EndToEndObjectNavTrainer

pytestmark = [
    pytest.mark.habitat,
    pytest.mark.gpu,
    pytest.mark.skipif(
        os.environ.get("STREAMNAV_INTEGRATION") != "1" or not torch.cuda.is_available(),
        reason="Set STREAMNAV_INTEGRATION=1 with CUDA and real data",
    ),
]


def test_frozen_visual_cache_preserves_real_replay_and_optimizer(tmp_path):
    with initialize_config_dir(version_base=None, config_dir=str(Path("configs").resolve())):
        cfg = compose(
            config_name="config",
            overrides=[
                "model=qwen35_0p8b_kda_stable",
                "trainer=ovsegdt_kda_cached",
                "data=hm3d_v1",
                "eval=hm3d_v1",
                f"run_dir={tmp_path}",
                "trainer.num_envs=2",
                "trainer.rollout_steps=2",
                "trainer.sequence_length=2",
                "trainer.sequence_batch_size=2",
                "trainer.actor_warmup_updates=0",
            ],
        )
    trainer = EndToEndObjectNavTrainer(OmegaConf.to_container(cfg, resolve=True))
    try:
        buffer = trainer.collect_rollout()
        assert buffer.visual_embeddings is not None
        sequences = next(buffer.sequence_batches(2, shuffle=False))
        with torch.no_grad(), trainer.autocast():
            cached = replay_sequences(trainer.policy, buffer, sequences)[:2]
            visual = buffer.visual_embeddings
            buffer.visual_embeddings = None
            uncached = replay_sequences(trainer.policy, buffer, sequences)[:2]
            buffer.visual_embeddings = visual
            for actual, expected in zip(cached, uncached, strict=True):
                torch.testing.assert_close(actual, expected, atol=0, rtol=0)
        metrics = trainer.update(buffer)
        assert metrics["preupdate_replay_log_prob_error_max"] == 0
        assert metrics["vision_grad_norm"] == 0
        assert metrics["kda_grad_norm"] > 0 and metrics["actor_grad_norm"] > 0
        # A cache cannot survive a change to which visual weights are trainable.
        parameter = next(trainer.policy.backbone.vision.parameters())
        parameter.requires_grad_(True)
        with pytest.raises(ValueError, match="after unfreezing vision"):
            replay_sequences(trainer.policy, buffer, sequences)
        parameter.requires_grad_(False)
    finally:
        trainer.envs.close()
