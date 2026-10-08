import json
import os
from pathlib import Path

import pytest
import torch
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

from streamnav.evaluation.runner import evaluate
from streamnav.training.trainer import EndToEndObjectNavTrainer

pytestmark = [
    pytest.mark.habitat,
    pytest.mark.gpu,
    pytest.mark.skipif(
        os.environ.get("STREAMNAV_INTEGRATION") != "1" or not torch.cuda.is_available(),
        reason="Set STREAMNAV_INTEGRATION=1 with CUDA and real data",
    ),
]


def test_real_habitat_recurrent_joint_optimizer_step(tmp_path):
    with initialize_config_dir(version_base=None, config_dir=str(Path("configs").resolve())):
        cfg = compose(
            config_name="config",
            overrides=[
                "data=hm3d_v1",
                "eval=hm3d_v1",
                f"run_dir={tmp_path}",
                "trainer.num_envs=1",
                "trainer.rollout_steps=2",
                "trainer.sequence_length=2",
                "trainer.sequence_batch_size=1",
                "trainer.update_epochs=1",
                "trainer.ealm.enabled=false",
                "trainer.ealm.fixed_alpha=0.5",
            ],
        )
    trainer = EndToEndObjectNavTrainer(OmegaConf.to_container(cfg, resolve=True))
    try:
        before = trainer.policy.actor_critic.actor.weight.detach().clone()
        buffer = trainer.collect_rollout()
        metrics = trainer.update(buffer)
        assert not torch.equal(before, trainer.policy.actor_critic.actor.weight)
        assert metrics["ealm_alpha"] == pytest.approx(0.5)
        for name in ("vision_grad_norm", "kda_grad_norm", "actor_grad_norm", "critic_grad_norm"):
            assert metrics[name] > 0
        assert torch.isfinite(buffer.returns).all()
        assert buffer.observations.shape[:2] == (2, 1)
        assert buffer.observations.shape[2:] == (270, 480, 3)
        del buffer
        trainer.optimizer.zero_grad(set_to_none=True)
        trainer.config["eval"]["num_envs"] = 2
        batched = evaluate(trainer.policy, trainer.config, update=1, episodes=3, max_steps=3)
        trainer.config["eval"]["num_envs"] = 1
        serial = evaluate(trainer.policy, trainer.config, update=2, episodes=3, max_steps=3)
        for split in batched:
            assert batched[split]["episodes"] == serial[split]["episodes"] == 3
            assert batched[split]["evaluation_batch_size"] == 2
            assert batched[split]["state_bytes"] == serial[split]["state_bytes"]

            def records(update):
                path = tmp_path / "evaluation" / f"update_{update:07d}" / f"{split}_episodes.jsonl"
                return {r["episode_id"]: r for r in map(json.loads, path.read_text().splitlines())}

            assert records(1) == records(2)
        assert trainer.policy.training
    finally:
        trainer.envs.close()
