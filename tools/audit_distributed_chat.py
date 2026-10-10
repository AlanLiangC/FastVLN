"""Verify chat readout DDP unfreezing and full-parameter equality on real rollouts."""

import argparse
import hashlib
import json
from pathlib import Path

import torch
import yaml

from streamnav.training import distributed as parallel
from streamnav.training.trainer import EndToEndObjectNavTrainer


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--gpu-offset", type=int, default=4)
    args = parser.parse_args()
    config = yaml.safe_load(Path(args.config).read_text())
    config.update(run_dir=args.run_dir, checkpoint=None)
    config["distributed"]["gpu_offset"] = args.gpu_offset
    config["trainer"].update(num_envs=2, rollout_steps=2, sequence_length=2, sequence_batch_size=2)
    parallel.initialize(config)
    trainer = None
    try:
        trainer = EndToEndObjectNavTrainer(config)
        records = []
        for update in range(1, 4):
            buffer = trainer.collect_rollout()
            metrics = trainer.update(buffer)
            trainer.update_index = update
            trainer.scheduler.step()
            if update == trainer.cfg["actor_warmup_updates"]:
                trainer._configure_replay()
            trainer.dagger.step()
            digest = hashlib.sha256()
            for name, tensor in trainer.policy.state_dict().items():
                digest.update(name.encode())
                digest.update(tensor.detach().cpu().contiguous().numpy().tobytes())
            payload = {
                "rank": parallel.rank(),
                "update": update,
                "all_parameters_sha256": digest.hexdigest(),
                **{
                    key: metrics[key]
                    for key in (
                        "preupdate_replay_log_prob_error_max",
                        "kda_grad_norm",
                        "actor_grad_norm",
                        "critic_grad_norm",
                        "vision_grad_norm",
                    )
                },
            }
            ranks = parallel.gather(payload)
            assert len({r["all_parameters_sha256"] for r in ranks}) == 1
            assert all(r["preupdate_replay_log_prob_error_max"] == 0 for r in ranks)
            if update > 1:
                assert all(r["kda_grad_norm"] > 0 and r["actor_grad_norm"] > 0 for r in ranks)
            assert all(r["vision_grad_norm"] == 0 for r in ranks)
            records.append(ranks)
            if parallel.rank() == 0:
                print(json.dumps(ranks), flush=True)
            del buffer
        if parallel.rank() == 0:
            (Path(args.run_dir) / "audit.json").write_text(
                json.dumps({"protocol": __doc__, "updates": records}, indent=2) + "\n"
            )
    finally:
        if trainer is not None:
            trainer.envs.close()
        if torch.distributed.is_initialized():
            torch.distributed.destroy_process_group()


if __name__ == "__main__":
    main()
