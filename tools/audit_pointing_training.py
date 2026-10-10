"""Full 100-frame real-Habitat DDP canary for perception and checkpoint consistency."""

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
    parser.add_argument("--updates", type=int, default=3)
    args = parser.parse_args()
    config = yaml.safe_load(Path(args.config).read_text())
    config["run_dir"] = args.run_dir
    config["trainer"].update(
        num_updates=args.updates,
        checkpoint_interval=10000,
        eval_interval=10000,
        early_eval_updates=[],
        eval_first_update=False,
    )
    parallel.initialize(config)
    trainer = None
    try:
        trainer = EndToEndObjectNavTrainer(config)
        assert config["trainer"]["rollout_steps"] == config["trainer"]["sequence_length"] == 100
        assert trainer.policy.perception is not None
        trainer.run()
        digest = hashlib.sha256()
        for name, tensor in trainer.policy.state_dict().items():
            digest.update(name.encode())
            digest.update(tensor.detach().cpu().contiguous().numpy().tobytes())
        hashes = parallel.gather({"rank": parallel.rank(), "sha256": digest.hexdigest()})
        assert len({r["sha256"] for r in hashes}) == 1, "DDP parameters diverged"
        if parallel.rank() == 0:
            rows = [
                json.loads(line)
                for line in (Path(args.run_dir) / "train_metrics.jsonl").read_text().splitlines()
            ]
            assert len(rows) == args.updates
            for row in rows:
                assert row["preupdate_replay_log_prob_error_max"] <= 0.05
                assert row["preupdate_clip_fraction"] == 0
                assert row["vision_grad_norm"] == 0
                assert row["perception_grad_norm"] > 0
                assert row["kda_grad_norm"] > 0 and row["actor_grad_norm"] > 0
            checkpoint = Path(args.run_dir) / "checkpoints/latest"
            assert (checkpoint / "perception.safetensors").is_file()
            report = {
                "scope": __doc__,
                "world_size": parallel.world_size(),
                "parameter_hashes": hashes,
                "metrics": rows,
                "checkpoint": str(checkpoint.resolve()),
                "passed": True,
            }
            (Path(args.run_dir) / "audit.json").write_text(json.dumps(report, indent=2) + "\n")
            print(
                json.dumps(
                    {
                        "passed": True,
                        "updates": args.updates,
                        "parameter_hashes": hashes,
                        "max_replay_error": max(
                            r["preupdate_replay_log_prob_error_max"] for r in rows
                        ),
                    }
                ),
                flush=True,
            )
        parallel.barrier()
    finally:
        if trainer is not None:
            trainer.envs.close()
        if torch.distributed.is_initialized():
            torch.distributed.destroy_process_group()


if __name__ == "__main__":
    main()
