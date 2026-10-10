"""Resume an isolated, finite speed canary using the exact saved training recipe."""

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
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--updates", type=int, default=5)
    parser.add_argument("--pack-frames", type=int, default=64)
    parser.add_argument("--batch-chat-body", action="store_true")
    parser.add_argument("--no-unused-parameter-search", action="store_true")
    parser.add_argument("--preflight", action="store_true")
    parser.add_argument("--force-first-preflight-rejection-rank", type=int, default=-1)
    args = parser.parse_args()
    checkpoint = Path(args.checkpoint).resolve()
    root = Path(args.run_dir).resolve()
    if checkpoint.parent.parent == root:
        parser.error("Speed canaries must use a separate run directory")
    if args.updates < 1:
        parser.error("Updates must be positive")
    if args.force_first_preflight_rejection_rank >= 0 and not args.preflight:
        parser.error("Synthetic preflight rejection requires --preflight")
    config = yaml.safe_load((checkpoint / "resolved_config.yaml").read_text())
    start_update = json.loads((checkpoint / "manifest.json").read_text())["update"]
    config.update(checkpoint=str(checkpoint), run_dir=str(root))
    config["trainer"].update(
        num_updates=start_update + args.updates,
        replay_pack_frames=args.pack_frames,
        batch_chat_body=args.batch_chat_body,
        ddp_find_unused_parameters=not args.no_unused_parameter_search,
        replay_preflight=args.preflight,
        eval_interval=100000000,
        early_eval_updates=[],
        checkpoint_interval=100000000,
        verify_data_hashes=False,
    )
    parallel.initialize(config)
    try:
        if args.force_first_preflight_rejection_rank >= parallel.world_size():
            raise ValueError("Synthetic rejection rank is outside the process group")
        trainer = EndToEndObjectNavTrainer(config)
        injected = [False]
        if parallel.rank() == args.force_first_preflight_rejection_rank:
            original_losses = trainer.compute_losses

            def synthetic_rejection(*inputs):
                loss, metrics = original_losses(*inputs)
                if not injected[0]:
                    # Exercise only the preflight decision. The actual logits,
                    # loss, labels and optimizer remain unmodified.
                    metrics["replay_log_prob_error_max"] = 0.1
                    injected[0] = True
                return loss, metrics

            trainer.compute_losses = synthetic_rejection
        trainer.run()
        digest = hashlib.sha256()
        for name, parameter in trainer.policy.named_parameters():
            digest.update(name.encode())
            digest.update(parameter.detach().cpu().contiguous().numpy().tobytes())
        hashes = parallel.gather(digest.hexdigest())
        injected_by_rank = parallel.gather(injected[0])
        if len(set(hashes)) != 1:
            raise RuntimeError("Rank parameters diverged during the speed canary")
        if parallel.rank() == 0:
            (root / "canary_verification.json").write_text(
                json.dumps(
                    {
                        "source_checkpoint": str(checkpoint),
                        "start_update": start_update,
                        "end_update": trainer.update_index,
                        "pack_frames": args.pack_frames,
                        "batch_chat_body": args.batch_chat_body,
                        "ddp_find_unused_parameters": not args.no_unused_parameter_search,
                        "replay_preflight": args.preflight,
                        "synthetic_preflight_rejection_rank": args.force_first_preflight_rejection_rank,
                        "synthetic_rejection_observed_by_rank": injected_by_rank,
                        "rank_parameter_sha256": hashes,
                        "all_rank_parameters_identical": True,
                        "main_training_modified": False,
                        "evaluation_disabled_for_speed_comparison": True,
                    },
                    indent=2,
                )
                + "\n"
            )
    finally:
        if torch.distributed.is_initialized():
            torch.distributed.destroy_process_group()


if __name__ == "__main__":
    main()
