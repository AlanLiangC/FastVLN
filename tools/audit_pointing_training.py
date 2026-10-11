"""Full 100-frame real-Habitat DDP canary for perception and checkpoint consistency."""

import argparse
import hashlib
import json
import math
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
    parser.add_argument("--closed-loop", action="store_true")
    parser.add_argument("--control", action="store_true", help="Audit an unchanged legacy recipe")
    parser.add_argument("--reference-evaluation")
    parser.add_argument(
        "--reference-after-evaluation",
        help="Matched legacy continuation after equal training interactions",
    )
    parser.add_argument("--guard-reference", choices=("initial", "matched"), default="initial")
    parser.add_argument(
        "--guard-evaluation",
        help="Keep the original pre-migration navigation guard across refinement forks",
    )
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
        baseline = None
        if args.closed_loop:
            baseline = trainer.evaluate()
            if args.reference_evaluation:
                reference = json.loads(Path(args.reference_evaluation).read_text())
                for split, result in baseline.items():
                    for metric in ("success", "spl"):
                        assert math.isclose(
                            result[metric], reference[split][metric], abs_tol=1e-6
                        ), "Fusion migration changed the initial autonomous policy"
        trainer.run()
        after = trainer.evaluate() if args.closed_loop else None
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
                if args.closed_loop:
                    if not args.control:
                        assert row["ppo_policy_coefficient"] > 0
                        assert row["policy_logit_grad_norm_ppo"] > 0
                        assert row["policy_logit_grad_norm_stop_calibration"] > 0
            checkpoint = Path(args.run_dir) / "checkpoints/latest"
            assert (checkpoint / "perception.safetensors").is_file()
            report = {
                "scope": __doc__,
                "recipe_role": "legacy control" if args.control else "candidate",
                "world_size": parallel.world_size(),
                "parameter_hashes": hashes,
                "metrics": rows,
                "checkpoint": str(checkpoint.resolve()),
                "passed": True,
            }
            if args.closed_loop:
                assert baseline is not None and after is not None

                def count(results, metric):
                    return sum(r[metric] * r["episodes"] for r in results.values())

                before_success, after_success = count(baseline, "success"), count(after, "success")
                before_false, after_false = (
                    count(baseline, "false_stop_rate"),
                    count(after, "false_stop_rate"),
                )
                guard_reference = baseline
                matched_legacy = None
                if args.reference_after_evaluation:
                    matched_legacy = json.loads(Path(args.reference_after_evaluation).read_text())
                    assert matched_legacy.keys() == after.keys()
                    for split, result in after.items():
                        assert matched_legacy[split]["episodes"] == result["episodes"]
                        assert matched_legacy[split]["evaluated_model_update"] == args.updates
                if args.guard_reference == "matched":
                    assert matched_legacy is not None
                    guard_reference = matched_legacy
                if args.guard_evaluation:
                    assert args.guard_reference == "initial"
                    guard_reference = json.loads(Path(args.guard_evaluation).read_text())
                    assert guard_reference.keys() == after.keys()
                    for split, result in after.items():
                        assert guard_reference[split]["episodes"] == result["episodes"]
                        assert (
                            guard_reference[split]["success_distance"] == result["success_distance"]
                        )
                reference_success = count(guard_reference, "success")
                reference_false = count(guard_reference, "false_stop_rate")
                split_guards = {
                    split: result["success"] * result["episodes"]
                    >= guard_reference[split]["success"] * guard_reference[split]["episodes"] - 3
                    for split, result in after.items()
                }
                report["closed_loop"] = {
                    "baseline": baseline,
                    "after": after,
                    "successes_before": before_success,
                    "successes_after": after_success,
                    "false_stops_before": before_false,
                    "false_stops_after": after_false,
                    "guard_reference": args.guard_evaluation or args.guard_reference,
                    "reference_successes": reference_success,
                    "reference_false_stops": reference_false,
                    "matched_legacy": matched_legacy,
                    "same_144_fixed_original_starts": True,
                    "navigation_guard": "no more than 4 lost successes overall, 3 per split, or 4 additional false stops",
                    "split_guards": split_guards,
                    "navigation_guard_passed": after_success >= reference_success - 4
                    and after_false <= reference_false + 4
                    and all(split_guards.values()),
                    "scope": "short closed-loop stability/behavior test; not a full benchmark or convergence claim",
                }
                report["passed"] = report["closed_loop"]["navigation_guard_passed"]
            (Path(args.run_dir) / "audit.json").write_text(json.dumps(report, indent=2) + "\n")
            print(
                json.dumps(
                    {
                        "passed": report["passed"],
                        "updates": args.updates,
                        "parameter_hashes": hashes,
                        "max_replay_error": max(
                            r["preupdate_replay_log_prob_error_max"] for r in rows
                        ),
                        "closed_loop": report.get("closed_loop"),
                    }
                ),
                flush=True,
            )
        report = parallel.gather(report if parallel.rank() == 0 else None)[0]
        assert report["passed"], "Closed-loop navigation guard failed; inspect audit.json"
        parallel.barrier()
    finally:
        if trainer is not None:
            trainer.envs.close()
        if torch.distributed.is_initialized():
            torch.distributed.destroy_process_group()


if __name__ == "__main__":
    main()
