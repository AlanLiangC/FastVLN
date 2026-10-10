"""Matched-state training recipe experiments, preserving the source checkpoint."""

import argparse
import hashlib
import json
from pathlib import Path

import torch
import yaml

from streamnav.training import distributed as parallel
from streamnav.training.trainer import EndToEndObjectNavTrainer

RECIPES = (
    "control",
    "ppo_floor",
    "auxiliary_il",
    "ppo_auxiliary_il",
    "backbone_eps",
    "executable_teacher",
)


def recipe_config(checkpoint, run_dir, recipe, updates, eval_interval=100):
    checkpoint, run_dir = Path(checkpoint).resolve(), Path(run_dir).resolve()
    config = yaml.safe_load((checkpoint / "resolved_config.yaml").read_text())
    start = json.loads((checkpoint / "manifest.json").read_text())["update"]
    if run_dir == Path(config["run_dir"]).resolve():
        raise ValueError("Recipe experiments must use a separate run directory")
    if recipe not in RECIPES or updates < 1 or eval_interval < 0:
        raise ValueError("Invalid training recipe, update count or evaluation interval")
    config.update(checkpoint=str(checkpoint), run_dir=str(run_dir))
    trainer = config["trainer"]
    trainer.update(
        num_updates=start + updates,
        replay_pack_frames=64,
        batch_chat_body=True,
        ddp_find_unused_parameters=False,
        replay_preflight=True,
        log_policy_gradient_terms=True,
        eval_interval=eval_interval or 100000000,
        early_eval_updates=[],
        eval_first_update=False,
        checkpoint_interval=10 if eval_interval else 100000000,
        keep_checkpoints=2,
        verify_data_hashes=False,
        fork_recipe_changes=[],
    )
    if recipe in ("ppo_floor", "ppo_auxiliary_il"):
        trainer["ealm"]["minimum_ppo_weight"] = 0.2
        trainer["fork_recipe_changes"].append("ealm")
    if recipe in ("auxiliary_il", "ppo_auxiliary_il"):
        trainer["auxiliary_il"] = {
            "num_envs": 1,
            "expert_episode_period": 2,
            "alternating_turn_limit": 8,
            "consecutive_turn_limit": 16,
            "recovery_steps": 32,
        }
        trainer["fork_recipe_changes"].append("auxiliary_il")
    if recipe == "backbone_eps":
        trainer["backbone_optimizer_eps"] = 1e-6
        trainer["fork_recipe_changes"].append("backbone_optimizer_eps")
    if recipe == "executable_teacher":
        if config["habitat"]["oracle"] != "objnav_explorer":
            raise ValueError("Executable teacher repair requires ObjNavExplorer")
        config["habitat"]["oracle_execution"] = "collision_safe"
        trainer["filter_blocked_forward_labels"] = True
        trainer["auxiliary_il"] = {"num_envs": 0}
        trainer["fork_recipe_changes"] += [
            "oracle_execution",
            "filter_blocked_forward_labels",
            "auxiliary_il",
        ]
    return config, start


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--recipe", choices=RECIPES, required=True)
    parser.add_argument("--updates", type=int, required=True)
    parser.add_argument("--eval-interval", type=int, default=100)
    parser.add_argument("--label", help="Descriptive label when continuing a saved recipe")
    args = parser.parse_args()
    config, start = recipe_config(
        args.checkpoint, args.run_dir, args.recipe, args.updates, args.eval_interval
    )
    parallel.initialize(config)
    try:
        trainer = EndToEndObjectNavTrainer(config)
        # Capture identical source state before collection changes the RNG or
        # samplers. Hashes avoid writing large duplicate tensor snapshots.
        source_rng = torch.random.get_rng_state().numpy().tobytes()
        source_cuda_rng = torch.cuda.get_rng_state(trainer.device).cpu().numpy().tobytes()
        source_sampler = json.dumps([s.state_dict() for s in trainer.sources], sort_keys=True)
        source = parallel.gather(
            {
                "rank": parallel.rank(),
                "torch_cpu_rng_sha256": hashlib.sha256(source_rng).hexdigest(),
                "torch_cuda_rng_sha256": hashlib.sha256(source_cuda_rng).hexdigest(),
                "sampler_state_sha256": hashlib.sha256(source_sampler.encode()).hexdigest(),
                "entropy_ema": trainer.mixer.entropy_ema.item(),
                "optimizer_step_min": min(
                    float(s["step"]) for s in trainer.optimizer.state.values()
                ),
            }
        )
        trainer.run()
        digest = hashlib.sha256()
        for name, parameter in trainer.policy.named_parameters():
            digest.update(name.encode())
            digest.update(parameter.detach().cpu().contiguous().numpy().tobytes())
        hashes = parallel.gather(digest.hexdigest())
        if len(set(hashes)) != 1:
            raise RuntimeError("Rank parameters diverged during the recipe experiment")
        if parallel.rank() == 0:
            (Path(args.run_dir) / "recipe_verification.json").write_text(
                json.dumps(
                    {
                        "recipe": args.label or args.recipe,
                        "source_checkpoint": str(Path(args.checkpoint).resolve()),
                        "start_update": start,
                        "end_update": trainer.update_index,
                        "expected_end_update": start + args.updates,
                        "source_rank_states": source,
                        "rank_parameter_sha256": hashes,
                        "all_rank_parameters_identical": True,
                        "source_checkpoint_modified": False,
                        "evaluation_enabled": args.eval_interval > 0,
                        "optimizer_groups": [
                            {k: group[k] for k in ("role", "lr", "eps")}
                            for group in trainer.optimizer.param_groups
                        ],
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
