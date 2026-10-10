"""Run short, full-size eight-GPU checks before launching recipe trials."""

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path

if __package__:
    from .benchmark_training_recipes import RECIPES
else:
    from benchmark_training_recipes import RECIPES


def verify(run_dir, recipe):
    state = json.loads((run_dir / "recipe_verification.json").read_text())
    rows = [json.loads(line) for line in (run_dir / "train_metrics.jsonl").read_text().splitlines()]
    assert state["end_update"] == state["expected_end_update"], state
    assert len(state["rank_parameter_sha256"]) == 8
    assert len(set(state["rank_parameter_sha256"])) == 1
    assert len(rows) == 2
    for row in rows:
        assert row["global_transitions"] == 3200
        assert row["preupdate_replay_log_prob_error_max"] <= 0.05
        assert row["preupdate_clip_fraction"] == 0
        assert row["vision_grad_norm"] == 0
        assert row["actor_grad_norm"] > 0 and row["kda_grad_norm"] > 0
        if recipe in ("ppo_floor", "ppo_auxiliary_il"):
            assert abs(row["ppo_policy_coefficient"] - 0.2) < 1e-6
            assert row["policy_logit_grad_norm_ppo"] > 0
        if recipe in ("auxiliary_il", "ppo_auxiliary_il"):
            assert row["on_policy_transitions"] == 2400
            assert row["auxiliary_il_transitions"] == 800
            assert row["auxiliary_expert_steps"] > 0
    return {"verification": state, "metrics": rows}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--recipes", nargs="+", choices=RECIPES, default=list(RECIPES))
    args = parser.parse_args()
    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    summary = {}
    source = None
    for recipe in args.recipes:
        run_dir = output / recipe
        run_dir.mkdir()
        command = [
            sys.executable,
            "-m",
            "torch.distributed.run",
            "--standalone",
            "--nnodes=1",
            "--nproc_per_node=8",
            "tools/benchmark_training_recipes.py",
            "--checkpoint",
            str(Path(args.checkpoint).resolve()),
            "--run-dir",
            str(run_dir),
            "--recipe",
            recipe,
            "--updates",
            "2",
            "--eval-interval",
            "0",
        ]
        with (run_dir / "training.log").open("w") as log:
            process = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT)
            (run_dir / "training.pid").write_text(str(process.pid) + "\n")
            print(json.dumps({"canary_started": recipe, "pid": process.pid}), flush=True)
            code = process.wait()
        if code:
            raise RuntimeError(f"{recipe} exited with {code}; inspect {run_dir / 'training.log'}")
        request = run_dir / "stop_request.json"
        if request.exists() and json.loads(request.read_text()).get("training_pid") == process.pid:
            raise InterruptedError("Manual stop requested: no further GPU tests will start")
        result = verify(run_dir, recipe)
        current_source = result["verification"]["source_rank_states"]
        if source is None:
            source = current_source
        assert current_source == source, "Source optimizer/RNG/samplers differ between recipes"
        groups = result["verification"]["optimizer_groups"]
        for group in groups:
            expected = 1e-6 if recipe == "backbone_eps" and group["role"] == "backbone" else 1e-5
            assert group["eps"] == expected
        summary[recipe] = result
        (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
        # These are disposable two-update tests. Keep all evidence and the
        # immutable source; remove only weights created inside this canary.
        shutil.rmtree(run_dir / "checkpoints")
        print(json.dumps({"canary_passed": recipe}), flush=True)


if __name__ == "__main__":
    main()
