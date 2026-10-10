"""Sequential matched-checkpoint recipe trials with monitoring and video evals.

Each arm gets the same weights, Adam state, RNG, samplers and frame budget.
Manual stops halt the queue. Selection is provisional on this diagnostic set;
it does not establish full-benchmark performance or a model ceiling.
"""

import argparse
import hashlib
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

import yaml

from streamnav.utils.process_health import (
    process_alive,
    training_process_alive,
    training_stop_targets,
)

if __package__:
    from .benchmark_training_recipes import RECIPES
else:
    from benchmark_training_recipes import RECIPES


def write_json(path, value):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    temporary.replace(path)


def rows(path):
    return [json.loads(line) for line in path.read_text().splitlines()] if path.exists() else []


def evaluation_scores(run_dir):
    grouped = {}
    for row in rows(run_dir / "eval_metrics.jsonl"):
        grouped.setdefault(row["update"], {})[row["split"]] = row
    scores = []
    for update, splits in sorted(grouped.items()):
        if len(splits) != 3:
            continue
        n = sum(row["episodes"] for row in splits.values())
        score = {"update": update, "episodes": n}
        for key in ("success", "spl", "oracle_success", "false_stop_rate"):
            score[key] = sum(row[key] * row["episodes"] for row in splits.values()) / n
        score["saved_video_cases"] = sum(row["saved_video_cases"] for row in splits.values())
        episodes = []
        for path in (run_dir / "evaluation" / f"update_{update:07d}").glob("*_episodes.jsonl"):
            episodes.extend(rows(path))
        if episodes:
            if len(episodes) != n:
                raise RuntimeError("Episode diagnostics do not cover the whole evaluation set")
            identities = sorted((row["episode_index"], row["episode_id"]) for row in episodes)
            score["episode_set_sha256"] = hashlib.sha256(
                json.dumps(identities).encode()
            ).hexdigest()
            score["long_turn_episode_fraction"] = sum(
                row.get("max_consecutive_turns", 0) >= 100 for row in episodes
            ) / len(episodes)
            score["long_alternating_turn_episode_fraction"] = sum(
                row.get("max_alternating_turns", 0) >= 100 for row in episodes
            ) / len(episodes)
        scores.append(score)
    if len(scores) < 2:
        raise RuntimeError(f"Two complete fixed-set evaluations are required: {run_dir}")
    return scores


def choose_recipe(results):
    """Conservative, explicitly provisional selection against matched control."""
    control = results["control"]["scores"]
    mean_control = sum(row["success"] for row in control) / len(control)
    candidates = []
    for recipe, result in results.items():
        if recipe == "control":
            continue
        scores = result["scores"]
        if [row["update"] for row in scores] != [row["update"] for row in control]:
            raise RuntimeError("Evaluation updates differ between trial arms")
        if any(
            row.get("episode_set_sha256") != base.get("episode_set_sha256")
            for row, base in zip(scores, control, strict=True)
        ):
            raise RuntimeError("Fixed episode identities differ between trial arms")
        mean_sr = sum(row["success"] for row in scores) / len(scores)
        last, base = scores[-1], control[-1]
        # Require a repeated SR gain of at least two episodes per 144 and
        # retain navigation efficiency, goal reach and false-stop behavior.
        eligible = (
            mean_sr >= mean_control + 2 / base["episodes"]
            and last["success"] >= base["success"]
            and last["spl"] >= base["spl"]
            and last["oracle_success"] >= base["oracle_success"] - 2 / base["episodes"]
            and last["false_stop_rate"] <= base["false_stop_rate"] + 2 / base["episodes"]
        )
        result["mean_success"] = mean_sr
        result["eligible_for_provisional_continuation"] = eligible
        if eligible:
            candidates.append((mean_sr, last["spl"], recipe))
    return max(candidates)[2] if candidates else "control"


class ExistingTrainingProcess:
    """Reconnect queue supervision without restarting the learner."""

    def __init__(self, pid):
        self.pid = pid

    def poll(self):
        return None if training_process_alive(self.pid) else 0

    def wait(self):
        while self.poll() is None:
            time.sleep(2)
        # An adopted process has no waitpid exit code. finish() still requires
        # its complete verification file, full budget and identical parameters.
        return 0


class TrialQueue:
    def __init__(self, output):
        self.output = output
        self.process = None
        self.run_dir = None
        self.stopping = False

    def stop(self, signum, frame):
        self.stopping = True
        if self.process is None or self.process.poll() is not None:
            return
        workers = []
        worker_file = self.run_dir / "training_workers.json"
        if worker_file.exists():
            workers = json.loads(worker_file.read_text())
        targets = training_stop_targets(workers, self.process.pid)
        write_json(
            self.run_dir / "stop_request.json",
            {"training_pid": self.process.pid, "time": time.time(), "targets": targets},
        )
        for pid in targets:
            try:
                os.kill(pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
        print(json.dumps({"queue_stopping": True, "signal": signum}), flush=True)

    def adopt(self, checkpoint, run_dir, recipe, expected_end):
        config = yaml.safe_load((run_dir / "resolved_config.yaml").read_text())
        if Path(config["checkpoint"]).resolve() != checkpoint.resolve():
            raise ValueError("Adopted learner has a different source checkpoint")
        if config["trainer"]["num_updates"] != expected_end:
            raise ValueError("Adopted learner has a different training budget")
        self.run_dir = run_dir
        self.process = ExistingTrainingProcess(int((run_dir / "training.pid").read_text()))
        write_json(
            self.output / "status.json",
            {
                "status": "training",
                "recipe": recipe,
                "run_dir": str(run_dir),
                "training_pid": self.process.pid,
                "queue_pid": os.getpid(),
                "adopted_without_learner_restart": True,
                "time": time.time(),
            },
        )
        print(json.dumps({"trial_attached": recipe, "pid": self.process.pid}), flush=True)
        return self.finish(run_dir, self.process.wait())

    def launch(self, checkpoint, run_dir, recipe, updates, label=None):
        if self.stopping:
            raise InterruptedError("Trial queue stop requested")
        run_dir.mkdir(parents=True, exist_ok=False)
        self.run_dir = run_dir
        command = [
            sys.executable,
            "-m",
            "torch.distributed.run",
            "--standalone",
            "--nnodes=1",
            "--nproc_per_node=8",
            "tools/benchmark_training_recipes.py",
            "--checkpoint",
            str(checkpoint),
            "--run-dir",
            str(run_dir),
            "--recipe",
            recipe,
            "--updates",
            str(updates),
            "--eval-interval",
            "100",
        ]
        if label:
            command.extend(["--label", label])
        temporary_link = Path("runs/.streamnav_active_trial")
        temporary_link.unlink(missing_ok=True)
        temporary_link.symlink_to(run_dir.parent)
        temporary_link.replace("runs/streamnav_active")
        with (run_dir / "training.log").open("w") as log:
            self.process = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT)
        (run_dir / "training.pid").write_text(str(self.process.pid) + "\n")
        with (run_dir / "monitor.log").open("w") as log:
            monitor = subprocess.Popen(
                [
                    sys.executable,
                    "tools/monitor_training.py",
                    "--pid",
                    str(self.process.pid),
                    "--run-dir",
                    str(run_dir),
                    "--interval",
                    "30",
                ],
                stdout=log,
                stderr=subprocess.STDOUT,
            )
        (run_dir / "monitor.pid").write_text(str(monitor.pid) + "\n")
        write_json(
            self.output / "status.json",
            {
                "status": "training",
                "recipe": label or recipe,
                "run_dir": str(run_dir),
                "training_pid": self.process.pid,
                "queue_pid": os.getpid(),
                "time": time.time(),
            },
        )
        print(json.dumps({"trial_started": label or recipe, "pid": self.process.pid}), flush=True)
        return self.finish(run_dir, self.process.wait())

    def finish(self, run_dir, code):
        request_path = run_dir / "stop_request.json"
        manual_stop = (
            request_path.exists()
            and json.loads(request_path.read_text()).get("training_pid") == self.process.pid
        )
        if self.stopping or manual_stop:
            raise InterruptedError("Manual stop: checkpoint saved; remaining queue is paused")
        if code:
            raise RuntimeError(f"Training exited {code}; inspect {run_dir / 'training.log'}")
        verification = json.loads((run_dir / "recipe_verification.json").read_text())
        if verification["end_update"] != verification["expected_end_update"]:
            raise RuntimeError("Training halted early; queue will not start another arm")
        if not verification["all_rank_parameters_identical"]:
            raise RuntimeError("Parameters diverged between ranks")
        self.process = None
        return verification


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--updates", type=int, default=200)
    parser.add_argument(
        "--recipes",
        nargs="+",
        choices=RECIPES,
        default=["ppo_floor", "auxiliary_il", "ppo_auxiliary_il", "backbone_eps", "control"],
    )
    parser.add_argument("--continue-to", type=int, default=2000)
    parser.add_argument("--attach-first", action="store_true", help="Reconnect to the first arm")
    args = parser.parse_args()
    checkpoint = Path(args.checkpoint).resolve()
    output = Path(args.output).resolve()
    start = json.loads((checkpoint / "manifest.json").read_text())["update"]
    if args.updates < 200 or args.updates % 100 or start % 100 or "control" not in args.recipes:
        raise ValueError("Use a round source update, at least two 100-update evals, and a control")
    if len(set(args.recipes)) != len(args.recipes):
        raise ValueError("Duplicate trial recipes")
    if args.continue_to <= start + args.updates:
        raise ValueError("Continuation target must exceed the trial budget")
    if args.attach_first:
        previous = int((output / "queue.pid").read_text())
        if process_alive(previous, "tools/run_training_recipe_trials.py"):
            raise RuntimeError("Previous queue must exit before adopting its learner")
        status = json.loads((output / "status.json").read_text())
        if status.get("recipe") != args.recipes[0]:
            raise RuntimeError("Only the first active arm can be adopted")
    output.mkdir(parents=True, exist_ok=args.attach_first)
    (output / "queue.pid").write_text(str(os.getpid()) + "\n")
    queue = TrialQueue(output)
    for signum in (signal.SIGTERM, signal.SIGINT):
        signal.signal(signum, queue.stop)
    results = {}
    source_states = None
    try:
        for index, recipe in enumerate(args.recipes):
            run_dir = output / recipe / "ealm"
            verification = (
                queue.adopt(checkpoint, run_dir, recipe, start + args.updates)
                if args.attach_first and index == 0
                else queue.launch(checkpoint, run_dir, recipe, args.updates)
            )
            if source_states is None:
                source_states = verification["source_rank_states"]
            if source_states != verification["source_rank_states"]:
                raise RuntimeError("Source Adam/RNG/sampler states differ between trial arms")
            results[recipe] = {"run_dir": str(run_dir), "scores": evaluation_scores(run_dir)}
            write_json(output / "comparison.json", {"complete": False, "trials": results})
        selected = choose_recipe(results)
        write_json(
            output / "comparison.json",
            {"complete": True, "provisional_selection": selected, "trials": results},
        )
        parent = Path(results[selected]["run_dir"]) / "checkpoints" / "latest"
        end = json.loads((parent / "manifest.json").read_text())["update"]
        # 'control' retains every recipe option in the saved configuration;
        # this is a continuation of the selected arm without further changes.
        queue.launch(
            parent.resolve(),
            output / "selected" / "ealm",
            "control",
            args.continue_to - end,
            label=f"continue:{selected}",
        )
        write_json(output / "status.json", {"status": "complete", "selected": selected})
    except InterruptedError as error:
        write_json(output / "status.json", {"status": "paused", "reason": str(error)})
        print(str(error), flush=True)
    except Exception as error:
        write_json(output / "status.json", {"status": "failed", "reason": str(error)})
        raise


if __name__ == "__main__":
    main()
