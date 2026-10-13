"""Streaming Qwen3.5 navigation with geometric points, DAgger/PPO and grounding.

Rollouts store the actual APOS/OPOS/action block, not teacher-forced points.
All recurrent fragments accumulate gradients before a single synchronized
optimizer step, preserving the pre-update probability check for every fragment.
Only student-selected transitions are used by PPO, after the expert warmup ends.
"""

import argparse
import json
import math
import os
import signal
import time
from dataclasses import replace
from hashlib import sha256
from pathlib import Path
from typing import Any

import torch
import yaml
from safetensors.torch import load_file, save_file
from torch.distributions import Categorical
from torch.nn import functional as F

from streamnav.contracts.action import NavigationAction
from streamnav.contracts.state import LayerState
from streamnav.data.grounding import GroundingSource, check_grounding_split
from streamnav.data.manifest import check_leakage
from streamnav.data.mixed_starts import MixedStartEpisodeSource
from streamnav.data.schema import HabitatEpisodeSource
from streamnav.envs.habitat_client import VectorHabitatEnvs
from streamnav.errors import OracleUnavailableError
from streamnav.models.point_cot.policy import Qwen35PointCoTPolicy, TokenStep
from streamnav.models.policy.perception import point_type_logits, point_type_targets
from streamnav.training import distributed as parallel
from streamnav.training.affordance_loss import (
    imitation_weight,
    validate_affordance_label,
)
from streamnav.training.diagnostics import (
    merge_diagnostics,
    observation_distance,
    summarize_rollout,
)
from streamnav.training.expert_stream import (
    dagger_environment,
    expert_environment,
    split_episode_metrics,
    student_rl_weights,
    validate_expert_streams,
)
from streamnav.training.point_supervision import (
    grounding_loss,
    grounding_loss_options,
    marginal_point_loss,
    supervised_fragment,
    targets_from_label,
)
from streamnav.training.ppo import clipped_ppo_loss
from streamnav.utils.progress import write_worker_progress
from streamnav.utils.provenance import snapshot_sources
from streamnav.utils.seed import restore_rng, rng_state, seed_everything


def cache_copy(state, device="cpu"):
    return replace(
        state,
        kda_cache=tuple(
            LayerState(
                None, s.recurrent.detach().to(device).clone() if s.recurrent is not None else None
            )
            for s in state.kda_cache
        ),
    )


def token_log_prob(logits, tokens):
    return sum(Categorical(logits=x).log_prob(y) for x, y in zip(logits, tokens, strict=True))


def replay_records(policy, records, state, *, resets=None, mode="packed"):
    resets = resets or {}
    tokens = [tuple(t.to(policy.device) for t in r["tokens"]) for r in records]
    if mode == "packed":
        return policy.replay_fragment(
            [r["features"] for r in records], state, tokens, resets=resets
        )
    if mode != "sequential":
        raise ValueError(f"Unknown replay mode: {mode}")
    steps = []
    for i, (record, token) in enumerate(zip(records, tokens, strict=True)):
        if i in resets:
            state = policy.start_episode(*resets[i])
        step = policy.step(record["features"], state, tokens=token)
        state = step.state
        steps.append(step)
    return TokenStep(
        torch.stack([s.apos for s in steps]),
        torch.stack([s.opos for s in steps]),
        torch.stack([s.action for s in steps]),
        torch.stack([s.apos_logits for s in steps]),
        torch.stack([s.opos_logits for s in steps]),
        torch.stack([s.action_logits for s in steps]),
        torch.stack([s.value for s in steps]),
        state,
    )


def replay_losses(policy, state, resets, step, records, adv, returns, cfg, ppo_weight):
    """Replay actual point histories; optionally mix teacher-prefixed action IL."""
    device = step.action_logits.device
    experts = torch.tensor([r["expert"] for r in records], device=device)
    labels = [targets_from_label(r.get("perception_label")) for r in records]
    _, teacher_apos, teacher_opos, teacher_il = supervised_fragment(
        policy,
        [r["features"] for r in records],
        state,
        [r["tokens"] for r in records],
        labels,
        resets=resets,
        experts=experts,
    )
    if cfg.get("point_supervision_prefix", "teacher") == "actual":
        apos = marginal_point_loss(step.apos_logits, [label["apos_targets"] for label in labels])
        opos = marginal_point_loss(step.opos_logits, [label["opos_targets"] for label in labels])
    else:
        apos, opos = teacher_apos, teacher_opos
    apos_targets = torch.tensor(
        [label["apos_targets"][0] if label["apos_targets"] else 0 for label in labels],
        device=device,
    )
    apos_valid = torch.tensor([bool(label["apos_targets"]) for label in labels], device=device)
    apos_type = (
        F.cross_entropy(
            point_type_logits(step.apos_logits), point_type_targets(apos_targets), reduction="none"
        )
        * apos_valid
    )
    predicted_il = F.cross_entropy(step.action_logits, experts, reduction="none")
    mix = cfg["teacher_action_mix"]
    il = ((1 - mix) * predicted_il + mix * teacher_il) * predicted_il.new_tensor(
        [r["il_weight"] for r in records]
    )
    old = torch.stack([r["old_log_prob"] for r in records]).to(device)
    weights = student_rl_weights(records, cfg, device)
    eligible = torch.tensor([r["student_action"] for r in records], device=device)
    ppo, _ = clipped_ppo_loss(step.log_prob, old, adv.to(device), clip_eps=cfg["ppo_clip"])
    ppo *= eligible * weights
    # Expert-overridden and greedy DAgger transitions are never on-policy PPO.
    # The critic is fitted only on pure stochastic learner rollouts.
    value_loss = 0.5 * (step.value - returns.to(device)).square() * eligible * weights
    entropy = (
        sum(
            Categorical(logits=x).entropy()
            for x in (step.apos_logits, step.opos_logits, step.action_logits)
        )
        * eligible
        * weights
    )
    loss = (
        il
        + cfg["point_il_coef"] * (apos + opos + cfg.get("apos_type_il_coef", 0) * apos_type)
        + ppo_weight * ppo
        + cfg["value_loss_coef"] * value_loss
        - ppo_weight * cfg["entropy_coef"] * entropy
    )
    return (
        loss,
        {
            "il_loss": il,
            "apos_loss": apos,
            "apos_type_loss": apos_type,
            "apos_teacher_prefix_loss": teacher_apos,
            "opos_loss": opos,
            "ppo_loss": ppo,
            "value_loss": value_loss,
            "entropy": entropy,
            "loss": loss,
        },
        (step.log_prob.detach() - old).abs().max(),
    )


def guarded_replay_records(policy, records, state, *, resets=None, mode="packed", tolerance=0.05):
    step = replay_records(policy, records, state, resets=resets, mode=mode)
    old = torch.stack([r["old_log_prob"] for r in records]).to(policy.device)
    error = (step.log_prob.detach() - old).abs().max().item()
    fallback = mode == "packed" and (not math.isfinite(error) or error > tolerance)
    if fallback:
        # Recompute the entire fragment to preserve its recurrent gradients.
        del step
        step = replay_records(policy, records, state, resets=resets, mode="sequential")
    return step, error, fallback


def advantages(rewards, values, dones, last_values, gamma=0.99, tau=0.95, timeout_values=None):
    result = torch.zeros_like(rewards)
    running = torch.zeros_like(last_values)
    for t in reversed(range(len(rewards))):
        next_value = last_values if t == len(rewards) - 1 else values[t + 1]
        bootstrap = (1 - dones[t].float()) * next_value
        if timeout_values is not None:
            bootstrap = bootstrap + timeout_values[t]
        delta = rewards[t] + gamma * bootstrap - values[t]
        running = delta + gamma * tau * (1 - dones[t].float()) * running
        result[t] = running
    return result, result + values


def append_json(path, row):
    with open(path, "a") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_progress(root, update, phase, completed, total, started):
    write_worker_progress(root, update, phase, completed, total, started, rank=parallel.rank())


def save(policy, optimizer, source, config, update, frames=0, rank_payload=None):
    policy.loaded_checkpoint = str(
        (Path(config["run_dir"]) / "checkpoints" / f"update_{update:07d}").resolve()
    )
    adapters = policy.trainable_state()
    digest = sha256()
    for name, value in sorted(adapters.items()):
        digest.update(name.encode())
        digest.update(value.numpy().tobytes())
    source_state = source.state_dict()
    grounding_source = getattr(policy, "grounding_source", source)
    payload = {
        "rank": parallel.rank(),
        "rng": rng_state(),
        "source": source_state,
        "grounding": source_state if grounding_source is source else grounding_source.state_dict(),
        "adapter_sha256": digest.hexdigest(),
    }
    ranks = parallel.gather(payload if rank_payload is None else rank_payload)
    if len({r["adapter_sha256"] for r in ranks}) != 1:
        raise RuntimeError("Distributed ranks have different adapter parameters")
    root = Path(config["run_dir"])
    if parallel.rank() == 0:
        folder = root / "checkpoints" / f"update_{update:07d}"
        temp = folder.with_name(".writing-" + folder.name)
        if folder.exists():
            parallel.barrier()
            return
        temp.mkdir(parents=True, exist_ok=True)
        save_file(adapters, str(temp / "adapters.safetensors"))
        torch.save(
            {
                "optimizer": optimizer.state_dict(),
                "ranks": ranks,
                "update": update,
                "frames": frames,
                "world_size": parallel.world_size(),
            },
            temp / "optimizer.pt",
        )
        (temp / "resolved_config.yaml").write_text(yaml.safe_dump(config, sort_keys=False))
        (temp / "manifest.json").write_text(
            json.dumps(
                {
                    "format": "qwen35-pointcot-adapters-v1",
                    "update": update,
                    "base": policy.parameter_summary()["source_revision"],
                    "adapter_sha256": digest.hexdigest(),
                    "parameters": policy.parameter_summary(),
                    "source_snapshot": config.get("source_snapshot"),
                },
                indent=2,
            )
            + "\n"
        )
        if not folder.exists():
            temp.rename(folder)
        link = root / "checkpoints" / ".latest-new"
        link.unlink(missing_ok=True)
        link.symlink_to(folder.name)
        link.replace(root / "checkpoints" / "latest")
        (root / "latest_checkpoint.json").write_text(
            json.dumps({"update": update, "path": str(folder.resolve())})
        )
        keep = config["trainer"].get("keep_checkpoints", 2)
        candidates = sorted((root / "checkpoints").glob("update_*"))
        best = (
            (root / "checkpoints" / "best").resolve()
            if (root / "checkpoints" / "best").exists()
            else None
        )
        import shutil

        initial_update = config.get("fork_origin", {}).get("update", 0)
        initial_name = f"update_{initial_update:07d}"
        for old in candidates[:-keep]:
            if old.resolve() != best and old.name not in ("update_0000000", initial_name):
                shutil.rmtree(old)
    parallel.barrier()


def load_adapters(policy, folder):
    manifest = json.loads((Path(folder) / "manifest.json").read_text())
    if manifest.get("format") != "qwen35-pointcot-adapters-v1":
        raise ValueError("Not a Qwen3.5 point-CoT checkpoint")
    if manifest["base"] != policy.parameter_summary()["source_revision"]:
        raise ValueError("Checkpoint uses a different frozen Qwen3.5 base")
    weights = load_file(str(Path(folder) / "adapters.safetensors"), device=str(policy.device))
    digest = sha256()
    for name, value in sorted(weights.items()):
        digest.update(name.encode())
        digest.update(value.cpu().numpy().tobytes())
    if digest.hexdigest() != manifest["adapter_sha256"]:
        raise ValueError("Adapter checksum differs from the saved manifest")
    named = dict(policy.named_parameters())
    expected = {name for name, p in named.items() if p.requires_grad}
    if set(weights) != expected:
        raise ValueError("Adapter recipe differs from the saved checkpoint")
    with torch.no_grad():
        for name, value in weights.items():
            if value.shape != named[name].shape or not bool(torch.isfinite(value).all()):
                raise ValueError(f"Invalid saved parameter {name}")
            named[name].copy_(value)
    policy.loaded_checkpoint = str(Path(folder).resolve())


def resume_recipe(config):
    """Fields affecting optimization and observations, excluding run scheduling."""
    trainer = dict(config["trainer"])
    for key in (
        "num_updates",
        "checkpoint_interval",
        "eval_interval",
        "keep_checkpoints",
        "supervision",
        "replay_mode",
    ):
        trainer.pop(key, None)
    habitat = dict(config["habitat"])
    habitat.pop("gpu_device_id", None)
    # Historical checkpoints omitted this field and always disabled sliding.
    habitat.setdefault("allow_sliding", False)
    model = dict(config["model"])
    model.pop("language_compute_dtype", None)
    model.pop("recurrent_decode", None)
    model.pop("float32_matmul_precision", None)
    model.pop("point_decoding", None)
    # Recomputing activations affects memory/time, not the model or objective.
    model.pop("gradient_checkpointing", None)
    return {
        "grounding": config["grounding"],
        "curriculum": config["curriculum"],
        "model": model,
        "data": config["data"],
        "habitat": habitat,
        "trainer": trainer,
        "seed": config["seed"],
    }


def fork_signature(config):
    """A recipe fork may change objectives, but not observations or model rows."""
    recipe = resume_recipe(config)
    recipe.pop("trainer")
    recipe["habitat"].pop("reward", None)
    # Dynamics may change in an explicit recipe fork, never an exact resume.
    recipe["habitat"].pop("allow_sliding", None)
    # These sensors/labels are used only by the training annotation service.
    # RGB camera and robot geometry must still match the parent checkpoint.
    recipe["habitat"].pop("perception_labels", None)
    recipe["habitat"].pop("perception_object_labels", None)
    return recipe


def synchronized_optimizer_step(parameters, optimizer, max_grad_norm):
    parallel.average_gradients(parameters)
    # Finite local gradients can still overflow during collective reduction.
    # Reject the synchronized norm before modifying parameters or Adam moments.
    norm = torch.nn.utils.clip_grad_norm_(parameters, max_grad_norm, error_if_nonfinite=True)
    optimizer.step()
    return norm


def record_best(config, update, results):
    if parallel.rank() == 0:
        root = Path(config["run_dir"])
        score = sum(r["success"] for r in results.values()) / len(results)
        best_file = root / "best_evaluation.json"
        previous = json.loads(best_file.read_text())["success"] if best_file.exists() else -1
        if score > previous:
            folder = root / "checkpoints" / f"update_{update:07d}"
            if not folder.exists():
                raise RuntimeError("Best evaluation must refer to a saved checkpoint")
            link = folder.parent / ".best-new"
            link.unlink(missing_ok=True)
            link.symlink_to(folder.name)
            link.replace(folder.parent / "best")
            best_file.write_text(
                json.dumps({"update": update, "success": score, "results": results}, indent=2)
            )
    parallel.barrier()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/pointcot.yaml")
    parser.add_argument("--run-dir")
    parser.add_argument(
        "--initialize-from", help="Grounding adapters; fresh navigation optimizer and sampler"
    )
    parser.add_argument("--updates", type=int)
    parser.add_argument("--rollout-steps", type=int)
    parser.add_argument("--sequence-length", type=int)
    parser.add_argument("--num-envs", type=int)
    restoration = parser.add_mutually_exclusive_group()
    restoration.add_argument("--resume")
    restoration.add_argument("--fork-from")
    parser.add_argument("--eval-episodes", type=int)
    parser.add_argument("--eval-max-steps", type=int)
    parser.add_argument("--skip-initial-eval", action="store_true")
    args = parser.parse_args()
    if not (args.resume or args.fork_from or args.initialize_from):
        raise ValueError("Run grounding first, then use --initialize-from its checkpoint")
    config = yaml.safe_load(Path(args.config).read_text())
    config["launch"] = {
        "resume": args.resume,
        "fork_from": args.fork_from,
        "skip_initial_eval": args.skip_initial_eval,
        "initialize_from": args.initialize_from,
    }
    if args.run_dir:
        config["run_dir"] = args.run_dir
    for option, key in (
        (args.updates, "num_updates"),
        (args.rollout_steps, "rollout_steps"),
        (args.sequence_length, "sequence_length"),
        (args.num_envs, "num_envs"),
    ):
        if option is not None:
            config["trainer"][key] = option
    if args.eval_episodes:
        config["eval"]["episodes"] = args.eval_episodes
    if args.eval_max_steps:
        config["eval"]["max_steps"] = args.eval_max_steps
    parallel.initialize(config)
    cfg = config["trainer"]
    validate_expert_streams(cfg)
    if not 0 <= cfg["teacher_action_mix"] <= 1:
        raise ValueError("Teacher action mixture must be in [0, 1]")
    if config["curriculum"]["ready_distance"][1] > config["habitat"]["success_distance"]:
        raise ValueError("STOP-ready starts must be inside the success radius")
    config["grounding"]["manifest_sha256"] = {
        p: sha256(Path(p).read_bytes()).hexdigest()
        for p in (config["grounding"]["train_manifest"], config["grounding"]["val_manifest"])
    }
    config["data"]["manifest_sha256"] = {
        s["manifest"]: sha256(Path(s["manifest"]).read_bytes()).hexdigest()
        for s in config["data"]["sources"]
    }
    train_manifests = [
        json.loads(Path(s["manifest"]).read_text()) for s in config["data"]["sources"]
    ]
    validation_manifests = [json.loads(Path(p).read_text()) for p in config["eval"]["manifests"]]
    if len(train_manifests) != 1 or train_manifests[0]["split"] != "train":
        raise ValueError("This trainer requires one training-only dataset source")
    check_leakage(train_manifests, validation_manifests)
    root = Path(config["run_dir"])
    if not (args.resume or args.fork_from) and (root / "checkpoints" / "latest").exists():
        raise ValueError("Run already has a checkpoint; use --resume or a new run directory")
    if args.fork_from and (root / "checkpoints" / "latest").exists():
        if Path(args.fork_from).resolve() != (root / "checkpoints" / "latest").resolve():
            raise ValueError("An in-place recipe fork must start from this run's latest checkpoint")
    if cfg["rollout_steps"] % cfg["sequence_length"]:
        raise ValueError("Rollout must consist of complete recurrent fragments")
    if cfg.get("point_supervision_prefix", "teacher") not in ("actual", "teacher"):
        raise ValueError("Unknown point supervision prefix")
    type_coef = cfg.get("apos_type_il_coef", 0)
    if not math.isfinite(type_coef) or type_coef < 0:
        raise ValueError("APOS type coefficient must be finite and nonnegative")
    geometry = True
    for key in ("recovery_il_weight", "stop_il_weight"):
        if not math.isfinite(cfg.get(key, 1.0)) or cfg.get(key, 1.0) < 1:
            raise ValueError(f"{key} must be finite and at least one")
    seed_everything(config["seed"])
    policy = Qwen35PointCoTPolicy(config).train()
    groups: dict[str, list[torch.nn.Parameter]] = {
        "lora": [],
        "memory": [],
        "point": [],
        "action": [],
        "critic": [],
    }
    for name, parameter in policy.named_parameters():
        if parameter.requires_grad:
            group = (
                "memory"
                if name.startswith("memories.")
                else "point"
                if name.startswith(("apos_", "opos_"))
                else "action"
                if name.startswith("action_")
                else "critic"
                if name.startswith("critic.")
                else "lora"
            )
            groups[group].append(parameter)
    optimizer = torch.optim.AdamW(
        [
            {"params": params, "lr": cfg["learning_rates"][name], "name": name}
            for name, params in groups.items()
        ],
        weight_decay=0.0,
        eps=1e-8,
    )
    episode_source = HabitatEpisodeSource(
        config["data"]["sources"][0]["manifest"],
        seed=config["seed"] + 1000 * parallel.rank(),
        iterator_options=config["data"]["iterator_options"],
    )
    source = MixedStartEpisodeSource(episode_source, config["curriculum"])
    grounding = GroundingSource(
        config["grounding"]["train_manifest"],
        seed=config["seed"] + parallel.rank(),
        source_weights=config["grounding"].get("source_weights"),
        absence_probability=config["grounding"].get("absence_probability"),
    )
    grounding_val = GroundingSource(config["grounding"]["val_manifest"])
    check_grounding_split(grounding, grounding_val)
    policy.grounding_source = grounding
    update, frames = 0, 0
    restore_from = args.resume or args.fork_from
    if args.initialize_from:
        if restore_from:
            raise ValueError("Initialization and resume are mutually exclusive")
        load_adapters(policy, args.initialize_from)
        origin = json.loads(Path(args.initialize_from, "manifest.json").read_text())
        config["initialization"] = {
            "checkpoint": str(Path(args.initialize_from).resolve()),
            "update": origin["update"],
            "adapter_sha256": origin["adapter_sha256"],
            "source_snapshot": origin.get("source_snapshot"),
        }
    if restore_from:
        previous_config = yaml.safe_load((Path(restore_from) / "resolved_config.yaml").read_text())
        if "initialization" in previous_config:
            config["initialization"] = previous_config["initialization"]
        if args.resume and resume_recipe(config) != resume_recipe(previous_config):
            raise ValueError("Resume changes the model, observations or optimization recipe")
        if args.fork_from and fork_signature(config) != fork_signature(previous_config):
            raise ValueError("A recipe fork must preserve the model, observations and dataset")
        load_adapters(policy, restore_from)
        saved = torch.load(
            Path(restore_from) / "optimizer.pt", map_location="cpu", weights_only=False
        )
        if saved["world_size"] != parallel.world_size():
            raise ValueError("Exact resume requires the original world size")
        optimizer.load_state_dict(saved["optimizer"])
        if args.fork_from:
            for optimizer_group in optimizer.param_groups:
                optimizer_group["lr"] = cfg["learning_rates"][optimizer_group["name"]]
            parent = json.loads((Path(restore_from) / "manifest.json").read_text())
            config["fork_origin"] = {
                "checkpoint": str(Path(restore_from).resolve()),
                "update": saved["update"],
                "frames": saved["frames"],
                "adapter_sha256": parent["adapter_sha256"],
                "source_snapshot": parent.get("source_snapshot"),
                "previous_recipe": resume_recipe(previous_config),
            }
            if parallel.rank() == 0:
                root.mkdir(parents=True, exist_ok=True)
                append_json(root / "recipe_forks.jsonl", config["fork_origin"])
        elif "fork_origin" in previous_config:
            config["fork_origin"] = previous_config["fork_origin"]
        update = saved["update"]
        frames = saved.get(
            "frames", update * cfg["rollout_steps"] * cfg["num_envs"] * parallel.world_size()
        )
        local = saved["ranks"][parallel.rank()]
        source.load_state_dict(local["source"])
        if "grounding" in local:
            grounding.load_state_dict(local["grounding"])
        restore_rng(local["rng"])
    else:
        seed_everything(config["seed"] + parallel.rank())
    root.mkdir(parents=True, exist_ok=True)
    workers = parallel.gather(
        {
            "pid": os.getpid(),
            "rank": parallel.rank(),
            "launcher_pid": os.getppid(),
            "device": str(policy.device),
        }
    )
    if parallel.rank() == 0:
        (root / "training_workers.json").write_text(json.dumps(workers, indent=2))
        (root / "parameters.json").write_text(json.dumps(policy.parameter_summary(), indent=2))
        print(json.dumps({"qwen35_pointcot": policy.parameter_summary()}), flush=True)
    config["source_snapshot"] = snapshot_sources(root)
    if parallel.rank() == 0:
        (root / "resolved_config.yaml").write_text(yaml.safe_dump(config, sort_keys=False))
    stop = [False]

    def request_stop(signum, frame):
        stop[0] = True

    signal.signal(signal.SIGTERM, request_stop)
    signal.signal(signal.SIGINT, request_stop)
    from streamnav.evaluation.runner import evaluate

    if not restore_from or args.fork_from:
        save(policy, optimizer, source, config, update, frames)
    if not args.skip_initial_eval:
        results = evaluate(policy, config, update=update, video=True)
        record_best(config, update, results)
    envs = VectorHabitatEnvs(config["habitat"], cfg["num_envs"])
    episodes = [source.sample_episode() for _ in range(cfg["num_envs"])]
    observations = envs.reset(episodes)
    states = [policy.start_episode(e.uid, e.goal_text) for e in episodes]
    try:
        while update < cfg["num_updates"]:
            if parallel.any_rank(stop[0], policy.device):
                save(policy, optimizer, source, config, update, frames)
                break
            update += 1
            started = time.monotonic()
            tmax, count = cfg["rollout_steps"], len(states)
            write_progress(root, update, "rollout", 0, tmax, started)
            beta = cfg["expert_warmup_probability"] * max(
                0.0, 1 - update / max(1, cfg["warmup_updates"])
            )
            ppo_weight = cfg["ppo_weight"] * min(
                1.0, max(0.0, (update - cfg["warmup_updates"]) / cfg["ppo_ramp_updates"])
            )
            records: list[list[dict[str, Any]]] = []
            starts, resets, metrics = {}, {}, []
            timeout_values = torch.zeros(tmax, count)
            rollout_timings = dict.fromkeys(
                (
                    "rollout_oracle_seconds",
                    "rollout_policy_seconds",
                    "rollout_environment_seconds",
                    "rollout_bootstrap_seconds",
                ),
                0.0,
            )
            with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                for t in range(tmax):
                    phase_started = time.monotonic()
                    supervisions = envs.get_oracle_supervisions() if geometry else None
                    experts = (
                        [
                            s
                            if isinstance(s, OracleUnavailableError)
                            else NavigationAction(s["action"])
                            for s in supervisions
                        ]
                        if supervisions is not None
                        else envs.get_oracle_actions()
                    )
                    for i, expert in enumerate(experts):
                        retries = 0
                        while isinstance(expert, OracleUnavailableError):
                            if t:
                                records[t - 1][i]["done"] = True
                                if cfg.get("bootstrap_time_limits", False):
                                    terminal = policy.begin(
                                        policy.frame_features(
                                            observations[i]["rgb"], states[i].instruction
                                        )[0],
                                        states[i],
                                    )
                                    timeout_values[t - 1, i] = (
                                        policy.read_value(terminal.hidden).squeeze().cpu()
                                    )
                            episode = source.sample_episode()
                            observations[i] = envs.clients[i].reset(episode)
                            states[i] = policy.start_episode(episode.uid, episode.goal_text)
                            resets[(t, i)] = (episode.uid, episode.goal_text)
                            try:
                                if supervisions is not None:
                                    supervisions[i] = envs.clients[i].get_oracle_supervision()
                                    expert = NavigationAction(supervisions[i]["action"])
                                else:
                                    expert = envs.clients[i].get_oracle_action()
                            except OracleUnavailableError as exc:
                                expert = exc
                            retries += 1
                            if retries >= 3 and isinstance(expert, OracleUnavailableError):
                                raise expert
                        experts[i] = expert
                    rollout_timings["rollout_oracle_seconds"] += time.monotonic() - phase_started
                    phase_started = time.monotonic()
                    features = policy.navigation_features([o["rgb"] for o in observations], states)
                    overrides, greedy, stream_flags = [], [], []
                    for i, state in enumerate(states):
                        expert_action = experts[i]
                        if isinstance(expert_action, OracleUnavailableError):
                            raise expert_action
                        if t % cfg["sequence_length"] == 0:
                            starts[(t, i)] = cache_copy(state)
                        expert_stream = expert_environment(i, cfg)
                        dagger_stream = dagger_environment(i, cfg)
                        use_teacher = expert_stream or (
                            ppo_weight == 0 and not dagger_stream and bool(torch.rand(()) < beta)
                        )
                        forced = (
                            (None, None, torch.tensor(int(expert_action), device=policy.device))
                            if use_teacher
                            else None
                        )
                        overrides.append(forced)
                        greedy.append(dagger_stream)
                        stream_flags.append((expert_stream, dagger_stream, use_teacher))
                    steps = policy.step_batch(
                        features, states, tokens=overrides, deterministic=greedy
                    )
                    row = []
                    for i, (state, feature, step) in enumerate(
                        zip(states, features, steps, strict=True)
                    ):
                        expert_stream, dagger_stream, use_teacher = stream_flags[i]
                        expert_action = experts[i]
                        if isinstance(expert_action, OracleUnavailableError):
                            raise expert_action
                        label = (
                            validate_affordance_label(
                                supervisions[i]["perception"],
                                state.episode_id,
                                observations[i]["frame_id"],
                            )
                            if supervisions is not None
                            else None
                        )
                        row.append(
                            {
                                "features": feature.cpu(),
                                "tokens": tuple(
                                    x.cpu() for x in (step.apos, step.opos, step.action)
                                ),
                                "old_log_prob": step.log_prob.cpu(),
                                "value": step.value.cpu(),
                                "expert": int(expert_action),
                                "student_action": not use_teacher and not dagger_stream,
                                "expert_stream": expert_stream,
                                "dagger_stream": dagger_stream,
                                "perception_label": label,
                                "collision_before": bool(observations[i].get("collision", False)),
                                "episode_id": state.episode_id,
                                "conditional_greedy_action": int(step.action_logits.argmax()),
                                "apos_greedy_token": int(step.apos_logits.argmax()),
                                "distance_before": observation_distance(observations[i]),
                                "start_kind": (
                                    observations[i]
                                    .get("training_start", {})
                                    .get("kind", "original")
                                    if observations[i]
                                    .get("training_start", {})
                                    .get("applied", False)
                                    else "original"
                                ),
                            }
                        )
                    rollout_timings["rollout_policy_seconds"] += time.monotonic() - phase_started
                    phase_started = time.monotonic()
                    following = envs.step([NavigationAction(s.action.item()) for s in steps])
                    for i, (step, after) in enumerate(zip(steps, following, strict=True)):
                        row[i].update(
                            reward=after["reward"],
                            done=after["done"],
                            distance_after=observation_distance(after),
                            collision=after["collision"],
                            success=after["success"],
                            truncated=after["truncated"],
                        )
                        states[i] = step.state
                        if after["done"]:
                            metrics.append(
                                {
                                    **after["metrics"],
                                    "expert_stream": expert_environment(i, cfg),
                                    "dagger_stream": dagger_environment(i, cfg),
                                }
                            )
                            if after.get("truncated", False) and cfg.get(
                                "bootstrap_time_limits", False
                            ):
                                terminal = policy.begin(
                                    policy.frame_features(after["rgb"], states[i].instruction)[0],
                                    states[i],
                                )
                                timeout_values[t, i] = (
                                    policy.read_value(terminal.hidden).squeeze().cpu()
                                )
                            episode = source.sample_episode()
                            following[i] = envs.clients[i].reset(episode)
                            states[i] = policy.start_episode(episode.uid, episode.goal_text)
                            resets[(t + 1, i)] = (episode.uid, episode.goal_text)
                    observations = following
                    rollout_timings["rollout_environment_seconds"] += (
                        time.monotonic() - phase_started
                    )
                    records.append(row)
                    for _ in range(count):
                        source.step_taken()
                    if (t + 1) % cfg["sequence_length"] == 0:
                        write_progress(root, update, "rollout", t + 1, tmax, started)
                    if (t + 1) % cfg["sequence_length"] == 0 and parallel.any_rank(
                        stop[0], policy.device
                    ):
                        tmax = t + 1
                        break
                phase_started = time.monotonic()
                context = policy.begin_batch(
                    policy.navigation_features([o["rgb"] for o in observations], states), states
                )
                last = list(policy.read_value(context.hidden).cpu().unbind())
                rollout_timings["rollout_bootstrap_seconds"] += time.monotonic() - phase_started
            rollout_seconds = time.monotonic() - started
            # Normalize once over the complete distributed rollout, never per
            # replay fragment. Otherwise a fragment full of recovery examples
            # would cancel its own extra weight.
            mean_il_weight = parallel.average_scalar(
                torch.tensor(
                    sum(imitation_weight(r, cfg) for row in records for r in row) / (tmax * count),
                    device=policy.device,
                )
            ).item()
            for row in records:
                for record in row:
                    record["il_weight"] = imitation_weight(record, cfg) / mean_il_weight
            rewards = torch.tensor([[r["reward"] for r in row] for row in records])
            dones = torch.tensor([[r["done"] for r in row] for row in records])
            values = torch.stack([torch.stack([r["value"] for r in row]) for row in records])
            adv, returns = advantages(
                rewards,
                values,
                dones,
                torch.stack(last),
                timeout_values=timeout_values[:tmax]
                if cfg.get("bootstrap_time_limits", False)
                else None,
            )
            if ppo_weight:
                student_mask = torch.tensor([[r["student_action"] for r in row] for row in records])
                adv = parallel.normalize_advantages(adv, policy.device, mask=student_mask)
            optimizer.zero_grad(set_to_none=True)
            totals = {
                k: 0.0
                for k in (
                    "il_loss",
                    "apos_loss",
                    "apos_type_loss",
                    "apos_teacher_prefix_loss",
                    "opos_loss",
                    "ppo_loss",
                    "value_loss",
                    "entropy",
                    "loss",
                )
            }
            max_error, peak_point = 0.0, []
            packed_error_max, replay_fallbacks = 0.0, 0
            ppo_token_gradient_norm = 0.0
            fragments = [
                (t, i) for t in range(0, tmax, cfg["sequence_length"]) for i in range(count)
            ]
            write_progress(root, update, "optimization", 0, len(fragments), started)
            for fragment_index, (t0, i) in enumerate(fragments):
                state = cache_copy(starts[(t0, i)], policy.device)
                fragment = [records[t][i] for t in range(t0, t0 + cfg["sequence_length"])]
                fragment_resets = {
                    t - t0: episode
                    for (t, env), episode in resets.items()
                    if env == i and t0 <= t < t0 + cfg["sequence_length"]
                }
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    step, raw_error, fallback = guarded_replay_records(
                        policy,
                        fragment,
                        state,
                        resets=fragment_resets,
                        mode=cfg.get("replay_mode", "packed"),
                        tolerance=cfg["replay_tolerance"],
                    )
                    if cfg.get("replay_mode", "packed") == "packed":
                        packed_error_max = max(packed_error_max, raw_error)
                        replay_fallbacks += int(fallback)
                    combined, components, error = replay_losses(
                        policy,
                        state,
                        fragment_resets,
                        step,
                        fragment,
                        adv[t0 : t0 + cfg["sequence_length"], i],
                        returns[t0 : t0 + cfg["sequence_length"], i],
                        cfg,
                        ppo_weight,
                    )
                    max_error = max(max_error, error.item())
                    if ppo_weight and not ppo_token_gradient_norm:
                        eligible = [j for j, r in enumerate(fragment) if r["student_action"]]
                        if eligible:
                            witness = torch.autograd.grad(
                                ppo_weight * components["ppo_loss"][eligible[0]],
                                (step.apos_logits, step.opos_logits, step.action_logits),
                                retain_graph=True,
                            )
                            ppo_token_gradient_norm = (
                                torch.stack([g.float().square().sum() for g in witness])
                                .sum()
                                .sqrt()
                                .item()
                            )
                    loss = combined.sum() / (tmax * count)
                    statistics = torch.stack([x.detach().sum() for x in components.values()]).cpu()
                    for name, item in zip(components, statistics.tolist(), strict=True):
                        totals[name] += item / (tmax * count)
                    peak_point.append(step.opos_logits.softmax(-1).max(-1).values.mean().item())
                loss.backward()
                write_progress(
                    root, update, "optimization", fragment_index + 1, len(fragments), started
                )
            grounding_rows = grounding.sample(cfg["grounding_batch_size"])
            auxiliary, grounding_metrics, _ = grounding_loss(
                policy, grounding_rows, **grounding_loss_options(config["grounding"])
            )
            (cfg["grounding_coef"] * auxiliary).backward()
            invalid = not bool(
                torch.stack(
                    [
                        torch.isfinite(p.grad).all()
                        for p in policy._all_trainable
                        if p.grad is not None
                    ]
                ).all()
            )
            if parallel.any_rank(invalid or max_error > cfg["replay_tolerance"], policy.device):
                raise RuntimeError(
                    f"Rejected update: nonfinite={invalid}, replay_error={max_error:.6f}"
                )
            grad_norm = synchronized_optimizer_step(
                policy._all_trainable, optimizer, cfg["max_grad_norm"]
            )
            frames += tmax * count * parallel.world_size()
            for i, state in enumerate(states):
                states[i] = cache_copy(state, policy.device)
            local_metrics = {
                **totals,
                **rollout_timings,
                "grounding_loss": float(auxiliary.detach()),
                "grounding_opos_loss": grounding_metrics["opos_loss"],
                **summarize_rollout(records),
                "update": update,
                "ppo_coefficient": ppo_weight,
                "ppo_token_gradient_norm": ppo_token_gradient_norm,
                "expert_probability": beta,
                "grad_norm": float(grad_norm),
                "apos_parameter_gradient_norm": float(policy.apos_rows.grad.norm()),
                "replay_log_prob_error_max": max_error,
                "opos_peak_probability": sum(peak_point) / len(peak_point),
                "num_expert_envs_per_rank": cfg.get("num_expert_envs", 0),
                "num_dagger_envs_per_rank": cfg.get("num_dagger_envs", 0),
                "imitation_weight_mean": mean_il_weight,
                **split_episode_metrics(metrics),
                "transitions": tmax * count,
                "elapsed_seconds": time.monotonic() - started,
                "rollout_seconds": rollout_seconds,
                "optimization_seconds": time.monotonic() - started - rollout_seconds,
                "action_counts": torch.bincount(
                    torch.tensor([int(r["tokens"][2]) for row in records for r in row]), minlength=6
                ).tolist(),
                "expert_action_counts": torch.bincount(
                    torch.tensor([r["expert"] for row in records for r in row]), minlength=6
                ).tolist(),
                "start_kind_transitions": {
                    kind: sum(r["start_kind"] == kind for row in records for r in row)
                    for kind in ("original", "near", "ready")
                },
                "peak_vram_bytes": torch.cuda.max_memory_allocated(policy.device),
                "replay_mode": cfg.get("replay_mode", "packed"),
                "packed_replay_error_max": packed_error_max,
                "replay_fallback_fragments": replay_fallbacks,
            }
            all_metrics = parallel.gather(local_metrics)
            append_json(root / f"rank_{parallel.rank():03d}_metrics.jsonl", local_metrics)
            if parallel.rank() == 0:
                combined_metrics = {
                    key: sum(m[key] for m in all_metrics) / len(all_metrics) for key in totals
                }
                combined_metrics.update(merge_diagnostics(all_metrics))
                combined_metrics.update(
                    grounding_loss=sum(m["grounding_loss"] for m in all_metrics) / len(all_metrics),
                    grounding_opos_loss=sum(m["grounding_opos_loss"] for m in all_metrics)
                    / len(all_metrics),
                    update=update,
                    frames=frames,
                    ppo_coefficient=ppo_weight,
                    ppo_token_gradient_norm_max=max(
                        m["ppo_token_gradient_norm"] for m in all_metrics
                    ),
                    expert_probability=beta,
                    num_expert_envs_per_rank=cfg.get("num_expert_envs", 0),
                    num_dagger_envs_per_rank=cfg.get("num_dagger_envs", 0),
                    imitation_weight_mean=mean_il_weight,
                    replay_log_prob_error_max=max(
                        m["replay_log_prob_error_max"] for m in all_metrics
                    ),
                    grad_norm=max(m["grad_norm"] for m in all_metrics),
                    apos_parameter_gradient_norm=max(
                        m["apos_parameter_gradient_norm"] for m in all_metrics
                    ),
                    episodes=sum(m["episodes"] for m in all_metrics),
                    successes=sum(m["successes"] for m in all_metrics),
                    expert_episodes=sum(m["expert_episodes"] for m in all_metrics),
                    expert_successes=sum(m["expert_successes"] for m in all_metrics),
                    dagger_episodes=sum(m["dagger_episodes"] for m in all_metrics),
                    dagger_successes=sum(m["dagger_successes"] for m in all_metrics),
                    elapsed_seconds=max(m["elapsed_seconds"] for m in all_metrics),
                    rollout_seconds=max(m["rollout_seconds"] for m in all_metrics),
                    optimization_seconds=max(m["optimization_seconds"] for m in all_metrics),
                    action_counts=[
                        sum(m["action_counts"][i] for m in all_metrics) for i in range(6)
                    ],
                    expert_action_counts=[
                        sum(m["expert_action_counts"][i] for m in all_metrics) for i in range(6)
                    ],
                    start_kind_transitions={
                        kind: sum(m["start_kind_transitions"][kind] for m in all_metrics)
                        for kind in ("original", "near", "ready")
                    },
                    opos_peak_probability=sum(m["opos_peak_probability"] for m in all_metrics)
                    / len(all_metrics),
                    peak_vram_bytes=max(m["peak_vram_bytes"] for m in all_metrics),
                    replay_mode=cfg.get("replay_mode", "packed"),
                    packed_replay_error_max=max(m["packed_replay_error_max"] for m in all_metrics),
                    replay_fallback_fragments=sum(
                        m["replay_fallback_fragments"] for m in all_metrics
                    ),
                )
                combined_metrics.update(
                    {name: max(m[name] for m in all_metrics) for name in rollout_timings}
                )
                combined_metrics["apos_loss_per_labeled_frame"] = (
                    combined_metrics["apos_loss"]
                    * combined_metrics["transitions"]
                    / max(1, combined_metrics["apos_geometry_labels"])
                )
                append_json(root / "train_metrics.jsonl", combined_metrics)
                print(json.dumps({"training": combined_metrics}), flush=True)
            write_progress(root, update, "update_complete", 1, 1, started)
            del records, starts
            should_stop = parallel.any_rank(stop[0], policy.device)
            if (
                update == 1
                or update == config.get("fork_origin", {}).get("update", -2) + 1
                or update % cfg["checkpoint_interval"] == 0
                or should_stop
                or update == cfg["num_updates"]
            ):
                write_progress(root, update, "checkpoint", 0, 1, started)
                save(policy, optimizer, source, config, update, frames)
            if not should_stop and (
                update % cfg["eval_interval"] == 0 or update == cfg["num_updates"]
            ):
                save(policy, optimizer, source, config, update, frames)
                write_progress(root, update, "evaluation", 0, 1, started)
                results = evaluate(policy, config, update=update, video=True)
                record_best(config, update, results)
            if should_stop:
                break
    finally:
        envs.close()
        if torch.distributed.is_initialized():
            torch.distributed.destroy_process_group()


if __name__ == "__main__":
    torch.set_num_threads(4)
    main()
