"""Synchronous recurrent LightNav training with point retention, DAgger and PPO.

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
import zipfile
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
from streamnav.data.manifest import check_leakage
from streamnav.data.schema import HabitatEpisodeSource
from streamnav.envs.habitat_client import VectorHabitatEnvs
from streamnav.errors import OracleUnavailableError
from streamnav.models.lightnav_kda.policy import LightNavKDAPolicy, TokenStep
from streamnav.training import distributed as parallel
from streamnav.training.ppo import clipped_ppo_loss
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


def point_kl(logits, probabilities):
    target = probabilities.float()
    target = target / target.sum().clamp_min(1e-12)
    return F.kl_div(F.log_softmax(logits.float(), -1), target, reduction="sum")


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


def replay_losses(step, records, adv, returns, cfg, ppo_weight):
    """Same per-transition objective for sequential and packed replay."""
    device = step.action_logits.device
    experts = torch.tensor([r["expert"] for r in records], device=device)
    student = torch.tensor([r["student_action"] for r in records], device=device)
    old = torch.stack([r["old_log_prob"] for r in records]).to(device)
    il = F.cross_entropy(step.action_logits, experts, reduction="none")
    log_prob = step.log_prob
    ppo, _ = clipped_ppo_loss(log_prob, old, adv.to(device))
    ppo = ppo * student
    value_loss = 0.5 * (step.value - returns.to(device)).square()
    kl = torch.zeros_like(il)
    selected = [i for i, r in enumerate(records) if r["teacher"] is not None]
    if selected:
        indices = torch.tensor(selected, device=device)
        divergence = torch.zeros(len(selected), device=device)
        for j, logits in enumerate((step.apos_logits, step.opos_logits)):
            target = torch.stack([records[i]["teacher"][j] for i in selected]).to(device).float()
            target = target / target.sum(-1, keepdim=True).clamp_min(1e-12)
            divergence = divergence + F.kl_div(
                F.log_softmax(logits.index_select(0, indices).float(), -1),
                target,
                reduction="none",
            ).sum(-1)
        kl = kl.index_copy(0, indices, divergence * cfg["point_teacher_interval"])
    entropy = Categorical(logits=step.action_logits).entropy()
    point_entropy = sum(
        Categorical(logits=x).entropy() for x in (step.apos_logits, step.opos_logits)
    )
    combined = (
        il
        + ppo_weight * ppo
        + cfg["value_loss_coef"] * value_loss
        + cfg["point_retention_coef"] * kl
        - cfg["entropy_coef"] * entropy
        - cfg["point_entropy_coef"] * point_entropy
    )
    return (
        combined,
        {
            "il_loss": il,
            "ppo_loss": ppo,
            "point_kl": kl,
            "value_loss": value_loss,
            "entropy": entropy,
            "loss": combined,
        },
        (log_prob.detach() - old).abs().max(),
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
    if parallel.rank() != 0:
        return
    path = root / "training_progress.json"
    temp = path.with_suffix(".tmp")
    temp.write_text(
        json.dumps(
            {
                "worker_pid": os.getpid(),
                "time": time.time(),
                "update": update,
                "phase": phase,
                "completed": completed,
                "total": total,
                "elapsed_seconds": time.monotonic() - started,
            }
        )
    )
    temp.replace(path)


def save(policy, optimizer, source, config, update, frames=0, rank_payload=None):
    policy.loaded_checkpoint = str(
        (Path(config["run_dir"]) / "checkpoints" / f"update_{update:07d}").resolve()
    )
    adapters = policy.trainable_state()
    digest = sha256()
    for name, value in sorted(adapters.items()):
        digest.update(name.encode())
        digest.update(value.numpy().tobytes())
    payload = {
        "rank": parallel.rank(),
        "rng": rng_state(),
        "source": source.state_dict(),
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
                    "format": "lightnav-kda-adapters-v1",
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

        for old in candidates[:-keep]:
            if old.resolve() != best and old.name != "update_0000000":
                shutil.rmtree(old)
    parallel.barrier()


def load_adapters(policy, folder):
    manifest = json.loads((Path(folder) / "manifest.json").read_text())
    if manifest["base"] != policy.parameter_summary()["source_revision"]:
        raise ValueError("Checkpoint uses a different frozen LightNav base")
    weights = load_file(str(Path(folder) / "adapters.safetensors"), device=str(policy.device))
    named = dict(policy.named_parameters())
    expected = {name for name, p in named.items() if p.requires_grad}
    if set(weights) != expected:
        raise ValueError("Adapter recipe differs from the saved checkpoint")
    with torch.no_grad():
        for name, value in weights.items():
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
    model = dict(config["model"])
    model.pop("language_compute_dtype", None)
    model.pop("recurrent_decode", None)
    model.pop("float32_matmul_precision", None)
    return {
        "model": model,
        "data": config["data"],
        "habitat": habitat,
        "trainer": trainer,
        "seed": config["seed"],
    }


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
    parser.add_argument("--config", default="configs/lightnav_kda.yaml")
    parser.add_argument("--run-dir")
    parser.add_argument("--updates", type=int)
    parser.add_argument("--rollout-steps", type=int)
    parser.add_argument("--sequence-length", type=int)
    parser.add_argument("--num-envs", type=int)
    parser.add_argument("--resume")
    parser.add_argument("--eval-episodes", type=int)
    parser.add_argument("--eval-max-steps", type=int)
    parser.add_argument("--skip-initial-eval", action="store_true")
    args = parser.parse_args()
    config = yaml.safe_load(Path(args.config).read_text())
    config["launch"] = {"resume": args.resume, "skip_initial_eval": args.skip_initial_eval}
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
    if not args.resume and (root / "checkpoints" / "latest").exists():
        raise ValueError("Run already has a checkpoint; use --resume or a new run directory")
    if cfg["rollout_steps"] % cfg["sequence_length"]:
        raise ValueError("Rollout must consist of complete recurrent fragments")
    seed_everything(config["seed"])
    policy = LightNavKDAPolicy(config).train()
    groups: dict[str, list[torch.nn.Parameter]] = {
        "lora": [],
        "memory": [],
        "action": [],
        "critic": [],
    }
    for name, parameter in policy.named_parameters():
        if parameter.requires_grad:
            group = (
                "memory"
                if name.startswith("memories.")
                else (
                    "action"
                    if name.startswith("action_")
                    else ("critic" if name.startswith("critic.") else "lora")
                )
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
    source = HabitatEpisodeSource(
        config["data"]["sources"][0]["manifest"],
        seed=config["seed"] + 1000 * parallel.rank(),
        iterator_options=config["data"]["iterator_options"],
    )
    update, frames = 0, 0
    if args.resume:
        previous_config = yaml.safe_load((Path(args.resume) / "resolved_config.yaml").read_text())
        if resume_recipe(config) != resume_recipe(previous_config):
            raise ValueError("Resume changes the model, observations or optimization recipe")
        load_adapters(policy, args.resume)
        saved = torch.load(
            Path(args.resume) / "optimizer.pt", map_location="cpu", weights_only=False
        )
        if saved["world_size"] != parallel.world_size():
            raise ValueError("Exact resume requires the original world size")
        optimizer.load_state_dict(saved["optimizer"])
        update = saved["update"]
        frames = saved.get(
            "frames", update * cfg["rollout_steps"] * cfg["num_envs"] * parallel.world_size()
        )
        local = saved["ranks"][parallel.rank()]
        source.load_state_dict(local["source"])
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
        temp_source = root / ".source-writing.zip"
        with zipfile.ZipFile(temp_source, "w", zipfile.ZIP_DEFLATED) as archive:
            for folder in ("src", "services", "configs", "scripts", "tools", "docs", "tests"):
                for path in sorted(Path(folder).rglob("*")):
                    if path.is_file() and "__pycache__" not in str(path):
                        archive.write(path, str(path))
            for filename in ("README.md", "pyproject.toml", "uv.lock", "Makefile"):
                archive.write(filename, filename)
        source_hash = sha256(temp_source.read_bytes()).hexdigest()
        source_path = root / "source" / f"{source_hash}.zip"
        source_path.parent.mkdir(exist_ok=True)
        if source_path.exists():
            temp_source.unlink()
        else:
            temp_source.rename(source_path)
        link = root / ".source-new"
        link.unlink(missing_ok=True)
        link.symlink_to(f"source/{source_hash}.zip")
        link.replace(root / "source.zip")
        config["source_snapshot"] = {"sha256": source_hash, "path": str(source_path.resolve())}
        print(json.dumps({"lightnav_kda": policy.parameter_summary()}), flush=True)
    config["source_snapshot"] = parallel.gather(config.get("source_snapshot"))[0]
    if parallel.rank() == 0:
        (root / "resolved_config.yaml").write_text(yaml.safe_dump(config, sort_keys=False))
    stop = [False]

    def request_stop(signum, frame):
        stop[0] = True

    signal.signal(signal.SIGTERM, request_stop)
    signal.signal(signal.SIGINT, request_stop)
    from streamnav.evaluation.runner import evaluate

    if not args.resume:
        save(policy, optimizer, source, config, 0, frames)
    if not args.skip_initial_eval:
        results = evaluate(policy, config, update=update, video=True)
        record_best(config, update, results)
    envs = VectorHabitatEnvs(config["habitat"], cfg["num_envs"])
    episodes = [source.sample_episode() for _ in range(cfg["num_envs"])]
    observations = envs.reset(episodes)
    states = [policy.start_episode(e.uid, e.goal_text) for e in episodes]
    histories: list[list[torch.Tensor]] = [[] for _ in states]
    try:
        while update < cfg["num_updates"]:
            if parallel.any_rank(stop[0], policy.device):
                save(policy, optimizer, source, config, update, frames)
                break
            update += 1
            started = time.monotonic()
            tmax, count = cfg["rollout_steps"], len(states)
            write_progress(root, update, "rollout", 0, tmax, started)
            beta = cfg["expert_warmup_probability"] * max(0.0, 1 - update / cfg["warmup_updates"])
            ppo_weight = cfg["ppo_weight"] * min(
                1.0, max(0.0, (update - cfg["warmup_updates"]) / cfg["ppo_ramp_updates"])
            )
            records: list[list[dict[str, Any]]] = []
            starts, resets, metrics = {}, {}, []
            timeout_values = torch.zeros(tmax, count)
            with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                for t in range(tmax):
                    experts = envs.get_oracle_actions()
                    for i, expert in enumerate(experts):
                        retries = 0
                        while isinstance(expert, OracleUnavailableError):
                            if t:
                                records[t - 1][i]["done"] = True
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
                            histories[i] = []
                            resets[(t, i)] = (episode.uid, episode.goal_text)
                            try:
                                expert = envs.clients[i].get_oracle_action()
                            except OracleUnavailableError as exc:
                                expert = exc
                            retries += 1
                            if retries >= 3 and isinstance(expert, OracleUnavailableError):
                                raise expert
                        experts[i] = expert
                    row, steps = [], []
                    for i, state in enumerate(states):
                        if t % cfg["sequence_length"] == 0:
                            starts[(t, i)] = cache_copy(state)
                        feature, frame = policy.frame_features(
                            observations[i]["rgb"], state.instruction
                        )
                        use_teacher = bool(torch.rand(()) < beta)
                        forced = (
                            (None, None, torch.tensor(int(experts[i]), device=policy.device))
                            if use_teacher
                            else None
                        )
                        step = policy.step(feature, state, tokens=forced)
                        teacher = None
                        if t % cfg["point_teacher_interval"] == 0:
                            teacher_feature, _ = policy.frame_features(
                                observations[i]["rgb"],
                                state.instruction,
                                history=histories[i][-cfg["teacher_history_frames"] :],
                            )
                            teacher = tuple(
                                p.cpu().half()
                                for p in policy.teacher_points(teacher_feature, state, step.apos)
                            )
                        histories[i].append(frame)
                        histories[i] = histories[i][-cfg["teacher_history_frames"] :]
                        steps.append(step)
                        row.append(
                            {
                                "features": feature.cpu(),
                                "tokens": tuple(
                                    x.cpu() for x in (step.apos, step.opos, step.action)
                                ),
                                "old_log_prob": step.log_prob.cpu(),
                                "value": step.value.cpu(),
                                "expert": int(experts[i]),
                                "teacher": teacher,
                                "student_action": not use_teacher,
                            }
                        )
                    following = envs.step([NavigationAction(s.action.item()) for s in steps])
                    for i, (step, after) in enumerate(zip(steps, following, strict=True)):
                        row[i].update(reward=after["reward"], done=after["done"])
                        states[i] = step.state
                        if after["done"]:
                            metrics.append(after["metrics"])
                            if after.get("truncated", False):
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
                            histories[i] = []
                            resets[(t + 1, i)] = (episode.uid, episode.goal_text)
                    observations = following
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
                last = []
                for observation, state in zip(observations, states, strict=True):
                    context = policy.begin(
                        policy.frame_features(observation["rgb"], state.instruction)[0], state
                    )
                    last.append(policy.read_value(context.hidden).squeeze().cpu())
            rollout_seconds = time.monotonic() - started
            rewards = torch.tensor([[r["reward"] for r in row] for row in records])
            dones = torch.tensor([[r["done"] for r in row] for row in records])
            values = torch.stack([torch.stack([r["value"] for r in row]) for row in records])
            adv, returns = advantages(
                rewards, values, dones, torch.stack(last), timeout_values=timeout_values[:tmax]
            )
            if ppo_weight:
                student_mask = torch.tensor([[r["student_action"] for r in row] for row in records])
                adv = parallel.normalize_advantages(adv, policy.device, mask=student_mask)
            optimizer.zero_grad(set_to_none=True)
            totals = {
                k: 0.0 for k in ("il_loss", "ppo_loss", "point_kl", "value_loss", "entropy", "loss")
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
            parallel.average_gradients(policy._all_trainable)
            grad_norm = torch.nn.utils.clip_grad_norm_(policy._all_trainable, cfg["max_grad_norm"])
            optimizer.step()
            frames += tmax * count * parallel.world_size()
            for i, state in enumerate(states):
                states[i] = cache_copy(state, policy.device)
            local_metrics = {
                **totals,
                "update": update,
                "ppo_coefficient": ppo_weight,
                "ppo_token_gradient_norm": ppo_token_gradient_norm,
                "expert_probability": beta,
                "grad_norm": float(grad_norm),
                "replay_log_prob_error_max": max_error,
                "opos_peak_probability": sum(peak_point) / len(peak_point),
                "episodes": len(metrics),
                "successes": sum(m["success"] for m in metrics),
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
                combined_metrics.update(
                    update=update,
                    frames=frames,
                    ppo_coefficient=ppo_weight,
                    ppo_token_gradient_norm_max=max(
                        m["ppo_token_gradient_norm"] for m in all_metrics
                    ),
                    expert_probability=beta,
                    replay_log_prob_error_max=max(
                        m["replay_log_prob_error_max"] for m in all_metrics
                    ),
                    grad_norm=max(m["grad_norm"] for m in all_metrics),
                    episodes=sum(m["episodes"] for m in all_metrics),
                    successes=sum(m["successes"] for m in all_metrics),
                    elapsed_seconds=max(m["elapsed_seconds"] for m in all_metrics),
                    rollout_seconds=max(m["rollout_seconds"] for m in all_metrics),
                    optimization_seconds=max(m["optimization_seconds"] for m in all_metrics),
                    action_counts=[
                        sum(m["action_counts"][i] for m in all_metrics) for i in range(6)
                    ],
                    expert_action_counts=[
                        sum(m["expert_action_counts"][i] for m in all_metrics) for i in range(6)
                    ],
                    opos_peak_probability=sum(m["opos_peak_probability"] for m in all_metrics)
                    / len(all_metrics),
                    peak_vram_bytes=max(m["peak_vram_bytes"] for m in all_metrics),
                    replay_mode=cfg.get("replay_mode", "packed"),
                    packed_replay_error_max=max(m["packed_replay_error_max"] for m in all_metrics),
                    replay_fallback_fragments=sum(
                        m["replay_fallback_fragments"] for m in all_metrics
                    ),
                )
                append_json(root / "train_metrics.jsonl", combined_metrics)
                print(json.dumps({"training": combined_metrics}), flush=True)
            write_progress(root, update, "update_complete", 1, 1, started)
            del records, starts
            should_stop = parallel.any_rank(stop[0], policy.device)
            if (
                update == 1
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
