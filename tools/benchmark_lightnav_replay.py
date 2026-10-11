"""Compare sequential/packed recurrent replay on identical real Habitat frames."""

import argparse
import json
import statistics
import time
from pathlib import Path

import torch
import yaml

from streamnav.contracts.action import NavigationAction
from streamnav.data.schema import HabitatEpisodeSource
from streamnav.envs.habitat_client import HabitatClient
from streamnav.models.lightnav_kda.policy import LightNavKDAPolicy
from streamnav.training.lightnav_trainer import (
    advantages,
    cache_copy,
    guarded_replay_records,
    load_adapters,
    replay_losses,
    replay_records,
)
from streamnav.utils.seed import seed_everything


def collect(policy, cfg, steps):
    source = HabitatEpisodeSource(cfg["data"]["sources"][0]["manifest"], seed=cfg["seed"])
    env = HabitatClient(cfg["habitat"])
    episode = source.sample_episode()
    observation = env.reset(episode)
    state = policy.start_episode(episode.uid, episode.goal_text)
    records, resets, history = [], {}, []
    try:
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            # Exercise a carried cache rather than only the all-zero initial state.
            for _ in range(2):
                feature, _ = policy.frame_features(observation["rgb"], state.instruction)
                step = policy.step(feature, state, deterministic=True)
                state = step.state
            initial = cache_copy(state)
            for t in range(steps):
                if t == steps // 2:
                    records[-1]["done"] = True
                    episode = source.sample_episode()
                    observation = env.reset(episode)
                    state = policy.start_episode(episode.uid, episode.goal_text)
                    history = []
                    resets[t] = (episode.uid, episode.goal_text)
                feature, frame = policy.frame_features(observation["rgb"], state.instruction)
                expert = env.get_oracle_action()
                student = t % 3 != 0
                forced = (
                    None
                    if student
                    else (None, None, torch.tensor(int(expert), device=policy.device))
                )
                step = policy.step(feature, state, tokens=forced)
                teacher = None
                if t % cfg["trainer"]["point_teacher_interval"] == 0:
                    teacher_feature, _ = policy.frame_features(
                        observation["rgb"], state.instruction, history=history[-8:]
                    )
                    teacher = tuple(
                        p.cpu().half()
                        for p in policy.teacher_points(teacher_feature, state, step.apos)
                    )
                after = env.step(NavigationAction(int(step.action)))
                records.append(
                    {
                        "features": feature.cpu(),
                        "tokens": tuple(x.cpu() for x in (step.apos, step.opos, step.action)),
                        "old_log_prob": step.log_prob.cpu(),
                        "value": step.value.cpu(),
                        "expert": int(expert),
                        "student_action": student,
                        "teacher": teacher,
                        "reward": after["reward"],
                        "done": after["done"],
                    }
                )
                state = step.state
                history.append(frame)
                observation = after
                if after["done"]:
                    episode = source.sample_episode()
                    observation = env.reset(episode)
                    state = policy.start_episode(episode.uid, episode.goal_text)
                    history = []
                    resets[t + 1] = (episode.uid, episode.goal_text)
            context = policy.begin(
                policy.frame_features(observation["rgb"], state.instruction)[0], state
            )
            last = policy.read_value(context.hidden).squeeze().cpu()
        resets.pop(steps, None)
        adv, returns = advantages(
            torch.tensor([r["reward"] for r in records]),
            torch.stack([r["value"] for r in records]),
            torch.tensor([r["done"] for r in records]),
            last,
        )
        return records, initial, resets, adv, returns
    finally:
        env.close()


def gradient_comparison(current, reference):
    groups = {}
    for name, original in reference.items():
        actual = current[name]
        group = (
            "memory"
            if name.startswith("memories")
            else ("lora" if ".self_attn." in name else "action_critic")
        )
        d = groups.setdefault(
            group, {"reference_sq": 0.0, "actual_sq": 0.0, "delta_sq": 0.0, "dot": 0.0}
        )
        a, b = actual.double(), original.double()
        d["reference_sq"] += b.square().sum().item()
        d["actual_sq"] += a.square().sum().item()
        d["delta_sq"] += (a - b).square().sum().item()
        d["dot"] += (a * b).sum().item()
    return {
        g: {
            "relative_l2_difference": (d["delta_sq"] / max(d["reference_sq"], 1e-30)) ** 0.5,
            "cosine": d["dot"] / max((d["reference_sq"] * d["actual_sq"]) ** 0.5, 1e-30),
        }
        for g, d in groups.items()
    }


def validate_causality(policy, records, initial, resets):
    tokens = [tuple(x.to(policy.device) for x in r["tokens"]) for r in records]
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        state = cache_copy(initial, policy.device)
        saved = [s.recurrent.clone() if s.recurrent is not None else None for s in state.kda_cache]
        base = policy.replay_fragment(
            [r["features"] for r in records], state, tokens, resets=resets
        )
        changed = list(tokens)
        changed[-1] = (
            (tokens[-1][0] + 1) % 1300,
            (tokens[-1][1] + 1) % 1297,
            (tokens[-1][2] + 1) % 6,
        )
        future = policy.replay_fragment(
            [r["features"] for r in records], state, changed, resets=resets
        )
        future_error = max(
            (getattr(base, n)[:-1] - getattr(future, n)[:-1]).abs().max().item()
            for n in ("apos_logits", "opos_logits", "action_logits")
        )
        unchanged = all(
            torch.equal(a, s.recurrent)
            for a, s in zip(saved, state.kda_cache, strict=True)
            if a is not None
        )
        first_reset = min(i for i in resets if i > 0)
        perturbed = cache_copy(initial, policy.device)
        for s in perturbed.kda_cache:
            if s.recurrent is not None:
                s.recurrent.add_(0.1)
        different = policy.replay_fragment(
            [r["features"] for r in records], perturbed, tokens, resets=resets
        )
        reset_error = max(
            (getattr(base, n)[first_reset:] - getattr(different, n)[first_reset:])
            .abs()
            .max()
            .item()
            for n in ("apos_logits", "opos_logits", "action_logits")
        )
        prefix = records[:first_reset]
        sequential = replay_records(
            policy, prefix, cache_copy(initial, policy.device), mode="sequential"
        )
        packed = replay_records(policy, prefix, cache_copy(initial, policy.device), mode="packed")
        no_reset_error = (sequential.log_prob - packed.log_prob).abs().max().item()
        guarded, _raw_error, fallback = guarded_replay_records(
            policy, prefix, cache_copy(initial, policy.device)
        )
        guarded_error = (sequential.log_prob - guarded.log_prob).abs().max().item()
    policy.zero_grad(set_to_none=True)
    state = cache_copy(initial, policy.device)
    for s in state.kda_cache:
        if s.recurrent is not None:
            s.recurrent.requires_grad_(True)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        step = replay_records(policy, records, state, resets=resets)
        loss = -step.log_prob[first_reset:].mean()
    loss.backward()
    boundary_gradient = max(
        s.recurrent.grad.abs().max().item()
        for s in state.kda_cache
        if s.recurrent is not None and s.recurrent.grad is not None
    )
    result = {
        "future_tokens_affect_past_logits_max": future_error,
        "previous_episode_affects_post_reset_logits_max": reset_error,
        "post_reset_gradient_to_previous_episode_max": boundary_gradient,
        "cache_snapshot_unchanged": unchanged,
        "no_reset_replay_error_max": no_reset_error,
        "no_reset_guarded_error_max": guarded_error,
        "no_reset_guard_fallback": fallback,
    }
    result["passed"] = (
        future_error == 0
        and reset_error == 0
        and boundary_gradient == 0
        and unchanged
        and guarded_error < 0.05
    )
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", default="runs/lightnav_kda_active/checkpoints/latest")
    p.add_argument("--output", default="runtime/reports/lightnav_efficiency_20261011")
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--steps", type=int, default=20)
    p.add_argument("--repeats", type=int, default=2)
    p.add_argument("--buffer-in")
    p.add_argument("--checks-only", action="store_true")
    p.add_argument("--language-compute-dtype", choices=("float32", "bfloat16"), default="float32")
    p.add_argument("--float32-matmul-precision", choices=("highest", "high"), default="high")
    args = p.parse_args()
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    cfg = yaml.safe_load((Path(args.checkpoint) / "resolved_config.yaml").read_text())
    cfg["device"] = args.device
    cfg["model"]["language_compute_dtype"] = args.language_compute_dtype
    cfg["model"]["float32_matmul_precision"] = args.float32_matmul_precision
    cfg["habitat"]["gpu_device_id"] = int(args.device.split(":")[-1])
    seed_everything(cfg["seed"])
    policy = LightNavKDAPolicy(cfg).train()
    load_adapters(policy, args.checkpoint)
    if args.buffer_in:
        data = torch.load(args.buffer_in, map_location="cpu", weights_only=False)
    else:
        data = collect(policy, cfg, args.steps)
        torch.save(data, output / "real_rollout.pt")
    records, initial, resets, adv, returns = data
    if args.checks_only:
        checks = validate_causality(policy, records, initial, resets)
        (output / "causality_checks.json").write_text(json.dumps(checks, indent=2) + "\n")
        print(json.dumps(checks, indent=2), flush=True)
        if not checks["passed"]:
            raise RuntimeError("Packed replay violated causality or episode isolation")
        return
    results, reference = {}, None
    print(
        json.dumps(
            {
                "phase": "collected",
                "steps": len(records),
                "resets": sorted(resets),
                "prefix_lengths": [r["features"].ids.shape[-1] for r in records],
            }
        ),
        flush=True,
    )
    for mode in ("sequential", "packed"):
        samples = []
        for repeat in range(args.repeats):
            policy.zero_grad(set_to_none=True)
            state = cache_copy(initial, policy.device)
            torch.cuda.reset_peak_memory_stats(policy.device)
            torch.cuda.synchronize(policy.device)
            start = time.perf_counter()
            with torch.autocast("cuda", dtype=torch.bfloat16):
                step = replay_records(policy, records, state, resets=resets, mode=mode)
                loss, components, error = replay_losses(
                    step, records, adv, returns, cfg["trainer"], 0.5
                )
                loss = loss.mean()
            loss.backward()
            torch.cuda.synchronize(policy.device)
            elapsed = time.perf_counter() - start
            sample = {
                "mode": mode,
                "repeat": repeat,
                "seconds": elapsed,
                "peak_vram_bytes": torch.cuda.max_memory_allocated(policy.device),
                "loss": loss.item(),
                "replay_log_prob_error_max": error.item(),
            }
            samples.append(sample)
            print(json.dumps(sample), flush=True)
        gradients = {
            name: p.grad.detach().cpu().clone()
            for name, p in policy.named_parameters()
            if p.grad is not None
        }
        logits = [
            x.detach().cpu() for x in (step.apos_logits, step.opos_logits, step.action_logits)
        ]
        cache = [
            s.recurrent.detach().cpu() for s in step.state.kda_cache if s.recurrent is not None
        ]
        results[mode] = {
            "samples": samples,
            "warm_seconds": statistics.mean(s["seconds"] for s in samples[1:] or samples),
        }
        if reference is None:
            reference = (gradients, logits, cache)
        else:
            results["comparison"] = {
                "same_gradient_parameters": set(gradients) == set(reference[0]),
                "gradients": gradient_comparison(gradients, reference[0]),
                "logits_max_absolute_difference": [
                    (a.float() - b.float()).abs().max().item()
                    for a, b in zip(logits, reference[1], strict=True)
                ],
                "final_cache_relative_l2": (
                    sum(
                        (a - b).double().square().sum().item()
                        for a, b in zip(cache, reference[2], strict=True)
                    )
                    / max(sum(b.double().square().sum().item() for b in reference[2]), 1e-30)
                )
                ** 0.5,
            }
        del step, loss, components, state
        torch.cuda.empty_cache()
    results["checkpoint"] = str(Path(args.checkpoint).resolve())
    results["compute"] = {
        "language_dtype": str(policy.language_compute_dtype),
        "matmul_precision": torch.get_float32_matmul_precision(),
        "recurrent_decode": policy.settings.get("recurrent_decode", True),
    }
    results["speedup"] = results["sequential"]["warm_seconds"] / results["packed"]["warm_seconds"]
    results["passed"] = (
        results["packed"]["samples"][-1]["replay_log_prob_error_max"]
        < cfg["trainer"]["replay_tolerance"]
        and results["comparison"]["same_gradient_parameters"]
        and all(
            g["cosine"] > 0.995 and g["relative_l2_difference"] < 0.1
            for g in results["comparison"]["gradients"].values()
        )
    )
    (output / "replay_comparison.json").write_text(json.dumps(results, indent=2) + "\n")
    print(json.dumps(results, indent=2), flush=True)
    if not results["passed"]:
        raise RuntimeError("Packed replay did not match the sequential reference")


if __name__ == "__main__":
    torch.set_num_threads(4)
    main()
