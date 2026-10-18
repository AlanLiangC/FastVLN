"""Streaming navigation with instance points, optional hints, DAgger/PPO and grounding.

Rollouts store the actual APOS/OPOS/action block, not teacher-forced points.
All fragments pass a pre-update probability check before synchronized mini-batch
optimizer steps, each accumulating gradients across its recurrent fragments.
Only student-selected transitions are used by PPO, after the expert warmup ends.
"""

import argparse
import json
import math
import os
import random
import signal
import sys
import time
import traceback
from concurrent.futures import wait
from dataclasses import replace
from hashlib import sha256
from pathlib import Path
from typing import Any

import torch
import yaml
from safetensors.torch import load_file, save_file
from torch.distributions import Categorical
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint

from streamnav.contracts.action import NavigationAction
from streamnav.contracts.state import LayerState
from streamnav.data.grounding import GroundingSource, check_grounding_split
from streamnav.data.manifest import check_leakage
from streamnav.data.mixed_starts import MixedStartEpisodeSource
from streamnav.data.schema import HabitatEpisodeSource
from streamnav.envs.habitat_client import VectorHabitatEnvs
from streamnav.errors import OracleUnavailableError
from streamnav.models.point_cot.policy import Qwen35PointCoTPolicy, TokenStep
from streamnav.models.policy.perception import (
    decode_point_logits,
    point_type_logits,
    point_type_targets,
)
from streamnav.training import distributed as parallel
from streamnav.training.action_planning import attach_future_actions, future_action_loss
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
from streamnav.training.object_hints import episode_hint_enabled, simulated_object_hint
from streamnav.training.object_supervision import object_loss_weights, summarize_object_predictions
from streamnav.training.optimizer_migration import load_optimizer_with_added_groups
from streamnav.training.point_supervision import (
    forward_intent_loss,
    grounding_loss,
    grounding_loss_options,
    marginal_point_loss,
    navigation_object_losses,
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


class ReplayMismatch(RuntimeError):
    """A synchronized preflight rejection, before any optimizer mutation."""


def stack_steps(steps):
    return TokenStep(
        torch.stack([s.apos for s in steps]),
        torch.stack([s.opos for s in steps]),
        torch.stack([s.action for s in steps]),
        torch.stack([s.apos_logits for s in steps]),
        torch.stack([s.opos_logits for s in steps]),
        torch.stack([s.action_logits for s in steps]),
        torch.stack([s.value for s in steps]),
        steps[-1].state,
        torch.stack([s.action_plan_logits for s in steps])
        if steps[0].action_plan_logits is not None
        else None,
    )


def replay_batched_records(policy, records, states, *, resets=None):
    """Reproduce the rollout's environment batch, including reset/padding shapes.

    All tokens are forced, so replay consumes no sampling RNG. The recurrent
    graph remains connected across the whole fragment in differentiable replay.
    """
    resets = resets or {}
    states = list(states)
    histories: list[list[TokenStep]] = [[] for _ in states]
    for t, row in enumerate(records):
        for i in range(len(states)):
            if (t, i) in resets:
                states[i] = policy.start_episode(*resets[t, i])
        # Checkpoint recomputation must see this step's input states, not the
        # list updated below with this step's outputs.
        args = ([r["features"] for r in row], list(states))
        kwargs: dict[str, Any] = {
            "tokens": [tuple(x.to(policy.device) for x in r["tokens"]) for r in row],
            "deterministic": True,
        }
        # A step's native attention/conv caches are internal to that frame.
        # Retain only outputs and recurrent KDA states across time; recompute
        # internal activations during backward without detaching the trajectory.
        steps = (
            checkpoint(
                policy.step_batch, *args, **kwargs, use_reentrant=False, preserve_rng_state=False
            )
            if torch.is_grad_enabled()
            else policy.step_batch(*args, **kwargs)
        )
        for i, step in enumerate(steps):
            histories[i].append(step)
            states[i] = step.state
    return [stack_steps(history) for history in histories]


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
        step = (
            checkpoint(
                policy.step,
                record["features"],
                state,
                tokens=token,
                use_reentrant=False,
                preserve_rng_state=False,
            )
            if torch.is_grad_enabled()
            else policy.step(record["features"], state, tokens=token)
        )
        state = step.state
        steps.append(step)
    return stack_steps(steps)


def replay_probability_error(step, records, device):
    old = torch.stack([r["old_log_prob"] for r in records]).to(device)
    delta = (step.log_prob.detach() - old).abs()
    return float(delta.max()) if bool(torch.isfinite(delta).all()) else math.inf


def replay_losses(
    policy,
    state,
    resets,
    step,
    records,
    adv,
    returns,
    cfg,
    ppo_weight,
    *,
    checkpoint_teacher=False,
    include_teacher=True,
):
    """Replay actual point histories; optionally mix teacher-prefixed action IL."""
    device = step.action_logits.device
    experts = torch.tensor([r["expert"] for r in records], device=device)
    labels = [targets_from_label(r.get("perception_label")) for r in records]
    teacher_args = (
        policy,
        [r["features"] for r in records],
        state,
        [r["tokens"] for r in records],
        labels,
    )
    if include_teacher:
        teacher, teacher_apos, _, teacher_il = (
            checkpoint(
                supervised_fragment,
                *teacher_args,
                resets=resets,
                experts=experts,
                use_reentrant=False,
                preserve_rng_state=False,
            )
            if checkpoint_teacher
            else supervised_fragment(*teacher_args, resets=resets, experts=experts)
        )
    else:
        if cfg.get("point_supervision_prefix", "teacher") != "actual":
            raise ValueError("Separate teacher backward requires actual-history point supervision")
        teacher = None
        teacher_apos = teacher_il = torch.zeros_like(step.value)
    if cfg.get("point_supervision_prefix", "teacher") == "actual":
        apos = marginal_point_loss(step.apos_logits, [label["apos_targets"] for label in labels])
    else:
        apos = teacher_apos
    opos, opos_visibility, opos_location = navigation_object_losses(
        step.opos_logits,
        [label["opos_targets"] for label in labels],
        anchor_weight=cfg.get("opos_supervision", {}).get("anchor_weight", 0.0),
    )
    opos_valid = step.opos_logits.new_tensor([bool(label["opos_targets"]) for label in labels])
    visibility_weights = step.opos_logits.new_tensor(
        [r.get("opos_visibility_weight", r.get("opos_loss_weight", 1.0)) for r in records]
    )
    location_weights = step.opos_logits.new_tensor(
        [r.get("opos_location_weight", r.get("opos_loss_weight", 1.0)) for r in records]
    )
    opos_visibility = opos_visibility * visibility_weights
    opos_location = opos_location * location_weights
    opos = opos_visibility + opos_location
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
    forward, forward_mask = forward_intent_loss(step.apos_logits, labels, experts)
    future = (
        (1 - cfg["teacher_action_mix"]) * future_action_loss(step.action_plan_logits, records)
        + (
            cfg["teacher_action_mix"] * future_action_loss(teacher.action_plan_logits, records)
            if teacher is not None
            else 0
        )
        if step.action_plan_logits is not None
        else step.action_logits.sum(-1) * 0
    )
    predicted_il = F.cross_entropy(step.action_logits, experts, reduction="none")
    mix = cfg["teacher_action_mix"]
    il = ((1 - mix) * predicted_il + mix * teacher_il) * predicted_il.new_tensor(
        [r["il_weight"] for r in records]
    )
    old = torch.stack([r["old_log_prob"] for r in records]).to(device)
    weights = student_rl_weights(records, cfg, device)
    eligible = torch.tensor([r["student_action"] for r in records], device=device)
    log_probability = step.log_prob
    ppo, _ = clipped_ppo_loss(log_probability, old, adv.to(device), clip_eps=cfg["ppo_clip"])
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
        + cfg["point_il_coef"]
        * (
            apos
            + opos
            + cfg.get("apos_type_il_coef", 0) * apos_type
            + cfg.get("apos_forward_intent_coef", 0) * forward
        )
        + cfg.get("future_action_loss_coef", 0) * future
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
            "apos_forward_intent_loss": forward,
            "apos_forward_intent_fraction": forward_mask.float(),
            "apos_teacher_prefix_loss": teacher_apos,
            "future_action_loss": future,
            "opos_loss": opos,
            "opos_visibility_loss": opos_visibility,
            "opos_location_loss": opos_location,
            "opos_label_fraction": opos_valid,
            "opos_visible_fraction": opos_valid.new_tensor(
                [bool(label["opos_targets"]) and label["opos_targets"] != [0] for label in labels]
            ),
            "opos_absent_fraction": opos_valid.new_tensor(
                [label["opos_targets"] == [0] for label in labels]
            ),
            "object_hint_fraction": opos_valid.new_tensor(
                [bool(r.get("object_hint_token", 0)) for r in records]
            ),
            "ppo_loss": ppo,
            "value_loss": value_loss,
            "entropy": entropy,
            "loss": loss,
        },
        (log_probability.detach() - old).abs().max(),
    )


def teacher_fragment_losses(policy, state, resets, records, cfg):
    """The independent teacher branch, after releasing the actual-history graph.

    Its gradients accumulate into the same minibatch without an optimizer step
    between branches. This keeps the original objective and avoids holding two
    complete language-model activation graphs at once.
    """
    experts = torch.tensor([r["expert"] for r in records], device=policy.device)
    teacher, apos, _, action = supervised_fragment(
        policy,
        [r["features"] for r in records],
        state,
        [r["tokens"] for r in records],
        [targets_from_label(r.get("perception_label")) for r in records],
        resets=resets,
        experts=experts,
    )
    mix = cfg["teacher_action_mix"]
    il = mix * action * action.new_tensor([r["il_weight"] for r in records])
    future = (
        mix * future_action_loss(teacher.action_plan_logits, records)
        if teacher.action_plan_logits is not None
        else torch.zeros_like(il)
    )
    loss = il + cfg.get("future_action_loss_coef", 0) * future
    return loss, {
        "il_loss": il,
        "future_action_loss": future,
        "apos_teacher_prefix_loss": apos,
        "loss": loss,
    }


def guarded_replay_records(policy, records, state, *, resets=None, mode="packed", tolerance=0.05):
    step = replay_records(policy, records, state, resets=resets, mode=mode)
    error = replay_probability_error(step, records, policy.device)
    fallback = mode == "packed" and (not math.isfinite(error) or error > tolerance)
    if fallback:
        # Recompute the entire fragment to preserve its recurrent gradients.
        del step
        step = replay_records(policy, records, state, resets=resets, mode="sequential")
    return step, error, fallback


def optimize_rollout(
    policy,
    optimizer,
    grounding,
    records,
    starts,
    resets,
    adv,
    returns,
    cfg,
    ground_cfg,
    ppo_weight,
    root,
    update,
    started,
):
    """Validate behavior replay first, then allow legitimate PPO policy changes."""
    tmax, count, length = len(records), len(records[0]), cfg["sequence_length"]
    blocks = list(range(0, tmax, length))
    batches = min(cfg.get("ppo_num_mini_batches", 1), len(blocks))
    split_teacher = (
        cfg.get("split_teacher_replay", True)
        and cfg.get("point_supervision_prefix", "teacher") == "actual"
    )
    fragments = [(t, i) for t in blocks for i in range(count)]
    modes, packed_error, replay_error, fallbacks = {}, 0.0, 0.0, 0
    errors: dict[tuple[int, int], dict[str, float]] = {}
    batched_blocks: set[int] = set()

    def fragment_data(t0, i):
        return (
            cache_copy(starts[(t0, i)], policy.device),
            [records[t][i] for t in range(t0, t0 + length)],
            {
                t - t0: episode
                for (t, env), episode in resets.items()
                if env == i and t0 <= t < t0 + length
            },
        )

    def batch_replay(t0):
        return replay_batched_records(
            policy,
            records[t0 : t0 + length],
            [cache_copy(starts[t0, i], policy.device) for i in range(count)],
            resets={
                (t - t0, i): episode for (t, i), episode in resets.items() if t0 <= t < t0 + length
            },
        )

    def validate_errors(stage):
        replay_error = max(row.get(stage, 0.0) for row in errors.values())
        if not parallel.any_rank(replay_error > cfg["replay_tolerance"], policy.device):
            return replay_error
        folder = root / "replay_failures" / f"update_{update:07d}"
        folder.mkdir(parents=True, exist_ok=True)
        for t0 in blocks:
            if max(errors[t0, i].get(stage, 0.0) for i in range(count)) <= cfg["replay_tolerance"]:
                continue
            torch.save(
                {
                    "records": records[t0 : t0 + length],
                    "states": [starts[t0, i] for i in range(count)],
                    "resets": {
                        (t - t0, i): episode
                        for (t, i), episode in resets.items()
                        if t0 <= t < t0 + length
                    },
                },
                folder / f"rank_{parallel.rank():03d}_block_{t0:03d}.pt",
            )
        reports = parallel.gather(
            {
                "rank": parallel.rank(),
                "max_error": replay_error,
                "fragments": [{"t": t, "env": i, **row} for (t, i), row in errors.items()],
            }
        )
        if parallel.rank() == 0:
            (folder / f"{stage}_errors.json").write_text(json.dumps(reports, indent=2) + "\n")
        worst = max(reports, key=lambda row: row["max_error"])
        raise ReplayMismatch(
            f"Rejected behavior replay on rank {worst['rank']}: {worst['max_error']:.6f}; "
            f"stage: {stage}; diagnostics: {folder}"
        )

    # Old probabilities must match only before the first optimizer step.
    # A later minibatch intentionally evaluates a changed policy against them.
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        for index, (t0, i) in enumerate(fragments):
            state, fragment, boundaries = fragment_data(t0, i)
            step, raw, fallback = guarded_replay_records(
                policy,
                fragment,
                state,
                resets=boundaries,
                mode=cfg.get("replay_mode", "packed"),
                tolerance=cfg["replay_tolerance"],
            )
            error = replay_probability_error(step, fragment, policy.device)
            packed_error = max(packed_error, raw)
            errors[t0, i] = {"initial": raw, "sequential": error, "accepted": error}
            if error > cfg["replay_tolerance"]:
                batched_blocks.add(t0)
            fallbacks += int(fallback)
            modes[t0, i] = "sequential" if fallback else cfg.get("replay_mode", "packed")
            write_progress(root, update, "replay_preflight", index + 1, len(fragments), started)
            del step
        # Single-environment replay still changes GEMM dimensions and padding.
        # Retry the complete original batch before rejecting a valid rollout.
        for index, t0 in enumerate(sorted(batched_blocks)):
            steps = batch_replay(t0)
            for i, step in enumerate(steps):
                fragment = [records[t][i] for t in range(t0, t0 + length)]
                error = replay_probability_error(step, fragment, policy.device)
                errors[t0, i].update(batched=error, accepted=error)
                modes[t0, i] = "batched"
            del steps, step
            write_progress(
                root, update, "replay_batch_fallback", index + 1, len(batched_blocks), started
            )
    replay_error = validate_errors("accepted")

    if batched_blocks:
        # Original-batch BPTT retains all environments' activations at once.
        # Release inference graph pools first; they can be recaptured next rollout.
        policy.inference_graphs.entries.clear()
        torch.cuda.empty_cache()

    totals: dict[str, float] = {}
    peaks, norms, apos_norms, opos_norms, auxiliary_metrics = [], [], [], [], []
    opos_no_hint_gradient = 0.0
    policy_change, ppo_gradient, differentiable_error = 0.0, 0.0, 0.0
    clips, kls, eligible_count, completed = 0.0, 0.0, 0, 0
    for batch in range(batches):
        selected = [
            (t, i) for j, t in enumerate(blocks) if j % batches == batch for i in range(count)
        ]
        optimizer.zero_grad(set_to_none=True)
        write_progress(root, update, "optimization", completed, len(fragments), started)

        def replay_groups():
            for t0 in blocks:
                if (t0, 0) not in selected:
                    continue
                if t0 in batched_blocks:
                    with torch.autocast("cuda", dtype=torch.bfloat16):
                        steps = batch_replay(t0)
                    yield [(t0, i, step) for i, step in enumerate(steps)]
                    del steps
                else:
                    for i in range(count):
                        state, fragment, boundaries = fragment_data(t0, i)
                        with torch.autocast("cuda", dtype=torch.bfloat16):
                            step = replay_records(
                                policy, fragment, state, resets=boundaries, mode=modes[t0, i]
                            )
                        yield [(t0, i, step)]
                        del step, state, fragment, boundaries

        for group in replay_groups():
            group_loss = None
            for t0, i, step in group:
                state, fragment, boundaries = fragment_data(t0, i)
                # Accumulate the shared original-batch graph before backward.
                # Ordinary packed fragments remain one group each.
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    combined, components, change = replay_losses(
                        policy,
                        state,
                        boundaries,
                        step,
                        fragment,
                        adv[t0 : t0 + length, i],
                        returns[t0 : t0 + length, i],
                        cfg,
                        ppo_weight,
                        checkpoint_teacher=len(group) > 1,
                        include_teacher=not split_teacher,
                    )
                fragment_loss = combined.sum() / (len(selected) * length)
                group_loss = fragment_loss if group_loss is None else group_loss + fragment_loss
                policy_change = max(policy_change, float(change))
                if batch == 0:
                    errors[t0, i]["differentiable"] = replay_probability_error(
                        step, fragment, policy.device
                    )
                if not opos_no_hint_gradient:
                    visible_no_hint = [
                        j
                        for j, r in enumerate(fragment)
                        if not r.get("object_hint_token", 0)
                        and (r.get("perception_label") or {}).get("opos_valid")
                        and r["perception_label"]["opos"] > 0
                    ]
                    if visible_no_hint:
                        gradient = torch.autograd.grad(
                            cfg["point_il_coef"] * components["opos_loss"][visible_no_hint[0]],
                            step.opos_logits,
                            retain_graph=True,
                        )[0]
                        opos_no_hint_gradient = float(gradient.norm())
                eligible = [j for j, r in enumerate(fragment) if r["student_action"]]
                if eligible:
                    old = torch.stack([r["old_log_prob"] for r in fragment]).to(policy.device)
                    delta = (step.log_prob.detach() - old)[eligible]
                    ratio = delta.exp()
                    clips += float(((ratio - 1).abs() > cfg["ppo_clip"]).sum())
                    kls += float((ratio - 1 - delta).sum())
                    eligible_count += len(eligible)
                    if ppo_weight and not ppo_gradient:
                        witness = torch.autograd.grad(
                            ppo_weight * components["ppo_loss"][eligible[0]],
                            (step.apos_logits, step.opos_logits, step.action_logits),
                            retain_graph=True,
                        )
                        ppo_gradient = float(
                            torch.stack([g.float().square().sum() for g in witness]).sum().sqrt()
                        )
                statistics = torch.stack([x.detach().sum() for x in components.values()]).cpu()
                for name, value in zip(components, statistics.tolist(), strict=True):
                    totals[name] = totals.get(name, 0.0) + value / (tmax * count)
                peaks.append(float(step.opos_logits.detach().softmax(-1).max(-1).values.mean()))
            assert group_loss is not None
            group_loss.backward()
            jobs = [(t0, i) for t0, i, _ in group]
            del group_loss, group, step, combined, components, fragment_loss
            if split_teacher:
                for t0, i in jobs:
                    state, fragment, boundaries = fragment_data(t0, i)
                    with torch.autocast("cuda", dtype=torch.bfloat16):
                        teacher_loss, teacher_metrics = teacher_fragment_losses(
                            policy,
                            state,
                            boundaries,
                            fragment,
                            cfg,
                        )
                        weighted_teacher_loss = teacher_loss.sum() / (len(selected) * length)
                    statistics = torch.stack(
                        [x.detach().sum() for x in teacher_metrics.values()]
                    ).cpu()
                    for name, value in zip(teacher_metrics, statistics.tolist(), strict=True):
                        totals[name] = totals.get(name, 0.0) + value / (tmax * count)
                    weighted_teacher_loss.backward()
                    del teacher_loss, teacher_metrics, weighted_teacher_loss
            completed += len(jobs)
            write_progress(root, update, "optimization", completed, len(fragments), started)
        if batch == 0:
            differentiable_error = validate_errors("differentiable")
        auxiliary, metrics, _ = grounding_loss(
            policy,
            grounding.sample(cfg["grounding_batch_size"]),
            **grounding_loss_options(ground_cfg),
        )
        (cfg["grounding_coef"] * auxiliary).backward()
        auxiliary_metrics.append((float(auxiliary.detach()), metrics["opos_loss"]))
        invalid = not bool(
            torch.stack(
                [torch.isfinite(p.grad).all() for p in policy._all_trainable if p.grad is not None]
            ).all()
        )
        if parallel.any_rank(invalid, policy.device):
            raise RuntimeError("Rejected optimization: nonfinite gradients")
        norms.append(
            float(
                synchronized_optimizer_step(policy._all_trainable, optimizer, cfg["max_grad_norm"])
            )
        )
        apos_norms.append(float(policy.apos_rows.grad.norm()))
        opos_norms.append(float(policy.opos_rows.grad.norm()))
    return {
        **totals,
        "grounding_loss": sum(x[0] for x in auxiliary_metrics) / batches,
        "grounding_opos_loss": sum(x[1] for x in auxiliary_metrics) / batches,
        "grad_norm": max(norms),
        "apos_parameter_gradient_norm": max(apos_norms),
        "opos_parameter_gradient_norm": max(opos_norms),
        "opos_no_hint_positive_gradient_norm": opos_no_hint_gradient,
        "ppo_token_gradient_norm": ppo_gradient,
        "replay_log_prob_error_max": replay_error,
        "differentiable_replay_error_max": differentiable_error,
        "packed_replay_error_max": packed_error,
        "replay_fallback_fragments": fallbacks,
        "replay_batched_fallback_blocks": len(batched_blocks),
        "policy_log_probability_change_max": policy_change,
        "ppo_clip_fraction": clips / max(1, eligible_count),
        "ppo_approx_kl": kls / max(1, eligible_count),
        "optimizer_steps": batches,
        "split_teacher_replay": split_teacher,
        "opos_peak_probability": sum(peaks) / len(peaks),
    }


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


def archive_uncommitted_metrics(root, update):
    """Keep interrupted metrics, but exclude unsaved updates from active history."""
    archive = root / "interrupted" / time.strftime("%Y%m%d_%H%M%S", time.gmtime())
    discarded = {}
    for path in [root / "train_metrics.jsonl", *sorted(root.glob("rank_*_metrics.jsonl"))]:
        if not path.exists():
            continue
        rows = path.read_text().splitlines(keepends=True)
        retained: list[str] = []
        orphaned: list[str] = []
        for row in rows:
            try:
                committed = json.loads(row).get("update", 0) <= update
            except json.JSONDecodeError:
                committed = False
            (retained if committed else orphaned).append(row)
        if orphaned:
            archive.mkdir(parents=True, exist_ok=True)
            (archive / path.name).write_text("".join(orphaned))
            temporary = path.with_suffix(".resume-tmp")
            temporary.write_text("".join(retained))
            temporary.replace(path)
            discarded[path.name] = len(orphaned)
    if discarded:
        append_json(
            root / "resume_events.jsonl",
            {
                "resume_update": update,
                "archive": str(archive),
                "uncommitted_rows": discarded,
            },
        )


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
                    "world_size": parallel.world_size(),
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


def load_adapters(policy, folder, *, allow_new_plan=False):
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
    added = expected - set(weights)
    if set(weights) - expected or (
        added
        and not (
            allow_new_plan
            and all(name.startswith(("future_plan.", "object_hint.")) for name in added)
        )
    ):
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
        "split_teacher_replay",
        "overlap_perception_oracle",
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
    model.pop("apos_decoding", None)
    model.pop("opos_decoding", None)
    # Recomputing activations affects memory/time, not the model or objective.
    model.pop("gradient_checkpointing", None)
    model.pop("inference_cuda_graphs", None)
    model.pop("cache_frame_templates", None)
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
    recipe["model"].pop("future_action_steps", None)
    recipe["model"].pop("object_hint_token", None)
    recipe["data"] = dict(recipe["data"])
    recipe["data"].pop("iterator_options", None)
    recipe["habitat"].pop("reward", None)
    # Dynamics may change in an explicit recipe fork, never an exact resume.
    recipe["habitat"].pop("allow_sliding", None)
    # These sensors/labels are used only by the training annotation service.
    # RGB camera and robot geometry must still match the parent checkpoint.
    recipe["habitat"].pop("perception_labels", None)
    recipe["habitat"].pop("perception_object_labels", None)
    for key in (
        "object_label_source",
        "semantic_dataset_config",
        "opos_min_visible_pixels",
        "semantic_assets_sha256",
    ):
        recipe["habitat"].pop(key, None)
    return recipe


def policy_schedule_update(config, update):
    if not config["trainer"].get("schedule_from_fork", False):
        return update
    return update - config.get(
        "schedule_origin_update", config.get("fork_origin", {}).get("update", 0)
    )


def navigation_rank_state(saved, rank, world_size, *, allow_world_size_change=False):
    """Keep existing streams; added ranks start independent streams, never clones."""
    previous_size = saved["world_size"]
    if world_size != previous_size and not allow_world_size_change:
        raise ValueError(
            "Exact resume requires the original world size; use --allow-world-size-change"
        )
    if previous_size < 1 or world_size < 1 or not 0 <= rank < world_size:
        raise ValueError("Invalid navigation worker group")
    if len(saved["ranks"]) != previous_size or any(
        row.get("rank", i) != i for i, row in enumerate(saved["ranks"])
    ):
        raise ValueError("Invalid navigation checkpoint rank states")
    return saved["ranks"][rank] if rank < previous_size else None


def added_navigation_rank_seed(seed, update, rank):
    """Stable independent seeds, including a distinct stream on later expansions."""
    return (seed + 1000003 * (update + 1) + 1009 * rank) % (2**32)


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
        score = sum(r.get("selection_success", r["success"]) for r in results.values()) / len(
            results
        )
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
                json.dumps(
                    {
                        "update": update,
                        "success": score,
                        "selection_metric": "fixed_anchor_success",
                        "results": results,
                    },
                    indent=2,
                )
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
    parser.add_argument("--allow-world-size-change", action="store_true")
    parser.add_argument("--keep-schedule", action="store_true")
    parser.add_argument("--eval-episodes", type=int)
    parser.add_argument("--eval-max-steps", type=int)
    parser.add_argument("--skip-initial-eval", action="store_true")
    args = parser.parse_args()
    if args.keep_schedule and not args.fork_from:
        raise ValueError("--keep-schedule requires an explicit --fork-from")
    if args.allow_world_size_change and not (args.resume or args.fork_from):
        raise ValueError("--allow-world-size-change requires a saved navigation checkpoint")
    if not (args.resume or args.fork_from or args.initialize_from):
        raise ValueError("Run grounding first, then use --initialize-from its checkpoint")
    config = yaml.safe_load(Path(args.config).read_text())
    config["launch"] = {
        "resume": args.resume,
        "fork_from": args.fork_from,
        "keep_schedule": args.keep_schedule,
        "skip_initial_eval": args.skip_initial_eval,
        "initialize_from": args.initialize_from,
        "allow_world_size_change": args.allow_world_size_change,
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
    if config["habitat"].get("object_label_source") == "semantic_instance":
        assets = Path("runtime/data/scene_datasets/hm3d/semantic_assets.json")
        if not assets.exists():
            raise FileNotFoundError(
                "Prepare and validate semantic instance assets before navigation training"
            )
        config["habitat"]["semantic_assets_sha256"] = sha256(assets.read_bytes()).hexdigest()
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
    if (
        not 1
        <= cfg.get("ppo_num_mini_batches", 1)
        <= cfg["rollout_steps"] // cfg["sequence_length"]
    ):
        raise ValueError("Mini-batches must contain complete time blocks for every stream")
    for key in ("apos_forward_intent_coef", "future_action_loss_coef"):
        if not math.isfinite(cfg.get(key, 0)) or cfg.get(key, 0) < 0:
            raise ValueError(f"{key} must be finite and nonnegative")
    if cfg.get("point_supervision_prefix", "teacher") not in ("actual", "teacher"):
        raise ValueError("Unknown point supervision prefix")
    type_coef = cfg.get("apos_type_il_coef", 0)
    if not math.isfinite(type_coef) or type_coef < 0:
        raise ValueError("APOS type coefficient must be finite and nonnegative")
    geometry = True
    for key in ("recovery_il_weight", "stop_il_weight"):
        if not math.isfinite(cfg.get(key, 1.0)) or cfg.get(key, 1.0) < 1:
            raise ValueError(f"{key} must be finite and at least one")
    restore_from = args.resume or args.fork_from
    saved, local, added_seed = None, None, None
    if restore_from:
        saved = torch.load(
            Path(restore_from) / "optimizer.pt", map_location="cpu", weights_only=False
        )
        local = navigation_rank_state(
            saved,
            parallel.rank(),
            parallel.world_size(),
            allow_world_size_change=args.allow_world_size_change,
        )
        if local is None:
            added_seed = added_navigation_rank_seed(
                config["seed"], saved["update"], parallel.rank()
            )
    seed_everything(config["seed"])
    policy = Qwen35PointCoTPolicy(config).train()
    groups: dict[str, list[torch.nn.Parameter]] = {
        "lora": [],
        "memory": [],
        "point": [],
        "action": [],
        "critic": [],
        "planning": [],
        "hint": [],
    }
    for name, parameter in policy.named_parameters():
        if parameter.requires_grad:
            group = (
                "hint"
                if name.startswith("object_hint.")
                else "planning"
                if name.startswith("future_plan.")
                else "memory"
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
            if params
        ],
        weight_decay=0.0,
        eps=1e-8,
    )
    episode_source = HabitatEpisodeSource(
        config["data"]["sources"][0]["manifest"],
        seed=config["seed"] + 1000 * parallel.rank() if added_seed is None else added_seed + 1000,
        iterator_options=config["data"]["iterator_options"],
    )
    source = MixedStartEpisodeSource(episode_source, config["curriculum"])
    grounding = GroundingSource(
        config["grounding"]["train_manifest"],
        seed=config["seed"] + parallel.rank() if added_seed is None else added_seed + 1,
        source_weights=config["grounding"].get("source_weights"),
        absence_probability=config["grounding"].get("absence_probability"),
    )
    grounding_val = GroundingSource(config["grounding"]["val_manifest"])
    check_grounding_split(grounding, grounding_val)
    policy.grounding_source = grounding
    update, frames = 0, 0
    if args.initialize_from:
        if restore_from:
            raise ValueError("Initialization and resume are mutually exclusive")
        load_adapters(policy, args.initialize_from, allow_new_plan=True)
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
        load_adapters(policy, restore_from, allow_new_plan=bool(args.fork_from))
        assert saved is not None
        if args.fork_from:
            load_optimizer_with_added_groups(optimizer, saved["optimizer"])
        else:
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
        if args.keep_schedule or (args.resume and "schedule_origin_update" in previous_config):
            config["schedule_origin_update"] = previous_config.get(
                "schedule_origin_update", previous_config.get("fork_origin", {}).get("update", 0)
            )
        update = saved["update"]
        frames = saved.get(
            "frames", update * cfg["rollout_steps"] * cfg["num_envs"] * parallel.world_size()
        )
        if local is not None:
            source.load_state_dict(local["source"])
        if (
            args.fork_from
            and config["data"]["iterator_options"] != previous_config["data"]["iterator_options"]
        ):
            source.source._set_step_limit()
        if local is not None:
            if "grounding" in local:
                grounding.load_state_dict(local["grounding"])
            restore_rng(local["rng"])
        else:
            assert added_seed is not None
            seed_everything(added_seed)
        config["world_size_changes"] = list(previous_config.get("world_size_changes", []))
        if saved["world_size"] != parallel.world_size():
            change = {
                "from": saved["world_size"],
                "to": parallel.world_size(),
                "update": update,
                "frames": frames,
                "preserved_ranks": list(range(min(saved["world_size"], parallel.world_size()))),
                "added_rank_seeds": {
                    i: added_navigation_rank_seed(config["seed"], update, i)
                    for i in range(saved["world_size"], parallel.world_size())
                },
                "transitions_per_update_before": cfg["rollout_steps"]
                * cfg["num_envs"]
                * saved["world_size"],
                "transitions_per_update_after": cfg["rollout_steps"]
                * cfg["num_envs"]
                * parallel.world_size(),
                "schedule_policy": "preserve_update_budget_and_ppo_schedule",
            }
            config["world_size_changes"].append(change)
            if parallel.rank() == 0:
                append_json(root / "world_size_changes.jsonl", change)
        if parallel.rank() == 0:
            archive_uncommitted_metrics(root, update)
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

    if not restore_from or (
        args.fork_from and not (root / "checkpoints" / f"update_{update:07d}").exists()
    ):
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
            schedule_update = policy_schedule_update(config, update)
            tmax, count = cfg["rollout_steps"], len(states)
            write_progress(root, update, "rollout", 0, tmax, started)
            beta = cfg["expert_warmup_probability"] * max(
                0.0, 1 - schedule_update / max(1, cfg["warmup_updates"])
            )
            ppo_weight = cfg["ppo_weight"] * min(
                1.0, max(0.0, (schedule_update - cfg["warmup_updates"]) / cfg["ppo_ramp_updates"])
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
            graph_hits = policy.inference_graphs.hits
            graph_captures = policy.inference_graphs.misses
            with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                for t in range(tmax):
                    phase_started = time.monotonic()
                    supervisions: list[Any] | None
                    prefetched_features, prefetch_seconds = None, 0.0
                    if geometry and cfg.get("overlap_perception_oracle", False):
                        pending = envs.submit_oracle_supervisions()
                        prefetch_started = time.monotonic()
                        try:
                            # Vision and prompt encoding do not depend on the
                            # target hint. Apply the hint after labels arrive.
                            prefetched_features = policy.navigation_features(
                                [o["rgb"] for o in observations], states
                            )
                        finally:
                            prefetch_seconds = time.monotonic() - prefetch_started
                            # Even a feature-encoding failure must drain socket
                            # requests before the exception closes simulators.
                            wait(pending)
                        supervisions = [future.result() for future in pending]
                    else:
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
                            prefetched_features = None
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
                    rollout_timings["rollout_oracle_seconds"] += (
                        time.monotonic() - phase_started - prefetch_seconds
                    )
                    phase_started = time.monotonic()
                    hint_config = cfg.get("object_hints", {})
                    hint_tokens = [
                        simulated_object_hint(
                            s["perception"], random, **hint_config.get("noise", {})
                        )
                        if isinstance(s, dict)
                        and episode_hint_enabled(
                            states[i].episode_id,
                            config["seed"],
                            hint_config.get("episode_probability", 0.0),
                        )
                        else 0
                        for i, s in enumerate(supervisions or [None] * len(states))
                    ]
                    features = (
                        policy.condition_features(prefetched_features, hint_tokens)
                        if prefetched_features is not None
                        else policy.navigation_features(
                            [o["rgb"] for o in observations], states, hint_tokens
                        )
                    )
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
                                "object_hint_token": feature.hint_token,
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
                                "apos_greedy_token": int(
                                    decode_point_logits(
                                        step.apos_logits,
                                        hierarchical=policy.apos_decoding == "hierarchical",
                                    )
                                ),
                                "opos_greedy_token": int(
                                    decode_point_logits(
                                        step.opos_logits,
                                        hierarchical=policy.opos_decoding == "hierarchical",
                                    )
                                ),
                                "action_plan_tokens": step.action_plan_logits.argmax(-1)
                                .cpu()
                                .tolist()
                                if step.action_plan_logits is not None
                                else None,
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
                    rollout_timings["rollout_policy_seconds"] += (
                        time.monotonic() - phase_started + prefetch_seconds
                    )
                    phase_started = time.monotonic()
                    following = envs.step([NavigationAction(s.action.item()) for s in steps])
                    finished_episodes = {}
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
                            finished_episodes[i] = episode
                            states[i] = policy.start_episode(episode.uid, episode.goal_text)
                            resets[(t + 1, i)] = (episode.uid, episode.goal_text)
                    for i, reset_observation in envs.reset_at(finished_episodes).items():
                        following[i] = reset_observation
                    observations = following
                    rollout_timings["rollout_environment_seconds"] += (
                        time.monotonic() - phase_started
                    )
                    records.append(row)
                    for _ in range(count):
                        source.step_taken()
                    if (t + 1) % cfg["sequence_length"] == 0:
                        write_progress(root, update, "rollout", t + 1, tmax, started)
                    # Ranks can load different scenes here. A collective every
                    # fragment turns each rank's reset delay into a global stall.
                    # Stop collectively after the complete update instead.
                phase_started = time.monotonic()
                bootstrap_hints = None
                if policy.supports_object_hints:
                    object_labels = list(
                        envs.executor.map(
                            lambda client: client.get_object_supervision(), envs.clients
                        )
                    )
                    hint_config = cfg.get("object_hints", {})
                    bootstrap_hints = [
                        simulated_object_hint(label, random, **hint_config.get("noise", {}))
                        if episode_hint_enabled(
                            state.episode_id,
                            config["seed"],
                            hint_config.get("episode_probability", 0.0),
                        )
                        else 0
                        for label, state in zip(object_labels, states, strict=True)
                    ]
                context = policy.begin_batch(
                    policy.navigation_features(
                        [o["rgb"] for o in observations], states, bootstrap_hints
                    ),
                    states,
                )
                last = list(policy.read_value(context.hidden).cpu().unbind())
                rollout_timings["rollout_bootstrap_seconds"] += time.monotonic() - phase_started
            rollout_seconds = time.monotonic() - started
            attach_future_actions(records, policy.future_action_steps)
            object_fractions = parallel.average_scalar(
                torch.tensor(
                    [
                        sum(
                            bool((r.get("perception_label") or {}).get("opos_valid"))
                            and bool(r["perception_label"]["opos"] > 0) == visible
                            for row in records
                            for r in row
                        )
                        / (tmax * count)
                        for visible in (True, False)
                    ],
                    device=policy.device,
                )
            ).tolist()
            opos_density = sum(object_fractions)
            if (
                config["habitat"].get("object_label_source") == "semantic_instance"
                and opos_density == 0
            ):
                raise RuntimeError(
                    "Instance OPOS supervision has zero valid labels; check semantic assets and goal IDs"
                )
            positive_weight, negative_weight, location_weight = object_loss_weights(
                *object_fractions,
                balanced=cfg.get("opos_supervision", {}).get("balanced_visibility", False),
                power=cfg.get("opos_supervision", {}).get("visibility_balance_power", 1.0),
            )
            for row in records:
                for record in row:
                    visible = (record.get("perception_label") or {}).get("opos", 0) > 0
                    record["opos_visibility_weight"] = (
                        positive_weight if visible else negative_weight
                    )
                    record["opos_location_weight"] = location_weight
            future_density = parallel.average_scalar(
                torch.tensor(
                    sum(
                        any(a != -100 for a in r["future_actions"][1:])
                        for row in records
                        for r in row
                    )
                    / (tmax * count),
                    device=policy.device,
                )
            ).item()
            for row in records:
                for record in row:
                    record["future_loss_weight"] = 1 / max(future_density, 1e-8)
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
                    "apos_forward_intent_loss",
                    "apos_forward_intent_fraction",
                    "apos_teacher_prefix_loss",
                    "future_action_loss",
                    "opos_loss",
                    "opos_visibility_loss",
                    "opos_location_loss",
                    "opos_label_fraction",
                    "opos_visible_fraction",
                    "opos_absent_fraction",
                    "object_hint_fraction",
                    "ppo_loss",
                    "value_loss",
                    "entropy",
                    "loss",
                )
            }
            try:
                optimization = optimize_rollout(
                    policy,
                    optimizer,
                    grounding,
                    records,
                    starts,
                    resets,
                    adv,
                    returns,
                    cfg,
                    config["grounding"],
                    ppo_weight,
                    root,
                    update,
                    started,
                )
            except ReplayMismatch:
                # All ranks reject before the first optimizer step. Parameters
                # and Adam still belong to the previous completed update.
                write_progress(root, update - 1, "replay_recovery_checkpoint", 0, 1, started)
                save(
                    policy,
                    optimizer,
                    source,
                    {
                        **config,
                        "recovery": {
                            "reason": "replay_mismatch",
                            "rejected_rollout_update": update,
                            "sampler_advanced": True,
                        },
                    },
                    update - 1,
                    frames,
                )
                raise
            totals.update({key: optimization[key] for key in totals})
            frames += tmax * count * parallel.world_size()
            for i, state in enumerate(states):
                states[i] = cache_copy(state, policy.device)
            local_metrics = {
                **optimization,
                **rollout_timings,
                **summarize_rollout(records),
                **summarize_object_predictions(records),
                "update": update,
                "schedule_update": schedule_update,
                "ppo_coefficient": ppo_weight,
                "expert_probability": beta,
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
                "inference_graph_hits": policy.inference_graphs.hits - graph_hits,
                "inference_graph_captures": policy.inference_graphs.misses - graph_captures,
                "inference_graph_cache_entries": len(policy.inference_graphs.entries),
                "future_action_target_counts": [
                    sum(r["future_actions"][k] != -100 for row in records for r in row)
                    for k in range(policy.future_action_steps)
                ],
                "future_action_hits": [
                    sum(
                        r["future_actions"][k] != -100
                        and r["action_plan_tokens"] is not None
                        and r["action_plan_tokens"][k] == r["future_actions"][k]
                        for row in records
                        for r in row
                    )
                    for k in range(policy.future_action_steps)
                ],
                "sampled_scene_indices": [
                    int(k) for k, v in source.source.scene_cursors.items() if v
                ],
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
                    schedule_update=schedule_update,
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
                    differentiable_replay_error_max=max(
                        m["differentiable_replay_error_max"] for m in all_metrics
                    ),
                    grad_norm=max(m["grad_norm"] for m in all_metrics),
                    apos_parameter_gradient_norm=max(
                        m["apos_parameter_gradient_norm"] for m in all_metrics
                    ),
                    opos_parameter_gradient_norm=max(
                        m["opos_parameter_gradient_norm"] for m in all_metrics
                    ),
                    opos_no_hint_positive_gradient_norm=max(
                        m["opos_no_hint_positive_gradient_norm"] for m in all_metrics
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
                    inference_graph_hits=sum(m["inference_graph_hits"] for m in all_metrics),
                    inference_graph_captures=sum(
                        m["inference_graph_captures"] for m in all_metrics
                    ),
                    inference_graph_cache_entries=max(
                        m["inference_graph_cache_entries"] for m in all_metrics
                    ),
                    packed_replay_error_max=max(m["packed_replay_error_max"] for m in all_metrics),
                    replay_fallback_fragments=sum(
                        m["replay_fallback_fragments"] for m in all_metrics
                    ),
                    replay_batched_fallback_blocks=sum(
                        m["replay_batched_fallback_blocks"] for m in all_metrics
                    ),
                    optimizer_steps=max(m["optimizer_steps"] for m in all_metrics),
                    policy_log_probability_change_max=max(
                        m["policy_log_probability_change_max"] for m in all_metrics
                    ),
                    ppo_clip_fraction=sum(m["ppo_clip_fraction"] for m in all_metrics)
                    / len(all_metrics),
                    ppo_approx_kl=sum(m["ppo_approx_kl"] for m in all_metrics) / len(all_metrics),
                    training_scene_count=len(
                        {scene for m in all_metrics for scene in m["sampled_scene_indices"]}
                    ),
                    future_action_target_counts=[
                        sum(m["future_action_target_counts"][k] for m in all_metrics)
                        for k in range(policy.future_action_steps)
                    ],
                    future_action_hits=[
                        sum(m["future_action_hits"][k] for m in all_metrics)
                        for k in range(policy.future_action_steps)
                    ],
                )
                combined_metrics["future_action_accuracy"] = [
                    hits / max(1, total)
                    for hits, total in zip(
                        combined_metrics["future_action_hits"],
                        combined_metrics["future_action_target_counts"],
                        strict=True,
                    )
                ]
                for key in all_metrics[0]:
                    if key.startswith(("opos_no_hint_", "opos_hint_")) and key.endswith(
                        ("_count", "_hits")
                    ):
                        combined_metrics[key] = sum(m[key] for m in all_metrics)
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
            checkpoint_due = (
                update == 1
                or update == config.get("fork_origin", {}).get("update", -2) + 1
                or any(
                    change["update"] + 1 == update
                    for change in config.get("world_size_changes", [])
                )
                or update % cfg["checkpoint_interval"] == 0
                or should_stop
                or update == cfg["num_updates"]
            )
            if checkpoint_due:
                write_progress(root, update, "checkpoint", 0, 1, started)
                save(policy, optimizer, source, config, update, frames)
            if not should_stop and (
                update % cfg["eval_interval"] == 0 or update == cfg["num_updates"]
            ):
                if not checkpoint_due:
                    save(policy, optimizer, source, config, update, frames)
                write_progress(root, update, "evaluation", 0, 1, started)
                results = evaluate(policy, config, update=update, video=True)
                record_best(config, update, results)
            if should_stop:
                break
    except BaseException:
        # Emit the original failure before NCCL cleanup: a collective destructor
        # can otherwise hide it while the other ranks wait for synchronization.
        traceback.print_exc()
        raise
    finally:
        envs.close()
        if sys.exc_info()[0] is None and torch.distributed.is_initialized():
            torch.distributed.destroy_process_group()


if __name__ == "__main__":
    torch.set_num_threads(4)
    main()
