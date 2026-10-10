import json
import math
import os
import signal
import time
from pathlib import Path
from typing import cast

import hydra
import torch
import torch.nn.functional as F
import yaml
from omegaconf import OmegaConf

from streamnav.data.manifest import check_leakage, verify_manifest
from streamnav.data.mixture import make_source, partition_scenes
from streamnav.data.schema import read_json
from streamnav.envs.habitat_client import VectorHabitatEnvs
from streamnav.errors import ReplayConsistencyError
from streamnav.evaluation.runner import evaluate
from streamnav.training import distributed as parallel
from streamnav.training.auxiliary_il import AuxiliaryILController
from streamnav.training.checkpoint import (
    initialize_training_branches,
    load_policy,
    promote_best_checkpoint,
    provenance,
    restore_training,
    save_checkpoint,
    validate_resume_configuration,
)
from streamnav.training.dagger import DaggerBetaScheduler, behavior_log_prob
from streamnav.training.ealm import EntropyAdaptiveLossMixer
from streamnav.training.gae import compute_gae
from streamnav.training.optimizer import OVSegDTLRScheduler, build_optimizer, gradient_norm
from streamnav.training.perception_loss import perception_losses
from streamnav.training.ppo import clipped_ppo_loss, ovsegdt_value_loss, replay_consistency_metrics
from streamnav.training.rollout import RolloutCollector, SequenceReplay
from streamnav.utils.logging import ActionHistogramMetric, append_json
from streamnav.utils.seed import rng_state, seed_everything


class EndToEndObjectNavTrainer:
    def __init__(self, config):
        self.config, self.cfg = config, config["trainer"]
        self.run_dir = Path(config["run_dir"])
        self.run_dir.mkdir(parents=True, exist_ok=True)
        metrics_file = self.run_dir / "train_metrics.jsonl"
        if metrics_file.exists() and metrics_file.stat().st_size:
            if not config.get("checkpoint"):
                raise ValueError(
                    "Run already contains training: use a new run_dir or explicit checkpoint"
                )
            previous_update = json.loads(metrics_file.read_text().splitlines()[-1])["update"]
            saved_update = json.loads((Path(config["checkpoint"]) / "manifest.json").read_text())[
                "update"
            ]
            if previous_update != saved_update:
                raise ValueError("Checkpoint differs from logged update: resume into a new run_dir")
        fork_changes = (
            validate_resume_configuration(config["checkpoint"], config)
            if config.get("checkpoint")
            else {}
        )
        seed_everything(config["seed"])
        self.stop_requested = False
        self.update_index = 0
        if self.cfg["num_envs"] < 1 or self.cfg["sequence_batch_size"] < 1:
            raise ValueError("Environment count and sequence batch size must be positive")
        if self.cfg["rollout_steps"] % self.cfg["sequence_length"]:
            raise ValueError("rollout_steps must be divisible by sequence_length")
        self.auxiliary_controller = AuxiliaryILController(
            self.cfg.get("auxiliary_il"), self.cfg["num_envs"]
        )
        load = (
            verify_manifest
            if self.cfg["verify_data_hashes"] and parallel.rank() == 0
            else read_json
        )
        train = [load(s["manifest"]) for s in config["data"]["sources"]]
        val = [load(p) for p in config["eval"]["manifests"]]
        if any(m["split"] != "train" for m in train):
            raise ValueError("Only training splits can enter rollout sampling")
        check_leakage(train, val)
        self.manifest = provenance(config) if parallel.rank() == 0 else None
        self.manifest = parallel.gather(self.manifest)[0]
        if config.get("checkpoint"):
            parent_manifest = json.loads((Path(config["checkpoint"]) / "manifest.json").read_text())
            if "branch_initialization" in parent_manifest:
                self.manifest["branch_initialization"] = parent_manifest["branch_initialization"]
        if fork_changes:
            self.manifest["training_fork"] = {
                "parent_checkpoint": str(Path(config["checkpoint"]).resolve()),
                "recipe_changes": fork_changes,
                "restored": "weights, Adam moments, schedules, per-rank RNG/samplers/EALM",
                "episode_state": "simulator episodes and recurrent caches restart",
            }
        if parallel.rank() == 0:
            (self.run_dir / "manifest.json").write_text(json.dumps(self.manifest, indent=2))
            (self.run_dir / "resolved_config.yaml").write_text(yaml.safe_dump(config))
        self.policy = load_policy(config, training=True)
        if self.policy.perception is not None and not config["habitat"].get(
            "perception_labels", False
        ):
            raise ValueError("Perception training requires training-only simulator labels")
        self.policy.batch_chat_body = self.cfg.get("batch_chat_body", False)
        self.device = self.policy.backbone.device
        self.dtype = getattr(torch, config["model"]["dtype"])
        self.optimizer = build_optimizer(self.policy, self.cfg)
        seed_everything(config["seed"] + parallel.rank() * 1000003)
        self.scheduler = OVSegDTLRScheduler(self.optimizer, self.cfg)
        self.dagger = DaggerBetaScheduler(**self.cfg["dagger"])
        self.mixer = EntropyAdaptiveLossMixer(**self.cfg["ealm"])
        self.sources = [
            make_source(config["data"], config["seed"] + i * 1009 + parallel.rank() * 1000003)
            for i in range(self.cfg["num_envs"])
        ]
        if config["data"].get("iterator_options") is not None:
            partition_scenes(self.sources, config["seed"] + parallel.rank() * 1000003)
        if config.get("checkpoint"):
            self.update_index = restore_training(
                config["checkpoint"],
                self.optimizer,
                self.scheduler,
                self.dagger,
                self.sources,
                config,
                mixer=self.mixer,
                auxiliary_controller=self.auxiliary_controller,
            )
        elif self.cfg.get("initialize_optimizer_branches", False):
            initialization = initialize_training_branches(
                config["model"]["checkpoint"],
                self.optimizer,
                self.sources,
                config,
                self.mixer,
            )
            self.manifest["branch_initialization"] = initialization
            if parallel.rank() == 0:
                (self.run_dir / "manifest.json").write_text(json.dumps(self.manifest, indent=2))
                (self.run_dir / "branch_initialization.json").write_text(
                    json.dumps(initialization, indent=2)
                )
        self.replay: torch.nn.Module = SequenceReplay(
            self.policy,
            cache_embeddings=self.cfg.get("cache_replay_embeddings", False),
            pack_frames=self.cfg.get("replay_pack_frames", 1),
        )
        self.ddp = (
            parallel.world_size() > 1
            and config.get("distributed", {}).get("gradient_sync", "ddp") == "ddp"
        )
        self._configure_replay()
        self.histogram = ActionHistogramMetric(
            self.cfg["collapse_window"],
            self.cfg["collapse_threshold"],
            config["model"]["action_dim"],
        )
        self.envs = VectorHabitatEnvs(
            {**config["habitat"], "seed": config["seed"] + parallel.rank() * 1000003},
            self.cfg["num_envs"],
        )
        self.collector = RolloutCollector(
            self.policy, self.envs, self.sources, self.cfg, self.auxiliary_controller
        )
        workers = parallel.gather(
            {
                "rank": parallel.rank(),
                "pid": os.getpid(),
                "launcher_pid": os.getppid(),
                "device": str(self.device),
            }
        )
        if parallel.rank() == 0:
            (self.run_dir / "training_workers.json").write_text(json.dumps(workers, indent=2))

    def _configure_replay(self):
        # Rebuild after PIRLNav's first update unfreezes actor/backbone, so DDP
        # registers all newly trainable parameters, not just the initial critic.
        self.replay = SequenceReplay(
            self.policy,
            cache_embeddings=self.cfg.get("cache_replay_embeddings", False),
            pack_frames=self.cfg.get("replay_pack_frames", 1),
        )
        if self.ddp:
            self.replay = torch.nn.parallel.DistributedDataParallel(
                self.replay,
                device_ids=[self.device.index],
                broadcast_buffers=False,
                gradient_as_bucket_view=True,
                find_unused_parameters=self.cfg.get("ddp_find_unused_parameters", True),
            )

    def autocast(self):
        return torch.autocast(
            device_type=self.device.type, dtype=self.dtype, enabled=self.dtype != torch.float32
        )

    def collect_rollout(self):
        self.collector.update_index = self.update_index
        with self.autocast():
            return self.collector.collect(self.dagger.beta)

    def compute_losses(self, logits, values, buffer, sequences, predictions=None):
        actions = buffer.gather("executed_actions", sequences, self.device)
        expert = buffer.gather("expert_actions", sequences, self.device)
        dist = self.policy.distribution.build(logits)
        entropy = dist.entropy()
        il_loss = F.cross_entropy(logits.flatten(0, 1), expert.flatten(), reduction="none").view_as(
            expert
        )
        action_dim = logits.shape[-1]
        class_weights = logits.new_tensor(self.cfg.get("il_class_weights", [1.0] * action_dim))
        if class_weights.shape != (action_dim,) or not bool((class_weights > 0).all()):
            raise ValueError("One positive IL class weight per action is required")
        weights = class_weights[expert]
        il_mask = buffer.gather("il_mask", sequences, self.device)
        weights = weights * il_mask
        weighted_il = il_loss * weights / parallel.average_scalar(weights.mean()).clamp_min(1e-8)
        il_fraction = parallel.average_scalar(il_mask.float().mean()).clamp_min(1e-8)
        new_log_prob = behavior_log_prob(logits, actions, expert, buffer.beta)
        auxiliary = self.cfg.get("auxiliary_il", {}).get("num_envs", 0) > 0
        mask = buffer.gather("ppo_mask", sequences, self.device)
        fraction = (
            parallel.average_scalar(mask.float().mean()) if auxiliary else logits.new_tensor(1)
        )
        denominator = fraction.clamp_min(1e-8)
        old_log_prob = buffer.gather("old_log_probs", sequences, self.device)
        advantages = buffer.gather("advantages", sequences, self.device)
        if auxiliary:
            # Auxiliary actions are deterministic expert/greedy interventions.
            # Only sampled behavior contributes PPO, value fitting and entropy
            # exploration. Sanitize excluded values before nonlinear operations.
            new_log_prob = torch.where(mask, new_log_prob, 0)
            old_log_prob = torch.where(mask, old_log_prob, 0)
            advantages = torch.where(mask, advantages, 0)
        ppo_loss, ratio = clipped_ppo_loss(
            new_log_prob,
            old_log_prob,
            advantages,
            self.cfg["ppo"]["clip_eps"],
        )
        ppo_loss = ppo_loss / denominator
        policy_loss, alpha = self.mixer(weighted_il, ppo_loss, entropy)
        old_values = buffer.gather("old_values", sequences, self.device)
        returns = buffer.gather("returns", sequences, self.device)
        if auxiliary:
            values = torch.where(mask, values, 0)
            old_values, returns = torch.where(mask, old_values, 0), torch.where(mask, returns, 0)
        value_loss = ovsegdt_value_loss(
            values,
            old_values,
            returns,
            self.cfg["ppo"]["clip_eps"],
            self.cfg["ppo"].get("use_clipped_value_loss", True),
        )
        value_loss = value_loss / denominator
        entropy_bonus = (entropy * mask / denominator).mean() if auxiliary else entropy.mean()
        entropy_stat = (
            parallel.average_scalar((entropy * mask).mean()) / denominator
            if auxiliary
            else entropy.mean()
        )
        if buffer.replay_log_probs is None:
            replay_log_prob, old_replay_log_prob = new_log_prob, old_log_prob
        else:
            # Verify model probabilities on every IL/PPO frame, independently
            # of the deterministic auxiliary behavior distribution.
            replay_log_prob = dist.log_prob(actions)
            old_replay_log_prob = buffer.gather("replay_log_probs", sequences, self.device)
        total = (
            policy_loss.mean()
            + self.cfg["loss"]["value_coef"] * value_loss.mean()
            - self.cfg["loss"]["entropy_coef"] * entropy_bonus
        )
        metrics = {
            "total_loss": total.detach().item(),
            "il_loss": ((il_loss * il_mask).mean() / il_fraction).detach().item(),
            "il_loss_all_frames": il_loss.mean().detach().item(),
            "il_eligible_fraction": il_mask.float().mean().item(),
            "weighted_il_loss": weighted_il.mean().detach().item(),
            "ppo_loss": ppo_loss.mean().detach().item(),
            "value_loss": value_loss.mean().detach().item(),
            "entropy": entropy_stat.detach().item(),
            "entropy_all_frames": entropy.mean().detach().item(),
            "ealm_alpha": alpha.mean().item(),
            "ppo_policy_coefficient": (1 - alpha).mean().item(),
            "ppo_eligible_fraction": mask.float().mean().item(),
            "ppo_approx_kl": (((ratio - 1) - (new_log_prob - old_log_prob)) / denominator)
            .mean()
            .detach()
            .item(),
            "clip_fraction": (
                ((ratio - 1).abs() > self.cfg["ppo"]["clip_eps"]).float() / denominator
            )
            .mean()
            .item(),
            **replay_consistency_metrics(replay_log_prob, old_replay_log_prob),
        }
        if predictions is not None:
            auxiliary_loss, auxiliary_metrics = perception_losses(
                predictions,
                buffer.gather_perception(sequences, self.device),
                self.cfg.get("perception_loss"),
            )
            # Perception learning stays active independently of the IL/PPO gate.
            total = total + auxiliary_loss
            metrics.update(auxiliary_metrics)
            metrics["total_loss"] = total.detach().item()
        if self.cfg.get("log_policy_gradient_terms", False) and logits.requires_grad:
            # Target only logits, not DDP parameter leaves. No body backward or
            # reducer hooks run here; the later loss.backward() is unchanged.
            il_gradient = torch.autograd.grad(
                (alpha * weighted_il).mean(), logits, retain_graph=True
            )[0]
            ppo_gradient = torch.autograd.grad(
                ((1 - alpha) * ppo_loss).mean(), logits, retain_graph=True
            )[0]
            il_norm, ppo_norm = il_gradient.norm(), ppo_gradient.norm()
            metrics.update(
                policy_logit_grad_norm_il=il_norm.item(),
                policy_logit_grad_norm_ppo=ppo_norm.item(),
                policy_logit_grad_ppo_to_il_ratio=(ppo_norm / il_norm.clamp_min(1e-12)).item(),
                policy_logit_grad_il_ppo_cosine=(
                    (il_gradient * ppo_gradient).sum() / (il_norm * ppo_norm).clamp_min(1e-12)
                ).item(),
            )
        return total, metrics

    def _preflight_replay(self, buffer, sequences):
        # Probe the unwrapped module before DDP prepares a backward pass. A
        # second DDP forward after a rejected one would leave its reducer busy.
        # Keep autograd/AMP/checkpointing identical to the actual training pass,
        # discard each probe graph, and never update parameters or entropy EMA.
        replay = cast(SequenceReplay, self.replay.module if self.ddp else self.replay)
        batch_body = self.policy.batch_chat_body
        configured_pack = replay.pack_frames
        candidates = [(configured_pack, batch_body)]
        candidates.extend((n, batch_body) for n in (32, 16, 8, 4, 1) if n < configured_pack)
        if batch_body:
            candidates.append((1, False))
        tolerance = self.cfg["ppo"].get("replay_log_prob_tolerance", 0.05)
        attempts = []
        started = time.monotonic()
        for pack_frames, batch_chat_body in candidates:
            replay.pack_frames = pack_frames
            self.policy.batch_chat_body = batch_chat_body
            with self.autocast():
                result = replay(buffer, sequences)
                logits, values = result[:2]
                loss, metrics = self.compute_losses(logits, values, buffer, sequences, *result[2:])
            error = metrics["replay_log_prob_error_max"]
            rejected = parallel.any_rank(not math.isfinite(error) or error > tolerance, self.device)
            attempts.append(
                {
                    "pack_frames": pack_frames,
                    "batch_chat_body": batch_chat_body,
                    "local_log_prob_error": error,
                    "rejected_on_any_rank": rejected,
                }
            )
            del logits, values, loss, result
            if not rejected:
                if len(attempts) > 1:
                    records = parallel.gather({"rank": parallel.rank(), "attempts": attempts})
                    if parallel.rank() == 0:
                        record = {"update": self.update_index + 1, "ranks": records}
                        append_json(self.run_dir / "replay_fallbacks.jsonl", record)
                        print(json.dumps({"replay_fallback": record}), flush=True)
                return {
                    "replay_pack_frames_used": pack_frames,
                    "replay_used_serial_chat_body": not batch_chat_body,
                    "replay_preflight_attempts": len(attempts),
                    "replay_preflight_seconds": time.monotonic() - started,
                    "preupdate_replay_initial_error_max": attempts[0]["local_log_prob_error"],
                }
        raise ReplayConsistencyError(
            "Rollout/replay probabilities differ before any optimizer step even after "
            f"frame-by-frame fallback (local max log-prob error={error:.6f}, "
            f"tolerance={tolerance}). Checkpointing the last complete update."
        )

    def update(self, buffer):
        replay = cast(SequenceReplay, self.replay.module if self.ddp else self.replay)
        configured_pack = replay.pack_frames
        configured_batch = self.policy.batch_chat_body
        try:
            return self._update(buffer)
        finally:
            # A conservative replay is local to one update. Collection and the
            # next update retain the configured fast implementation.
            replay.pack_frames = configured_pack
            self.policy.batch_chat_body = configured_batch

    def _update(self, buffer):
        cfg = self.cfg
        advantages, buffer.returns = compute_gae(
            buffer.rewards,
            buffer.old_values,
            buffer.dones,
            buffer.last_values,
            cfg["ppo"]["gamma"],
            cfg["ppo"]["gae_lambda"],
            buffer.timeout_bootstrap,
        )
        buffer.advantages = (
            parallel.normalize_advantages(
                advantages,
                self.device,
                buffer.ppo_mask if self.auxiliary_controller.num_envs else None,
            )
            if cfg["ppo"].get("use_normalized_advantage", False)
            else advantages
        )
        self.policy.train()
        all_metrics: list[dict[str, float]] = []
        replay_check = {}
        preflight = {}
        start = time.monotonic()
        for _ in range(cfg["update_epochs"]):
            for sequences in buffer.sequence_batches(cfg["sequence_batch_size"]):
                self.optimizer.zero_grad(set_to_none=True)
                if not all_metrics and cfg.get("replay_preflight", False):
                    preflight = self._preflight_replay(buffer, sequences)
                with self.autocast():
                    result = self.replay(buffer, sequences)
                    logits, values = result[:2]
                    loss, metrics = self.compute_losses(
                        logits, values, buffer, sequences, *result[2:]
                    )
                if not all_metrics:
                    # Before the first optimizer step, the behavior probabilities
                    # must match collection. Kernel/precision drift is not PPO
                    # learning and must never be hidden by clipping the ratio.
                    replay_check = {
                        "preupdate_" + key: value
                        for key, value in metrics.items()
                        if key.startswith("replay_") or key == "clip_fraction"
                    }
                    tolerance = cfg["ppo"].get("replay_log_prob_tolerance", 0.05)
                    error = metrics["replay_log_prob_error_max"]
                    if parallel.any_rank(
                        not math.isfinite(error) or error > tolerance, self.device
                    ):
                        raise ReplayConsistencyError(
                            "Rollout/replay probabilities differ before any optimizer step "
                            f"(local max log-prob error={error:.6f}, tolerance={tolerance}). "
                            "Use matching recurrence kernels and precision; checkpoint not updated."
                        )
                if parallel.any_rank(not bool(torch.isfinite(loss)), self.device):
                    raise FloatingPointError("Nonfinite loss; refusing to update checkpoint")
                loss.backward()
                if not self.ddp:
                    parallel.average_gradients(self.policy.parameters())
                metrics.update(
                    {
                        "vision_grad_norm": gradient_norm(self.policy.backbone.vision.parameters()),
                        "kda_grad_norm": gradient_norm(
                            p
                            for layer in self.policy.backbone.layers
                            if hasattr(layer.mixer, "beta_proj")
                            for p in layer.mixer.parameters()
                        ),
                        "actor_grad_norm": gradient_norm(
                            self.policy.actor_critic.actor.parameters()
                        ),
                        "critic_grad_norm": gradient_norm(
                            self.policy.actor_critic.critic.parameters()
                        ),
                    }
                )
                if getattr(self.policy, "perception", None) is not None:
                    metrics["perception_grad_norm"] = gradient_norm(
                        self.policy.perception.parameters()
                    )
                norm = torch.nn.utils.clip_grad_norm_(
                    self.policy.parameters(), cfg["max_grad_norm"], error_if_nonfinite=True
                )
                metrics["grad_norm"] = norm.item()
                self.optimizer.step()
                self.mixer.observe_entropy(metrics["entropy"])
                all_metrics.append(metrics)
        metrics = {k: sum(m[k] for m in all_metrics) / len(all_metrics) for k in all_metrics[0]}
        metrics.update(replay_check)
        metrics.update(preflight)
        metrics["entropy_ema"] = self.mixer.entropy_ema.item()
        metrics["optimization_seconds"] = time.monotonic() - start
        metrics["training_fps"] = (
            buffer.steps * buffer.num_envs * cfg["update_epochs"] / metrics["optimization_seconds"]
        )
        return metrics

    def evaluate(self):
        self.optimizer.zero_grad(set_to_none=True)
        results = evaluate(
            self.policy,
            self.config,
            self.update_index,
            episodes=self.cfg["eval_episodes"],
            max_steps=self.cfg["eval_max_steps"],
            video=True,
        )
        self.save_checkpoint()
        if parallel.rank() == 0:
            promote_best_checkpoint(self.run_dir, self.update_index, results)
        parallel.barrier()
        return results

    def save_checkpoint(self):
        states = parallel.gather(
            {
                "rng": rng_state(),
                "sources": [s.state_dict() for s in self.sources],
                "mixer": self.mixer.state_dict(),
                "auxiliary_il": self.auxiliary_controller.state_dict(),
            }
        )
        result = None
        if parallel.rank() == 0:
            result = save_checkpoint(
                self.policy,
                self.optimizer,
                self.scheduler,
                self.dagger,
                self.sources,
                self.update_index,
                self.config,
                self.manifest,
                rank_states=states,
            )
        parallel.barrier()
        return result

    def run(self):
        def request_stop(signum, frame):
            self.stop_requested = True
            print(f"Signal {signum}: will checkpoint after the current update.", flush=True)

        old_handlers = {s: signal.signal(s, request_stop) for s in (signal.SIGTERM, signal.SIGINT)}
        try:
            while self.update_index < self.cfg["num_updates"]:
                if parallel.any_rank(self.stop_requested, self.device):
                    break
                start = time.monotonic()
                buffer = self.collect_rollout()
                metrics = self.update(buffer)
                self.update_index += 1
                self.scheduler.step()
                if self.update_index == self.cfg.get("actor_warmup_updates", 0):
                    self._configure_replay()
                self.dagger.step()
                histogram, collapsed = self.histogram.update(buffer.greedy_actions[buffer.ppo_mask])
                action_dim = self.config["model"]["action_dim"]
                sampled_hist = torch.bincount(
                    buffer.actions[buffer.ppo_mask], minlength=action_dim
                ).float()
                expert_hist = torch.bincount(
                    buffer.expert_actions.flatten(), minlength=action_dim
                ).float()
                matches = buffer.greedy_actions == buffer.expert_actions
                confusion = torch.bincount(
                    (buffer.expert_actions * action_dim + buffer.greedy_actions).flatten(),
                    minlength=action_dim * action_dim,
                ).view(action_dim, action_dim)
                stop_labels = buffer.expert_actions == 0
                recalls = [
                    matches[buffer.expert_actions == a].float().mean().item()
                    if (buffer.expert_actions == a).any()
                    else None
                    for a in range(action_dim)
                ]
                prior = expert_hist / expert_hist.sum()
                prior_ce = -(prior * prior.clamp_min(1e-8).log()).sum().item()
                metrics.update(
                    {
                        "update": self.update_index,
                        "dagger_beta": buffer.beta,
                        "reward": buffer.rewards[buffer.ppo_mask].mean().item(),
                        "episodes_completed": len(buffer.episode_metrics),
                        "oracle_skipped_episodes": len(buffer.oracle_failures),
                        "oracle_navigation_repairs": buffer.oracle_navigation_repairs,
                        "invalid_forward_labels_filtered": int((~buffer.il_mask).sum()),
                        "greedy_action_histogram": histogram,
                        "greedy_oracle_accuracy": matches.float().mean().item(),
                        "oracle_class_recall": recalls,
                        "expert_action_counts": expert_hist.long().tolist(),
                        "greedy_action_confusion": confusion.tolist(),
                        "stop_probability_on_teacher_stop": buffer.stop_probabilities[stop_labels]
                        .mean()
                        .item()
                        if stop_labels.any()
                        else None,
                        "stop_probability_on_teacher_nonstop": buffer.stop_probabilities[
                            ~stop_labels
                        ]
                        .mean()
                        .item()
                        if (~stop_labels).any()
                        else None,
                        "oracle_prior_cross_entropy": prior_ce,
                        "il_gain_over_prior": prior_ce - metrics["il_loss"],
                        "curriculum_warmup_steps": buffer.curriculum_warmup_steps,
                        "curriculum_fallbacks": buffer.curriculum_fallbacks,
                        "rollout_behavior": (
                            "sampled_policy_ppo_with_training_only_auxiliary_il"
                            if self.auxiliary_controller.num_envs
                            else "policy_on_policy_expert_labels_only"
                            if buffer.beta == 0
                            else "dagger_mixture_not_autonomous_evaluation"
                        ),
                        "on_policy_transitions": int(buffer.ppo_mask.sum()),
                        "auxiliary_il_transitions": int((~buffer.ppo_mask).sum()),
                        "auxiliary_episodes_completed": len(buffer.auxiliary_episode_metrics),
                        "auxiliary_success": sum(
                            m["success"] for m in buffer.auxiliary_episode_metrics
                        )
                        / max(len(buffer.auxiliary_episode_metrics), 1),
                        "auxiliary_expert_steps": buffer.auxiliary_expert_steps,
                        "auxiliary_recovery_triggers": buffer.auxiliary_recovery_triggers,
                        "teacher_stop_count_on_policy": int((stop_labels & buffer.ppo_mask).sum()),
                        "teacher_stop_count_auxiliary": int((stop_labels & ~buffer.ppo_mask).sum()),
                        "success": sum(m["success"] for m in buffer.episode_metrics)
                        / max(len(buffer.episode_metrics), 1),
                        "spl": sum(m["spl"] for m in buffer.episode_metrics)
                        / max(len(buffer.episode_metrics), 1),
                        "collision_rate": (buffer.collisions - buffer.auxiliary_collisions)
                        / int(buffer.ppo_mask.sum()),
                        "auxiliary_collision_rate": buffer.auxiliary_collisions
                        / max(int((~buffer.ppo_mask).sum()), 1),
                        "policy_action_histogram": (sampled_hist / sampled_hist.sum()).tolist(),
                        "expert_action_histogram": (expert_hist / expert_hist.sum()).tolist(),
                        "action_collapse": collapsed,
                        "rollout_fps": buffer.rewards.numel() / buffer.elapsed_s,
                        "rollout_seconds": buffer.elapsed_s,
                        "reset_seconds": buffer.reset_seconds,
                        "update_seconds": time.monotonic() - start,
                        "gpu_memory_bytes": torch.cuda.max_memory_allocated(self.device),
                    }
                )
                metrics = parallel.combine_metrics(parallel.gather(metrics))
                metrics["world_size"] = parallel.world_size()
                metrics["gradient_sync"] = "ddp" if self.ddp else "flat_allreduce"
                metrics["global_transitions"] = (
                    parallel.world_size() * self.cfg["num_envs"] * self.cfg["rollout_steps"]
                )
                metrics["total_env_steps"] = self.update_index * metrics["global_transitions"]
                metrics["global_rollout_fps"] = (
                    metrics["global_transitions"] / metrics["rollout_seconds"]
                )
                metrics["global_update_fps"] = (
                    metrics["global_transitions"] / metrics["update_seconds"]
                )
                if parallel.rank() == 0:
                    append_json(self.run_dir / "train_metrics.jsonl", metrics)
                    print(json.dumps(metrics), flush=True)
                # Release rollout graphs and snapshots before checkpoint/evaluation.
                del buffer
                if (
                    self.update_index == 1
                    or self.update_index % self.cfg["checkpoint_interval"] == 0
                ):
                    self.save_checkpoint()
                if (
                    self.update_index % self.cfg["eval_interval"] == 0
                    or self.update_index in self.cfg.get("early_eval_updates", [])
                    or (self.update_index == 1 and self.cfg.get("eval_first_update", False))
                ):
                    if not parallel.any_rank(self.stop_requested, self.device):
                        self.evaluate()
            self.save_checkpoint()
        except ReplayConsistencyError:
            # This error is raised collectively only before the first optimizer
            # step. The model/Adam still belong to the last completed update.
            self.save_checkpoint()
            raise
        finally:
            self.envs.close()
            for sig, handler in old_handlers.items():
                signal.signal(sig, handler)


@hydra.main(version_base=None, config_path="../../../configs", config_name="config")
def main(cfg):
    config = OmegaConf.to_container(cfg, resolve=True)
    parallel.initialize(config)
    try:
        EndToEndObjectNavTrainer(config).run()
    finally:
        if torch.distributed.is_initialized():
            torch.distributed.destroy_process_group()


if __name__ == "__main__":
    main()
