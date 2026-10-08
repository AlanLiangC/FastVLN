import json
import os
import signal
import time
from pathlib import Path

import hydra
import torch
import torch.nn.functional as F
import yaml
from omegaconf import OmegaConf

from streamnav.data.manifest import check_leakage, verify_manifest
from streamnav.data.mixture import make_source, partition_scenes
from streamnav.data.schema import read_json
from streamnav.envs.habitat_client import VectorHabitatEnvs
from streamnav.evaluation.runner import evaluate
from streamnav.training import distributed as parallel
from streamnav.training.checkpoint import (
    load_policy,
    promote_best_checkpoint,
    provenance,
    restore_training,
    save_checkpoint,
)
from streamnav.training.dagger import DaggerBetaScheduler, behavior_log_prob
from streamnav.training.ealm import EntropyAdaptiveLossMixer
from streamnav.training.gae import compute_gae
from streamnav.training.optimizer import OVSegDTLRScheduler, build_optimizer, gradient_norm
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
        seed_everything(config["seed"])
        self.stop_requested = False
        self.update_index = 0
        if self.cfg["num_envs"] < 1 or self.cfg["sequence_batch_size"] < 1:
            raise ValueError("Environment count and sequence batch size must be positive")
        if self.cfg["rollout_steps"] % self.cfg["sequence_length"]:
            raise ValueError("rollout_steps must be divisible by sequence_length")
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
        if parallel.rank() == 0:
            (self.run_dir / "manifest.json").write_text(json.dumps(self.manifest, indent=2))
            (self.run_dir / "resolved_config.yaml").write_text(yaml.safe_dump(config))
        self.policy = load_policy(config, training=True)
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
            )
        self.replay: torch.nn.Module = SequenceReplay(self.policy)
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
        self.collector = RolloutCollector(self.policy, self.envs, self.sources, self.cfg)
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
        self.replay = SequenceReplay(self.policy)
        if self.ddp:
            self.replay = torch.nn.parallel.DistributedDataParallel(
                self.replay,
                device_ids=[self.device.index],
                broadcast_buffers=False,
                gradient_as_bucket_view=True,
                find_unused_parameters=True,
            )

    def autocast(self):
        return torch.autocast(
            device_type=self.device.type, dtype=self.dtype, enabled=self.dtype != torch.float32
        )

    def collect_rollout(self):
        self.collector.update_index = self.update_index
        with self.autocast():
            return self.collector.collect(self.dagger.beta)

    def compute_losses(self, logits, values, buffer, sequences):
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
        weighted_il = il_loss * weights / parallel.average_scalar(weights.mean())
        new_log_prob = behavior_log_prob(logits, actions, expert, buffer.beta)
        ppo_loss, ratio = clipped_ppo_loss(
            new_log_prob,
            buffer.gather("old_log_probs", sequences, self.device),
            buffer.gather("advantages", sequences, self.device),
            self.cfg["ppo"]["clip_eps"],
        )
        policy_loss, alpha = self.mixer(weighted_il, ppo_loss, entropy)
        value_loss = ovsegdt_value_loss(
            values,
            buffer.gather("old_values", sequences, self.device),
            buffer.gather("returns", sequences, self.device),
            self.cfg["ppo"]["clip_eps"],
            self.cfg["ppo"].get("use_clipped_value_loss", True),
        )
        total = (
            policy_loss.mean()
            + self.cfg["loss"]["value_coef"] * value_loss.mean()
            - self.cfg["loss"]["entropy_coef"] * entropy.mean()
        )
        metrics = {
            "total_loss": total.detach().item(),
            "il_loss": il_loss.mean().detach().item(),
            "weighted_il_loss": weighted_il.mean().detach().item(),
            "ppo_loss": ppo_loss.mean().detach().item(),
            "value_loss": value_loss.mean().detach().item(),
            "entropy": entropy.mean().detach().item(),
            "ealm_alpha": alpha.mean().item(),
            "clip_fraction": ((ratio - 1).abs() > self.cfg["ppo"]["clip_eps"])
            .float()
            .mean()
            .item(),
            **replay_consistency_metrics(
                new_log_prob, buffer.gather("old_log_probs", sequences, self.device)
            ),
        }
        return total, metrics

    def update(self, buffer):
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
            parallel.normalize_advantages(advantages, self.device)
            if cfg["ppo"].get("use_normalized_advantage", False)
            else advantages
        )
        self.policy.train()
        all_metrics: list[dict[str, float]] = []
        replay_check = {}
        start = time.monotonic()
        for _ in range(cfg["update_epochs"]):
            for sequences in buffer.sequence_batches(cfg["sequence_batch_size"]):
                self.optimizer.zero_grad(set_to_none=True)
                with self.autocast():
                    logits, values = self.replay(buffer, sequences)
                    loss, metrics = self.compute_losses(logits, values, buffer, sequences)
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
                    if parallel.any_rank(error > tolerance, self.device):
                        raise RuntimeError(
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
                norm = torch.nn.utils.clip_grad_norm_(
                    self.policy.parameters(), cfg["max_grad_norm"], error_if_nonfinite=True
                )
                metrics["grad_norm"] = norm.item()
                self.optimizer.step()
                self.mixer.observe_entropy(metrics["entropy"])
                all_metrics.append(metrics)
        metrics = {k: sum(m[k] for m in all_metrics) / len(all_metrics) for k in all_metrics[0]}
        metrics.update(replay_check)
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
                histogram, collapsed = self.histogram.update(buffer.greedy_actions)
                action_dim = self.config["model"]["action_dim"]
                sampled_hist = torch.bincount(
                    buffer.actions.flatten(), minlength=action_dim
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
                        "reward": buffer.rewards.mean().item(),
                        "episodes_completed": len(buffer.episode_metrics),
                        "oracle_skipped_episodes": len(buffer.oracle_failures),
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
                        "rollout_behavior": "policy_on_policy_expert_labels_only"
                        if buffer.beta == 0
                        else "dagger_mixture_not_autonomous_evaluation",
                        "success": sum(m["success"] for m in buffer.episode_metrics)
                        / max(len(buffer.episode_metrics), 1),
                        "spl": sum(m["spl"] for m in buffer.episode_metrics)
                        / max(len(buffer.episode_metrics), 1),
                        "collision_rate": buffer.collisions / buffer.rewards.numel(),
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
