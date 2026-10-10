"""Benchmark frozen-vision chat replay on a shared real Habitat rollout."""

import argparse
import json
import statistics
import time
from dataclasses import replace
from pathlib import Path

import torch
import yaml

from streamnav.contracts.state import LayerState
from streamnav.training import distributed as parallel
from streamnav.training.gae import compute_gae
from streamnav.training.rollout import replay_sequences
from streamnav.training.trainer import EndToEndObjectNavTrainer


def move_boundaries(buffer, device):
    buffer.initial_states = {
        key: replace(
            state,
            kda_cache=tuple(
                LayerState(
                    layer.conv.to(device) if layer.conv is not None else None,
                    layer.recurrent.to(device) if layer.recurrent is not None else None,
                )
                for layer in state.kda_cache
            ),
        )
        for key, state in buffer.initial_states.items()
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--pack-frames", type=int, default=1)
    parser.add_argument("--no-checkpointing", action="store_true")
    parser.add_argument("--buffer-in")
    parser.add_argument("--buffer-out")
    parser.add_argument("--reference-in")
    parser.add_argument("--reference-out")
    args = parser.parse_args()
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    config = yaml.safe_load((Path(args.checkpoint) / "resolved_config.yaml").read_text())
    config.update(checkpoint=None, device=args.device, run_dir=str(output.with_suffix("")))
    config["model"]["checkpoint"] = args.checkpoint
    config["habitat"]["gpu_device_id"] = int(args.device.split(":")[-1])
    config["trainer"].update(
        num_envs=2,
        rollout_steps=args.steps,
        sequence_length=args.steps,
        sequence_batch_size=2,
        actor_warmup_updates=0,
        verify_data_hashes=False,
        cache_frozen_vision=True,
    )
    parallel.initialize(config)
    trainer = EndToEndObjectNavTrainer(config)
    trainer.policy.backbone.gradient_checkpointing = not args.no_checkpointing
    try:
        if args.buffer_in:
            buffer = torch.load(args.buffer_in, weights_only=False, map_location="cpu")
            move_boundaries(buffer, trainer.device)
        else:
            buffer = trainer.collect_rollout()
            if args.buffer_out:
                move_boundaries(buffer, "cpu")
                torch.save(buffer, args.buffer_out)
                move_boundaries(buffer, trainer.device)
                print(json.dumps({"buffer_saved": args.buffer_out}), flush=True)
        ppo = config["trainer"]["ppo"]
        buffer.advantages, buffer.returns = compute_gae(
            buffer.rewards,
            buffer.old_values,
            buffer.dones,
            buffer.last_values,
            ppo["gamma"],
            ppo["gae_lambda"],
            buffer.timeout_bootstrap,
        )
        sequences = next(buffer.sequence_batches(2, shuffle=False))
        trainer.policy.train()
        samples = []
        for repeat in range(args.repeats):
            trainer.optimizer.zero_grad(set_to_none=True)
            torch.cuda.reset_peak_memory_stats(trainer.device)
            torch.cuda.synchronize(trainer.device)
            start = time.perf_counter()
            with trainer.autocast():
                logits, values = replay_sequences(
                    trainer.policy,
                    buffer,
                    sequences,
                    cache_embeddings=True,
                    pack_frames=args.pack_frames,
                )
                loss, metrics = trainer.compute_losses(logits, values, buffer, sequences)
            loss.backward()
            torch.cuda.synchronize(trainer.device)
            samples.append(
                {
                    "repeat": repeat,
                    "seconds": time.perf_counter() - start,
                    "peak_allocated_gib": torch.cuda.max_memory_allocated(trainer.device) / 2**30,
                    "loss": loss.item(),
                    "replay_log_prob_error_max": metrics["replay_log_prob_error_max"],
                }
            )
            print(json.dumps(samples[-1]), flush=True)
            if repeat < args.repeats - 1:
                del logits, values, loss
        reference = {
            "logits": logits.detach().cpu(),
            "values": values.detach().cpu(),
            "gradients": {
                name: p.grad.detach().cpu()
                for name, p in trainer.policy.named_parameters()
                if p.grad is not None
            },
        }
        comparison = {}
        if args.reference_out:
            torch.save(reference, args.reference_out)
        if args.reference_in:
            expected = torch.load(args.reference_in, weights_only=True, map_location="cpu")
            comparison["logits_max_absolute_difference"] = (
                (reference["logits"] - expected["logits"]).abs().max().item()
            )
            comparison["values_max_absolute_difference"] = (
                (reference["values"] - expected["values"]).abs().max().item()
            )
            comparison["same_gradient_parameter_names"] = set(reference["gradients"]) == set(
                expected["gradients"]
            )
            groups = {}
            for name, gradient in reference["gradients"].items():
                group = "body" if name.startswith("backbone.") else "head"
                stats = groups.setdefault(
                    group, {"reference_l2_squared": 0.0, "delta_l2_squared": 0.0}
                )
                original = expected["gradients"][name]
                stats["reference_l2_squared"] += original.double().square().sum().item()
                stats["delta_l2_squared"] += (
                    (gradient.double() - original.double()).square().sum().item()
                )
            for stats in groups.values():
                stats["relative_l2_difference"] = (
                    stats["delta_l2_squared"] / max(stats["reference_l2_squared"], 1e-30)
                ) ** 0.5
            comparison["gradients"] = groups
        report = {
            "checkpoint": args.checkpoint,
            "pack_frames": args.pack_frames,
            "gradient_checkpointing": not args.no_checkpointing,
            "steps": buffer.steps,
            "environments": buffer.num_envs,
            "resets": len(buffer.resets),
            "samples": samples,
            "mean_warm_seconds": statistics.mean(s["seconds"] for s in samples[1:] or samples),
            "comparison": comparison,
            "protocol": "Same saved real Habitat buffer and unchanged weights, full forward/backward; excludes optimizer/DDP. First repeat warms kernels. No navigation-score claim.",
        }
        output.write_text(json.dumps(report, indent=2) + "\n")
        print(json.dumps(report), flush=True)
    finally:
        trainer.envs.close()


if __name__ == "__main__":
    main()
