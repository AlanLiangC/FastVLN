"""Compare complete real-rollout replay outputs, gradients and token lookup cost."""

import argparse
import json
import statistics
import time
from pathlib import Path

import torch
import yaml

from streamnav.training import distributed as parallel
from streamnav.training.gae import compute_gae
from streamnav.training.rollout import replay_sequences
from streamnav.training.trainer import EndToEndObjectNavTrainer


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--repeats", type=int, default=2)
    args = parser.parse_args()
    if args.steps < 1 or args.repeats < 1:
        parser.error("Steps and repeats must be positive")
    output = Path(args.output)
    config = yaml.safe_load((Path(args.checkpoint) / "resolved_config.yaml").read_text())
    config.update(checkpoint=None, device=args.device, run_dir=str(output.parent / "replay_probe"))
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
    calls = []
    hook = trainer.policy.backbone.embeddings.register_forward_hook(lambda *args: calls.append(1))
    try:
        buffer = trainer.collect_rollout()
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
        samples, reference_gradients, reference_logits, reference_values = [], {}, None, None
        gradient_squared, delta_squared, max_gradient_delta = 0.0, 0.0, 0.0
        for repeat in range(args.repeats):
            for cached in (False, True):
                trainer.optimizer.zero_grad(set_to_none=True)
                calls.clear()
                torch.cuda.synchronize(trainer.device)
                start = time.perf_counter()
                with trainer.autocast():
                    logits, values = replay_sequences(
                        trainer.policy, buffer, sequences, cache_embeddings=cached
                    )
                    loss, _ = trainer.compute_losses(logits, values, buffer, sequences)
                loss.backward()
                torch.cuda.synchronize(trainer.device)
                sample = {
                    "repeat": repeat,
                    "cached": cached,
                    "forward_backward_seconds": time.perf_counter() - start,
                    "embedding_lookups": len(calls),
                    "loss": loss.item(),
                }
                samples.append(sample)
                print(json.dumps(sample), flush=True)
                if repeat == 0:
                    if not cached:
                        reference_logits, reference_values = (
                            logits.detach().cpu(),
                            values.detach().cpu(),
                        )
                        reference_gradients = {
                            name: p.grad.detach().cpu()
                            for name, p in trainer.policy.named_parameters()
                            if p.grad is not None
                        }
                    else:
                        torch.testing.assert_close(logits.cpu(), reference_logits, atol=0, rtol=0)
                        torch.testing.assert_close(values.cpu(), reference_values, atol=0, rtol=0)
                        actual_names = set()
                        for name, p in trainer.policy.named_parameters():
                            if p.grad is None:
                                continue
                            actual_names.add(name)
                            expected = reference_gradients[name]
                            delta = p.grad.detach().cpu() - expected
                            gradient_squared += expected.square().sum().item()
                            delta_squared += delta.square().sum().item()
                            max_gradient_delta = max(max_gradient_delta, delta.abs().max().item())
                        assert actual_names == set(reference_gradients)
                        assert (delta_squared / gradient_squared) ** 0.5 < 2e-5
                        assert max_gradient_delta < 1e-4
                        reference_gradients.clear()
                del logits, values, loss
        original = statistics.mean(
            s["forward_backward_seconds"] for s in samples if not s["cached"]
        )
        cached_seconds = statistics.mean(
            s["forward_backward_seconds"] for s in samples if s["cached"]
        )
        report = {
            "checkpoint": args.checkpoint,
            "steps": args.steps,
            "environments": 2,
            "samples": samples,
            "outputs_exactly_equal": True,
            "all_gradient_relative_l2": (delta_squared / gradient_squared) ** 0.5,
            "max_gradient_absolute_difference": max_gradient_delta,
            "mean_original_seconds": original,
            "mean_cached_seconds": cached_seconds,
            "relative_time_reduction": 1 - cached_seconds / original,
            "protocol": "Same real Habitat buffer and unchanged weights. Whole replay forward/backward; no optimizer or navigation-score claim. FP32 gradient accumulation order may differ.",
        }
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(report, indent=2) + "\n")
        print(json.dumps(report), flush=True)
    finally:
        hook.remove()
        trainer.envs.close()


if __name__ == "__main__":
    main()
