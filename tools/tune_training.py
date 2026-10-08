"""Measure real Habitat rollout/recurrent optimization; never save benchmark weights."""

import argparse
import gc
import json
import statistics
import subprocess
import threading
import time
from pathlib import Path

import torch
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

from streamnav.training.trainer import EndToEndObjectNavTrainer


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--envs", type=int, default=8)
    parser.add_argument("--batches", type=int, nargs="+", default=[4, 8, 16])
    parser.add_argument("--repeats", type=int, default=2)
    parser.add_argument("--unfused", action="store_true")
    parser.add_argument("--no-grad-checkpoint", action="store_true")
    parser.add_argument("--output", default="runtime/reports/training_tuning.json")
    args = parser.parse_args()
    destination = Path(args.output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with initialize_config_dir(version_base=None, config_dir=str(Path("configs").resolve())):
        cfg = compose(
            config_name="config",
            overrides=[
                f"device=cuda:{args.gpu}",
                f"habitat.gpu_device_id={args.gpu}",
                f"trainer.num_envs={args.envs}",
                f"run_dir=runtime/tuning/{destination.stem}",
                f"trainer.fused_optimizer={str(not args.unfused).lower()}",
                f"model.gradient_checkpointing={str(not args.no_grad_checkpoint).lower()}",
            ],
        )
    trainer = EndToEndObjectNavTrainer(OmegaConf.to_container(cfg, resolve=True))
    records = []
    try:
        buffer = trainer.collect_rollout()
        rollout_fps = buffer.steps * buffer.num_envs / buffer.elapsed_s
        for batch in args.batches:
            trainer.cfg["sequence_batch_size"] = batch
            samples, done = [], threading.Event()

            def sample_gpu():
                while not done.is_set():
                    result = subprocess.run(
                        [
                            "nvidia-smi",
                            "-i",
                            str(args.gpu),
                            "--query-gpu=utilization.gpu,memory.used",
                            "--format=csv,noheader,nounits",
                        ],
                        capture_output=True,
                        text=True,
                    )
                    if result.returncode == 0:
                        samples.append([float(x) for x in result.stdout.strip().split(",")])
                    done.wait(0.5)

            watcher = threading.Thread(target=sample_gpu, daemon=True)
            watcher.start()
            record = {
                "sequence_batch_size": batch,
                "num_envs": args.envs,
                "fused_optimizer": not args.unfused,
                "gradient_checkpointing": not args.no_grad_checkpoint,
                "rollout_fps_cold": rollout_fps,
            }
            try:
                trainer.update(buffer)  # Warm Triton kernels and optimizer state.
                torch.cuda.synchronize(trainer.device)
                torch.cuda.reset_peak_memory_stats(trainer.device)
                metrics = [trainer.update(buffer) for _ in range(args.repeats)]
                record.update(
                    {
                        "training_fps": statistics.median(m["training_fps"] for m in metrics),
                        "peak_allocated_bytes": torch.cuda.max_memory_allocated(trainer.device),
                        "finite_loss": all(
                            torch.isfinite(torch.tensor(m["total_loss"])) for m in metrics
                        ),
                        "status": "ok",
                    }
                )
            except torch.OutOfMemoryError:
                trainer.optimizer.zero_grad(set_to_none=True)
                gc.collect()
                torch.cuda.empty_cache()
                record["status"] = "oom"
            finally:
                done.set()
                watcher.join()
            record["gpu_utilization_mean"] = (
                statistics.mean(s[0] for s in samples) if samples else None
            )
            record["device_memory_peak_mib"] = max((s[1] for s in samples), default=0)
            records.append(record)
            destination.write_text(
                json.dumps(
                    {
                        "purpose": "benchmark_not_training",
                        "time": time.time(),
                        "sensor": {"width": cfg.habitat.width, "height": cfg.habitat.height},
                        "records": records,
                    },
                    indent=2,
                )
            )
            print(json.dumps(record), flush=True)
    finally:
        trainer.envs.close()


if __name__ == "__main__":
    main()
