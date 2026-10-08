"""User-requested allocation keepalive, separate from useful model training.

Triggers only after prolonged low utilization and exits with the learner.
Synthetic GEMMs never count as training progress or benchmark samples.
"""

import argparse
import json
import os
import subprocess
import time
from pathlib import Path


def alive(pid):
    try:
        return not Path(f"/proc/{pid}/stat").read_text().split(")", 1)[1].strip().startswith("Z")
    except FileNotFoundError:
        return False


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pid", type=int, required=True)
    parser.add_argument("--gpu", type=int, required=True)
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--idle-seconds", type=float, default=7200)
    parser.add_argument("--burst-seconds", type=float, default=15)
    parser.add_argument("--interval", type=float, default=10)
    parser.add_argument(
        "--once", action="store_true", help="Run one labeled diagnostic burst and exit"
    )
    args = parser.parse_args()
    if min(args.idle_seconds, args.burst_seconds, args.interval) <= 0:
        parser.error("Durations must be positive")
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    physical = visible.split(",")[args.gpu] if visible else str(args.gpu)
    last_busy = time.monotonic()
    output = Path(args.run_dir) / "keepalive_events.jsonl"
    output.parent.mkdir(parents=True, exist_ok=True)
    while alive(args.pid):
        result = subprocess.run(
            [
                "nvidia-smi",
                "-i",
                physical,
                "--query-gpu=utilization.gpu",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
        )
        if result.returncode:
            raise RuntimeError("Cannot read GPU utilization; keepalive stopped")
        if float(result.stdout.strip()) >= 50:
            last_busy = time.monotonic()
        if args.once or time.monotonic() - last_busy >= args.idle_seconds:
            import torch

            device = torch.device(f"cuda:{args.gpu}")
            with torch.cuda.device(device), torch.no_grad():
                a = torch.randn(4096, 4096, dtype=torch.float16, device=device)
                b, c = torch.randn_like(a), torch.empty_like(a)
                start, count = time.monotonic(), 0
                while time.monotonic() - start < args.burst_seconds and alive(args.pid):
                    torch.mm(a, b, out=c)
                    torch.cuda.synchronize(device)
                    count += 1
                del a, b, c
                torch.cuda.empty_cache()
            event = {
                "time": time.time(),
                "gpu": args.gpu,
                "training_pid": args.pid,
                "purpose": "synthetic_allocation_keepalive_not_training",
                "duration_s": time.monotonic() - start,
                "matmuls": count,
            }
            with output.open("a") as f:
                f.write(json.dumps(event) + "\n")
            print(json.dumps(event), flush=True)
            last_busy = time.monotonic()
            if args.once:
                return
        time.sleep(args.interval)


if __name__ == "__main__":
    main()
