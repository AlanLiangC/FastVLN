"""Detach jobs with a new OS session, including non-interactive tool terminals."""

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", required=True)
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--monitor", action="store_true")
    parser.add_argument("--keepalive-gpu", type=int, action="append", default=[])
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    directory = Path(args.run_dir)
    directory.mkdir(parents=True, exist_ok=True)
    pid_file = directory / f"{args.name}.pid"
    if pid_file.exists():
        pid = int(pid_file.read_text())
        command_file = Path(f"/proc/{pid}/cmdline")
        if command_file.exists() and b"streamnav" in command_file.read_bytes():
            raise RuntimeError(f"{args.name} already running as PID {pid}")
    command = args.command[1:] if args.command and args.command[0] == "--" else args.command
    if not command:
        parser.error("Supply a Python command after --")
    log_path = directory / f"{args.name}.log"
    with open(log_path, "a") as log:
        log.write(f"\nStarting at {time.strftime('%Y-%m-%d %H:%M:%S UTC', time.gmtime())}\n")
        log.flush()
        process = subprocess.Popen(
            [sys.executable, *command],
            stdin=subprocess.DEVNULL,
            stdout=log,
            stderr=subprocess.STDOUT,
            start_new_session=True,
            env=os.environ.copy(),
        )
    pid_file.write_text(str(process.pid) + "\n")
    if args.monitor:
        with open(directory / "monitor.log", "a") as log:
            monitor = subprocess.Popen(
                [
                    sys.executable,
                    "tools/monitor_training.py",
                    "--pid",
                    str(process.pid),
                    "--run-dir",
                    str(directory),
                ],
                stdin=subprocess.DEVNULL,
                stdout=log,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
        (directory / "monitor.pid").write_text(str(monitor.pid) + "\n")
    keepalive_pids = {}
    for gpu in args.keepalive_gpu:
        with open(directory / f"keepalive_gpu{gpu}.log", "a") as log:
            keepalive = subprocess.Popen(
                [
                    sys.executable,
                    "tools/gpu_keepalive.py",
                    "--pid",
                    str(process.pid),
                    "--gpu",
                    str(gpu),
                    "--run-dir",
                    str(directory),
                ],
                stdin=subprocess.DEVNULL,
                stdout=log,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
        keepalive_pids[gpu] = keepalive.pid
        (directory / f"keepalive_gpu{gpu}.pid").write_text(str(keepalive.pid) + "\n")
    if keepalive_pids:
        (directory / "keepalive_pids.json").write_text(json.dumps(keepalive_pids))
        if len(keepalive_pids) == 1:
            (directory / "keepalive.pid").write_text(
                str(next(iter(keepalive_pids.values()))) + "\n"
            )
    print(f"{args.name} PID: {process.pid}\nLog: {log_path.resolve()}")


if __name__ == "__main__":
    main()
