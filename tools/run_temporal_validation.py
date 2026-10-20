"""Run paired controller variants for one case shard on one fixed GPU."""

import argparse
import subprocess
import sys
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True)
    parser.add_argument("--shard-index", type=int, required=True)
    parser.add_argument("--shards", type=int, default=8)
    args = parser.parse_args()
    root = Path(args.root)
    for name, temporal, arrival in (
        ("baseline", 0, 0),
        ("temporal", 0.25, 0),
        ("temporal_arrival", 0.25, 0.1),
    ):
        print(f"Starting {name} shard {args.shard_index}/{args.shards}", flush=True)
        subprocess.run(
            [
                sys.executable,
                "tools/validate_action_chunks.py",
                "--checkpoint",
                str(root / "checkpoint"),
                "--cases",
                str(root / "cases.json"),
                "--horizon",
                "1",
                "--output",
                str(root / name / f"shard_{args.shard_index:03d}"),
                "--temporal-weight",
                str(temporal),
                "--arrival-weight",
                str(arrival),
                "--shard-index",
                str(args.shard_index),
                "--shards",
                str(args.shards),
            ],
            check=True,
        )


if __name__ == "__main__":
    main()
