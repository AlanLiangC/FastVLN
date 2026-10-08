"""Record selected real viewer episodes using only autonomous policy actions."""

import argparse
import base64
import io
import json
import urllib.request
from pathlib import Path

import imageio.v2 as imageio
import numpy as np
from PIL import Image


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default="http://127.0.0.1:8765")
    parser.add_argument("--indices", type=int, nargs="+", required=True)
    parser.add_argument("--split", type=int, default=0)
    parser.add_argument("--max-steps", type=int, default=500)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    root = Path(args.output)
    root.mkdir(parents=True, exist_ok=True)

    def call(route, payload):
        request = urllib.request.Request(
            args.url + route, json.dumps(payload).encode(), {"Content-Type": "application/json"}
        )
        with urllib.request.urlopen(request, timeout=120) as response:
            return json.load(response)

    def frame(response):
        return np.asarray(Image.open(io.BytesIO(base64.b64decode(response["rgb_base64"]))))

    for index in args.indices:
        first = call("/demo/reset", {"episode_index": index, "split_index": args.split})
        records = []
        stem = f"split{args.split}_sample{index:02d}"
        with imageio.get_writer(str(root / f"{stem}.mp4"), fps=6, macro_block_size=2) as writer:
            writer.append_data(frame(first))
            for _ in range(args.max_steps):
                result = call("/demo/step", {})
                records.append({k: v for k, v in result.items() if k != "rgb_base64"})
                writer.append_data(frame(result))
                if result["done"]:
                    break
        report = {
            "index": index,
            "episode_id": first["episode_id"],
            "checkpoint": first["checkpoint"],
            "initial_distance": first["distance"],
            "metrics": result["metrics"],
            "steps": records,
        }
        (root / f"{stem}.json").write_text(json.dumps(report, indent=2))
        print(json.dumps({k: v for k, v in report.items() if k != "steps"}), flush=True)


if __name__ == "__main__":
    main()
