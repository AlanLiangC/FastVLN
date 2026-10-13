"""Fetch the pinned official multimodal base; conversions are unnecessary."""

import argparse
import json
from pathlib import Path

from huggingface_hub import snapshot_download

REVISION = "2fc06364715b967f1860aea9cf38778875588b17"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default="checkpoints/qwen35_0p8b_base")
    args = parser.parse_args()
    snapshot_download("Qwen/Qwen3.5-0.8B", revision=REVISION, local_dir=args.output)
    Path(args.output, "source_revision.json").write_text(
        json.dumps({"repo": "Qwen/Qwen3.5-0.8B", "revision": REVISION}, indent=2)
    )


if __name__ == "__main__":
    main()
