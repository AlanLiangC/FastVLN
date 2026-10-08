import argparse
import json

from streamnav.models.qwen35_kda.conversion import convert

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", default="Qwen/Qwen3.5-0.8B")
    parser.add_argument("--output", default="checkpoints/qwen35_0p8b_kda")
    args = parser.parse_args()
    print(json.dumps(convert(args.source, args.output), indent=2))
