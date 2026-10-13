"""Fetch annotation-only RefCOCO mirrors; reuse local COCO RGB and masks."""

import argparse
import json
from pathlib import Path

import pyarrow.parquet as pq
from huggingface_hub import HfApi, snapshot_download


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default="runtime/data/grounding/referring")
    args = parser.parse_args()
    root = Path(args.root)
    root.mkdir(parents=True, exist_ok=True)
    sources = {}
    with (root / "annotations.jsonl.writing").open("w") as out:
        for name in ("refcoco", "refcocoplus", "refcocog"):
            repo = f"jxu124/{name}"
            revision = HfApi().dataset_info(repo).sha
            folder = root / name
            snapshot_download(
                repo,
                repo_type="dataset",
                revision=revision,
                local_dir=folder,
                allow_patterns=["README.md", "data/*.parquet"],
            )
            count = 0
            for path in sorted((folder / "data").glob("*.parquet")):
                for batch in pq.ParquetFile(path).iter_batches(
                    batch_size=2048, columns=["image_id", "ann_id", "split", "captions"]
                ):
                    for row in batch.to_pylist():
                        out.write(json.dumps({**row, "source": name}, ensure_ascii=False) + "\n")
                        count += 1
            sources[name] = {"repo": repo, "revision": revision, "object_references": count}
            print(json.dumps({name: sources[name]}), flush=True)
    (root / "annotations.jsonl.writing").replace(root / "annotations.jsonl")
    (root / "sources.json").write_text(json.dumps(sources, indent=2) + "\n")


if __name__ == "__main__":
    main()
