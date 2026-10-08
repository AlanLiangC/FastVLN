import argparse
import json
from pathlib import Path

from streamnav.data.manifest import build_manifest, check_leakage


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", default="runtime/data")
    parser.add_argument("--output", default="runtime/data/manifests")
    args = parser.parse_args()
    root = Path(args.data_root)
    manifests = []
    for dataset, subdir, splits in [
        ("hm3d_v1", "datasets/objectnav/hm3d/v1/objectnav_hm3d_v1", ["train", "val"]),
        ("hm3d_v2", "datasets/objectnav/hm3d/v2/objectnav_hm3d_v2", ["train", "val"]),
        (
            "hm3d_ovon",
            "datasets/ovon/hm3d",
            ["train", "val_seen", "val_seen_synonyms", "val_unseen"],
        ),
    ]:
        for split in splits:
            m = build_manifest(
                dataset,
                split,
                root / subdir / split / "content",
                root / "scene_datasets",
                Path(args.output) / f"{dataset}_{split}.json",
            )
            print(json.dumps({k: m[k] for k in ("dataset_id", "split", "episodes")}), flush=True)
            manifests.append(m)
    # Raw closed-vocabulary HM3D includes 'plant', an OVON unseen category.
    # Preserve raw manifests for standalone baselines; make explicit filtered ones
    # for mixed training, never mutate the upstream episode archives.
    unseen = next(m for m in manifests if m["split"] == "val_unseen")
    train = []
    for manifest in (m for m in manifests if m["split"] == "train"):
        if manifest["dataset_id"] != "hm3d_ovon":
            excluded = set(manifest["categories"]) & set(unseen["categories"])
            manifest = build_manifest(
                manifest["dataset_id"],
                "train",
                Path(manifest["files"][0]["path"]).parent,
                root / "scene_datasets",
                Path(args.output) / f"{manifest['dataset_id']}_train_ovon_safe.json",
                excluded,
            )
        train.append(manifest)
    check_leakage(train, [m for m in manifests if m["split"] != "train"])
    print("Mixed training scene and unseen-category leakage checks passed.")


if __name__ == "__main__":
    main()
