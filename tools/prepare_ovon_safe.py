"""Explicitly exclude OVON unseen target categories from the closed-vocabulary sources."""

import json
from pathlib import Path

from streamnav.data.manifest import build_manifest, check_leakage
from streamnav.data.schema import read_json

if __name__ == "__main__":
    root = Path("runtime/data")
    manifests = root / "manifests"
    unseen = read_json(manifests / "hm3d_ovon_val_unseen.json")
    safe = []
    for version in ("v1", "v2"):
        original = read_json(manifests / f"hm3d_{version}_train.json")
        excluded = sorted(set(original["categories"]) & set(unseen["categories"]))
        target = manifests / f"hm3d_{version}_train_ovon_safe.json"
        m = build_manifest(
            f"hm3d_{version}",
            "train",
            Path(original["files"][0]["path"]).parent,
            root / "scene_datasets",
            target,
            excluded,
        )
        safe.append(m)
        print(
            json.dumps(
                {
                    k: m[k]
                    for k in ("dataset_id", "episodes", "excluded_categories", "excluded_episodes")
                }
            ),
            flush=True,
        )
    safe.append(read_json(manifests / "hm3d_ovon_train.json"))
    val = [read_json(p) for p in manifests.glob("*val*.json")]
    check_leakage(safe, val)
    print("Mixed training leakage checks passed.")
