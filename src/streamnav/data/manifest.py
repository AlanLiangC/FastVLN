from __future__ import annotations

import hashlib
import json
from pathlib import Path

from streamnav.data.schema import read_json
from streamnav.errors import DatasetIntegrityError


def file_hash(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def build_manifest(dataset_id, split, content_dir, scene_root, output, excluded_categories=()):
    assets = {p.name: str(p.resolve()) for p in Path(scene_root).rglob("*.basis.glb")}
    files, scenes, categories, paths = [], set(), set(), {}
    excluded = set(excluded_categories)
    removed = 0
    for path in sorted(Path(content_dir).glob("*.json.gz")):
        if path.name.startswith("."):
            continue
        data = read_json(path)
        before = len(data["episodes"])
        if excluded:
            data["episodes"] = [
                e
                for e in data["episodes"]
                if not excluded.intersection(
                    [
                        e["object_category"].strip().casefold(),
                        *(c.strip().casefold() for c in e.get("children_object_categories", [])),
                    ]
                )
            ]
        removed += before - len(data["episodes"])
        if not data["episodes"]:
            continue
        for episode in data["episodes"]:
            name = Path(episode["scene_id"]).name
            if name not in paths:
                if name not in assets:
                    raise DatasetIntegrityError(f"Missing scene asset {name} for {path}")
                navmesh = Path(assets[name]).with_suffix(".navmesh")
                if not navmesh.exists():
                    raise DatasetIntegrityError(f"Missing navmesh {navmesh}")
                paths[name] = assets[name]
            scenes.add(name)
            categories.add(episode["object_category"].strip().casefold())
            categories.update(
                c.strip().casefold() for c in episode.get("children_object_categories", [])
            )
        files.append(
            {
                "path": str(path.resolve()),
                "sha256": file_hash(path),
                "episodes": len(data["episodes"]),
            }
        )
    manifest = {
        "format_version": 1,
        "dataset_id": dataset_id,
        "split": split,
        "files": files,
        "scene_paths": paths,
        "scenes": sorted(scenes),
        "categories": sorted(categories),
        "episodes": sum(f["episodes"] for f in files),
        "excluded_categories": sorted(excluded),
        "excluded_episodes": removed,
    }
    if not files:
        raise DatasetIntegrityError(f"No episodes found: {content_dir}")
    Path(output).parent.mkdir(parents=True, exist_ok=True)
    Path(output).write_text(json.dumps(manifest, indent=2))
    return manifest


def check_leakage(train_manifests, validation_manifests):
    train_scenes = set().union(*(set(m["scenes"]) for m in train_manifests))
    train_categories = set().union(*(set(m["categories"]) for m in train_manifests))
    for val in validation_manifests:
        overlap = train_scenes.intersection(val["scenes"])
        if overlap:
            raise DatasetIntegrityError(f"Train/validation scene leakage: {sorted(overlap)}")
        if val["split"] == "val_unseen":
            overlap = train_categories.intersection(val["categories"])
            if overlap:
                raise DatasetIntegrityError(f"OVON unseen-category leakage: {sorted(overlap)}")


def verify_manifest(path):
    manifest = read_json(path)
    for entry in manifest["files"]:
        if file_hash(entry["path"]) != entry["sha256"]:
            raise DatasetIntegrityError(f"Episode file hash changed: {entry['path']}")
    for scene in manifest["scene_paths"].values():
        if not Path(scene).is_file():
            raise DatasetIntegrityError(f"Missing scene: {scene}")
    return manifest
