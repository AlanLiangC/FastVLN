"""Enrich existing immutable splits with true visible-instance grid support."""

import argparse
import hashlib
import io
import json
import shutil
import zipfile
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
from PIL import Image
from pycocotools import mask as coco_mask

from streamnav.data.grounding import validate_record
from streamnav.data.grounding_support import mask_grid, pack_grid


def coco_support(item):
    image_id, height, width, annotations, categories, references = item
    unions, instances = {}, {}
    for ann in annotations:
        if ann["iscrowd"] or ann["area"] < max(64, 0.001 * width * height):
            continue
        if ann["category_id"] not in categories and ann["id"] not in references:
            continue
        segmentation = ann["segmentation"]
        rle = (
            coco_mask.frPyObjects(segmentation, height, width)
            if isinstance(segmentation, list)
            else segmentation
        )
        mask = coco_mask.decode(rle)
        grid = mask_grid(mask.any(-1) if mask.ndim == 3 else mask)
        if not grid.any():
            continue
        if ann["category_id"] in categories:
            unions.setdefault(ann["category_id"], np.zeros_like(grid))[:] |= grid
        if ann["id"] in references:
            instances[ann["id"]] = pack_grid(grid)
    return image_id, {k: pack_grid(v) for k, v in unions.items()}, instances


def scan_support(item):
    public, scan, requests = item
    path = Path(public) / "ScanNet/scans" / scan
    labels = {
        g["objectId"] + 1: g["label"].strip().lower()
        for g in json.loads((path / f"{scan}.aggregation.json").read_text())["segGroups"]
    }
    result = {}
    with zipfile.ZipFile(path / f"{scan}_2d-instance-filt.zip") as archive:
        for frame, names in requests.items():
            mask = np.asarray(Image.open(io.BytesIO(archive.read(f"instance-filt/{frame}.png"))))
            h, w = mask.shape
            unions = {}
            for instance, area in zip(*np.unique(mask, return_counts=True), strict=True):
                name = labels.get(int(instance))
                if name not in names or area < max(256, 0.002 * w * h):
                    continue
                visible = mask == instance
                if (
                    min(np.count_nonzero(visible.any(0)) / w, np.count_nonzero(visible.any(1)) / h)
                    < 0.025
                ):
                    continue
                grid = mask_grid(visible)
                if grid.any():
                    unions.setdefault(name, np.zeros_like(grid))[:] |= grid
            for name, grid in unions.items():
                result[(scan, frame, name)] = pack_grid(grid)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", default="runtime/data/grounding/public_v1")
    parser.add_argument("--output", default="runtime/data/grounding/public_v2")
    parser.add_argument("--workers", type=int, default=12)
    args = parser.parse_args()
    source, dest = Path(args.input).resolve(), Path(args.output).resolve()
    if source == dest or (dest / "train.jsonl").exists():
        raise ValueError("Publish into a new directory; never overwrite a live manifest")
    dest.mkdir(parents=True, exist_ok=True)
    prep = json.loads((source / "preparation.json").read_text())
    public = prep["public_root"]
    wants_cats, wants_refs, wants_scans = defaultdict(set), defaultdict(set), defaultdict(dict)
    for split in ("train", "val"):
        for line in (source / f"{split}.jsonl").open():
            row = json.loads(line)
            if row["source"] == "coco-instance":
                wants_cats[int(row["image_id"].split(":")[-1])].add(
                    row["instruction"].removeprefix("Find a ").removesuffix(".")
                )
            elif row["source"] == "referring":
                wants_refs[int(row["image_id"].split(":")[-1])].add(row["annotation_id"])
            elif row["source"] == "scannet-instance":
                wants_scans[row["scan"]].setdefault(row["frame"], set()).add(
                    row["instruction"].removeprefix("Find a ").removesuffix(".")
                )
    category_support, instance_support = {}, {}
    annotations = Path("runtime/data/grounding/coco2014")
    for split in ("train", "val"):
        data = json.loads((annotations / f"instances_{split}2014.json").read_text())
        categories = {c["name"]: c["id"] for c in data["categories"]}
        grouped = defaultdict(list)
        for ann in data["annotations"]:
            if ann["image_id"] in wants_cats or ann["image_id"] in wants_refs:
                grouped[ann["image_id"]].append(ann)
        items = (
            (
                im["id"],
                im["height"],
                im["width"],
                grouped[im["id"]],
                {categories[n] for n in wants_cats[im["id"]]},
                wants_refs[im["id"]],
            )
            for im in data["images"]
            if im["id"] in grouped
        )
        with ProcessPoolExecutor(args.workers) as pool:
            for i, (image_id, cats, refs) in enumerate(pool.map(coco_support, items, chunksize=32)):
                category_support.update({(image_id, cat): support for cat, support in cats.items()})
                instance_support.update(refs)
                if (i + 1) % 10000 == 0:
                    print(json.dumps({"coco_split": split, "images": i + 1}), flush=True)
        del data, grouped
    print(
        json.dumps(
            {"coco_categories": len(category_support), "ref_instances": len(instance_support)}
        ),
        flush=True,
    )
    scan_cells = {}
    with ProcessPoolExecutor(args.workers) as pool:
        for i, values in enumerate(
            pool.map(scan_support, [(public, s, r) for s, r in wants_scans.items()])
        ):
            scan_cells.update(values)
            if (i + 1) % 50 == 0:
                print(json.dumps({"scan_captures": i + 1, "total": len(wants_scans)}), flush=True)
    report = {
        **prep,
        "format": "pointcot-public-grounding-v2",
        "parent": str(source),
        "support": "48x27 original-mask membership; canonical points and splits unchanged",
        "supported_queries": {},
    }
    for split in ("train", "val"):
        count = 0
        temporary = dest / f"{split}.jsonl.writing"
        with temporary.open("w") as f:
            for line in (source / f"{split}.jsonl").open():
                row = json.loads(line)
                support = None
                if row["source"] == "coco-instance":
                    name = row["instruction"].removeprefix("Find a ").removesuffix(".")
                    support = category_support[
                        (int(row["image_id"].split(":")[-1]), categories[name])
                    ]
                elif row["source"] == "referring":
                    support = instance_support[row["annotation_id"]]
                elif row["source"] == "scannet-instance":
                    name = row["instruction"].removeprefix("Find a ").removesuffix(".")
                    support = scan_cells[(row["scan"], row["frame"], name)]
                if support:
                    row["opos_support"] = support
                    validate_record(row)
                    count += 1
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
        temporary.replace(dest / f"{split}.jsonl")
        report["supported_queries"][split] = count
        report[split + "_sha256"] = hashlib.sha256(
            (dest / f"{split}.jsonl").read_bytes()
        ).hexdigest()
    for name in ("coco_split.json",):
        if (source / name).exists():
            shutil.copyfile(source / name, dest / name)
    (dest / "preparation.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(report, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
