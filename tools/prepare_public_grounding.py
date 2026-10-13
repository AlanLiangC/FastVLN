"""Prepare navigation-style grounding from local RGB, true masks and weak boxes."""

import argparse
import csv
import hashlib
import io
import json
import struct
import zipfile
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from itertools import islice
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
from PIL import Image
from pycocotools import mask as coco_mask

from streamnav.data.grounding_preparation import (
    box_tokens,
    goal_instruction,
    group_split,
    mask_point,
)


def record(image, image_id, label, source, targets, **metadata):
    return {
        "image": str(image),
        "image_id": image_id,
        "instruction": goal_instruction(label),
        "source": source,
        "apos_targets": [],
        "opos_targets": targets,
        **metadata,
    }


def coco_image(item):
    image, annotations, public = item
    h, w = image["height"], image["width"]
    split = "train2014" if "train2014" in image["file_name"] else "val2014"
    path = Path(public) / "LaSco" / split / image["file_name"]
    if not path.exists():
        return image["id"], str(path), {}, {}
    targets, ann_points = defaultdict(list), {}
    for annotation in annotations:
        if annotation["iscrowd"] or annotation["area"] < max(64, 0.001 * w * h):
            continue
        segmentation = annotation["segmentation"]
        rle = (
            coco_mask.frPyObjects(segmentation, h, w)
            if isinstance(segmentation, list)
            else segmentation
        )
        mask = coco_mask.decode(rle)
        if mask.ndim == 3:
            mask = mask.any(-1)
        point = mask_point(mask)
        if point is not None:
            targets[annotation["category_id"]].append(point)
            ann_points[annotation["id"]] = point
    return image["id"], str(path), dict(targets), ann_points


def scannet_capture(item):
    path, output, stride, max_frames = item
    path, output = Path(path), Path(output)
    name = path.name
    groups = json.loads((path / f"{name}.aggregation.json").read_text())["segGroups"]
    labels = {g["objectId"] + 1: g["label"].strip().lower() for g in groups}
    skip = {"wall", "floor", "ceiling", "object", "objects", "structure", "furniture"}
    results = []
    with (
        (path / f"{name}.sens").open("rb") as f,
        zipfile.ZipFile(path / f"{name}_2d-instance-filt.zip") as z,
    ):
        assert struct.unpack("<I", f.read(4))[0] == 4
        name_length = struct.unpack("<Q", f.read(8))[0]
        f.seek(name_length + 256, 1)
        color_compression, _ = struct.unpack("<ii", f.read(8))
        assert color_compression == 2
        w, h, _, _ = struct.unpack("<IIII", f.read(16))
        f.read(4)
        frames = struct.unpack("<Q", f.read(8))[0]
        last_pose = None
        selected = 0
        available = set(z.namelist())
        for index in range(frames):
            pose = np.frombuffer(f.read(64), dtype="<f4").reshape(4, 4)
            f.read(16)
            color_size, depth_size = struct.unpack("<QQ", f.read(16))
            mask_name = f"instance-filt/{index}.png"
            candidate = (
                index % stride == 0
                and selected < max_frames
                and mask_name in available
                and np.isfinite(pose).all()
            )
            if candidate and last_pose is not None:
                moved = np.linalg.norm(pose[:3, 3] - last_pose[:3, 3])
                rotated = np.trace(pose[:3, :3].T @ last_pose[:3, :3]) < 2.93
                candidate = moved >= 0.25 or rotated
            if not candidate:
                f.seek(color_size + depth_size, 1)
                continue
            payload = f.read(color_size)
            f.seek(depth_size, 1)
            image = Image.open(io.BytesIO(payload)).convert("RGB")
            if image.size != (w, h):
                raise ValueError(f"RGB header differs in {name}/{index}")
            small = image.copy()
            small.thumbnail((192, 144))
            gray = np.asarray(small.convert("L"), dtype=np.float32)
            sharpness = np.var(
                gray[1:-1, :-2]
                + gray[1:-1, 2:]
                + gray[:-2, 1:-1]
                + gray[2:, 1:-1]
                - 4 * gray[1:-1, 1:-1]
            )
            if sharpness < 70 or gray.mean() < 20:
                continue
            mask = np.array(Image.open(io.BytesIO(z.read(mask_name))))
            if mask.shape != (h, w):
                raise ValueError(f"Mask/RGB mismatch in {name}/{index}")
            targets = defaultdict(list)
            ids, areas = np.unique(mask, return_counts=True)
            for instance, area in zip(ids, areas, strict=True):
                label = labels.get(int(instance))
                if not label or label in skip or area < max(256, 0.002 * w * h):
                    continue
                m = mask == instance
                # Truncated slivers and distant small objects are noisy targets.
                if min(np.count_nonzero(m.any(0)) / w, np.count_nonzero(m.any(1)) / h) < 0.025:
                    continue
                token = mask_point(m, min_pixels=256)
                if token is not None:
                    targets[label].append(token)
            if not targets:
                continue
            digest = hashlib.sha256(payload).hexdigest()
            dest = output / "images" / f"{name}_{index:06d}.jpg"
            dest.parent.mkdir(parents=True, exist_ok=True)
            image.thumbnail((448, 256), Image.Resampling.BILINEAR)
            image.save(dest, quality=93)
            for label, points in targets.items():
                results.append(
                    record(
                        dest.resolve(),
                        digest,
                        label,
                        "scannet-instance",
                        sorted(set(points)),
                        split_group=name.split("_")[0],
                        scan=name,
                        frame=index,
                        supervision="point",
                    )
                )
            selected += 1
            last_pose = pose.copy()
    return name, results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--public-root", default="/inspire/hdd/global_public/public_datas")
    parser.add_argument("--output", default="runtime/data/grounding/public_v1")
    parser.add_argument("--sources", default="coco,referring,scannet,imagenet,pixmo")
    parser.add_argument("--workers", type=int, default=12)
    parser.add_argument("--scannet-stride", type=int, default=80)
    parser.add_argument("--scannet-max-frames", type=int, default=64)
    parser.add_argument("--merge-only", action="store_true")
    args = parser.parse_args()
    public, output = Path(args.public_root), Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    sources = args.sources.split(",")
    counts = Counter()

    def write_rows(name, rows):
        handles = {s: (output / f"{name}.{s}.jsonl.writing").open("w") for s in ("train", "val")}
        try:
            for split, row in rows:
                handles[split].write(json.dumps(row, ensure_ascii=False) + "\n")
                counts[f"{name}.{split}"] += 1
        finally:
            for split, handle in handles.items():
                handle.close()
                (output / f"{name}.{split}.jsonl.writing").replace(output / f"{name}.{split}.jsonl")
        print(json.dumps(dict(counts)), flush=True)

    refs_path = Path("runtime/data/grounding/referring/annotations.jsonl")
    refs = (
        [json.loads(line) for line in refs_path.read_text().splitlines()]
        if refs_path.exists()
        and not args.merge_only
        and any(s in sources for s in ("coco", "referring"))
        else []
    )
    heldout = {r["image_id"] for r in refs if r["split"] != "train"}
    tests = {r["image_id"] for r in refs if r["split"].startswith("test")}

    if not args.merge_only and any(s in sources for s in ("coco", "referring")):
        annotations = Path("runtime/data/grounding/coco2014")
        metadata, grouped, categories = {}, defaultdict(list), {}
        for split in ("train", "val"):
            data = json.loads((annotations / f"instances_{split}2014.json").read_text())
            metadata.update({i["id"]: i for i in data["images"]})
            categories.update({c["id"]: c["name"] for c in data["categories"]})
            for ann in data["annotations"]:
                grouped[ann["image_id"]].append(ann)

        def split_coco(image_id):
            if image_id in tests:
                return None
            return (
                "val"
                if image_id in heldout or "val2014" in metadata[image_id]["file_name"]
                else group_split(f"coco:{image_id}")
            )

        cached_points = {}

        def coco_rows():
            items = (
                (im, grouped[im["id"]], str(public))
                for im in metadata.values()
                if split_coco(im["id"])
            )
            with ProcessPoolExecutor(args.workers) as pool:
                for i, (image_id, path, targets, ann_points) in enumerate(
                    pool.map(coco_image, items, chunksize=32)
                ):
                    cached_points[image_id] = (path, ann_points)
                    for category, points in targets.items():
                        yield (
                            split_coco(image_id),
                            record(
                                path,
                                f"coco:{image_id}",
                                categories[category],
                                "coco-instance",
                                sorted(set(points)),
                                supervision="point",
                            ),
                        )
                    if (i + 1) % 10000 == 0:
                        print(json.dumps({"coco_images_processed": i + 1}), flush=True)

        write_rows("coco", coco_rows())
        if "referring" in sources:
            if not refs:
                raise ValueError("Prepare referring annotations before the mixed COCO split")

            def referring_rows():
                seen = set()
                for ref in refs:
                    split = split_coco(ref["image_id"])
                    path, points = cached_points.get(ref["image_id"], ("", {}))
                    token = points.get(ref["ann_id"])
                    if not split or token is None:
                        continue
                    for label in ref["captions"]:
                        key = (ref["image_id"], ref["ann_id"], label)
                        if key in seen:
                            continue
                        seen.add(key)
                        row = record(
                            path,
                            f"coco:{ref['image_id']}",
                            label,
                            "referring",
                            [token],
                            supervision="point",
                            referring_dataset=ref["source"],
                            annotation_id=ref["ann_id"],
                        )
                        row["instruction"] = goal_instruction(label, referring=True)
                        yield split, row

            write_rows("referring", referring_rows())
        (output / "coco_split.json").write_text(
            json.dumps(
                {
                    "heldout_ref_images": len(heldout),
                    "excluded_test_images": len(tests),
                    "ref_sources": json.loads(refs_path.with_name("sources.json").read_text())
                    if refs
                    else {},
                }
            )
        )

    if not args.merge_only and "scannet" in sources:

        def scannet_rows():
            scans = sorted(
                p
                for p in (public / "ScanNet/scans").iterdir()
                if p.is_dir() and (p / f"{p.name}.sens").exists()
            )
            items = (
                (str(p), str(output), args.scannet_stride, args.scannet_max_frames) for p in scans
            )
            with ProcessPoolExecutor(args.workers) as pool:
                for i, (name, rows) in enumerate(pool.map(scannet_capture, items)):
                    for row in rows:
                        yield group_split(row["split_group"]), row
                    if (i + 1) % 25 == 0:
                        print(
                            json.dumps({"scans_processed": i + 1, "total": len(scans)}), flush=True
                        )

        write_rows("scannet", scannet_rows())

    if not args.merge_only and "imagenet" in sources:
        mapping = {}
        for line in (public / "LOC_synset_mapping.txt").read_text().splitlines():
            key, label = line.split(" ", 1)
            mapping[key] = label.split(",")[0]

        def imagenet_item(item):
            parts = item["PredictionString"].split()
            boxes = defaultdict(list)
            for start in range(0, len(parts), 5):
                boxes[parts[start]].append(list(map(int, parts[start + 1 : start + 5])))
            first = next(iter(boxes))
            path = public / "ILSVRC/Data/CLS-LOC/train" / first / (item["ImageId"] + ".JPEG")
            try:
                with Image.open(path) as im:
                    w, h = im.size
            except FileNotFoundError:
                return []
            result = []
            for synset, b in boxes.items():
                tokens = box_tokens(b, w, h)
                if not tokens:
                    continue
                normalized = [
                    [max(0, x1) / w, max(0, y1) / h, min(w, x2) / w, min(h, y2) / h]
                    for x1, y1, x2, y2 in b
                ]
                result.append(
                    (
                        group_split("imagenet:" + item["ImageId"]),
                        record(
                            path,
                            "imagenet:" + item["ImageId"],
                            mapping[synset],
                            "imagenet-loc",
                            [],
                            supervision="region",
                            opos_boxes=normalized,
                            loss_weight=0.25,
                            synset=synset,
                        ),
                    )
                )
            return result

        def imagenet_rows():
            with (
                (public / "LOC_train_solution.csv").open() as f,
                ThreadPoolExecutor(args.workers) as pool,
            ):
                reader = csv.DictReader(f)
                count = 0
                while items := list(islice(reader, 2048)):
                    for rows in pool.map(imagenet_item, items):
                        yield from rows
                    count += len(items)
                    print(json.dumps({"imagenet_images_processed": count}), flush=True)

        write_rows("imagenet", imagenet_rows())

    if not args.merge_only and "pixmo" in sources:
        old_images = Path("runtime/data/grounding/processed/images")
        present = {p.stem: p.resolve() for p in old_images.glob("*.img")}

        def pixmo_rows():
            seen = set()
            for path in sorted(Path("runtime/data/grounding/pixmo-points/data").glob("*.parquet")):
                for batch in pq.ParquetFile(path).iter_batches(batch_size=8192):
                    for r in batch.to_pylist():
                        key = r["image_sha256"]
                        if key not in present or len(r["points"]) > 8:
                            continue
                        points = r["points"]
                        if any(not 0 <= p[k] <= 100 for p in points for k in ("x", "y")):
                            continue
                        from streamnav.contracts.perception import encode_pixel

                        targets = sorted(
                            {
                                encode_pixel(min(p["x"], 99.999), min(p["y"], 99.999), 100, 100)
                                for p in points
                            }
                        ) or [0]
                        unique = (key, r["label"], tuple(targets))
                        if unique in seen:
                            continue
                        seen.add(unique)
                        yield (
                            group_split(key),
                            record(
                                present[key],
                                key,
                                r["label"],
                                "pixmo-points",
                                targets,
                                supervision="point",
                            ),
                        )

        write_rows("pixmo", pixmo_rows())

    # Source jobs may run separately; publish only fully written source shards.
    if args.merge_only or set(sources) == {"coco", "referring", "scannet", "imagenet", "pixmo"}:
        for split in ("train", "val"):
            with (output / f"{split}.jsonl.writing").open("wb") as dest:
                for name in ("coco", "referring", "scannet", "imagenet", "pixmo"):
                    with (output / f"{name}.{split}.jsonl").open("rb") as src:
                        import shutil

                        shutil.copyfileobj(src, dest)
            (output / f"{split}.jsonl.writing").replace(output / f"{split}.jsonl")
        report = {
            "format": "pointcot-public-grounding-v1",
            "public_root": str(public),
            "sources": {},
            "coordinate_grid": [48, 27],
            "prompt": "shared navigation system prompt + Find a/the <target>.",
            "scannet": {
                "stride": args.scannet_stride,
                "max_frames_per_capture": args.scannet_max_frames,
            },
            "no_actions_or_apos_fabricated": True,
        }
        for split in ("train", "val"):
            stats = defaultdict(lambda: {"queries": 0, "images": set(), "absence_queries": 0})
            for line in (output / f"{split}.jsonl").open():
                r = json.loads(line)
                s = stats[r["source"]]
                s["queries"] += 1
                s["images"].add(r["image_id"])
                s["absence_queries"] += r["opos_targets"] == [0]
            report["sources"][split] = {
                name: {**s, "images": len(s["images"])} for name, s in stats.items()
            }
            report[split + "_sha256"] = hashlib.sha256(
                (output / f"{split}.jsonl").read_bytes()
            ).hexdigest()
        (output / "preparation.json").write_text(json.dumps(report, indent=2) + "\n")
        print(json.dumps(report), flush=True)


if __name__ == "__main__":
    main()
