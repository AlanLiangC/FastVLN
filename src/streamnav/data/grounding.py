"""Explicitly masked image point supervision, split by image identity."""

import json
import math
import random
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from streamnav.contracts.perception import APOS_SIZE, OPOS_SIZE
from streamnav.data.grounding_preparation import box_tokens
from streamnav.data.grounding_support import unpack_grid


def validate_record(row):
    if not row.get("instruction") or not row.get("image") or not row.get("image_id"):
        raise ValueError("Grounding samples need image identity, path and instruction")
    for name, size in (("apos", APOS_SIZE), ("opos", OPOS_SIZE)):
        targets = row.get(name + "_targets", [])
        if any(type(t) is not int or not 0 <= t < size for t in targets):
            raise ValueError(f"Invalid {name} point tokens")
    boxes = row.get("opos_boxes", [])
    if row.get("opos_support"):
        grid = unpack_grid(row["opos_support"]).reshape(-1)
        targets = row.get("opos_targets", [])
        if not targets or not grid.any() or any(t == 0 or not grid[t - 1] for t in targets):
            raise ValueError("Canonical points must lie inside their visible instance support")
        if row.get("supervision") != "point" or boxes:
            raise ValueError("True instance support cannot be a weak box or absence label")
    if boxes and (
        row.get("supervision") != "region"
        or any(
            len(b) != 4
            or not all(math.isfinite(v) and 0 <= v <= 1 for v in b)
            or b[2] <= b[0]
            or b[3] <= b[1]
            for b in boxes
        )
        or not box_tokens(boxes, 1, 1)
    ):
        raise ValueError("Invalid normalized OPOS regions")
    if not math.isfinite(row.get("loss_weight", 1.0)) or row.get("loss_weight", 1.0) <= 0:
        raise ValueError("Invalid grounding loss weight")
    if not row.get("apos_targets") and not row.get("opos_targets") and not boxes:
        raise ValueError("A grounding sample must supervise at least one point")
    return row


class GroundingSource:
    def __init__(self, manifest, seed=0, source_weights=None, absence_probability=None):
        self.path = Path(manifest).resolve()
        self.rows = [validate_record(json.loads(s)) for s in self.path.open() if s.strip()]
        if not self.rows:
            raise ValueError(f"Empty grounding manifest: {manifest}")
        for row in self.rows:
            path = Path(row["image"])
            row["image"] = str(path if path.is_absolute() else self.path.parent / path)
        self.rng = random.Random(seed)
        self.order = list(range(len(self.rows)))
        self.rng.shuffle(self.order)
        self.cursor = 0
        self.source_weights = source_weights
        self.absence_probability = absence_probability
        self.groups = {}
        self.group_cursors = {}
        if source_weights:
            if any(not math.isfinite(w) or w <= 0 for w in source_weights.values()):
                raise ValueError("Grounding source weights must be positive and finite")
            for name in source_weights:
                indices = [
                    i
                    for i, row in enumerate(self.rows)
                    if row.get("source") == name
                    and (absence_probability is None or row.get("opos_targets") != [0])
                ]
                if not indices:
                    raise ValueError(f"No grounding samples from {name}")
                self.rng.shuffle(indices)
                self.groups[name] = indices
                self.group_cursors[name] = 0
        if absence_probability is not None:
            if not source_weights or not 0 <= absence_probability < 1:
                raise ValueError("Balanced absence requires weighted positive sources")
            negatives = [i for i, r in enumerate(self.rows) if r.get("opos_targets") == [0]]
            if absence_probability and not negatives:
                raise ValueError("No explicitly annotated absence samples")
            self.rng.shuffle(negatives)
            self.groups["__absence__"] = negatives
            self.group_cursors["__absence__"] = 0

    @staticmethod
    def materialize(row):
        if row.get("opos_boxes"):
            return {**row, "opos_targets": box_tokens(row["opos_boxes"], 1, 1)}
        return row

    def sample(self, count):
        result = []
        for _ in range(count):
            if self.source_weights:
                name = (
                    "__absence__"
                    if self.absence_probability is not None
                    and self.rng.random() < self.absence_probability
                    else self.rng.choices(
                        list(self.source_weights), weights=list(self.source_weights.values())
                    )[0]
                )
                cursor = self.group_cursors[name]
                if cursor == len(self.groups[name]):
                    self.rng.shuffle(self.groups[name])
                    cursor = 0
                result.append(self.materialize(self.rows[self.groups[name][cursor]]))
                self.group_cursors[name] = cursor + 1
                self.cursor += 1
                continue
            if self.cursor == len(self.order):
                self.rng.shuffle(self.order)
                self.cursor = 0
            result.append(self.materialize(self.rows[self.order[self.cursor]]))
            self.cursor += 1
        return result

    def state_dict(self):
        return {
            "rng": self.rng.getstate(),
            "order": list(self.order),
            "cursor": self.cursor,
            "groups": {name: list(indices) for name, indices in self.groups.items()},
            "group_cursors": dict(self.group_cursors),
        }

    def load_state_dict(self, state):
        if sorted(state["order"]) != list(range(len(self.rows))):
            raise ValueError("Grounding manifest changed since checkpoint")
        self.rng.setstate(state["rng"])
        self.order, self.cursor = state["order"], state["cursor"]
        if set(state.get("groups", {})) != set(self.groups):
            raise ValueError("Grounding sampling recipe changed")
        self.groups, self.group_cursors = state.get("groups", {}), state.get("group_cursors", {})

    @staticmethod
    def features(policy, row):
        with Image.open(row["image"]) as image:
            rgb = torch.from_numpy(np.array(image.convert("RGB")))
        return policy.frame_features(rgb, row["instruction"])[0]

    @staticmethod
    def features_many(policy, rows):
        if not hasattr(policy, "frame_features_batch"):
            return [GroundingSource.features(policy, row) for row in rows]
        if not hasattr(policy, "_grounding_image_pool"):
            policy._grounding_image_pool = ThreadPoolExecutor(max_workers=8)

        def read(row):
            with Image.open(row["image"]) as image:
                rgb = image.convert("RGB")
                h, w = policy.settings["image_size"]
                rgb.thumbnail((w, h), Image.Resampling.BILINEAR)
                return rgb

        images = list(policy._grounding_image_pool.map(read, rows))
        return policy.frame_features_batch(images, [row["instruction"] for row in rows])


def check_grounding_split(train, val):
    overlap = {r["image_id"] for r in train.rows} & {r["image_id"] for r in val.rows}
    if overlap:
        raise ValueError("Grounding train/val share images")
    groups = {r["split_group"] for r in train.rows if r.get("split_group")} & {
        r["split_group"] for r in val.rows if r.get("split_group")
    }
    if groups:
        raise ValueError("Grounding train/val share physical spaces")
