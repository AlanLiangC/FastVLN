"""Compact visible-instance support for point tokens; no segmentation model."""

import base64
import math
from functools import lru_cache

import numpy as np

from streamnav.contracts.perception import GRID_HEIGHT, GRID_WIDTH, POINT_CELLS


def mask_grid(mask):
    mask = np.asarray(mask, dtype=bool)
    h, w = mask.shape
    xs = np.minimum(((np.arange(GRID_WIDTH) + 0.5) / GRID_WIDTH * w).astype(int), w - 1)
    ys = np.minimum(((np.arange(GRID_HEIGHT) + 0.5) / GRID_HEIGHT * h).astype(int), h - 1)
    return mask[ys[:, None], xs[None, :]]


def pack_grid(grid):
    grid = np.asarray(grid, dtype=bool)
    if grid.shape != (GRID_HEIGHT, GRID_WIDTH) or not grid.any():
        raise ValueError("Instance support must contain visible grid cells")
    return base64.b64encode(np.packbits(grid.reshape(-1), bitorder="little")).decode("ascii")


def unpack_grid(encoded):
    payload = base64.b64decode(encoded, validate=True)
    if len(payload) != POINT_CELLS // 8:
        raise ValueError("Invalid instance-support bitmap size")
    return (
        np.unpackbits(np.frombuffer(payload, dtype=np.uint8), bitorder="little")
        .reshape(GRID_HEIGHT, GRID_WIDTH)
        .astype(bool)
    )


@lru_cache(maxsize=8192)
def point_neighborhood(targets, radius):
    if not math.isfinite(radius) or not 0 <= radius <= 0.05:
        raise ValueError("Point supervision radius must lie in [0, 0.05]")
    if targets == (0,) or not targets or radius == 0:
        return targets
    xs = (np.arange(GRID_WIDTH) + 0.5) / GRID_WIDTH
    ys = (np.arange(GRID_HEIGHT) + 0.5) / GRID_HEIGHT
    grid = np.zeros((GRID_HEIGHT, GRID_WIDTH), dtype=bool)
    for token in targets:
        if not 1 <= token <= POINT_CELLS:
            raise ValueError("Visible point neighborhoods cannot contain absence tokens")
        y, x = divmod(token - 1, GRID_WIDTH)
        grid |= (ys[:, None] - ys[y]) ** 2 + (xs[None, :] - xs[x]) ** 2 <= radius**2
    return tuple((np.flatnonzero(grid) + 1).tolist())


def support_targets(row, radius=0.0):
    if row.get("opos_support"):
        return (np.flatnonzero(unpack_grid(row["opos_support"])) + 1).tolist()
    targets = row.get("opos_targets", [])
    if row.get("supervision") == "region":
        return targets
    return list(point_neighborhood(tuple(sorted(set(targets))), radius))
