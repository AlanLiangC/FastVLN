"""Offline targets with explicit point/region provenance and grouped splits."""

import hashlib

import numpy as np
from scipy.ndimage import distance_transform_edt

from streamnav.contracts.perception import GRID_HEIGHT, GRID_WIDTH, decode_pixel


def goal_instruction(label, *, referring=False):
    label = " ".join(label.strip().rstrip(".").split())
    if not label:
        raise ValueError("Empty object description")
    if label.lower().startswith(("a ", "an ", "the ", "this ", "that ")):
        return f"Find {label}."
    return f"Find {'the' if referring else 'a'} {label}."


def group_split(group, val_fraction=0.05, seed=2025):
    value = int(hashlib.sha256(f"{seed}:{group}".encode()).hexdigest()[:16], 16)
    return "val" if value / 2**64 < val_fraction else "train"


def mask_point(mask, min_pixels=64):
    """Select a quantized point inside the actual visible instance, not its box.

    Work on the small prediction grid, then verify the selected grid center and
    nearby original pixels. This avoids full-resolution distance transforms.
    """
    mask = np.asarray(mask, dtype=bool)
    if mask.sum() < min_pixels:
        return None
    h, w = mask.shape
    xs = np.minimum(((np.arange(GRID_WIDTH) + 0.5) / GRID_WIDTH * w).astype(int), w - 1)
    ys = np.minimum(((np.arange(GRID_HEIGHT) + 0.5) / GRID_HEIGHT * h).astype(int), h - 1)
    inside = mask[ys[:, None], xs[None, :]].copy()
    for dx, dy in ((-2, 0), (2, 0), (0, -2), (0, 2)):
        inside &= mask[np.clip(ys[:, None] + dy, 0, h - 1), np.clip(xs[None, :] + dx, 0, w - 1)]
    if not inside.any():
        return None
    scores = distance_transform_edt(np.pad(inside, 1))[1:-1, 1:-1]
    y, x = np.unravel_index(scores.argmax(), scores.shape)
    token = 1 + int(y) * GRID_WIDTH + int(x)
    u, v = decode_pixel(token, w, h)
    assert mask[int(v), int(u)]
    return token


def box_tokens(boxes, width, height):
    """Weak localization: the union of box interiors, never fake point centers."""
    grid = np.zeros((GRID_HEIGHT, GRID_WIDTH), dtype=bool)
    xs = (np.arange(GRID_WIDTH) + 0.5) / GRID_WIDTH * width
    ys = (np.arange(GRID_HEIGHT) + 0.5) / GRID_HEIGHT * height
    for x1, y1, x2, y2 in boxes:
        if x2 <= x1 or y2 <= y1:
            continue
        grid |= (
            (ys[:, None] >= max(0, y1))
            & (ys[:, None] < min(height, y2))
            & (xs[None, :] >= max(0, x1))
            & (xs[None, :] < min(width, x2))
        )
    return (np.flatnonzero(grid) + 1).tolist()
