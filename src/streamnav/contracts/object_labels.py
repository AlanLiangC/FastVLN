"""Instance-rendered visible points; incomplete instance mapping stays unknown."""

import numpy as np

from streamnav.contracts.perception import GRID_HEIGHT, GRID_WIDTH


def instance_object_label(semantic, goal_ids, known_ids, *, min_pixels=9):
    result = {
        "opos": 0,
        "opos_valid": False,
        "opos_confidence": 0.0,
        "object_label_source": "semantic_instance",
        "opos_targets": [],
    }
    if not goal_ids:
        return result
    known = set(known_ids)
    matched = set(goal_ids) & known
    mask = np.isin(semantic, list(matched))
    y, x = np.nonzero(mask)
    result["opos_visible_pixels"] = len(x)
    if len(x) == 0:
        if set(goal_ids) <= known:
            result.update(opos_valid=True, opos_confidence=1.0, opos_targets=[0])
        return result
    if len(x) < min_pixels:
        return result
    height, width = semantic.shape
    xs = np.minimum(((np.arange(GRID_WIDTH) + 0.5) / GRID_WIDTH * width).astype(int), width - 1)
    ys = np.minimum(((np.arange(GRID_HEIGHT) + 0.5) / GRID_HEIGHT * height).astype(int), height - 1)
    cells = np.flatnonzero(mask[ys[:, None], xs[None, :]]) + 1
    if not len(cells):
        return result
    # Pick an interior grid cell nearest the visible support's centroid.
    cx, cy = (
        ((cells - 1) % GRID_WIDTH + 0.5) / GRID_WIDTH,
        ((cells - 1) // GRID_WIDTH + 0.5) / GRID_HEIGHT,
    )
    chosen = int(cells[np.argmin((cx - x.mean() / width) ** 2 + (cy - y.mean() / height) ** 2)])
    result.update(
        opos=chosen,
        opos_valid=True,
        opos_confidence=1.0,
        opos_targets=[chosen, *(int(c) for c in cells if c != chosen)],
    )
    return result
