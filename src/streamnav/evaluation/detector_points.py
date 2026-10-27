"""Compress a detector mask into one supported input point, without GT labels."""

import numpy as np
from scipy.ndimage import distance_transform_edt, label

from streamnav.contracts.perception import GRID_HEIGHT, GRID_WIDTH


def detector_point(mask):
    mask = np.asarray(mask, dtype=bool)
    if mask.ndim != 2 or not all(mask.shape):
        raise ValueError("Detector mask must be a nonempty HxW array")
    height, width = mask.shape
    result = {"token": 0, "source": "yoloe_unavailable", "mask_pixels": int(mask.sum())}
    if not mask.any():
        return result
    components, _ = label(mask)
    sizes = np.bincount(components.ravel())
    sizes[0] = 0
    primary = components == sizes.argmax()
    ys = np.floor((np.arange(GRID_HEIGHT) + 0.5) * height / GRID_HEIGHT).astype(int)
    xs = np.floor((np.arange(GRID_WIDTH) + 0.5) * width / GRID_WIDTH).astype(int)
    supported = primary[ys[:, None], xs[None, :]]
    if not supported.any():
        primary = mask
        supported = mask[ys[:, None], xs[None, :]]
    if not supported.any():
        result["source"] = "yoloe_no_grid_supported_point"
        return result
    interior = distance_transform_edt(np.pad(primary, 1))[1:-1, 1:-1]
    distances = interior[ys[:, None], xs[None, :]]
    distances[~supported] = -1
    y, x = np.unravel_index(distances.argmax(), distances.shape)
    assert mask[ys[y], xs[x]]
    result.update(token=1 + int(y) * GRID_WIDTH + int(x), source="yoloe_point")
    return result
