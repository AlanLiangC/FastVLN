"""One optional current-frame point; NULL is not an absence label."""

import math

from streamnav.contracts.perception import POINT_CELLS, encode_pixel


def object_hint_token(hint, width, height, *, frame_id=None, episode_id=None):
    if hint is None:
        return 0
    if type(hint) is int:
        if not 0 <= hint <= POINT_CELLS:
            raise ValueError("Object hint token must be NULL or a grid point")
        return hint
    if not isinstance(hint, dict):
        raise ValueError("Object hint must be a point packet or null")
    if frame_id is not None and hint.get("frame_id") != frame_id:
        return 0
    if episode_id is not None and hint.get("episode_id", episode_id) != episode_id:
        return 0
    if "token" in hint:
        if hint.get("xy_px") is not None:
            raise ValueError("Provide either a point token or pixel coordinates")
        return object_hint_token(hint["token"], width, height)
    xy = hint.get("xy_px")
    if xy is None:
        return 0
    if (
        not isinstance(xy, (list, tuple))
        or len(xy) != 2
        or not all(type(v) in (int, float) and math.isfinite(v) for v in xy)
    ):
        return 0
    x, y = xy
    return encode_pixel(x, y, width, height) if 0 <= x < width and 0 <= y < height else 0
