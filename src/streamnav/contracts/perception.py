"""Frame-relative pointing vocabulary, shared by labels, heads and visualizations."""

GRID_WIDTH = 48
GRID_HEIGHT = 27
POINT_CELLS = GRID_WIDTH * GRID_HEIGHT
APOS_LEFT = POINT_CELLS + 1
APOS_RIGHT = POINT_CELLS + 2
APOS_STOP = POINT_CELLS + 3
APOS_SIZE = POINT_CELLS + 4
OPOS_SIZE = POINT_CELLS + 1
ARRIVAL_FAR = 0
ARRIVAL_NEAR = 1
ARRIVAL_READY = 2


def encode_pixel(u, v, width, height):
    # Outside/behind-camera points must be masked, never clamped onto an edge.
    if not (0 <= u < width and 0 <= v < height):
        raise ValueError("Point is outside the current RGB frame")
    return 1 + int(v / height * GRID_HEIGHT) * GRID_WIDTH + int(u / width * GRID_WIDTH)


def decode_pixel(token, width, height):
    if not 1 <= token <= POINT_CELLS:
        return None
    index = token - 1
    return (
        (index % GRID_WIDTH + 0.5) / GRID_WIDTH * width,
        (index // GRID_WIDTH + 0.5) / GRID_HEIGHT * height,
    )


def perception_config(config=None):
    result = {"enabled": False, "embedding_dim": 64}
    if config:
        unknown = set(config) - set(result)
        if unknown:
            raise ValueError(f"Unknown perception model options: {sorted(unknown)}")
        result.update(config)
    if not isinstance(result["enabled"], bool) or not 1 <= result["embedding_dim"] <= 256:
        raise ValueError("Invalid perception model configuration")
    return result
