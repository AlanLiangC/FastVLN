"""Training-only geometric pointing labels; uncertain object visibility is masked.

No semantic sensor, learned detector, simulator step or observation transfer to the
policy is used. Object points are weak 3D-anchor/depth labels, not instance masks.
"""

from __future__ import annotations

import habitat_sim
import numpy as np
from habitat_sim.utils.common import quat_rotate_vector

from streamnav.contracts.action import NavigationAction as A
from streamnav.contracts.perception import (
    APOS_LEFT,
    APOS_RIGHT,
    APOS_STOP,
    ARRIVAL_FAR,
    ARRIVAL_NEAR,
    ARRIVAL_READY,
    encode_pixel,
)


def project_camera(point, camera, width, height, hfov):
    local = quat_rotate_vector(camera.rotation.inverse(), np.asarray(point) - camera.position)
    z = -float(local[2])
    if z <= 0.05:
        return None, z
    focal = width / (2 * np.tan(np.deg2rad(hfov) / 2))
    return (width / 2 + focal * local[0] / z, height / 2 - focal * local[1] / z), z


def depth_agrees(depth, pixel, expected_z, tolerance, far=20.0):
    u, v = pixel
    height, width = depth.shape
    if not (1 <= u < width - 1 and 1 <= v < height - 1):
        return False
    x, y = int(u), int(v)
    patch = depth[y - 1 : y + 2, x - 1 : x + 2]
    valid = patch[np.isfinite(patch) & (patch > 0.05) & (patch < far * 0.99)]
    # Median avoids accepting an anchor through a single depth discontinuity pixel.
    return bool(valid.size >= 5 and abs(float(np.median(valid)) - expected_z) <= tolerance)


def path_point(points, distance):
    for first, second in zip(points[:-1], points[1:]):
        length = float(np.linalg.norm(second - first))
        if distance <= length and length > 1e-8:
            return first + (second - first) * distance / length
        distance -= length
    return np.array(points[-1], copy=True)


class PerceptionLabeler:
    def __init__(self, env):
        self.env = env

    def labels(self, action):
        env = self.env
        state = env.sim.get_agent(0).get_state()
        camera = state.sensor_states["rgb"]
        depth = env._sensor_observations()["perception_depth"]
        height, width = depth.shape
        hfov = env.config.get("hfov", 120)
        near = env.previous_distance < env.config.get("success_distance", 0.25)
        result = {
            "apos": 0,
            "apos_valid": False,
            "apos_confidence": 0.0,
            "opos": 0,
            "opos_valid": False,
            "opos_confidence": 0.0,
            "arrival": ARRIVAL_READY
            if near and action == A.STOP
            else (ARRIVAL_NEAR if near else ARRIVAL_FAR),
            "arrival_valid": True,
            "arrival_confidence": 1.0,
            "object_label_source": "geometry_depth_weak",
            "episode_id": env.uid,
            "frame_id": env.steps,
        }
        if action == A.STOP:
            # Local-frontier STOP is never arrival supervision.
            if near:
                result.update(apos=APOS_STOP, apos_valid=True, apos_confidence=1.0)
        else:
            target = (
                env.explorer.current_navigation_target
                if env.explorer is not None
                else env.oracle_goal
            )
            point = self.affordance(target, camera, depth, width, height, hfov)
            if point is not None:
                result.update(
                    apos=encode_pixel(*point, width, height), apos_valid=True, apos_confidence=1.0
                )
            elif action in (A.TURN_LEFT, A.TURN_RIGHT):
                result.update(
                    apos=APOS_LEFT if action == A.TURN_LEFT else APOS_RIGHT,
                    apos_valid=True,
                    apos_confidence=1.0,
                )
        candidates, definitely_out = [], []
        for goal in env.episode["goals"]:
            position = np.asarray(goal["position"])
            pixel, z = project_camera(position, camera, width, height, hfov)
            distance = float(np.linalg.norm(position - camera.position))
            # A centroid just outside the image can belong to a visible large object.
            outside = distance > 1.0 and (
                z < -0.5
                or (
                    pixel is not None
                    and (
                        pixel[0] < -0.2 * width
                        or pixel[0] > 1.2 * width
                        or pixel[1] < -0.2 * height
                        or pixel[1] > 1.2 * height
                    )
                )
            )
            definitely_out.append(outside)
            if pixel is None or not 0.1 < z < 10.0:
                continue
            tolerance = 0.30 + 0.04 * z
            if depth_agrees(depth, pixel, z, tolerance):
                candidates.append((distance, pixel))
        if candidates:
            # Stable geometric choice among equivalent goal instances, independent
            # of agent action, RGB teacher predictions and sampled learner actions.
            _, pixel = min(candidates, key=lambda item: item[0])
            result.update(
                opos=encode_pixel(*pixel, width, height), opos_valid=True, opos_confidence=0.5
            )
        elif definitely_out and all(definitely_out):
            result.update(opos=0, opos_valid=True, opos_confidence=0.25)
        # Occlusion, invalid depth and boundary ambiguity remain unlabelled.
        return result

    def affordance(self, target, camera, depth, width, height, hfov):
        if target is None:
            return None
        path = habitat_sim.ShortestPath()
        path.requested_start = self.env.sim.get_agent(0).get_state().position
        path.requested_end = target
        if not self.env.sim.pathfinder.find_path(path) or len(path.points) < 2:
            return None
        points = np.asarray(path.points)
        for distance in (2.0, 1.75, 1.5, 1.25, 1.0, 0.75, 0.5):
            point = path_point(points, distance)
            point[1] += 0.02
            pixel, z = project_camera(point, camera, width, height, hfov)
            if pixel is not None and depth_agrees(depth, pixel, z, 0.15 + 0.03 * z):
                return pixel
        return None
