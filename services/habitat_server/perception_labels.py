"""Training annotations for geometric affordances and visible object instances.

Instance mode uses aligned semantic rendering; the legacy depth projection remains
available for old recipes. Only compact labels leave this service, never masks.
Label queries do not step the simulator or invoke a learned detector.
"""

from __future__ import annotations

import habitat_sim
import numpy as np
from habitat_sim.utils.common import quat_rotate_vector

from streamnav.contracts.action import NavigationAction as A
from streamnav.contracts.object_labels import instance_object_label
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

    def object_labels(self, goal_ids=None):
        env = self.env
        if env.config.get("object_label_source") != "semantic_instance":
            return {"opos": 0, "opos_valid": False, "opos_confidence": 0.0}
        known = env.semantic_known_ids
        if goal_ids is None:
            try:
                goal_ids = {
                    int(str(g["object_id"]).rsplit("_", 1)[-1]) for g in env.episode["goals"]
                }
            except (KeyError, ValueError):
                goal_ids = set()
        label = instance_object_label(
            env._sensor_observations()["perception_semantic"],
            goal_ids,
            known,
            min_pixels=env.config.get("opos_min_visible_pixels", 9),
        )
        label.update(episode_id=env.uid, frame_id=env.steps)
        return label

    def labels(self, action, *, navigation_target=None, goal_ids=None):
        env = self.env
        state = env.sim.get_agent(0).get_state()
        camera = state.sensor_states["rgb"]
        depth = env._sensor_observations()["perception_depth"]
        height, width = depth.shape
        hfov = env.config.get("hfov", 120)
        in_success = env.previous_distance < env.config.get("success_distance", 0.1)
        success_aligned = env.config.get("oracle_success_stop", False)
        near = (
            env.previous_distance < env.config.get("arrival_near_distance", 0.75)
            if success_aligned
            else in_success
        )
        ready = in_success and (success_aligned or action == A.STOP)
        result = {
            "apos": 0,
            "apos_valid": False,
            "apos_confidence": 0.0,
            "opos": 0,
            "opos_valid": False,
            "opos_confidence": 0.0,
            "arrival": ARRIVAL_READY if ready else (ARRIVAL_NEAR if near else ARRIVAL_FAR),
            "arrival_valid": True,
            "arrival_confidence": 1.0,
            "object_label_source": "geometry_depth_weak",
            "episode_id": env.uid,
            "frame_id": env.steps,
        }
        if action == A.STOP:
            # Local-frontier STOP is never arrival supervision.
            if in_success:
                result.update(apos=APOS_STOP, apos_valid=True, apos_confidence=1.0)
        else:
            target = (
                navigation_target
                if navigation_target is not None
                else (
                    env.explorer.current_navigation_target
                    if env.explorer is not None
                    else env.oracle_goal
                )
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
        if not env.config.get("perception_object_labels", True):
            return result
        if env.config.get("object_label_source") == "semantic_instance":
            result.update(self.object_labels(goal_ids))
            return result
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
