"""Sample directly on the robot navmesh, checking actual goal geodesics."""

import habitat_sim
import numpy as np
from habitat_sim.utils.common import quat_from_angle_axis


def apply_mixed_start(env, specification):
    if env.episode["split"] != "train":
        raise ValueError("Evaluation must retain its official start pose")
    lower, upper = specification["distance_band"]
    if not 0 <= lower < upper <= 20:
        raise ValueError("Invalid training start band")
    rng = np.random.RandomState(specification["seed"])
    pathfinder = env.sim.pathfinder
    agent = env.sim.get_agent(0)
    original = agent.get_state()
    viewpoints = [(goal, view) for goal in env.episode["goals"] for view in goal["view_points"]]
    for _ in range(48):
        goal, view = viewpoints[rng.randint(len(viewpoints))]
        endpoint = np.asarray(view["agent_state"]["position"], dtype=np.float32)
        wanted = rng.uniform(lower, upper)
        angle = rng.uniform(-np.pi, np.pi)
        candidate = pathfinder.snap_point(
            endpoint + wanted * np.array([np.sin(angle), 0, np.cos(angle)])
        )
        if not np.isfinite(candidate).all() or abs(candidate[1] - endpoint[1]) > 0.3:
            continue
        path = habitat_sim.MultiGoalShortestPath()
        path.requested_start, path.requested_ends = candidate, env.goal_positions
        if not pathfinder.find_path(path) or not lower <= path.geodesic_distance < upper:
            continue
        state = habitat_sim.AgentState()
        state.position = candidate
        ready = specification["kind"] == "ready"
        facing = ready or rng.rand() < specification["goal_facing_probability"]
        if facing:
            direction = np.asarray(goal["position"]) - candidate
            # Camera forward is -Z. atan2(-dx,-dz) is yaw in Habitat coordinates.
            yaw = np.arctan2(-direction[0], -direction[2]) + rng.uniform(-np.pi / 6, np.pi / 6)
            state.rotation = quat_from_angle_axis(float(yaw), np.array([0.0, 1.0, 0.0]))
        else:
            state.rotation = quat_from_angle_axis(
                float(rng.uniform(-np.pi, np.pi)), np.array([0.0, 1.0, 0.0])
            )
        agent.set_state(state)
        env._cached_observations = None
        if ready or (specification["kind"] == "boundary" and facing):
            # A fresh cache cannot infer that an invisible object is behind it.
            # Align the camera in 3D, including high objects at a low mounting
            # height, then reject occluded goal anchors. These are sampling
            # checks only; weak centroid projections never become OPOS labels.
            from perception_labels import depth_agrees, project_camera

            aligned = agent.get_state()
            camera = aligned.sensor_states["rgb"]
            delta = np.asarray(goal["position"]) - camera.position
            pitch = np.arctan2(delta[1], np.linalg.norm(delta[[0, 2]]))
            for sensor in aligned.sensor_states.values():
                sensor.rotation = state.rotation * quat_from_angle_axis(
                    float(pitch), np.array([1.0, 0.0, 0.0])
                )
            agent.set_state(aligned, reset_sensors=False, infer_sensor_states=False)
            env._cached_observations = None
            observations = env._sensor_observations()
            if "perception_depth" not in observations:
                raise ValueError("STOP-ready sampling requires training visibility checks")
            camera = agent.get_state().sensor_states["rgb"]
            depth = observations["perception_depth"]
            h, w = depth.shape
            pixel, z = project_camera(
                np.asarray(goal["position"]), camera, w, h, env.config.get("hfov", 120)
            )
            if env.config.get("object_label_source") == "semantic_instance":
                from perception_labels import PerceptionLabeler

                label = PerceptionLabeler(env).object_labels()
                if not label.get("opos_valid") or not label.get("opos"):
                    continue
            elif pixel is None or not depth_agrees(depth, pixel, z, 0.3 + 0.04 * z):
                continue
        return {
            "kind": specification["kind"],
            "applied": True,
            "distance": float(path.geodesic_distance),
            "goal_facing": bool(facing),
        }
    # Failed sampling retains the exact official start, never an unchecked pose.
    agent.set_state(original)
    env._cached_observations = None
    return {"kind": specification["kind"], "applied": False, "reason": "no_valid_pose_in_band"}
