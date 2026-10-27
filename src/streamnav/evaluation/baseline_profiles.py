"""Use each method's observation and execution settings for the same task."""


def method_profiles(fast_config, saved_ovsegdt, official_eval, *, seed, max_steps):
    task = {
        "seed": seed,
        "max_episode_steps": max_steps,
        "success_distance": official_eval["habitat"]["task"]["measurements"]["success"][
            "success_distance"
        ],
    }
    fast = {**fast_config["habitat"], **task, "gpu_device_id": 0, "oracle": "greedy"}
    sim = official_eval["habitat"]["simulator"]
    agent = sim["agents"]["main_agent"]
    camera = agent["sim_sensors"]["rgb_sensor"]
    # Do not inherit anything from the competing method's simulator profile.
    original = {
        **task,
        "backend": "ovsegdt_original",
        "python": "runtime/ovsegdt-sim-env/bin/python",
        "gpu_device_id": 0,
        "width": camera["width"],
        "height": camera["height"],
        "hfov": camera["hfov"],
        "sensor_height": camera["position"][1],
        "sensor_pitch_deg": 0,
        "agent_height": agent["height"],
        "agent_radius": agent["radius"],
        "allow_sliding": sim["habitat_sim_v0"]["allow_sliding"],
        "forward_step": saved_ovsegdt["habitat"]["simulator"]["forward_step_size"],
        "turn_angle": sim["turn_angle"],
        "tilt_angle": saved_ovsegdt["habitat"]["simulator"]["tilt_angle"],
        "navmesh_agent_max_climb": sim["navmesh_settings"]["agent_max_climb"],
        "navmesh_cell_height": sim["navmesh_settings"]["cell_height"],
        "perception_labels": False,
        "object_label_source": "none",
        "oracle": "greedy",
        "timeout_s": 180,
    }
    return {"fastvln": fast, "ovsegdt": original}
