"""Original OVSegDT simulator version and OVONSim navmesh construction."""

import habitat_sim
import numpy as np
from env_factory import HabitatObjectNavEnv


class OVSegDTEvalEnv(HabitatObjectNavEnv):
    def __init__(self, config):
        if habitat_sim.__version__ != "0.2.3":
            raise RuntimeError("OVSegDT evaluation requires Habitat-Sim 0.2.3")
        super().__init__(config)

    def _configure_navmesh(self, scene, agent):
        # Match ovon/task/simulator.py. In particular, use 0.2.3 defaults and
        # never reuse a FastVLN / Habitat-Sim 0.3.3 cached mesh.
        settings = habitat_sim.NavMeshSettings()
        settings.set_defaults()
        settings.agent_height = agent.height
        settings.agent_radius = agent.radius
        settings.agent_max_climb = self.config["navmesh_agent_max_climb"]
        settings.cell_height = self.config["navmesh_cell_height"]
        # In 0.2.3 this option is a recompute_navmesh argument, whereas the
        # vendor's newer OVONSim source assigns it to NavMeshSettings.
        if not self.sim.recompute_navmesh(
            self.sim.pathfinder, settings, include_static_objects=False
        ):
            raise RuntimeError("OVSegDT navmesh construction failed: " + scene)
        actual = self.sim.pathfinder.nav_mesh_settings
        for name in ("agent_height", "agent_radius", "agent_max_climb", "cell_height"):
            if not np.isclose(getattr(actual, name), getattr(settings, name)):
                raise RuntimeError("OVSegDT navmesh settings differ: " + name)

    def robot_configuration(self):
        return {**super().robot_configuration(), "habitat_sim_version": habitat_sim.__version__}

    def distance(self):
        path = habitat_sim.MultiGoalShortestPath()
        path.requested_start = self.sim.get_agent(0).get_state().position
        path.requested_ends = self.goal_positions
        if not self.sim.pathfinder.find_path(path):
            raise RuntimeError("OVSegDT goal viewpoints are unreachable")
        # 0.2.3 exposes the path endpoint, without the newer closest-end index.
        # The geodesic itself is the original OVON VIEW_POINTS task metric.
        self.closest_goal = np.asarray(path.points[-1], dtype=np.float32)
        return float(path.geodesic_distance)
