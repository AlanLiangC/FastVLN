import argparse
import json

from PIL import Image

from streamnav.data.schema import HabitatEpisodeSource
from streamnav.envs.habitat_client import HabitatClient

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", default="runtime/data/manifests/hm3d_v1_val.json")
    parser.add_argument("--steps", type=int, default=500)
    args = parser.parse_args()
    source = HabitatEpisodeSource(args.manifest, seed=17)
    episode = next(source.evaluation_episodes(limit=1))
    env = HabitatClient(
        {
            "gpu_device_id": 1,
            "width": 480,
            "height": 270,
            "hfov": 120,
            "sensor_height": 0.88,
            "sensor_pitch_deg": 0.0,
            "agent_height": 0.88,
            "agent_radius": 0.18,
            "timeout_s": 120,
        }
    )
    try:
        obs = env.reset(episode)
        Image.fromarray(obs["rgb"].numpy()).save("runtime/logs/habitat_reset.png")
        print("reset", episode.uid, episode.goal_text, obs["distance"], flush=True)
        print(json.dumps({"robot": obs["robot"]}), flush=True)
        for i in range(args.steps):
            action = env.get_oracle_action()
            obs = env.step(action)
            print(i, action.name, obs["geodesic_distance"], flush=True)
            if obs["done"]:
                print(json.dumps(obs["metrics"]), flush=True)
                break
    finally:
        env.close()
