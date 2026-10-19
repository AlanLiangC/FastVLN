"""Training-only mixture of untouched, near-goal and STOP-ready starts."""

import math
from dataclasses import replace


def validate_curriculum(config):
    keys = ("original_probability", "near_probability", "ready_probability")
    ps = [config[k] for k in keys] + [config.get("boundary_probability", 0)]
    if any(not math.isfinite(p) or p < 0 for p in ps) or not math.isclose(sum(ps), 1):
        raise ValueError("Start probabilities must be finite, nonnegative and sum to one")
    if ps[0] <= 0:
        raise ValueError("Preserve original starts to learn search")
    bands = ["near_distance", "ready_distance"]
    if config.get("boundary_probability", 0):
        bands.append("boundary_distance")
    for key in bands:
        lo, hi = config[key]
        if not 0 <= lo < hi <= 20:
            raise ValueError("Invalid geodesic start-distance band")
    if not 0 <= config["goal_facing_probability"] <= 1:
        raise ValueError("Invalid start orientation probability")
    if not 0 <= config.get("boundary_goal_facing_probability", 1) <= 1:
        raise ValueError("Invalid boundary orientation probability")


class MixedStartEpisodeSource:
    def __init__(self, source, config):
        validate_curriculum(config)
        self.source, self.config = source, config

    def sample_episode(self):
        episode = self.source.sample_episode()
        if episode.split != "train":
            raise ValueError("Mixed starts are training-only")
        cfg, rng = self.config, self.source.rng
        value = rng.random()
        if value < cfg["original_probability"]:
            return episode
        if value < cfg["original_probability"] + cfg["near_probability"]:
            kind = "near"
        elif value < sum(
            cfg[k] for k in ("original_probability", "near_probability", "ready_probability")
        ):
            kind = "ready"
        else:
            kind = "boundary"
        return replace(
            episode,
            metadata={
                **episode.metadata,
                "training_start": {
                    "kind": kind,
                    "distance_band": cfg[kind + "_distance"],
                    "goal_facing_probability": cfg.get("boundary_goal_facing_probability", 1)
                    if kind == "boundary"
                    else cfg["goal_facing_probability"],
                    "seed": rng.getrandbits(31),
                },
            },
        )

    def state_dict(self):
        return self.source.state_dict()

    def load_state_dict(self, state):
        self.source.load_state_dict(state)

    def step_taken(self):
        self.source.step_taken()
