from __future__ import annotations

import math
import random

from streamnav.data.schema import HabitatEpisodeSource


class MixedEpisodeSource:
    def __init__(self, sources, weights, seed=0):
        if len(sources) != len(weights) or not sources:
            raise ValueError("Sources/weights must be nonempty with matching lengths")
        if any(not math.isfinite(w) or w < 0 for w in weights) or sum(weights) <= 0:
            raise ValueError("Mixture weights must be finite nonnegative and sum to > 0")
        self.sources, self.weights, self.rng = sources, weights, random.Random(seed)

    def sample_episode(self):
        return self.rng.choices(self.sources, self.weights)[0].sample_episode()

    def state_dict(self):
        return {"rng": self.rng.getstate(), "sources": [s.state_dict() for s in self.sources]}

    def load_state_dict(self, state):
        self.rng.setstate(state["rng"])
        for source, saved in zip(self.sources, state["sources"], strict=True):
            source.load_state_dict(saved)


def make_source(config, seed):
    sources = [
        HabitatEpisodeSource(item["manifest"], seed + i, config.get("scene_repeat", 16))
        for i, item in enumerate(config["sources"])
    ]
    return MixedEpisodeSource(sources, [item["weight"] for item in config["sources"]], seed)
