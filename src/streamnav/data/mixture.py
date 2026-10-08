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
        self.current = 0

    def sample_episode(self):
        self.current = self.rng.choices(range(len(self.sources)), self.weights)[0]
        return self.sources[self.current].sample_episode()

    def step_taken(self):
        self.sources[self.current].step_taken()

    def state_dict(self):
        return {
            "rng": self.rng.getstate(),
            "sources": [s.state_dict() for s in self.sources],
            "current": self.current,
        }

    def load_state_dict(self, state):
        self.rng.setstate(state["rng"])
        self.current = state.get("current", 0)
        for source, saved in zip(self.sources, state["sources"], strict=True):
            source.load_state_dict(saved)


def make_source(config, seed):
    sources = [
        HabitatEpisodeSource(
            item["manifest"],
            seed + i,
            config.get("scene_repeat", 16),
            config.get("iterator_options"),
        )
        for i, item in enumerate(config["sources"])
    ]
    return MixedEpisodeSource(sources, [item["weight"] for item in config["sources"]], seed)


def partition_scenes(sources, seed):
    """VER worker allocation: shuffled scenes, round-robin, minimum 16/worker."""
    rng = random.Random(seed)
    for dataset in range(len(sources[0].sources)):
        files = list(sources[0].sources[dataset].files)
        per_worker = max(math.ceil(len(files) / len(sources)), 16)
        assignments: list[list[dict]] = [[] for _ in sources]
        cursor = len(files)
        for index in range(per_worker * len(sources)):
            if cursor == len(files):
                rng.shuffle(files)
                cursor = 0
            assignments[index % len(sources)].append(files[cursor])
            cursor += 1
        for source, entries in zip(sources, assignments, strict=True):
            # Habitat's content_scenes filter uses scene membership, so repeated
            # scenes in a small dataset do not duplicate episodes.
            source.sources[dataset].files = list({e["path"]: e for e in entries}.values())
