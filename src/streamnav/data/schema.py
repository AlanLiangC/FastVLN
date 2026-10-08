from __future__ import annotations

import gzip
import json
import random
from collections import OrderedDict
from pathlib import Path
from typing import Protocol

from streamnav.contracts.episode import NavigationEpisode
from streamnav.errors import DatasetIntegrityError


class EpisodeSource(Protocol):
    def sample_episode(self) -> NavigationEpisode: ...


def read_json(path):
    with gzip.open(path, "rt") if str(path).endswith(".gz") else open(path) as f:
        return json.load(f)


class HabitatEpisodeSource:
    """Lazy per-scene loading; no eager multi-million-episode Python list."""

    def __init__(self, manifest, seed=0, scene_repeat=16):
        self.path = Path(manifest)
        self.manifest = read_json(self.path)
        self.files = self.manifest["files"]
        if not self.files:
            raise DatasetIntegrityError(f"Empty manifest {manifest}")
        self.rng = random.Random(seed)
        self.cache = OrderedDict()
        self.scene_repeat = scene_repeat
        self.remaining = 0
        self.current = None

    def _load(self, entry):
        path = entry["path"]
        if path not in self.cache:
            data = read_json(path)
            excluded = set(self.manifest.get("excluded_categories", []))
            if excluded:
                data["episodes"] = [
                    e
                    for e in data["episodes"]
                    if not excluded.intersection(
                        [
                            e["object_category"].strip().casefold(),
                            *(
                                c.strip().casefold()
                                for c in e.get("children_object_categories", [])
                            ),
                        ]
                    )
                ]
            self.cache[path] = data
        self.cache.move_to_end(path)
        while len(self.cache) > 2:
            self.cache.popitem(last=False)
        return self.cache[path]

    def decode(self, raw, data):
        scene = Path(raw["scene_id"]).name
        category = raw["object_category"]
        goals = list(
            raw.get("goals") or data.get("goals_by_category", {}).get(f"{scene}_{category}", [])
        )
        for child in raw.get("children_object_categories", []):
            goals.extend(data.get("goals_by_category", {}).get(f"{scene}_{child}", []))
        if not goals or not any(g.get("view_points") for g in goals):
            raise DatasetIntegrityError(f"No goal viewpoints for {scene}/{category}")
        scene_path = self.manifest["scene_paths"][scene]
        return NavigationEpisode(
            self.manifest["dataset_id"],
            self.manifest["split"],
            scene_path,
            str(raw["episode_id"]),
            f"Find a {category}.",
            raw["start_position"],
            raw["start_rotation"],
            goals,
            category,
            raw.get("info", {}),
        )

    def sample_episode(self):
        if self.remaining <= 0:
            self.current = self.rng.choices(self.files, [f["episodes"] for f in self.files])[0]
            self.remaining = self.scene_repeat
        self.remaining -= 1
        data = self._load(self.current)
        return self.decode(self.rng.choice(data["episodes"]), data)

    def evaluation_episodes(self, limit=None, stratified=False, seed=17):
        if stratified and limit is not None:
            rng = random.Random(seed)
            entries = sorted(self.files, key=lambda f: f["path"])
            rng.shuffle(entries)
            count = 0
            selected: dict[str, set[int]] = {e["path"]: set() for e in entries}
            while count < limit:
                progress = False
                for entry in entries:
                    data = self._load(entry)
                    choices = [
                        i for i in range(len(data["episodes"])) if i not in selected[entry["path"]]
                    ]
                    if not choices:
                        continue
                    index = rng.choice(choices)
                    selected[entry["path"]].add(index)
                    yield self.decode(data["episodes"][index], data)
                    count += 1
                    progress = True
                    if count >= limit:
                        return
                if not progress:
                    return
            return
        count = 0
        for entry in sorted(self.files, key=lambda f: f["path"]):
            data = self._load(entry)
            for raw in data["episodes"]:
                if limit is not None and count >= limit:
                    return
                yield self.decode(raw, data)
                count += 1

    def state_dict(self):
        return {"rng": self.rng.getstate(), "remaining": self.remaining, "current": self.current}

    def load_state_dict(self, state):
        self.rng.setstate(state["rng"])
        self.remaining, self.current = state["remaining"], state["current"]
