"""Real observations supply hints; simulator annotations are never detector inputs."""

from collections import OrderedDict
from typing import Any

from streamnav.evaluation.detector_client import DetectorClient
from streamnav.training.object_hints import episode_hint_mode


class TrainingDetectorHints:
    def __init__(self, config, directory):
        self.settings = config["trainer"].get("object_hints", {})
        self.seed, self.directory = config["seed"], directory
        self.client = None
        self.cache = OrderedDict()

    def packets(self, observations, states):
        if not self.settings.get("real_probability", 0):
            return None
        packets: list[dict[str, Any] | None] = []
        pending = []
        for i, (observation, state) in enumerate(zip(observations, states, strict=True)):
            key = (state.episode_id, observation["frame_id"])
            mode = episode_hint_mode(state.episode_id, self.seed, self.settings)
            if mode == "simulated":
                packets.append(None)
            elif mode == "null":
                packets.append(
                    {"token": 0, "source": "episode_null", "episode_id": key[0], "frame_id": key[1]}
                )
            elif key in self.cache:
                packets.append(self.cache[key])
            else:
                packets.append(None)
                pending.append(i)
        if pending:
            if self.client is None:
                self.client = DetectorClient(self.directory)
            categories = [
                states[i].instruction.removeprefix("Find a ").removesuffix(".") for i in pending
            ]
            predictions = self.client.predict(
                [observations[i]["rgb"] for i in pending], categories, "train"
            )
            for i, packet in zip(pending, predictions, strict=True):
                key = (states[i].episode_id, observations[i]["frame_id"])
                packets[i] = {**packet, "episode_id": key[0], "frame_id": key[1]}
                self.cache[key] = packets[i]
                if len(self.cache) > 64:
                    self.cache.popitem(last=False)
        return packets

    def close(self):
        if self.client is not None:
            self.client.close()
