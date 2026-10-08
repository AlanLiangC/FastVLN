from __future__ import annotations

from dataclasses import asdict, dataclass, field


@dataclass(frozen=True)
class NavigationEpisode:
    dataset_id: str
    split: str
    scene_id: str
    episode_id: str
    goal_text: str
    start_position: list[float]
    start_rotation: list[float]
    goals: list[dict]
    object_category: str
    metadata: dict = field(default_factory=dict)

    @property
    def uid(self):
        return f"{self.dataset_id}:{self.split}:{self.scene_id}:{self.episode_id}"

    def to_dict(self):
        return asdict(self)
