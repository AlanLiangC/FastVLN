import gzip
import json

import pytest

from streamnav.data.manifest import check_leakage
from streamnav.data.mixture import MixedEpisodeSource
from streamnav.data.schema import HabitatEpisodeSource
from streamnav.errors import DatasetIntegrityError


class ConstantSource:
    def __init__(self, value):
        self.value = value

    def sample_episode(self):
        return self.value


def test_mixture_weights_and_reproducibility():
    sources = [ConstantSource("a"), ConstantSource("b")]
    a, b = [MixedEpisodeSource(sources, [0.25, 0.75], seed=17) for _ in range(2)]
    draws = [a.sample_episode() for _ in range(2000)]
    assert draws == [b.sample_episode() for _ in range(2000)]
    assert 0.20 < draws.count("a") / len(draws) < 0.30
    for weights in ([0, 0], [-1, 1], [float("nan"), 1]):
        with pytest.raises(ValueError):
            MixedEpisodeSource(sources, weights)


def test_scene_and_unseen_leakage():
    train = [{"scenes": ["a"], "categories": ["plant"]}]
    check_leakage(train, [{"scenes": ["b"], "categories": ["chair"], "split": "val_unseen"}])
    for val in (
        {"scenes": ["a"], "categories": [], "split": "val"},
        {"scenes": ["b"], "categories": ["plant"], "split": "val_unseen"},
    ):
        with pytest.raises(DatasetIntegrityError):
            check_leakage(train, [val])


def test_explicit_exclusion_and_goal_resolution(tmp_path):
    path = tmp_path / "scene.json.gz"
    episodes = [
        {
            "scene_id": "scene.basis.glb",
            "object_category": category,
            "goals": [],
            "episode_id": str(i),
            "start_position": [0, 0, 0],
            "start_rotation": [0, 0, 0, 1],
        }
        for i, category in enumerate(["plant", "chair"])
    ]
    data = {
        "episodes": episodes,
        "goals_by_category": {
            "scene.basis.glb_chair": [{"view_points": [{"agent_state": {"position": [1, 0, 0]}}]}]
        },
    }
    with gzip.open(path, "wt") as f:
        json.dump(data, f)
    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "dataset_id": "hm3d_v1",
                "split": "train",
                "files": [{"path": str(path), "episodes": 1}],
                "excluded_categories": ["plant"],
                "scene_paths": {"scene.basis.glb": "scene.basis.glb"},
            }
        )
    )
    source = HabitatEpisodeSource(manifest)
    assert all(source.sample_episode().object_category == "chair" for _ in range(20))
    assert len(list(source.evaluation_episodes())) == 1


def test_limited_validation_is_reproducible_and_covers_scenes(tmp_path):
    files, scene_paths = [], {}
    for scene in range(4):
        scene_name = f"scene_{scene}.glb"
        path = tmp_path / f"scene_{scene}.json"
        path.write_text(
            json.dumps(
                {
                    "episodes": [
                        {
                            "scene_id": scene_name,
                            "object_category": "chair",
                            "episode_id": str(i),
                            "start_position": [0, 0, 0],
                            "start_rotation": [0, 0, 0, 1],
                            "goals": [{"view_points": [{"agent_state": {"position": [1, 0, 0]}}]}],
                        }
                        for i in range(5)
                    ]
                }
            )
        )
        files.append({"path": str(path), "episodes": 5})
        scene_paths[scene_name] = scene_name
    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        json.dumps(
            {"dataset_id": "test", "split": "val", "files": files, "scene_paths": scene_paths}
        )
    )
    source = HabitatEpisodeSource(manifest)
    first = list(source.evaluation_episodes(4, stratified=True))
    second = list(source.evaluation_episodes(4, stratified=True))
    assert [e.uid for e in first] == [e.uid for e in second]
    assert len({e.scene_id for e in first}) == 4
    exhausted = list(source.evaluation_episodes(30, stratified=True))
    assert len(exhausted) == len({e.uid for e in exhausted}) == 20
