from types import SimpleNamespace

import torch
from fastapi.testclient import TestClient

from streamnav.contracts.state import StreamingState
from streamnav.serving import server


class FakePolicy:
    def __init__(self, path):
        self.loaded_checkpoint = str(path)

    def eval(self):
        return self

    def start_episode(self, episode, instruction):
        return StreamingState((), episode, instruction, 0, instruction)

    reset = start_episode


class FakeEnv:
    def __init__(self, *args):
        pass

    def reset(self, episode):
        return {"rgb": torch.zeros(2, 2, 3, dtype=torch.uint8), "distance": 1, "robot": {}}

    def close(self):
        pass


def test_viewer_reloads_latest_on_reset_and_invalidates_sessions(tmp_path, monkeypatch):
    versions = [tmp_path / f"update_{i}" for i in (1, 2)]
    for version in versions:
        version.mkdir()
        (version / "actor_critic.safetensors").touch()
    latest = tmp_path / "latest"
    latest.symlink_to(versions[0])
    config = {
        "run_dir": str(tmp_path),
        "checkpoint": str(latest),
        "serving": {"max_sessions": 16, "session_ttl_s": 900, "reload_on_demo_reset": True},
        "eval": {"manifests": ["val.json"]},
        "habitat": dict.fromkeys(
            [
                "width",
                "height",
                "hfov",
                "sensor_height",
                "sensor_pitch_deg",
                "agent_height",
                "agent_radius",
            ],
            1,
        ),
    }
    episode = SimpleNamespace(uid="episode", goal_text="Find a chair.")
    monkeypatch.setattr(server, "HabitatClient", FakeEnv)
    monkeypatch.setattr(
        server,
        "HabitatEpisodeSource",
        lambda path: SimpleNamespace(evaluation_episodes=lambda *args, **kwargs: iter([episode])),
    )
    loaded = []

    def load(config, preserve_master_weights=False):
        assert preserve_master_weights
        loaded.append(config["checkpoint"])
        return FakePolicy(config["checkpoint"])

    monkeypatch.setattr(server, "load_policy", load)
    with TestClient(server.create_app(FakePolicy(versions[0]), config)) as client:
        sid = client.post("/sessions/start", json={"instruction": "chair"}).json()["session_id"]
        latest.unlink()
        latest.symlink_to(versions[1])
        assert client.get("/health").json()["new_checkpoint_available"]
        result = client.post("/demo/reset", json={})
        assert result.status_code == 200
        assert result.json()["checkpoint"] == str(versions[1])
        assert loaded == [str(versions[1])]
        assert not client.get("/health").json()["new_checkpoint_available"]
        assert (
            client.post(f"/sessions/{sid}/reset", json={"instruction": "chair"}).status_code == 404
        )
        assert client.post("/demo/reset", json={}).status_code == 200
        assert len(loaded) == 1
        catalog = client.get("/demo/episodes").json()
        assert catalog["episodes"] == [
            {"index": 0, "episode_id": "episode", "goal": "Find a chair.", "success": None}
        ]
        assert client.get("/demo/episodes?split_index=-1").status_code == 400
