import json
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from types import SimpleNamespace

import pytest
import torch

from streamnav.contracts.state import StreamingState
from streamnav.evaluation import runner


@pytest.mark.parametrize("collisions", [False, True])
@pytest.mark.parametrize("pointing", [False, True])
def test_evaluation_updates_live_progress_during_long_episodes_and_cleans_up(
    tmp_path, monkeypatch, collisions, pointing
):
    episodes = [
        SimpleNamespace(uid=str(i), goal_text="chair", scene_id="room", object_category="chair")
        for i in range(2)
    ]

    class Source:
        manifest = {"dataset_id": "toy", "split": "val"}

        def __init__(self, *args, **kwargs):
            pass

        def evaluation_episodes(self, *args):
            return iter(episodes)

    class Client:
        def observation(self, done=False):
            return {
                "rgb": torch.zeros(4, 4, 3, dtype=torch.uint8),
                "frame_id": self.frame,
                "geodesic_distance": 0.2,
                "collision": collisions and 0 < self.frame <= 21,
                "done": done,
                "success": done,
                "metrics": {
                    "success": float(done),
                    "spl": 1.0,
                    "soft_spl": 1.0,
                    "distance_to_goal": 0.2,
                    "episode_length": self.frame,
                    "collision_rate": 0.0,
                    "oracle_success": 1.0,
                },
            }

        def reset(self, episode):
            self.frame = 0
            self.uid = episode.uid
            return self.observation()

        def get_object_supervision(self):
            token = 649 if self.uid == "0" else 0
            return {"opos_valid": True, "opos": token, "opos_targets": [token]}

        def step(self, action):
            self.frame += 1
            return self.observation(done=self.frame == 22)

    class Envs:
        def __init__(self, *args):
            assert args[0]["perception_labels"] == pointing
            self.clients = [Client()]
            self.executor = ThreadPoolExecutor(1)
            self.closed = False
            instances.append(self)

        def close(self):
            self.closed = True
            self.executor.shutdown()

    class Policy(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.parameter = torch.nn.Parameter(torch.ones(1))
            self.backbone = SimpleNamespace(device=torch.device("cpu"))
            self.perception = True if pointing else None
            self.supports_object_hints = pointing
            self.received = []

        def start_episode(self, uid, goal):
            return StreamingState((), uid, "hash", 0, goal)

        def forward_batch(self, frames, states, return_perception=False, hint_tokens=None):
            self.received.append(hint_tokens)
            logits = torch.full((len(states), 6), -10.0)
            for i, state in enumerate(states):
                logits[i, 0 if state.step_index == 21 else 1] = 10
            output = (
                logits,
                torch.zeros(len(states)),
                [replace(s, step_index=s.step_index + 1) for s in states],
            )
            if not return_perception:
                return output
            return (
                *output,
                {
                    "point_cot": True,
                    "apos": torch.zeros(len(states), 1300),
                    "opos": torch.zeros(len(states), 1297),
                    "object_hint_tokens": hint_tokens,
                },
            )

    instances, progress_calls = [], []
    write_progress = runner.write_worker_progress

    def recording_progress(*args, **kwargs):
        write_progress(*args, **kwargs)
        progress_calls.append(json.loads((tmp_path / "training_progress.json").read_text()))

    monkeypatch.setattr(runner, "HabitatEpisodeSource", Source)
    monkeypatch.setattr(runner, "VectorHabitatEnvs", Envs)
    monkeypatch.setattr(runner, "write_worker_progress", recording_progress)
    monkeypatch.setattr(runner.parallel, "rank", lambda: 0)
    monkeypatch.setattr(runner.parallel, "world_size", lambda: 1)
    monkeypatch.setattr(runner.parallel, "gather", lambda payload: [payload])
    monkeypatch.setattr(torch.cuda, "synchronize", lambda *args: None)
    monkeypatch.setattr(torch.cuda, "empty_cache", lambda: None)
    monkeypatch.setattr(torch.cuda, "max_memory_allocated", lambda *args: 0)
    policy = Policy()
    config = {
        "run_dir": str(tmp_path),
        "seed": 1,
        "habitat": {"success_distance": 0.25},
        "model": {"dtype": "float32", "action_dim": 6},
        "eval": {"manifests": ["toy"], "episodes": 2, "max_steps": 25},
    }
    if pointing:
        config["habitat"]["object_label_source"] = "semantic_instance"
        config["trainer"] = {
            "object_hints": {"noise": {"dropout": 1.0, "jitter": 1, "false_positive": 1.0}}
        }
        config["eval"].update(
            settings=["no_hint", "simulated_hint"],
            hint_noise={"dropout": 0.0, "jitter": 0, "false_positive": 0.0},
        )
    results = runner.evaluate(policy, config, update=100)
    assert results["toy_val"]["episodes"] == 2
    assert results["toy_val"]["success"] == 1
    assert results["toy_val"]["false_stop_rate"] == 0
    if collisions:
        assert results["toy_val"]["max_forward_collision_streak"] == 21
        assert results["toy_val"]["blocked_forward_episode_rate"] == 1
        assert results["toy_val"]["after_collision_forward_rate"] == pytest.approx(20 / 21)
    else:
        assert results["toy_val"]["after_collision_forward_rate"] is None
    assert any(r["completed"] == 20 and r.get("episodes_completed") == 0 for r in progress_calls)
    assert progress_calls[-1]["phase"] == "evaluation_wait"
    assert progress_calls[-1]["episodes_completed"] == 2
    assert progress_calls[-1]["completed"] == 44
    assert instances[0].closed
    assert policy.training
    if pointing:
        assert policy.received[:44] == [[0]] * 44
        assert policy.received[44:66] == [[649]] * 22
        assert policy.received[66:] == [[0]] * 22
        assert results["toy_val"]["object_hint_count"] == 0
        hints = json.loads((tmp_path / "eval_hint_metrics.jsonl").read_text())
        assert hints["object_hint_count"] == 22
        assert hints["object_hint_off_target_count"] == hints["object_hint_random_count"] == 0
        assert hints["hint_noise"] == config["eval"]["hint_noise"]
