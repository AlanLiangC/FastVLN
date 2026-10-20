import json
from pathlib import Path

import pytest
import torch

from streamnav.evaluation import runner
from streamnav.serving.batching import batch_steps
from streamnav.serving.session import SessionManager


def test_paired_settings_use_same_suite_and_keep_primary_metrics_separate(tmp_path, monkeypatch):
    calls = []

    def evaluate(policy, config, update, episodes, max_steps, video, setting):
        calls.append((setting, config["eval"]["episodes"], config["seed"]))
        path = Path(config["run_dir"]) / "evaluation" / f"update_{update:07d}"
        path.mkdir(parents=True, exist_ok=True)
        return {"seen": {"success": 0.1 if setting == "no_hint" else 0.9}}

    monkeypatch.setattr(runner, "_evaluate_setting", evaluate)
    config = {
        "run_dir": str(tmp_path),
        "seed": 7,
        "eval": {
            "settings": ["no_hint", "simulated_hint"],
            "episodes": 12,
            "full_episodes": 48,
            "full_eval_interval": 500,
        },
    }
    result = runner.evaluate(None, config, update=1500)
    assert result["seen"]["success"] == 0.1
    assert calls == [("no_hint", 48, 7), ("simulated_hint", 48, 7)]
    assert config["eval"]["episodes"] == 12
    summary = json.loads((tmp_path / "evaluation/update_0001500/settings_summary.json").read_text())
    assert summary["simulated_hint"]["seen"]["success"] == 0.9


def test_serving_routes_one_hint_and_resets_without_leaking_session_state():
    from streamnav.contracts.state import StreamingState

    class Policy(torch.nn.Module):
        supports_object_hints = True

        def __init__(self):
            super().__init__()
            self.received = []

        def start_episode(self, sid, instruction):
            return StreamingState((), sid, "hash", 0, instruction)

        def forward_batch(self, rgb, states, return_perception=False, hint_tokens=None):
            from dataclasses import replace

            self.received.append(hint_tokens)
            count = len(states)
            logits = torch.zeros(count, 6)
            logits[:, 1] = 2
            return (
                logits,
                torch.zeros(count),
                [replace(s, step_index=s.step_index + 1) for s in states],
                {
                    "point_cot": True,
                    "apos": torch.zeros(count, 1300),
                    "opos": torch.zeros(count, 1297),
                    "object_hint_tokens": hint_tokens,
                },
            )

    policy = Policy()
    manager = SessionManager(policy)
    a, b = manager.create("chair"), manager.create("bed")
    rgb = torch.zeros(270, 480, 3, dtype=torch.uint8)
    packet = {"frame_id": 0, "xy_px": [240, 135]}
    result = manager.step(a, rgb, packet)
    assert result["object_hint_token"] == 649
    assert manager.step(a, rgb, packet)["object_hint_token"] == 0  # stale
    results = batch_steps(manager, [(a, rgb, None), (b, rgb, packet)])
    assert [r["object_hint_token"] for r in results] == [0, 649]
    manager.reset(a, "lamp")
    assert manager.sessions[a].state.step_index == 0


def test_best_can_use_hint_while_both_settings_keep_the_same_fixed_anchors(tmp_path, monkeypatch):
    calls = []

    def evaluate(policy, config, update, episodes, max_steps, video, setting):
        calls.append((setting, config["eval"]["episodes"], config["eval"]["selection_episodes"]))
        path = Path(config["run_dir"]) / "evaluation" / f"update_{update:07d}"
        path.mkdir(parents=True, exist_ok=True)
        return {"seen": {"success": 0.25 if setting == "no_hint" else 0.75}}

    monkeypatch.setattr(runner, "_evaluate_setting", evaluate)
    config = {
        "run_dir": str(tmp_path),
        "eval": {
            "settings": ["no_hint", "simulated_hint"],
            "best_setting": "simulated_hint",
            "episodes": 12,
            "full_episodes": 48,
        },
    }
    results = runner.evaluate(None, config, update=9500)
    assert results["seen"]["success"] == 0.75
    assert calls == [("no_hint", 48, 12), ("simulated_hint", 48, 12)]
    config["eval"]["settings"] = ["no_hint"]
    with pytest.raises(ValueError, match="included"):
        runner.evaluate(None, config)
