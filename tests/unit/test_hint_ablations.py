import importlib.util
import sys
from pathlib import Path

import pytest

TOOLS = Path(__file__).resolve().parents[2] / "tools"
sys.path.insert(0, str(TOOLS))
SPEC = importlib.util.spec_from_file_location("hint_ablations", TOOLS / "compare_hint_ablations.py")
ablation = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(ablation)


def test_sample_is_unique_reproducible_and_independent_per_split():
    indices = [
        {"key": f"{name}_{i}", "benchmark": name} for name in ablation.BENCHMARKS for i in range(40)
    ]
    a = ablation.select_keys(indices, 30, 17)
    assert a == ablation.select_keys(list(reversed(indices)), 30, 17)
    assert a != ablation.select_keys(indices, 30, 18)
    assert all(len(keys) == len(set(keys)) == 30 for keys in a.values())


def test_paired_statistics_count_gains_losses_and_keep_the_sign():
    s = ablation.paired_stats([1, 1, 0, 0], [0, 1, 1, 0])
    assert s["wins"] == s["losses"] == 1
    assert s["delta_sr"] == 0
    assert s["mcnemar_exact_p"] == 1
    s = ablation.paired_stats([1] * 10, [0] * 10)
    assert s["paired_bootstrap_95ci"] == [1, 1]
    assert s["mcnemar_exact_p"] == pytest.approx(2 / 1024)
    s = ablation.paired_stats([0] * 10, [1] * 10)
    assert s["delta_sr"] == -1
    assert s["paired_bootstrap_95ci"] == [-1, -1]


def test_partial_completion_does_not_publish_biased_sr_or_unmatched_comparison():
    protocol = {"selection": {"seen": ["a", "b"]}}
    results = {("a", arm): {"success": 1} for arm in ablation.ARMS}
    groups, contrasts = ablation.summarize(protocol, results)
    assert groups["seen"] == {"paired_completed": 1, "expected": 2}
    assert not contrasts


def test_full_comparison_preserves_all_four_arms_and_holm_tests():
    protocol = {"selection": {"seen": ["a"]}}
    row = {
        "success": 1,
        "spl": 0.5,
        "episode_length": 50,
        "collision_rate": 0.1,
        "oracle_success": 1,
        "false_stop": False,
        "min_distance_to_goal": 0.1,
        "max_alternating_turns": 0,
    }
    results = {("a", arm): dict(row) for arm in ablation.ARMS}
    groups, contrasts = ablation.summarize(protocol, results)
    assert all(groups["seen"][arm]["success"] == 1 for arm in ablation.ARMS)
    assert len(contrasts) == 4
    assert all(c["delta_sr"] == 0 and c["holm_adjusted_p"] == 1 for c in contrasts.values())
