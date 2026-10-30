"""Select deployable navigation checkpoints using one explicit evaluation setting."""

import json
import math
import time
from fractions import Fraction
from pathlib import Path


def best_setting(config):
    setting = config.get("eval", {}).get("best_setting", "no_hint")
    if setting not in ("no_hint", "simulated_hint", "yoloe_point"):
        raise ValueError("Unknown navigation best setting")
    return setting


def anchor_score(results):
    """Compare integer success counts exactly, including unequal split sizes."""
    if not results:
        raise ValueError("Cannot select a checkpoint without evaluation results")
    scores = []
    for row in results.values():
        count = row.get("selection_episodes", row.get("episodes", 0))
        rate = row.get("selection_success", row["success"])
        if (
            not isinstance(count, int)
            or count <= 0
            or not math.isfinite(rate)
            or not 0 <= rate <= 1
        ):
            raise ValueError("Invalid navigation selection counts")
        successes = round(rate * count)
        if not math.isclose(rate, successes / count, abs_tol=1e-9):
            raise ValueError("Navigation success rate disagrees with integer counts")
        scores.append(Fraction(successes, count))
    return sum(scores, Fraction()) / len(scores)


def record_setting(record):
    settings = {r.get("setting", "no_hint") for r in record["results"].values()}
    if len(settings) != 1:
        raise ValueError("Best checkpoint mixes evaluation settings")
    setting = record.get("selection_setting", next(iter(settings)))
    if settings != {setting}:
        raise ValueError("Best metadata disagrees with result settings")
    return setting


def make_record(update, results, setting):
    score = anchor_score(results)
    record = {
        "update": update,
        "success": float(score),
        "selection_metric": "fixed_anchor_success",
        "selection_setting": setting,
        "score_numerator": score.numerator,
        "score_denominator": score.denominator,
        "results": results,
    }
    if record_setting(record) != setting:
        raise ValueError("Evaluation returned the wrong best setting")
    return record


def write_best(root, record, *, link_name="best", record_name="best_evaluation.json"):
    root = Path(root)
    folder = root / "checkpoints" / f"update_{record['update']:07d}"
    if not all(
        (folder / n).is_file()
        for n in ("manifest.json", "adapters.safetensors", "optimizer.pt", "resolved_config.yaml")
    ):
        raise RuntimeError("Best evaluation must refer to a complete saved checkpoint")
    if json.loads((folder / "manifest.json").read_text())["update"] != record["update"]:
        raise ValueError("Best checkpoint update disagrees with evaluation")
    temp = root / ("." + record_name + ".new")
    temp.write_text(json.dumps(record, indent=2) + "\n")
    link = folder.parent / ("." + link_name + "-new")
    link.unlink(missing_ok=True)
    link.symlink_to(folder.name)
    link.replace(folder.parent / link_name)
    temp.replace(root / record_name)


def update_real_hint_best(root, update, results):
    path = Path(root) / "best_real_hint_evaluation.json"
    previous = json.loads(path.read_text()) if path.exists() else None
    candidate = make_record(update, results, "yoloe_point")
    if previous is None or anchor_score(results) > anchor_score(previous["results"]):
        write_best(root, candidate, link_name="best_real_hint", record_name=path.name)
        return True
    return False


def reselect_best(root, setting):
    """Scope changes use available checkpoints, while recording missing historical leaders."""
    root = Path(root)
    previous_file = root / "best_evaluation.json"
    previous = json.loads(previous_file.read_text()) if previous_file.exists() else None
    candidates, available = [], []
    for path in sorted((root / "evaluation").glob("update_*/settings_summary.json")):
        summary = json.loads(path.read_text())
        if setting not in summary:
            continue
        update = int(path.parent.name.split("_")[1])
        record = make_record(update, summary[setting], setting)
        candidates.append(record)
        folder = root / "checkpoints" / path.parent.name
        if all(
            (folder / n).is_file()
            for n in (
                "manifest.json",
                "adapters.safetensors",
                "optimizer.pt",
                "resolved_config.yaml",
            )
        ):
            available.append(record)
    if not available:
        raise RuntimeError("No evaluated saved checkpoint is available for this best setting")
    # Sorted ascending updates and exact rational scores keep the earliest tie.
    chosen = max(available, key=lambda r: anchor_score(r["results"]))
    historical = max(candidates, key=lambda r: anchor_score(r["results"]))
    write_best(root, chosen)
    event = {
        "time": time.time(),
        "selection_setting": setting,
        "previous_update": previous["update"] if previous else None,
        "previous_setting": record_setting(previous) if previous else None,
        "selected_update": chosen["update"],
        "selected_success": chosen["success"],
        "historical_best_update": historical["update"],
        "historical_best_success": historical["success"],
        "historical_best_checkpoint_available": any(
            r["update"] == historical["update"] for r in available
        ),
    }
    with (root / "best_selection_changes.jsonl").open("a") as f:
        f.write(json.dumps(event) + "\n")
    return event


def ensure_best_setting(root, setting):
    path = Path(root) / "best_evaluation.json"
    if path.exists() and record_setting(json.loads(path.read_text())) != setting:
        reselect_best(root, setting)


def update_best(root, update, results, setting):
    ensure_best_setting(root, setting)
    candidate = make_record(update, results, setting)
    path = Path(root) / "best_evaluation.json"
    previous = json.loads(path.read_text()) if path.exists() else None
    if previous is None or anchor_score(results) > anchor_score(previous["results"]):
        write_best(root, candidate)
        return True
    return False
