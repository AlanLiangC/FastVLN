"""Stage I: open-vocabulary point training before navigation reinforcement."""

import argparse
import json
import math
import os
import signal
import time
from collections import Counter
from hashlib import sha256
from pathlib import Path

import torch
import yaml

from streamnav.contracts.perception import decode_pixel
from streamnav.data.grounding import GroundingSource, check_grounding_split
from streamnav.data.grounding_support import support_targets
from streamnav.evaluation.grounding_cases import case_record, write_gallery
from streamnav.models.point_cot.policy import Qwen35PointCoTPolicy
from streamnav.training import distributed as parallel
from streamnav.training.point_supervision import (
    grounding_loss,
    grounding_loss_options,
    grounding_opos_losses,
)
from streamnav.training.pointcot_trainer import (
    append_json,
    load_adapters,
    save,
    synchronized_optimizer_step,
)
from streamnav.utils.progress import write_worker_progress
from streamnav.utils.provenance import snapshot_sources
from streamnav.utils.seed import restore_rng, seed_everything


def optimizer_for(policy, learning_rates):
    groups: dict[str, list[torch.nn.Parameter]] = {
        k: [] for k in ("lora", "memory", "point", "action", "critic")
    }
    for name, p in policy.named_parameters():
        if p.requires_grad:
            group = (
                "memory"
                if name.startswith("memories.")
                else "point"
                if name.startswith(("apos_", "opos_"))
                else "action"
                if name.startswith("action_")
                else "critic"
                if name.startswith("critic.")
                else "lora"
            )
            groups[group].append(p)
    return torch.optim.AdamW(
        [{"params": ps, "name": name, "lr": learning_rates[name]} for name, ps in groups.items()],
        weight_decay=0,
        eps=1e-8,
        fused=policy.device.type == "cuda",
    )


def configure_sample_schedule(cfg, world_size, *, saved=None, previous=None, updates=None):
    """Keep the sample budget and LR curve continuous when shrinking the worker group."""
    global_batch = cfg["batch_size"] * world_size
    if saved is None:
        budget = cfg["pretrain_updates"] * global_batch
        warmup = cfg.get("warmup_updates", 0) * global_batch
        start_update, frames = 0, 0
    else:
        old = previous["grounding"]
        old_batch = old["batch_size"] * saved["world_size"]
        budget = old.get("sample_budget", old["pretrain_updates"] * old_batch)
        warmup = old.get("warmup_samples", old.get("warmup_updates", 0) * old_batch)
        start_update, frames = saved["update"], saved["frames"]
        if updates is not None:
            if updates < start_update:
                raise ValueError("The requested update budget precedes the checkpoint")
            budget = frames + (updates - start_update) * global_batch
    cfg["sample_budget"] = budget
    cfg["warmup_samples"] = warmup
    cfg["pretrain_updates"] = start_update + math.ceil(max(0, budget - frames) / global_batch)


def sample_lr_scale(cfg, frames, global_batch):
    warmup = cfg["warmup_samples"]
    if frames < warmup:
        return min(1.0, (frames + global_batch) / warmup)
    progress = min(1.0, max(0.0, (frames - warmup) / max(1, cfg["sample_budget"] - warmup)))
    minimum = cfg.get("min_lr_ratio", 0.1)
    return minimum + (1 - minimum) * 0.5 * (1 + math.cos(math.pi * progress))


def resumed_rank_state(saved, rank, world_size, *, allow_world_size_change=False):
    old_size = saved["world_size"]
    if world_size != old_size and not (allow_world_size_change and 0 < world_size < old_size):
        raise ValueError("Resume requires the same world size; explicitly allow shrinking workers")
    if len(saved["ranks"]) != old_size or not 0 <= rank < world_size:
        raise ValueError("Invalid grounding checkpoint rank states")
    # Every grounding rank samples the full manifest independently. Surviving
    # streams keep their RNG/cursors; no fixed data shards are discarded.
    return saved["ranks"][rank]


def validation_indices(source, count):
    # Equal coverage of source/label kind, fixed throughout the run. Weak boxes
    # have their own score and must not inflate the strong point score.
    groups: dict[tuple[str, str], list[int]] = {}
    for i in source.order:
        row = source.rows[i]
        kind = (
            "region"
            if row.get("supervision") == "region"
            else "absence"
            if row.get("opos_targets") == [0]
            else "point"
        )
        groups.setdefault((row.get("source", "unknown"), kind), []).append(i)
    selected: list[int] = []
    cursor = 0
    while len(selected) < min(count, len(source.rows)):
        for key in sorted(groups):
            if cursor < len(groups[key]):
                selected.append(groups[key][cursor])
                if len(selected) == count:
                    return selected
        cursor += 1
    return selected


def summarize_grounding(totals):
    return {
        **totals,
        "cell_accuracy": totals["cell_hits"] / max(1, totals["count"]),
        "point_accuracy_5pct": totals["near_hits"]
        / max(1, totals["count"] - totals["region_count"]),
        "visible_point_accuracy_5pct": totals["visible_near_hits"]
        / max(1, totals["visible_count"]),
        "absence_accuracy": totals["absence_hits"] / max(1, totals["absence_count"]),
        "region_accuracy": totals["region_hits"] / max(1, totals["region_count"]),
        "visible_grounding_accuracy": totals["visible_grounding_hits"]
        / max(1, totals["visible_count"]),
        "instance_accuracy": totals["instance_hits"] / max(1, totals["instance_count"]),
        "false_absence_rate": totals["false_absence_count"] / max(1, totals["visible_count"]),
        "false_presence_rate": totals["false_presence_count"] / max(1, totals["absence_count"]),
        "validation_loss": totals["validation_loss_sum"] / max(1, totals["nll_count"]),
        "anchor_nll": totals["anchor_nll_sum"] / max(1, totals["nll_count"]),
        "support_nll": totals["support_nll_sum"] / max(1, totals["nll_count"]),
    }


@torch.no_grad()
def validate_grounding(
    policy,
    source,
    count=128,
    *,
    output_dir=None,
    update=0,
    case_count=24,
    batch_size=16,
    anchor_weight=1.0,
    point_radius=0.0,
):
    selected = validation_indices(source, count)
    rows = [
        (index, GroundingSource.materialize(source.rows[i])) for index, i in enumerate(selected)
    ][parallel.rank() :: parallel.world_size()]
    cases = []
    started = time.monotonic()
    progress_root = Path(output_dir).parent.parent if output_dir is not None else None
    by_source: dict[str, dict[str, float]] = {}
    zero = {
        k: 0
        for k in (
            "count",
            "cell_hits",
            "near_hits",
            "absence_count",
            "absence_hits",
            "visible_count",
            "visible_near_hits",
            "region_count",
            "region_hits",
            "visible_grounding_hits",
            "instance_count",
            "instance_hits",
            "false_absence_count",
            "false_presence_count",
            "validation_loss_sum",
            "anchor_nll_sum",
            "support_nll_sum",
            "nll_count",
        )
    }
    for offset in range(0, len(rows), batch_size):
        if progress_root is not None:
            write_worker_progress(
                progress_root,
                update,
                "grounding_eval",
                offset,
                len(rows),
                started,
                rank=parallel.rank(),
            )
        batch = rows[offset : offset + batch_size]
        features = GroundingSource.features_many(policy, [r for _, r in batch])
        for (index, row), feature in zip(batch, features, strict=True):
            step = policy.step(
                feature,
                policy.start_episode(row["image_id"], row["instruction"]),
                deterministic=True,
            )
            if output_dir is not None and case_count:
                cases.append(case_record(row, step, index))
            totals = by_source.setdefault(row.get("source", "unknown"), dict(zero))
            if row.get("opos_targets") and hasattr(step, "opos_logits"):
                loss, anchor, support = grounding_opos_losses(
                    step.opos_logits.reshape(1, -1),
                    [row],
                    anchor_weight=anchor_weight,
                    point_radius=point_radius,
                )
                totals["validation_loss_sum"] += float(loss[0]) * row.get("loss_weight", 1.0)
                totals["anchor_nll_sum"] += float(anchor[0])
                totals["support_nll_sum"] += float(support[0])
                totals["nll_count"] += 1
            for name in ("apos", "opos"):
                targets = row.get(name + "_targets", [])
                if not targets:
                    continue
                token = int(getattr(step, name))
                hit = token in targets
                totals["count"] += 1
                totals["cell_hits"] += hit
                if row.get("supervision") == "region":
                    totals["region_count"] += 1
                    totals["region_hits"] += hit
                    continue
                point = decode_pixel(token, 1, 1)
                close = point is not None and any(
                    ((point[0] - other[0]) ** 2 + (point[1] - other[1]) ** 2) ** 0.5 <= 0.05
                    for t in targets
                    if (other := decode_pixel(t, 1, 1)) is not None
                )
                totals["near_hits"] += bool(hit or close)
                if targets == [0]:
                    totals["absence_count"] += 1
                    totals["absence_hits"] += token == 0
                    totals["false_presence_count"] += token != 0
                else:
                    totals["visible_count"] += 1
                    totals["visible_near_hits"] += bool(hit or close)
                    totals["false_absence_count"] += token == 0
                    if name == "opos" and row.get("opos_support"):
                        inside = token in support_targets(row)
                        totals["instance_count"] += 1
                        totals["instance_hits"] += inside
                        totals["visible_grounding_hits"] += inside
                    else:
                        totals["visible_grounding_hits"] += bool(hit or close)
    gathered = parallel.gather(by_source)
    merged_sources: dict[str, dict[str, float]] = {}
    for shard in gathered:
        for name, counts in shard.items():
            dest = merged_sources.setdefault(name, dict(zero))
            for key, value in counts.items():
                dest[key] += value
    merged = {key: sum(s[key] for s in merged_sources.values()) for key in zero}
    result = summarize_grounding(merged)
    result["by_source"] = {name: summarize_grounding(s) for name, s in merged_sources.items()}
    strong = [
        s["visible_point_accuracy_5pct"] for s in result["by_source"].values() if s["visible_count"]
    ]
    result["macro_visible_point_accuracy_5pct"] = sum(strong) / max(1, len(strong))
    object_scores = [
        s["visible_grounding_accuracy"] for s in result["by_source"].values() if s["visible_count"]
    ]
    result["macro_visible_grounding_accuracy"] = sum(object_scores) / max(1, len(object_scores))
    if output_dir is not None and case_count:
        gathered_cases = parallel.gather(cases)
        if parallel.rank() == 0:
            output_dir = Path(output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
            with (output_dir / "predictions.jsonl").open("w") as handle:
                for case in sorted(
                    [c for shard in gathered_cases for c in shard], key=lambda c: c["index"]
                ):
                    compact = {
                        k: v
                        for k, v in case.items()
                        if k not in ("opos_targets", "apos_targets", "opos_boxes", "opos_support")
                    }
                    handle.write(json.dumps(compact, ensure_ascii=False) + "\n")
            write_gallery(
                output_dir,
                [c for shard in gathered_cases for c in shard],
                update,
                result,
                count=case_count,
            )
        parallel.barrier()
        write_worker_progress(
            progress_root,
            update,
            "grounding_eval",
            len(rows),
            len(rows),
            started,
            rank=parallel.rank(),
        )
    return result


def promote_grounding_best(root, update, result, *, min_absence_accuracy=0.0):
    if parallel.rank() == 0:
        record = root / "best_grounding.json"
        previous = json.loads(record.read_text()) if record.exists() else None
        metric = result.get("selection_metric", "macro_visible_point_accuracy_5pct")
        score = (
            result.get(metric, result["visible_point_accuracy_5pct"]),
            result["absence_accuracy"],
        )
        old_score = (
            (
                previous.get(metric, previous["visible_point_accuracy_5pct"]),
                previous["absence_accuracy"],
            )
            if previous
            else (-1, -1)
        )
        eligible = previous is None or result["absence_accuracy"] >= min_absence_accuracy
        if eligible and score > old_score:
            link = root / "checkpoints" / ".best-new"
            link.unlink(missing_ok=True)
            link.symlink_to(f"update_{update:07d}")
            link.replace(root / "checkpoints" / "best")
            record.write_text(json.dumps({"update": update, **result}, indent=2))
    parallel.barrier()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/pointcot.yaml")
    parser.add_argument("--run-dir", default="runs/pointcot_grounding_refined")
    parser.add_argument("--updates", type=int)
    parser.add_argument("--batch-size", type=int)
    initialization = parser.add_mutually_exclusive_group()
    initialization.add_argument("--resume")
    initialization.add_argument(
        "--initialize-from", help="Adapters only; new optimizer, sampler and LR schedule"
    )
    parser.add_argument("--allow-world-size-change", action="store_true")
    parser.add_argument("--skip-validation", action="store_true")
    args = parser.parse_args()
    config = yaml.safe_load(Path(args.config).read_text())
    config["run_dir"] = args.run_dir
    config["cuda_visible_devices"] = os.environ.get("CUDA_VISIBLE_DEVICES")
    cfg = config["grounding"]
    config["model"]["gradient_checkpointing"] = cfg.get(
        "gradient_checkpointing", config["model"]["gradient_checkpointing"]
    )
    cfg["batch_size"] = args.batch_size or cfg["batch_size"]
    parallel.initialize(config)
    root = Path(config["run_dir"])
    root.mkdir(parents=True, exist_ok=True)
    if not args.resume and (root / "checkpoints" / "latest").exists():
        raise ValueError("Existing grounding run; use --resume or a new directory")
    seed_everything(config["seed"])
    started = time.monotonic()
    write_worker_progress(root, 0, "grounding_model_load", 0, 1, started, rank=parallel.rank())
    policy = Qwen35PointCoTPolicy(config).train()
    write_worker_progress(root, 0, "grounding_data_load", 0, 1, started, rank=parallel.rank())
    source = GroundingSource(
        cfg["train_manifest"],
        seed=config["seed"] + parallel.rank(),
        source_weights=cfg.get("source_weights"),
        absence_probability=cfg.get("absence_probability"),
    )
    cfg["pretrain_updates"] = (
        args.updates
        or cfg.get("pretrain_updates")
        or math.ceil(
            len(source.rows) * cfg["pretrain_epochs"] / (cfg["batch_size"] * parallel.world_size())
        )
    )
    val = GroundingSource(cfg["val_manifest"])
    check_grounding_split(source, val)
    policy.grounding_source = source
    config["stage"] = "grounding"
    config["grounding_manifest_sha256"] = {
        p: sha256(Path(p).read_bytes()).hexdigest()
        for p in (cfg["train_manifest"], cfg["val_manifest"])
    }
    optimizer = optimizer_for(policy, cfg["learning_rates"])
    loss_options = grounding_loss_options(cfg)
    update, frames = 0, 0
    if args.resume:
        previous = yaml.safe_load(Path(args.resume, "resolved_config.yaml").read_text())
        if previous.get("stage") != "grounding" or any(
            previous.get(k) != config[k] for k in ("model", "grounding_manifest_sha256")
        ):
            raise ValueError("Grounding base or data changed since checkpoint")
        for key in (
            "batch_size",
            "learning_rates",
            "source_weights",
            "absence_probability",
            "warmup_updates",
            "min_lr_ratio",
            "anchor_weight",
            "point_radius",
            "best_metric",
            "best_min_absence_accuracy",
        ):
            if previous["grounding"].get(key) != cfg.get(key):
                raise ValueError(f"Grounding resume changes {key}")
        load_adapters(policy, args.resume)
        saved = torch.load(
            Path(args.resume, "optimizer.pt"), map_location="cpu", weights_only=False
        )
        local = resumed_rank_state(
            saved,
            parallel.rank(),
            parallel.world_size(),
            allow_world_size_change=args.allow_world_size_change,
        )
        configure_sample_schedule(
            cfg, parallel.world_size(), saved=saved, previous=previous, updates=args.updates
        )
        if saved["world_size"] != parallel.world_size():
            config["world_size_change"] = {
                "from": saved["world_size"],
                "to": parallel.world_size(),
                "update": saved["update"],
                "frames": saved["frames"],
                "preserved_sample_budget": cfg["sample_budget"],
            }
        optimizer.load_state_dict(saved["optimizer"])
        source.load_state_dict(local["source"])
        restore_rng(local["rng"])
        update, frames = saved["update"], saved["frames"]
    else:
        configure_sample_schedule(cfg, parallel.world_size())
        if args.initialize_from:
            load_adapters(policy, args.initialize_from)
            origin = json.loads(Path(args.initialize_from, "manifest.json").read_text())
            config["initialization"] = {
                "checkpoint": str(Path(args.initialize_from).resolve()),
                "update": origin["update"],
                "adapter_sha256": origin["adapter_sha256"],
                "optimizer_and_sampler": "fresh",
            }
        seed_everything(config["seed"] + parallel.rank())
    root.mkdir(parents=True, exist_ok=True)
    config["source_snapshot"] = snapshot_sources(root)
    workers = parallel.gather(
        {
            "pid": os.getpid(),
            "rank": parallel.rank(),
            "launcher_pid": os.getppid(),
            "device": str(policy.device),
        }
    )
    if parallel.rank() == 0:
        (root / "training_workers.json").write_text(json.dumps(workers, indent=2))
        (root / "resolved_config.yaml").write_text(yaml.safe_dump(config, sort_keys=False))
        (root / "parameters.json").write_text(json.dumps(policy.parameter_summary(), indent=2))
        print(
            json.dumps(
                {
                    "phase": "grounding_ready",
                    "train_queries": len(source.rows),
                    "val_queries": len(val.rows),
                    "batch_size_per_rank": cfg["batch_size"],
                    "world_size": parallel.world_size(),
                    "updates": cfg["pretrain_updates"],
                    "resumed_at": update,
                }
            ),
            flush=True,
        )
    stop = [False]

    def request_stop(signum, frame):
        stop[0] = True

    signal.signal(signal.SIGTERM, request_stop)
    signal.signal(signal.SIGINT, request_stop)
    try:
        if not args.resume and not args.skip_validation:
            result = validate_grounding(
                policy,
                val,
                cfg["val_samples"],
                output_dir=root / "grounding_visualizations" / "update_0000000",
                update=0,
                case_count=cfg.get("visualization_cases", 24),
                **loss_options,
            )
            result["selection_metric"] = cfg.get("best_metric", "macro_visible_point_accuracy_5pct")
            result["selection_min_absence_accuracy"] = cfg.get("best_min_absence_accuracy", 0.0)
            save(policy, optimizer, source, config, 0, 0)
            promote_grounding_best(
                root, 0, result, min_absence_accuracy=cfg.get("best_min_absence_accuracy", 0.0)
            )
            if parallel.rank() == 0:
                append_json(root / "grounding_eval.jsonl", {"update": 0, **result})
        while update < cfg["pretrain_updates"] and not parallel.any_rank(stop[0], policy.device):
            started = time.monotonic()
            scale = sample_lr_scale(cfg, frames, cfg["batch_size"] * parallel.world_size())
            for group in optimizer.param_groups:
                group["lr"] = cfg["learning_rates"][group["name"]] * scale
            optimizer.zero_grad(set_to_none=True)
            rows = source.sample(cfg["batch_size"])
            loss, detail, _ = grounding_loss(policy, rows, **loss_options)
            loss.backward()
            norm = synchronized_optimizer_step(
                policy._all_trainable, optimizer, config["trainer"]["max_grad_norm"]
            )
            update += 1
            frames += cfg["batch_size"] * parallel.world_size()
            local = {
                "update": update,
                "loss": float(loss.detach()),
                **detail,
                "grad_norm": float(norm),
                "seconds": time.monotonic() - started,
                "lr_scale": scale,
                "peak_vram_gib": torch.cuda.max_memory_allocated(policy.device) / 1024**3,
                "absence_samples": sum(row.get("opos_targets") == [0] for row in rows),
                "region_samples": sum(row.get("supervision") == "region" for row in rows),
            }
            local["source_counts"] = dict(Counter(row["source"] for row in rows))
            collected = parallel.gather(local)
            if update % 10 == 0:
                write_worker_progress(
                    root,
                    update,
                    "grounding_train",
                    update,
                    cfg["pretrain_updates"],
                    started,
                    rank=parallel.rank(),
                    frames=frames,
                )
            if parallel.rank() == 0:
                record = {
                    k: sum(r[k] for r in collected) / len(collected)
                    for k in local
                    if k != "source_counts"
                }
                record["source_counts"] = {
                    name: sum(r["source_counts"].get(name, 0) for r in collected)
                    for name in sorted({name for r in collected for name in r["source_counts"]})
                }
                for key in ("absence_samples", "region_samples"):
                    record[key] = sum(r[key] for r in collected)
                record["peak_vram_gib"] = max(r["peak_vram_gib"] for r in collected)
                record.update(
                    update=update, frames=frames, seconds=max(r["seconds"] for r in collected)
                )
                record["samples_per_second"] = (
                    cfg["batch_size"] * parallel.world_size() / record["seconds"]
                )
                append_json(root / "train_metrics.jsonl", record)
                print(json.dumps(record), flush=True)
            checkpoint_saved = (
                update % cfg["checkpoint_interval"] == 0 or update == cfg["pretrain_updates"]
            )
            if checkpoint_saved:
                save(policy, optimizer, source, config, update, frames)
            if not args.skip_validation and (
                update % cfg["eval_interval"] == 0 or update == cfg["pretrain_updates"]
            ):
                result = validate_grounding(
                    policy,
                    val,
                    cfg["val_samples"],
                    update=update,
                    output_dir=root / "grounding_visualizations" / f"update_{update:07d}",
                    case_count=cfg.get("visualization_cases", 24),
                    **loss_options,
                )
                result["selection_metric"] = cfg.get(
                    "best_metric", "macro_visible_point_accuracy_5pct"
                )
                result["selection_min_absence_accuracy"] = cfg.get("best_min_absence_accuracy", 0.0)
                if not checkpoint_saved:
                    save(policy, optimizer, source, config, update, frames)
                promote_grounding_best(
                    root,
                    update,
                    result,
                    min_absence_accuracy=cfg.get("best_min_absence_accuracy", 0.0),
                )
                if parallel.rank() == 0:
                    append_json(root / "grounding_eval.jsonl", {"update": update, **result})
        save(policy, optimizer, source, config, update, frames)
    finally:
        if torch.distributed.is_initialized():
            torch.distributed.destroy_process_group()


if __name__ == "__main__":
    torch.set_num_threads(4)
    main()
