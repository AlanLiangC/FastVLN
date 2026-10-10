import hashlib
import importlib.metadata
import json
import math
import os
import shutil
import subprocess
import tempfile
import zipfile
from dataclasses import asdict
from pathlib import Path

import torch
import yaml
from safetensors.torch import load_file, save_file

from streamnav.contracts.perception import perception_config
from streamnav.data.manifest import file_hash
from streamnav.models.policy.streaming_policy import StreamingObjectNavPolicy
from streamnav.models.qwen35_kda.backbone import Qwen35KDABackbone
from streamnav.utils.seed import restore_rng, rng_state


def load_policy(config, training=False, preserve_master_weights=False):
    device = torch.device(config["device"])
    if device.type == "cuda":
        torch.cuda.set_device(device)
    path = Path(config.get("checkpoint") or config["model"]["checkpoint"]).resolve()
    model_cfg = dict(config["model"])
    saved_config = path / "resolved_config.yaml"
    if not training and saved_config.exists():
        saved_model = yaml.safe_load(saved_config.read_text())["model"]
        for key in ("action_dim", "value_hidden_dim"):
            model_cfg[key] = saved_model[key]
        model_cfg["critic_type"] = saved_model.get("critic_type", "mlp")
        model_cfg["goal_conditioning"] = saved_model.get("goal_conditioning", "episode")
        model_cfg["kda_output_norm"] = saved_model.get("kda_output_norm", False)
        model_cfg["critic_gain"] = saved_model.get("critic_gain", 1.0)
        model_cfg["perception"] = saved_model.get("perception")
    if model_cfg["action_dim"] not in (4, 6) or model_cfg["dtype"] not in ("bfloat16", "float32"):
        raise ValueError("Only six/legacy-four actions and bf16/fp32 precision are supported")
    # FP32 master parameters/moments for the optimizer, BF16 autocast compute.
    dtype = (
        torch.float32 if training or preserve_master_weights else getattr(torch, model_cfg["dtype"])
    )
    backbone = Qwen35KDABackbone.from_converted(
        path,
        device=config["device"],
        dtype=dtype,
        image_size=model_cfg["image_size"],
        gradient_checkpointing=model_cfg["gradient_checkpointing"],
        inference_mode=model_cfg.get("inference_mode", "auto"),
        goal_conditioning=model_cfg.get("goal_conditioning", "episode"),
        kda_output_norm=model_cfg.get("kda_output_norm", False),
    )
    if model_cfg["freeze_vision_encoder"]:
        backbone.vision.requires_grad_(False)
    if training and model_cfg["gradient_checkpointing"] and not model_cfg["freeze_vision_encoder"]:
        backbone.vision.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False}
        )
    policy = StreamingObjectNavPolicy(
        backbone,
        model_cfg["value_hidden_dim"],
        model_cfg["action_dim"],
        model_cfg.get("critic_type", "mlp"),
        model_cfg.get("critic_gain", 1.0),
        perception=model_cfg.get("perception"),
    )
    policy.inference_compute_dtype = getattr(torch, model_cfg["dtype"])
    policy.loaded_checkpoint = str(path)
    heads = Path(path) / "actor_critic.safetensors"
    if heads.exists():
        policy.actor_critic.load_state_dict(load_file(str(heads)))
    perception_weights = path / "perception.safetensors"
    if policy.perception is not None:
        if perception_weights.exists():
            policy.perception.load_state_dict(load_file(str(perception_weights)))
        elif config.get("checkpoint"):
            raise ValueError(
                "Resume checkpoint lacks perception weights; use explicit weight initialization"
            )
    elif perception_weights.exists():
        raise ValueError("Cannot discard a checkpoint's perception action branch")
    return policy


def provenance(config):
    def git(*args):
        result = subprocess.run(["git", *args], capture_output=True, text=True)
        return result.stdout.strip() if result.returncode == 0 else "unavailable"

    manifests = [s["manifest"] for s in config["data"]["sources"]] + config["eval"]["manifests"]
    layout = Path(config["model"]["checkpoint"]) / "kda_layout.json"
    layout_data = json.loads(layout.read_text())
    source = Path(layout_data["source"])
    revision_file = source / ".cache/huggingface/download/config.json.metadata"
    source_revision = layout_data.get("source_revision")
    if source_revision is None and revision_file.exists():
        source_revision = revision_file.read_text().splitlines()[0]
    serialized = json.dumps(config, sort_keys=True)
    initialization = Path(config.get("checkpoint") or config["model"]["checkpoint"]).resolve()
    calibration = initialization / "conversion_calibration.json"
    distillation = initialization / "conversion_distillation.json"
    initialization_record = {
        "checkpoint": str(initialization),
        "weights_sha256": {p.name: file_hash(p) for p in initialization.glob("*.safetensors")},
        "conversion_calibration": json.loads(calibration.read_text())
        if calibration.exists()
        else None,
        "conversion_distillation": json.loads(distillation.read_text())
        if distillation.exists()
        else None,
    }
    source_hash = hashlib.sha256()
    source_files = []
    for folder in (
        "src",
        "services",
        "tools",
        "configs",
        "scripts",
        "runtime/vendor/frontier_exploration/frontier_exploration",
    ):
        for path in sorted(Path(folder).rglob("*")):
            if path.is_file() and "__pycache__" not in str(path):
                source_hash.update(str(path).encode())
                source_hash.update(path.read_bytes())
                source_files.append(path)
    for path in (Path("pyproject.toml"), Path("uv.lock")):
        if path.exists():
            source_hash.update(str(path).encode())
            source_hash.update(path.read_bytes())
            source_files.append(path)
    archive = Path(config["run_dir"]) / "source" / f"{source_hash.hexdigest()}.zip"
    archive.parent.mkdir(parents=True, exist_ok=True)
    if not archive.exists():
        with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_DEFLATED) as snapshot:
            for path in source_files:
                snapshot.write(path, str(path))
    return {
        "git_sha": git("rev-parse", "HEAD"),
        "git_dirty": bool(git("status", "--porcelain")),
        "ovsegdt_reference_commit": git("-C", "third_party/OVSegDT", "rev-parse", "HEAD"),
        "frontier_exploration_commit": git(
            "-C", "runtime/vendor/frontier_exploration", "rev-parse", "HEAD"
        ),
        "source_tree_sha256": source_hash.hexdigest(),
        "source_archive": str(archive.resolve()),
        "source_archive_sha256": file_hash(archive),
        "qwen_revision": source_revision or "unknown-local-source",
        "qwen_source_config_sha256": layout_data["source_config_sha256"],
        "model_initialization": initialization_record,
        "qwen_source_weight_sha256": {p.name: file_hash(p) for p in source.glob("*.safetensors")},
        "fla_revision": f"PyPI:{importlib.metadata.version('flash-linear-attention')}",
        "versions": {
            n: importlib.metadata.version(n) for n in ("torch", "transformers", "fla-core")
        },
        "config_hash": hashlib.sha256(serialized.encode()).hexdigest(),
        "kda_layout_hash": file_hash(layout),
        "dataset_manifests": {p: file_hash(p) for p in manifests},
        "resume_semantics": "optimizer/schedules/RNG/samplers restored; simulator episodes and caches restart",
    }


def save_checkpoint(
    policy, optimizer, scheduler, dagger, sources, update, config, manifest, rank_states=None
):
    root = Path(config["run_dir"]) / "checkpoints"
    root.mkdir(parents=True, exist_ok=True)
    target = root / f"update_{update:07d}"
    if target.exists():
        return target
    temp = Path(tempfile.mkdtemp(prefix=".writing-", dir=root))
    try:
        save_file(
            {k: v.detach().cpu().contiguous() for k, v in policy.backbone.state_dict().items()},
            str(temp / "model.safetensors"),
        )
        save_file(
            {k: v.detach().cpu().contiguous() for k, v in policy.actor_critic.state_dict().items()},
            str(temp / "actor_critic.safetensors"),
        )
        if getattr(policy, "perception", None) is not None:
            save_file(
                {
                    k: v.detach().cpu().contiguous()
                    for k, v in policy.perception.state_dict().items()
                },
                str(temp / "perception.safetensors"),
            )
        policy.backbone.config.save_pretrained(temp)
        policy.backbone.tokenizer.save_pretrained(temp / "tokenizer")
        shutil.copy2(
            Path(config["model"]["checkpoint"]) / "kda_layout.json", temp / "kda_layout.json"
        )
        torch.save(
            {
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "rng": rng_state(),
                "sources": [s.state_dict() for s in sources],
                "update": update,
                "rank_states": rank_states,
                "world_size": len(rank_states) if rank_states else 1,
            },
            temp / "optimizer.pt",
        )
        (temp / "dagger_scheduler.json").write_text(json.dumps(asdict(dagger)))
        (temp / "resolved_config.yaml").write_text(yaml.safe_dump(config))
        (temp / "manifest.json").write_text(json.dumps({**manifest, "update": update}, indent=2))
        if manifest.get("source_archive"):
            shutil.copy2(manifest["source_archive"], temp / "source.zip")
        temp.rename(target)
        link = root / ".latest-new"
        link.unlink(missing_ok=True)
        link.symlink_to(target.name)
        os.replace(link, root / "latest")
        keep = config["trainer"]["keep_checkpoints"]
        if keep > 0:
            best = (root / "best").resolve()
            for old in sorted(root.glob("update_*"))[:-keep]:
                if old.resolve() != best:
                    shutil.rmtree(old)
    finally:
        if temp.exists():
            shutil.rmtree(temp)
    return target


def promote_best_checkpoint(run_dir, update, results):
    """Preserve a positive-SR checkpoint selected on all configured validation splits."""
    root = Path(run_dir)
    score = (
        sum(r["success"] for r in results.values()) / len(results),
        sum(r["spl"] for r in results.values()) / len(results),
    )
    if not all(math.isfinite(s) for s in score) or score[0] <= 0:
        return False
    record = root / "best_evaluation.json"
    if record.exists():
        previous = json.loads(record.read_text())
        if score <= tuple(previous["score"]):
            return False
    target = root / "checkpoints" / f"update_{update:07d}"
    saved_update = json.loads((target / "manifest.json").read_text())["update"]
    if saved_update != update:
        raise ValueError("Best checkpoint must match the evaluated model update")
    link = root / "checkpoints" / ".best-new"
    link.unlink(missing_ok=True)
    link.symlink_to(target.name)
    os.replace(link, root / "checkpoints" / "best")
    temporary = record.with_suffix(".tmp")
    temporary.write_text(
        json.dumps(
            {
                "update": update,
                "score": score,
                "selection": "macro autonomous SR, then macro SPL; fixed diagnostic subset",
                "results": results,
            },
            indent=2,
        )
    )
    os.replace(temporary, record)
    return True


def validate_resume_configuration(path, config):
    """Validate before touching a run; recipe experiments require an empty fork."""
    changes = {}
    if config is not None:
        previous = yaml.safe_load((Path(path) / "resolved_config.yaml").read_text())
        if previous["trainer"].get("revision", 1) != config["trainer"].get("revision", 1):
            raise ValueError("Training revision changed; restart from converted initialization")
        if previous["data"] != config["data"]:
            raise ValueError(
                "Resume requires the same training manifests, weights and sampler configuration"
            )
        for key in (
            "width",
            "height",
            "hfov",
            "sensor_height",
            "sensor_pitch_deg",
            "agent_height",
            "agent_radius",
            "navmesh_agent_max_climb",
            "navmesh_cell_height",
            "success_distance",
            "forward_step",
            "turn_angle",
        ):
            if previous["habitat"].get(key) != config["habitat"].get(key):
                raise ValueError(f"Resume robot configuration changed: {key}; start a new run")
        for key in (
            "image_size",
            "freeze_vision_encoder",
            "action_dim",
            "value_hidden_dim",
            "critic_type",
            "inference_mode",
        ):
            if previous["model"][key] != config["model"][key]:
                raise ValueError(f"Resume model configuration changed: {key}")
        if previous["model"].get("goal_conditioning", "episode") != config["model"].get(
            "goal_conditioning", "episode"
        ):
            raise ValueError("Resume model configuration changed: goal_conditioning")
        if previous["model"].get("kda_output_norm", False) != config["model"].get(
            "kda_output_norm", False
        ):
            raise ValueError("Resume model configuration changed: kda_output_norm")
        if perception_config(previous["model"].get("perception")) != perception_config(
            config["model"].get("perception")
        ):
            raise ValueError(
                "Resume perception architecture changed; use explicit weight initialization"
            )
        for section, key in (
            ("trainer", "perception_loss"),
            ("trainer", "perception_lr"),
            ("habitat", "perception_labels"),
        ):
            if previous[section].get(key) != config[section].get(key):
                raise ValueError(f"Resume perception recipe changed: {section}.{key}")
        if previous["trainer"]["dagger"] != config["trainer"]["dagger"]:
            raise ValueError("Resume requires the same DAgger schedule")
        allowed = config["trainer"].get("fork_recipe_changes", [])
        permitted = {
            "ealm",
            "auxiliary_il",
            "backbone_optimizer_eps",
            "oracle_execution",
            "filter_blocked_forward_labels",
        }
        if not isinstance(allowed, list) or not set(allowed).issubset(permitted):
            raise ValueError(f"Recipe forks permit only {', '.join(sorted(permitted))}")
        for key in (
            "ppo",
            "ealm",
            "loss",
            "optimizer",
            "optimizer_eps",
            "backbone_lr",
            "head_lr",
            "critic_lr",
            "vision_lr",
            "weight_decay",
            "max_grad_norm",
            "il_class_weights",
            "curriculum",
            "actor_warmup_updates",
            "num_envs",
            "rollout_steps",
            "sequence_length",
            "sequence_batch_size",
            "update_epochs",
            "backbone_optimizer_eps",
            "auxiliary_il",
            "filter_blocked_forward_labels",
        ):
            before, after = previous["trainer"].get(key), config["trainer"].get(key)
            if key == "ealm":
                before = {"minimum_ppo_weight": 0.0, **before}
                after = {"minimum_ppo_weight": 0.0, **after}
            if key == "auxiliary_il":
                from streamnav.training.auxiliary_il import auxiliary_il_config

                before, after = auxiliary_il_config(before), auxiliary_il_config(after)
            if key == "filter_blocked_forward_labels":
                before, after = bool(before), bool(after)
            if before != after:
                if key not in allowed:
                    raise ValueError(f"Resume training recipe changed: {key}")
                changes[key] = {"before": before, "after": after}
        before = previous["habitat"].get("oracle_execution", "upstream")
        after = config["habitat"].get("oracle_execution", "upstream")
        if after not in ("upstream", "collision_safe"):
            raise ValueError("Unknown oracle_execution")
        if before != after:
            if "oracle_execution" not in allowed:
                raise ValueError("Resume environment recipe changed: oracle_execution")
            changes["oracle_execution"] = {"before": before, "after": after}
        if changes:
            destination = Path(config["run_dir"]).resolve()
            metrics = destination / "train_metrics.jsonl"
            if destination == Path(previous["run_dir"]).resolve() or (
                metrics.exists() and metrics.stat().st_size
            ):
                raise ValueError("Training recipe changes require a new, empty run_dir")
        for key in ("oracle", "oracle_config", "reward", "tilt_angle", "max_episode_steps"):
            if previous["habitat"].get(key) != config["habitat"].get(key):
                raise ValueError(f"Resume environment recipe changed: {key}")
        saved_manifest = json.loads((Path(path) / "manifest.json").read_text())
        for item in config["data"]["sources"]:
            manifest_path = item["manifest"]
            if saved_manifest["dataset_manifests"][manifest_path] != file_hash(manifest_path):
                raise ValueError(f"Training manifest changed since checkpoint: {manifest_path}")
    return changes


def restore_training(
    path, optimizer, scheduler, dagger, sources, config=None, mixer=None, auxiliary_controller=None
):
    # Only load checkpoints created by this project and trusted by the user.
    fork_changes = validate_resume_configuration(path, config)
    saved = torch.load(Path(path) / "optimizer.pt", map_location="cpu", weights_only=False)
    from streamnav.training.distributed import rank, world_size

    if saved.get("world_size", 1) != world_size():
        raise ValueError("Resume requires the same distributed world size")
    optimizer.load_state_dict(saved["optimizer"])
    if config is not None:
        from streamnav.training.optimizer import optimizer_epsilon

        for group in optimizer.param_groups:
            group["eps"] = optimizer_epsilon(config["trainer"], group.get("role"))
    scheduler.load_state_dict(saved["scheduler"])
    dagger.update = json.loads((Path(path) / "dagger_scheduler.json").read_text())["update"]
    per_rank = saved["rank_states"][rank()] if saved.get("rank_states") else saved
    if mixer is not None:
        if "mixer" not in per_rank:
            raise ValueError("Checkpoint lacks the per-rank EALM entropy EMA")
        mixer.load_state_dict(per_rank["mixer"])
    if (
        auxiliary_controller is not None
        and "auxiliary_il" in per_rank
        and "auxiliary_il" not in fork_changes
    ):
        auxiliary_controller.load_state_dict(per_rank["auxiliary_il"])
    for source, state in zip(sources, per_rank["sources"], strict=True):
        source.load_state_dict(state)
    restore_rng(per_rank["rng"])
    return saved["update"]


def initialize_training_branches(path, optimizer, sources, config, mixer):
    """Explicit new-run initialization: preserve old Adam/RNG/samplers, add new heads.

    The new run has its own update counter and recipe; this is not full resume.
    Existing optimizer groups must match exactly and never silently lose moments.
    """
    from streamnav.training.distributed import rank, world_size

    path = Path(path)
    previous = yaml.safe_load((path / "resolved_config.yaml").read_text())
    if previous["data"] != config["data"]:
        raise ValueError("Branch initialization requires identical training data and samplers")
    manifest = json.loads((path / "manifest.json").read_text())
    for item in config["data"]["sources"]:
        filename = item["manifest"]
        if manifest["dataset_manifests"][filename] != file_hash(filename):
            raise ValueError("Branch initialization training manifest changed")
    for section, keys in (
        (
            "model",
            (
                "image_size",
                "action_dim",
                "value_hidden_dim",
                "critic_type",
                "goal_conditioning",
                "kda_output_norm",
                "freeze_vision_encoder",
            ),
        ),
        (
            "habitat",
            (
                "width",
                "height",
                "hfov",
                "sensor_height",
                "sensor_pitch_deg",
                "agent_height",
                "agent_radius",
                "success_distance",
                "forward_step",
                "turn_angle",
                "oracle",
                "oracle_execution",
                "oracle_config",
                "reward",
            ),
        ),
    ):
        for key in keys:
            if previous[section].get(key) != config[section].get(key):
                raise ValueError(f"Branch initialization changed {section}.{key}")
    saved = torch.load(path / "optimizer.pt", map_location="cpu", weights_only=False)
    if saved.get("world_size", 1) != world_size():
        raise ValueError("Branch initialization requires the original distributed world size")
    restore_optimizer_branches(optimizer, saved["optimizer"])
    per_rank = saved["rank_states"][rank()] if saved.get("rank_states") else saved
    mixer.load_state_dict(per_rank["mixer"])
    for source, state in zip(sources, per_rank["sources"], strict=True):
        source.load_state_dict(state)
    restore_rng(per_rank["rng"])
    return {
        "parent_update": saved["update"],
        "checkpoint": str(path.resolve()),
        "restored": "existing Adam moments, per-rank RNG/samplers/EALM",
        "new": "perception branch and new-run update/schedule counters",
        "episode_state": "simulator episodes and recurrent caches restart",
    }


def restore_optimizer_branches(optimizer, saved):
    current = optimizer.state_dict()
    old_groups = {group["role"]: group for group in saved["param_groups"]}
    current_roles = {group["role"] for group in current["param_groups"]}
    if set(old_groups) - current_roles:
        raise ValueError("Cannot discard an existing optimizer branch")
    for actual, group in zip(optimizer.param_groups, current["param_groups"], strict=True):
        old = old_groups.get(group["role"])
        if old is None:
            if group["role"] != "perception":
                raise ValueError("Only the perception optimizer branch may be newly initialized")
            continue
        if len(old["params"]) != len(group["params"]):
            raise ValueError(f"Optimizer parameter count changed in {group['role']}")
        for parameter, new_id, old_id in zip(
            actual["params"], group["params"], old["params"], strict=True
        ):
            state = saved["state"].get(old_id)
            if state is not None:
                for key in ("exp_avg", "exp_avg_sq"):
                    if key in state and state[key].shape != parameter.shape:
                        raise ValueError(f"Optimizer shape changed in {group['role']}")
                current["state"][new_id] = state
    optimizer.load_state_dict(current)
