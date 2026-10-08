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

from streamnav.data.manifest import file_hash
from streamnav.models.policy.streaming_policy import StreamingObjectNavPolicy
from streamnav.models.qwen35_kda.backbone import Qwen35KDABackbone
from streamnav.utils.seed import restore_rng, rng_state


def load_policy(config, training=False, preserve_master_weights=False):
    device = torch.device(config["device"])
    if device.type == "cuda":
        torch.cuda.set_device(device)
    path = Path(config.get("checkpoint") or config["model"]["checkpoint"]).resolve()
    model_cfg = config["model"]
    if model_cfg["action_dim"] != 4 or model_cfg["dtype"] not in ("bfloat16", "float32"):
        raise ValueError("Only four actions and bf16/fp32 precision are supported")
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
    )
    if model_cfg["freeze_vision_encoder"]:
        backbone.vision.requires_grad_(False)
    if training and model_cfg["gradient_checkpointing"]:
        backbone.vision.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False}
        )
    policy = StreamingObjectNavPolicy(backbone, model_cfg["value_hidden_dim"])
    policy.inference_compute_dtype = getattr(torch, model_cfg["dtype"])
    policy.loaded_checkpoint = str(path)
    heads = Path(path) / "actor_critic.safetensors"
    if heads.exists():
        policy.actor_critic.load_state_dict(load_file(str(heads)))
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
    source_hash = hashlib.sha256()
    source_files = []
    for folder in ("src", "services", "tools", "configs"):
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
        "source_tree_sha256": source_hash.hexdigest(),
        "source_archive": str(archive.resolve()),
        "source_archive_sha256": file_hash(archive),
        "qwen_revision": source_revision or "unknown-local-source",
        "qwen_source_config_sha256": layout_data["source_config_sha256"],
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


def restore_training(path, optimizer, scheduler, dagger, sources, config=None):
    # Only load checkpoints created by this project and trusted by the user.
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
            "success_distance",
            "forward_step",
            "turn_angle",
        ):
            if previous["habitat"].get(key) != config["habitat"].get(key):
                raise ValueError(f"Resume robot configuration changed: {key}; start a new run")
        for key in ("image_size", "freeze_vision_encoder", "action_dim", "value_hidden_dim"):
            if previous["model"][key] != config["model"][key]:
                raise ValueError(f"Resume model configuration changed: {key}")
        if previous["trainer"]["dagger"] != config["trainer"]["dagger"]:
            raise ValueError("Resume requires the same DAgger schedule")
        saved_manifest = json.loads((Path(path) / "manifest.json").read_text())
        for item in config["data"]["sources"]:
            manifest_path = item["manifest"]
            if saved_manifest["dataset_manifests"][manifest_path] != file_hash(manifest_path):
                raise ValueError(f"Training manifest changed since checkpoint: {manifest_path}")
    saved = torch.load(Path(path) / "optimizer.pt", map_location="cpu", weights_only=False)
    from streamnav.training.distributed import rank, world_size

    if saved.get("world_size", 1) != world_size():
        raise ValueError("Resume requires the same distributed world size")
    optimizer.load_state_dict(saved["optimizer"])
    scheduler.load_state_dict(saved["scheduler"])
    dagger.update = json.loads((Path(path) / "dagger_scheduler.json").read_text())["update"]
    per_rank = saved["rank_states"][rank()] if saved.get("rank_states") else saved
    for source, state in zip(sources, per_rank["sources"], strict=True):
        source.load_state_dict(state)
    restore_rng(per_rank["rng"])
    return saved["update"]
