"""Load current point-CoT adapters on the pinned official Qwen3.5 base."""

from pathlib import Path

import torch
import yaml


def load_policy(config, training=False, preserve_master_weights=False):
    from streamnav.models.point_cot.policy import Qwen35PointCoTPolicy
    from streamnav.training.pointcot_trainer import load_adapters

    resolved = dict(config)
    checkpoint = config.get("checkpoint")
    if checkpoint and Path(checkpoint, "resolved_config.yaml").exists():
        saved = yaml.safe_load(Path(checkpoint, "resolved_config.yaml").read_text())
        resolved["model"] = saved["model"]
    if resolved["model"].get("family") != "qwen35_pointcot":
        raise ValueError("Only the current qwen35_pointcot architecture is supported")
    if str(config["device"]).startswith("cuda"):
        torch.cuda.set_device(torch.device(config["device"]))
    policy = Qwen35PointCoTPolicy(resolved)
    if checkpoint:
        load_adapters(policy, checkpoint)
    return policy.train(training)
