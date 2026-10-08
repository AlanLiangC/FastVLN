"""Localize single-frame batch sensitivity without modifying the checkpoint."""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf
from PIL import Image

from streamnav.models.qwen35_kda.cache import clone_state
from streamnav.training.checkpoint import load_policy


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--image", required=True)
    parser.add_argument("--goal", default="Find a couch.")
    parser.add_argument("--device", default="cuda:3")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    with initialize_config_dir(version_base=None, config_dir=str(Path("configs").resolve())):
        cfg = compose(
            config_name="config",
            overrides=[
                f"checkpoint={args.checkpoint}",
                f"device={args.device}",
            ],
        )
    policy = load_policy(
        OmegaConf.to_container(cfg, resolve=True), preserve_master_weights=True
    ).eval()
    rgb = torch.from_numpy(np.asarray(Image.open(args.image).convert("RGB")).copy())
    traces = {}

    def hook(name):
        def capture(module, inputs, output):
            tensor = output.pooler_output if name == "vision" else output[0]
            if name == "vision":
                tensor = tensor.reshape(-1, 135, tensor.shape[-1])
            traces[name] = tensor[0].detach().float().cpu()

        return capture

    hooks = [policy.backbone.vision.register_forward_hook(hook("vision"))]
    hooks.extend(
        layer.register_forward_hook(hook(f"layer_{i:02d}"))
        for i, layer in enumerate(policy.backbone.layers)
    )
    reports = {}
    try:
        vision_forward = policy.backbone.vision.forward
        for name, dtype in (
            ("bf16_default", torch.bfloat16),
            ("bf16_full_reduction", torch.bfloat16),
            ("bf16_fp32_vision", torch.bfloat16),
            ("fp32", torch.float32),
        ):
            torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = (
                name == "bf16_default"
            )

            def full_precision_vision(*args, **kwargs):
                with torch.autocast("cuda", enabled=False):
                    return vision_forward(*args, **kwargs)

            policy.backbone.vision.forward = (
                full_precision_vision if name == "bf16_fp32_vision" else vision_forward
            )
            with torch.autocast(
                "cuda", dtype=dtype, enabled=dtype != torch.float32, cache_enabled=False
            ):
                state = policy.start_episode("probe", args.goal)
                traces.clear()
                solo, _, _ = policy.forward_batch(rgb.unsqueeze(0), [clone_state(state)])
                single_trace = dict(traces)
                traces.clear()
                batch, _, _ = policy.forward_batch(
                    rgb.unsqueeze(0).repeat(2, 1, 1, 1), [clone_state(state), clone_state(state)]
                )
            layers = {}
            for layer_name, reference in single_trace.items():
                difference = traces[layer_name] - reference
                layers[layer_name] = {
                    "max_abs": difference.abs().max().item(),
                    "relative_rms": (
                        difference.square().mean().sqrt()
                        / reference.square().mean().sqrt().clamp_min(1e-9)
                    ).item(),
                }
            report = {
                "solo_probabilities": solo.softmax(-1)[0].tolist(),
                "batch_probabilities": batch.softmax(-1).tolist(),
                "probability_max_difference": (solo.softmax(-1)[0] - batch.softmax(-1)[0])
                .abs()
                .max()
                .item(),
                "layers": layers,
            }
            reports[name] = report
            print(json.dumps({"mode": name, "dtype": str(dtype), **report}), flush=True)
    finally:
        for handle in hooks:
            handle.remove()
        Path(args.output).write_text(json.dumps(reports, indent=2))


if __name__ == "__main__":
    main()
