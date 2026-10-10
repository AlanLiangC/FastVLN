"""Adapt six converted mixers to native Qwen features across chat contexts.

Only training RGB/text and frozen native hidden states are used. No action,
semantic, validation labels, navigation optimizer or extra navigation loss.
All GDN/MLP/embedding/vision parameters remain unchanged.
"""

import argparse
import json
import os
import shutil
import tempfile
from pathlib import Path

import torch
import yaml
from audit_architecture_capacity import write_json
from safetensors.torch import save_file
from torch.nn import functional as F

from streamnav.data.manifest import file_hash
from streamnav.training.checkpoint import load_policy
from streamnav.utils.seed import seed_everything


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--native-features", required=True)
    parser.add_argument("--candidate", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="cuda:5")
    parser.add_argument("--steps", type=int, default=90)
    parser.add_argument("--preserve-contrasts", action="store_true")
    args = parser.parse_args()
    destination = Path(args.candidate)
    if destination.exists():
        raise FileExistsError(f"Refusing to overwrite {destination}")
    torch.set_num_threads(1)
    seed_everything(11921)
    torch.cuda.set_device(args.device)
    config = yaml.safe_load(Path(args.config).read_text())
    config.update(device=args.device, checkpoint=None)
    policy = load_policy(config, training=True).eval()
    policy.requires_grad_(False)
    indices = [3, 7, 11, 15, 19, 23]
    for index in indices:
        policy.backbone.layers[index].mixer.requires_grad_(True)
    parameters = [p for p in policy.parameters() if p.requires_grad]
    optimizer = torch.optim.Adam(parameters, lr=1e-4, eps=1e-8)
    path = Path(args.data_root) / "train.pt"
    clips = torch.load(path, map_location="cpu", weights_only=False)["clips"]
    features = torch.load(args.native_features, map_location="cpu", weights_only=False)["train"]
    frames = [1, 4, 16]
    train_features = torch.cat([features[n] for n in frames])
    scale = train_features.std(0).clamp_min(0.25).to(args.device)
    scales = {n: features[n].std(0).clamp_min(0.25).to(args.device) for n in frames}
    group_size = 4 if args.preserve_contrasts else 2
    assert len(clips) % 2 == 0 and features[1].shape[0] == len(clips)
    for i in range(0, len(clips), 2):
        assert clips[i]["goal"] != clips[i + 1]["goal"]
        assert torch.equal(clips[i]["visual"], clips[i + 1]["visual"])
    if args.preserve_contrasts:
        assert len(clips) % 4 == 0
        for i in range(0, len(clips), 4):
            assert clips[i]["goal"] == clips[i + 2]["goal"]
            assert clips[i + 1]["goal"] == clips[i + 3]["goal"]
    readouts = []
    hook = policy.actor_critic.register_forward_pre_hook(
        lambda _, inputs: readouts.append(inputs[0])
    )

    # Keep state propagation explicit so this is the production streaming path.
    def stream(pair, count, gradients):
        batch = clips[pair * group_size : (pair + 1) * group_size]
        rgb = torch.cat([c["rgb"] for c in batch]).to(args.device)
        visual = torch.cat([c["visual"] for c in batch]).to(args.device)
        guard = torch.enable_grad() if gradients else torch.no_grad()
        with (
            guard,
            torch.autocast("cuda", dtype=torch.bfloat16),
            policy.backbone.reuse_token_embeddings(),
        ):
            states = [
                policy.start_episode(str(pair * 2 + i), c["goal"]) for i, c in enumerate(batch)
            ]
            for _ in range(count):
                _, _, states = policy.forward_batch(rgb, states, visual_embeddings=visual)
                output = readouts.pop()
        return output.float()

    def losses(output, target, count):
        normalization = scales[count] if args.preserve_contrasts else scale
        weighted = ((output - target) / normalization).square().mean()
        relative = F.mse_loss(output, target) / target.square().mean().clamp_min(1e-6)
        loss = weighted + 0.2 * relative
        if args.preserve_contrasts:
            # Preserve native responses to changing text with fixed RGB, and
            # changing RGB with fixed text. No class/action labels are accessed.
            for actual, expected in (
                (output[::2] - output[1::2], target[::2] - target[1::2]),
                (output[:2] - output[2:], target[:2] - target[2:]),
            ):
                loss = loss + F.mse_loss(actual, expected) / expected.square().mean().clamp_min(
                    1e-3
                )
        return loss, relative

    generator = torch.Generator().manual_seed(11923)
    fixed_pairs = [0, 6, 12, 18] if args.preserve_contrasts else [0, 12, 24, 36]

    def measure():
        values = {}
        for count in frames:
            values[count] = sum(
                losses(
                    stream(pair, count, False),
                    features[count][pair * group_size : (pair + 1) * group_size].to(args.device),
                    count,
                )[1].item()
                for pair in fixed_pairs
            ) / len(fixed_pairs)
        return values

    before = measure()
    records = []
    for step in range(args.steps):
        count = frames[step % len(frames)]
        pair = torch.randint(len(clips) // group_size, (), generator=generator).item()
        optimizer.zero_grad(set_to_none=True)
        output = stream(pair, count, True)
        target = features[count][pair * group_size : (pair + 1) * group_size].to(args.device)
        loss, relative = losses(output, target, count)
        if not torch.isfinite(loss):
            raise FloatingPointError("Nonfinite conversion loss")
        loss.backward()
        norm = torch.nn.utils.clip_grad_norm_(parameters, 1.0, error_if_nonfinite=True)
        optimizer.step()
        record = {
            "step": step + 1,
            "context_frames": count,
            "pair": pair,
            "loss": loss.item(),
            "relative_mse": relative.item(),
            "grad_norm": norm.item(),
        }
        records.append(record)
        print(json.dumps(record), flush=True)
    after = measure()
    hook.remove()
    metadata = {
        "protocol": __doc__,
        "source": config["model"]["checkpoint"],
        "train_data": str(path),
        "native_features": args.native_features,
        "native_features_sha256": file_hash(Path(args.native_features)),
        "train_frames": len(clips),
        "steps": args.steps,
        "trainable_layers": indices,
        "trainable_parameters": sum(p.numel() for p in parameters),
        "contexts": frames,
        "preserve_rgb_and_goal_contrasts": args.preserve_contrasts,
        "fixed_training_relative_mse_before": before,
        "fixed_training_relative_mse_after": after,
        "navigation_optimizer_included": False,
        "not_promoted": True,
        "records": records,
    }
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=".temporal-distillation-", dir=destination.parent))
    try:
        save_file(
            {k: v.detach().cpu().contiguous() for k, v in policy.backbone.state_dict().items()},
            str(temporary / "model.safetensors"),
        )
        policy.backbone.config.save_pretrained(temporary)
        policy.backbone.tokenizer.save_pretrained(temporary / "tokenizer")
        source = Path(config["model"]["checkpoint"])
        shutil.copy2(source / "kda_layout.json", temporary / "kda_layout.json")
        shutil.copy2(
            source / "conversion_calibration.json", temporary / "conversion_calibration.json"
        )
        config["model"]["checkpoint"] = str(destination)
        config["checkpoint"] = str(destination)
        config["run_dir"] = "runs/streamnav_conversion_temporal_candidate_20261008/ealm"
        (temporary / "resolved_config.yaml").write_text(yaml.safe_dump(config))
        (temporary / "conversion_distillation.json").write_text(
            json.dumps(metadata, indent=2) + "\n"
        )
        os.rename(temporary, destination)
    finally:
        if temporary.exists():
            shutil.rmtree(temporary)
    write_json(args.output, metadata)


if __name__ == "__main__":
    main()
