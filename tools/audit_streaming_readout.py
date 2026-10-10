"""Check a static diagnostic head through the actual streaming readout.

Repeated stationary views test context drift, not exploration or navigation SR.
The head is the saved training-only Adam probe; nothing enters the active run.
"""

import argparse
import json
from pathlib import Path

import torch
import yaml
from audit_architecture_capacity import write_json
from audit_cross_scene_capacity import dataset, score

from streamnav.models.qwen35_kda.cache import state_bytes
from streamnav.training.checkpoint import load_policy


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="cuda:6")
    parser.add_argument("--frames", type=int, default=16)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--include-train", action="store_true")
    args = parser.parse_args()
    torch.set_num_threads(1)
    torch.cuda.set_device(args.device)
    config = yaml.safe_load(Path(args.config).read_text())
    config.update(device=args.device, checkpoint=None)
    policy = load_policy(config, training=True).eval()
    root = Path(args.data_root)
    features = torch.load(
        root / "fit_natural_calibrated.features.pt", map_location="cpu", weights_only=False
    )
    policy.actor_critic.actor.load_state_dict(features["probe"])
    clips = dataset(root)
    records = {}
    feature_records = {}
    splits = (
        ("train", "val_seen", "val_unseen") if args.include_train else ("val_seen", "val_unseen")
    )
    for split in splits:
        snapshots = {i: [] for i in (1, 4, args.frames)}
        feature_snapshots = {i: [] for i in snapshots}
        hidden = []
        hook = policy.actor_critic.register_forward_pre_hook(
            lambda _, inputs: hidden.append(inputs[0].float().cpu())
        )
        first_readouts = []
        final_readouts = []
        sizes = []
        for start in range(0, len(clips[split]), args.batch_size):
            batch = clips[split][start : start + args.batch_size]
            rgb = torch.cat([c["rgb"] for c in batch]).to(args.device)
            visual = torch.cat([c["visual"] for c in batch]).to(args.device)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                states = [
                    policy.start_episode(str(start + i), c["goal"]) for i, c in enumerate(batch)
                ]
                for frame in range(1, args.frames + 1):
                    logits, _, states = policy.forward_batch(rgb, states, visual)
                    if frame in snapshots:
                        snapshots[frame].append(logits.cpu())
                    readout = hidden.pop()
                    if frame in feature_snapshots:
                        feature_snapshots[frame].append(readout)
                    if frame == 1:
                        first_readouts.append(readout)
                    if frame == args.frames:
                        final_readouts.append(readout)
                    sizes.extend(state_bytes(s) for s in states)
        hook.remove()
        first = torch.cat(first_readouts)
        final = torch.cat(final_readouts)
        reference = features["features"][split]
        actor = policy.actor_critic.actor
        reference_logits = torch.nn.functional.linear(
            reference, actor.weight.cpu(), actor.bias.cpu()
        )
        records[split] = {
            "whole_zero_state_reference": score(reference_logits, clips[split]),
            "streaming_frames": {
                frame: score(torch.cat(values), clips[split]) for frame, values in snapshots.items()
            },
            "first_vs_whole_relative_l2_mean": (
                (first - reference).norm(dim=-1) / reference.norm(dim=-1).clamp_min(1e-6)
            )
            .mean()
            .item(),
            "final_vs_first_relative_l2_mean": (
                (final - first).norm(dim=-1) / first.norm(dim=-1).clamp_min(1e-6)
            )
            .mean()
            .item(),
            "state_bytes_min": min(sizes),
            "state_bytes_max": max(sizes),
        }
        feature_records[split] = {
            frame: torch.cat(rows) for frame, rows in feature_snapshots.items()
        }
        print(json.dumps({split: records[split]}), flush=True)
    torch.save(feature_records, Path(args.output).with_suffix(".features.pt"))
    write_json(
        args.output,
        {
            "protocol": __doc__,
            "config": args.config,
            "initialization": config["model"]["checkpoint"],
            "head": "fit_natural_calibrated.features.pt/probe (500-step training-only Adam)",
            "frames": args.frames,
            "records": records,
            "limitations": [
                "Same prior diagnostic scenes; not an independent generalization estimate.",
                "Stationary repeated views; not an exploration or route-memory benchmark.",
                "Frozen initialization + diagnostic head; not the active navigation actor.",
            ],
        },
    )


if __name__ == "__main__":
    main()
