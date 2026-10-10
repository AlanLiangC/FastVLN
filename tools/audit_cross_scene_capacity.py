"""Matched RGB/text STOP capacity controls across scenes and target splits.

Teacher viewpoint labels describe stopping, not semantic visibility. Diagnostic
starts/labels never enter autonomous validation or the active navigation run.
"""

import argparse
import hashlib
import json
from pathlib import Path

import torch
from audit_architecture_capacity import configuration, write_json
from audit_native_goal_pair_fit import FrameReadout, metrics
from safetensors.torch import load_file
from torch import nn
from torch.nn import functional as F

from streamnav.models.policy.streaming_policy import StreamingObjectNavPolicy
from streamnav.training.checkpoint import load_policy
from streamnav.utils.seed import seed_everything


def dataset(root):
    data = {}
    for split in ("train", "val_seen", "val_unseen"):
        raw = torch.load(root / f"{split}.pt", map_location="cpu", weights_only=False)
        data[split] = raw["clips"]
        if split == "train":
            data["train_new_yaw"] = raw["holdout_clips"]
    train_scenes = {c["scene"] for c in data["train"]}
    train_goals = {c["goal"] for c in data["train"]}
    for split in ("val_seen", "val_unseen"):
        assert not train_scenes.intersection(c["scene"] for c in data[split])
    assert not train_goals.intersection(c["goal"] for c in data["val_unseen"])
    for clips in data.values():
        assert len(clips) % 4 == 0
        for i in range(0, len(clips), 2):
            assert torch.equal(clips[i]["rgb"], clips[i + 1]["rgb"])
            assert clips[i]["goal"] != clips[i + 1]["goal"]
            assert (clips[i]["actions"].item() == 0) != (clips[i + 1]["actions"].item() == 0)
    return data


def score(logits, clips):
    labels = torch.cat([c["actions"] for c in clips]).to(logits.device)
    result = metrics(logits, labels)
    probabilities = logits.softmax(-1)[:, 0]
    positive = labels == 0
    result.update(
        frames=len(clips),
        scenes=len({c["scene"] for c in clips}),
        categories=len({c["category"] for c in clips}),
        stop_probability_on_positive=probabilities[positive].mean().item(),
        stop_probability_on_negative=probabilities[~positive].mean().item(),
    )
    paired_probabilities = probabilities.view(-1, 2)
    paired_labels = positive.view(-1, 2)
    gap = (paired_probabilities * torch.where(paired_labels, 1, -1)).sum(-1)
    result["same_rgb_goal_pair_ranking_accuracy"] = (
        ((gap > 0).float() + 0.5 * (gap == 0).float()).mean().item()
    )
    result["same_rgb_goal_pair_stop_probability_gap"] = gap.mean().item()
    return result


@torch.no_grad()
def extract(model, clips, batch_size):
    model.eval()
    chunks = []
    for start in range(0, len(clips), batch_size):
        with torch.autocast("cuda", dtype=torch.bfloat16):
            chunks.append(model(clips[start : start + batch_size], return_hidden=True).float())
    return torch.cat(chunks)


@torch.no_grad()
def evaluate(model, data, batch_size):
    model.eval()
    scores, outputs = {}, {}
    for split, clips in data.items():
        logits = model.actor(extract(model, clips, batch_size)).float()
        scores[split], outputs[split] = score(logits, clips), logits.cpu()
    return scores, outputs


@torch.no_grad()
def deployed_readout(model, checkpoint, data):
    # Reuse the trained backbone with its actual NAV, goal embeddings and
    # actor, preserving production's separate prefill/frame calls. No second
    # model copy or gradient/optimizer changes.
    policy = StreamingObjectNavPolicy(model.body).eval()
    policy.actor_critic.load_state_dict(
        load_file(str(Path(checkpoint) / "actor_critic.safetensors"))
    )
    scores = {}
    for split, clips in data.items():
        logits = []
        for start in range(0, len(clips), 2):
            batch = clips[start : start + 2]
            with torch.autocast("cuda", dtype=torch.bfloat16):
                states = [policy.start_episode(c["episode"], c["goal"]) for c in batch]
                out, _, _ = policy.forward_batch(
                    torch.cat([c["rgb"] for c in batch]),
                    states,
                    visual_embeddings=torch.cat([c["visual"] for c in batch]).to(model.nav.device),
                )
            logits.append(out)
        scores[split] = score(torch.cat(logits), clips)
    return scores


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True, help="Configuration anchor")
    parser.add_argument("--trained-checkpoint")
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--variant",
        choices=["native", "native_no_rope", "kda", "kda_raw", "trained", "calibrated"],
        required=True,
    )
    parser.add_argument("--seed", type=int, default=10913)
    parser.add_argument("--steps", type=int, default=60)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--probe-steps", type=int, default=500)
    parser.add_argument("--initialize-from-probe", action="store_true")
    parser.add_argument("--natural-readout", action="store_true")
    args = parser.parse_args()
    assert args.batch_size % 4 == 0
    seed_everything(args.seed)
    torch.cuda.set_device(args.device)
    data = dataset(Path(args.data_root))
    construction = argparse.Namespace(**vars(args))
    if args.variant in ("trained", "calibrated"):
        construction.variant = "kda"
    if args.variant == "native_no_rope":
        construction.variant = "native"
        construction.disable_language_rope = True
    if args.variant == "kda_raw":
        construction.variant = "kda"
    model = FrameReadout(construction)
    if args.variant == "kda_raw":
        model.body.kda_output_norm = False
        for layer in model.body.layers:
            if hasattr(layer.mixer, "output_norm"):
                layer.mixer.output_norm = False
    if args.variant in ("trained", "calibrated"):
        if not args.trained_checkpoint:
            raise ValueError("trained variant requires an immutable checkpoint")
        config = configuration(args.trained_checkpoint, args.device)
        trained = load_policy(config, training=True)
        model.body = trained.backbone
        del trained
    shared_init = hashlib.sha256()
    for value in (model.nav, *model.actor.state_dict().values()):
        shared_init.update(value.detach().float().cpu().contiguous().numpy().tobytes())
    report = {
        "protocol": "RGB/text only. Balanced teacher STOP/non-STOP goal pairs at official viewpoints, negative distances >=0.5m. Labels are stopping decisions, not semantic visibility. Independent training/validation scenes; val_unseen goal strings absent from diagnostic training. Same fresh NAV/actor initialization across variants. Frozen-feature linear probe then equal-budget whole-backbone pure IL; original NAV+actor separately evaluated for trained checkpoint. No semantic inputs, no heldout optimization or checkpoint selection, no autonomous SR claim.",
        "variant": args.variant,
        "seed": args.seed,
        "trained_checkpoint": args.trained_checkpoint,
        "shared_nav_actor_initialization_sha256": shared_init.hexdigest(),
        "steps": args.steps,
        "batch_size": args.batch_size,
        "backbone_lr": 1e-5,
        "actor_lr": 2.5e-4,
        "initialize_actor_from_probe": args.initialize_from_probe,
        "readout": "Official user-image-goal chat, final assistant prefix token"
        if args.natural_readout
        else "Image appended after assistant prefix, learned NAV plus goal embedding mean",
        "dataset": {
            split: {
                "frames": len(clips),
                "scenes": sorted({c["scene"] for c in clips}),
                "goals": sorted({c["goal"] for c in clips}),
                "goals_seen_in_diagnostic_fit": sorted(
                    {c["goal"] for c in clips}.intersection(c["goal"] for c in data["train"])
                ),
            }
            for split, clips in data.items()
        },
    }
    if args.variant == "trained":
        report["original_navigation_head"] = deployed_readout(model, args.trained_checkpoint, data)
        print(
            json.dumps({"original_navigation_head": report["original_navigation_head"]}), flush=True
        )
        write_json(args.output, report)
    features = {
        split: extract(model, clips, args.batch_size).detach() for split, clips in data.items()
    }
    labels = torch.cat([c["actions"] for c in data["train"]]).to(args.device)
    # A standalone probe never mutates the matched end-to-end actor or backbone.
    probe = nn.Linear(features["train"].shape[-1], 6, device=args.device)
    probe.load_state_dict(model.actor.state_dict())
    optimizer = torch.optim.Adam(probe.parameters(), lr=2.5e-4, eps=1e-5)
    for _ in range(args.probe_steps):
        optimizer.zero_grad(set_to_none=True)
        F.cross_entropy(probe(features["train"]), labels).backward()
        optimizer.step()
    with torch.no_grad():
        report["frozen_feature_probe"] = {
            split: score(probe(features[split]), clips) for split, clips in data.items()
        }
    torch.save(
        {"features": {k: v.cpu() for k, v in features.items()}, "probe": probe.state_dict()},
        Path(args.output).with_suffix(".features.pt"),
    )
    print(json.dumps({"frozen_feature_probe": report["frozen_feature_probe"]}), flush=True)
    write_json(args.output, report)
    if args.steps == 0:
        return
    if args.initialize_from_probe:
        model.actor.load_state_dict(probe.state_dict())
    del features, probe, optimizer
    parameters = [p for p in model.body.parameters() if p.requires_grad] + [model.nav]
    optimizer = torch.optim.Adam(
        [{"params": parameters, "lr": 1e-5}, {"params": model.actor.parameters(), "lr": 2.5e-4}],
        eps=1e-5,
        fused=True,
    )
    # Keep both counterfactual poses/goals together, and shuffle only complete
    # four-frame scene groups. Sampler independent of model initialization RNG.
    generator = torch.Generator().manual_seed(10921)
    groups = len(data["train"]) // 4
    group_batch = args.batch_size // 4
    order, cursor = [], 0
    history = []
    for step in range(1, args.steps + 1):
        if cursor + group_batch > len(order):
            order, cursor = torch.randperm(groups, generator=generator).tolist(), 0
        batch = [
            data["train"][4 * g + i] for g in order[cursor : cursor + group_batch] for i in range(4)
        ]
        cursor += group_batch
        model.train()
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            loss = F.cross_entropy(
                model(batch), torch.cat([c["actions"] for c in batch]).to(args.device)
            )
        loss.backward()
        norm = torch.nn.utils.clip_grad_norm_(
            parameters + list(model.actor.parameters()), 0.2, error_if_nonfinite=True
        )
        optimizer.step()
        if step % 10 == 0 or step == args.steps:
            row = {"step": step, "minibatch_loss": loss.item(), "grad_norm": norm.item()}
            history.append(row)
            report["history"] = history
            print(json.dumps(row), flush=True)
            write_json(args.output, report)
    report["finetuned"], outputs = evaluate(model, data, args.batch_size)
    torch.save(outputs, Path(args.output).with_suffix(".outputs.pt"))
    write_json(args.output, report)
    print(json.dumps({"finetuned": report["finetuned"]}), flush=True)


if __name__ == "__main__":
    main()
