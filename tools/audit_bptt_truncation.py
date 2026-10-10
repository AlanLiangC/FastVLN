"""Compare 100/200/full BPTT with identical forward chunks and real labels."""

import argparse
import json
from pathlib import Path

import torch
import torch.nn.functional as F
import yaml

from streamnav.contracts.state import LayerState
from streamnav.training.checkpoint import load_policy
from streamnav.training.rollout import replay_sequences
from streamnav.training.rollout_buffer import RecurrentRolloutBuffer, SequenceIndex


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--buffer", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="cuda:5")
    parser.add_argument("--frames", type=int, default=0)
    args = parser.parse_args()
    if args.frames and (args.frames < 200 or args.frames % 50):
        parser.error("Frames must be zero or a multiple of 50 of at least 200")
    config = yaml.safe_load((Path(args.checkpoint) / "resolved_config.yaml").read_text())
    config.update(checkpoint=args.checkpoint, device=args.device)
    policy = load_policy(config, training=True).eval()
    body = policy.backbone
    source = torch.load(args.buffer, map_location="cpu", weights_only=False)
    results = []
    original_forward = body.recurrent_forward
    controls = {"frames": 0, "truncate": 0}

    def forward(tokens, cache=None, mode="auto", lengths=None):
        # 50-frame body calls have >1000 tokens; the system prefill is short.
        if tokens.shape[1] > 1000:
            if controls["truncate"] and controls["frames"] % controls["truncate"] == 0:
                if controls["frames"] > 0:
                    cache = tuple(
                        LayerState(
                            s.conv.detach() if s.conv is not None else None,
                            s.recurrent.detach() if s.recurrent is not None else None,
                        )
                        for s in cache
                    )
            controls["frames"] += 50
        return original_forward(tokens, cache, mode, lengths)

    body.recurrent_forward = forward
    try:
        for env in range(source.num_envs):
            starts = sorted({0, *(t for e, t in source.resets if e == env)})
            begin, end = max(
                zip(starts, starts[1:] + [source.steps], strict=True), key=lambda x: x[1] - x[0]
            )
            length = (end - begin) // 50 * 50
            if args.frames:
                length = min(length, args.frames)
            if length < 200:
                continue
            goal = (
                source.initial_states[(env, 0)].instruction
                if begin == 0
                else source.resets[(env, begin)][1]
            )
            with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                initial = policy.start_episode(f"bptt-{env}", goal)
            buffer = RecurrentRolloutBuffer(length, 1, (1, 1, 3), length, source.beta)
            buffer.initial_states[(0, 0)] = initial
            buffer.visual_embeddings = source.visual_embeddings[
                begin : begin + length, env : env + 1
            ]
            labels = source.expert_actions[begin : begin + length, env].to(body.device)
            sequences = [SequenceIndex(0, 0, length)]
            reference_gradients = {}
            reference_logits = None
            samples = []
            for truncate in (0, 100, 200):
                policy.zero_grad(set_to_none=True)
                controls.update(frames=0, truncate=truncate)
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    logits, _ = replay_sequences(
                        policy, buffer, sequences, cache_embeddings=True, pack_frames=50
                    )
                    loss = F.cross_entropy(logits[:, 0], labels)
                assert controls["frames"] == length
                loss.backward()
                sample = {
                    "bptt_limit": truncate or length,
                    "state_detach_interval": truncate,
                    "il_loss": loss.item(),
                }
                if truncate == 0:
                    reference_logits = logits.detach().cpu()
                    reference_gradients = {
                        name: p.grad.detach().cpu()
                        for name, p in policy.named_parameters()
                        if p.grad is not None
                    }
                else:
                    torch.testing.assert_close(logits.cpu(), reference_logits, atol=0, rtol=0)
                    groups = {}
                    for name, parameter in policy.named_parameters():
                        if parameter.grad is None:
                            continue
                        group = (
                            "head"
                            if name.startswith("actor_critic.")
                            else "embedding"
                            if "embeddings." in name
                            else "gdn"
                            if ".mixer.original." in name
                            else "kda"
                            if ".mixer." in name
                            else "other_body"
                        )
                        stats = groups.setdefault(
                            group,
                            {
                                "reference_squared": 0.0,
                                "actual_squared": 0.0,
                                "delta_squared": 0.0,
                                "dot": 0.0,
                            },
                        )
                        expected = reference_gradients[name]
                        actual = parameter.grad.detach().cpu()
                        stats["reference_squared"] += expected.square().sum().item()
                        stats["actual_squared"] += actual.square().sum().item()
                        stats["delta_squared"] += (actual - expected).square().sum().item()
                        stats["dot"] += (actual * expected).sum().item()
                    for stats in groups.values():
                        stats["relative_l2_difference"] = (
                            stats["delta_squared"] / max(stats["reference_squared"], 1e-30)
                        ) ** 0.5
                        stats["cosine_similarity"] = stats["dot"] / max(
                            (stats["reference_squared"] * stats["actual_squared"]) ** 0.5, 1e-30
                        )
                    sample.update(logits_exactly_equal=True, gradients=groups)
                samples.append(sample)
                print(json.dumps({"env": env, "frames": length, **sample}), flush=True)
                del logits, loss
            results.append(
                {"env": env, "goal": goal, "start": begin, "frames": length, "samples": samples}
            )
            reference_gradients.clear()
    finally:
        body.recurrent_forward = original_forward
    report = {
        "checkpoint": args.checkpoint,
        "buffer": args.buffer,
        "records": results,
        "protocol": "Same real continuous episode and 50-frame chunks for all controls; only cache.detach at 100/200 boundaries changes. All per-frame teacher IL losses averaged once, no optimizer. Exact logits equality asserted. This isolates gradient truncation from BF16 chunk-boundary changes.",
        "limitations": "Two sequences and teacher IL only. Does not establish convergence or SR. A shorter final BPTT segment can affect 200-step comparisons when sequence length is 350.",
    }
    Path(args.output).write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
