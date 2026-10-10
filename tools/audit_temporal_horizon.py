"""Read-only input-gradient, visual-history and decay audit on real saved rollouts."""

import argparse
import json
import math
from dataclasses import replace
from pathlib import Path

import torch
import torch.nn.functional as F
import yaml

from streamnav.contracts.state import LayerState
from streamnav.training.checkpoint import load_policy
from streamnav.training.rollout import replay_sequences
from streamnav.training.rollout_buffer import RecurrentRolloutBuffer, SequenceIndex


def longest_episode(buffer, env):
    starts = sorted({0, *(step for e, step in buffer.resets if e == env)})
    return max(zip(starts, starts[1:] + [buffer.steps], strict=True), key=lambda x: x[1] - x[0])


def gradient_summary(gradient):
    norms = gradient[:, 0].detach().double().flatten(1).norm(dim=1).flip(0).cpu()
    energy = norms.square()
    total = energy.sum().item()
    cumulative = energy.cumsum(0) / max(total, 1e-300)
    return {
        "norm_by_lag": norms.tolist(),
        "energy_fraction_from_lag_at_least": {
            str(lag): energy[lag:].sum().item() / max(total, 1e-300)
            for lag in (1, 2, 4, 8, 16, 32, 64, 100, 200)
        },
        "lag_containing_99_percent_energy": int((cumulative < 0.99).sum()),
        "finite": bool(torch.isfinite(gradient).all()),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--buffer", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--max-frames", type=int, default=400)
    args = parser.parse_args()
    config = yaml.safe_load((Path(args.checkpoint) / "resolved_config.yaml").read_text())
    config.update(checkpoint=args.checkpoint, device=args.device)
    policy = load_policy(config, preserve_master_weights=True).eval().requires_grad_(False)
    body = policy.backbone
    source = torch.load(args.buffer, map_location="cpu", weights_only=False)
    if source.visual_embeddings is None:
        raise ValueError("A real rollout with frozen visual embeddings is required")
    torch.manual_seed(9022)
    records = []
    for env in range(source.num_envs):
        start, stop = longest_episode(source, env)
        start = max(start, stop - args.max_frames)
        # If a max-length crop moves the start, reconstruct its true prefix.
        episode_start = max([0, *(t for e, t in source.resets if e == env and t <= start)])
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            if episode_start == 0:
                initial = source.initial_states[(env, 0)]
                initial = replace(
                    initial,
                    kda_cache=tuple(
                        LayerState(
                            s.conv.to(body.device) if s.conv is not None else None,
                            s.recurrent.to(body.device) if s.recurrent is not None else None,
                        )
                        for s in initial.kda_cache
                    ),
                )
                if initial.step_index == 0:
                    initial = policy.start_episode(initial.episode_id, initial.instruction)
            else:
                initial = policy.start_episode(*source.resets[(env, episode_start)])
            for t in range(episode_start, start):
                _, _, states = policy.forward_batch(
                    source.observations[t, env].unsqueeze(0),
                    [initial],
                    source.visual_embeddings[t, env].unsqueeze(0).to(body.device),
                )
                initial = states[0]
        length = stop - start
        buffer = RecurrentRolloutBuffer(length, 1, (1, 1, 3), length, source.beta)
        buffer.initial_states[(0, 0)] = initial
        visual = source.visual_embeddings[start:stop, env : env + 1].to(body.device)
        buffer.visual_embeddings = visual.detach().clone().requires_grad_(True)
        sequences = [SequenceIndex(0, 0, length)]
        readouts = []
        hook = policy.actor_critic.register_forward_pre_hook(lambda _, x: readouts.append(x[0]))
        try:
            with torch.autocast("cuda", dtype=torch.bfloat16):
                logits, _ = replay_sequences(
                    policy, buffer, sequences, cache_embeddings=True, pack_frames=64
                )
                teacher = source.expert_actions[stop - 1, env].to(body.device).unsqueeze(0)
                loss = F.cross_entropy(logits[-1], teacher)
                feature = readouts[-1].float()
                projection = torch.randn_like(feature) / math.sqrt(feature.shape[-1])
            loss.backward(retain_graph=True)
            actor_gradient = gradient_summary(buffer.visual_embeddings.grad)
            buffer.visual_embeddings.grad = None
            (feature * projection).sum().backward()
            feature_gradient = gradient_summary(buffer.visual_embeddings.grad)
            baseline_logits = logits[-1].detach()
            baseline_feature = feature.detach()
            del logits, loss, feature
            readouts.clear()
            buffer.visual_embeddings = visual
            controls = []
            with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                for keep in (1, 4, 16, 64, 100, length):
                    if keep > length:
                        continue
                    modified = visual.clone()
                    modified[: length - keep] = 0
                    buffer.visual_embeddings = modified
                    readouts.clear()
                    altered, _ = replay_sequences(
                        policy, buffer, sequences, cache_embeddings=True, pack_frames=64
                    )
                    controls.append(
                        {
                            "recent_frames_retained": keep,
                            "older_visual_embeddings_zeroed": length - keep,
                            "final_action_probability_tv": float(
                                (altered[-1].softmax(-1) - baseline_logits.softmax(-1)).abs().sum()
                                / 2
                            ),
                            "final_readout_relative_l2_change": float(
                                (readouts[-1].float() - baseline_feature).norm()
                                / baseline_feature.norm().clamp_min(1e-20)
                            ),
                        }
                    )
        finally:
            hook.remove()
        # Gate-only bounds exclude delta overwrites and inter-layer routes.
        gate_records = [[] for _ in body.layers]
        hooks = []

        def capture(layer_index, gdn=None):
            def record(_, inputs, output):
                raw = output.detach().float()
                if gdn is None:
                    mixer = body.layers[layer_index].mixer
                    g = -F.softplus(raw).reshape(1, -1, mixer.num_heads, mixer.head_dim)
                    log_bound = g.amax(-1).sum(1).amax(-1)
                else:
                    g = -gdn.A_log.float().exp() * F.softplus(raw + gdn.dt_bias.float())
                    log_bound = g.sum(1).amax(-1)
                gate_records[layer_index].append(float(log_bound))

            return record

        for i, layer in enumerate(body.layers):
            mixer = layer.mixer
            module = mixer.decay_proj if hasattr(mixer, "decay_proj") else mixer.original.in_proj_a
            hooks.append(
                module.register_forward_hook(
                    capture(i, None if hasattr(mixer, "decay_proj") else mixer.original)
                )
            )
        try:
            state = initial
            with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                for frame in visual[:, 0]:
                    _, _, states = policy.forward_batch(
                        buffer.observations[0], [state], frame.unsqueeze(0)
                    )
                    state = states[0]
        finally:
            for h in hooks:
                h.remove()
        gates = []
        for i, values in enumerate(gate_records):
            half_lives = math.log(2) / -torch.tensor(values, dtype=torch.float64)
            gates.append(
                {
                    "layer": i,
                    "mixer": type(body.layers[i].mixer).__name__,
                    "slowest_head_gate_only_half_life_frames_p05_p50_p95": half_lives.quantile(
                        torch.tensor([0.05, 0.5, 0.95], dtype=torch.float64)
                    ).tolist(),
                }
            )
        record = {
            "env": env,
            "start": start,
            "stop": stop,
            "frames": length,
            "goal": initial.instruction,
            "last_teacher_action": int(teacher),
            "last_actor_ce_gradient": actor_gradient,
            "last_random_readout_gradient": feature_gradient,
            "visual_history_controls": controls,
            "gate_only_bounds": gates,
        }
        records.append(record)
        print(
            json.dumps(
                {
                    "env": env,
                    "frames": length,
                    "actor_99_percent_lag": actor_gradient["lag_containing_99_percent_energy"],
                    "readout_99_percent_lag": feature_gradient["lag_containing_99_percent_energy"],
                    "history_controls": controls,
                }
            ),
            flush=True,
        )
    report = {
        "checkpoint": args.checkpoint,
        "buffer": args.buffer,
        "records": records,
        "protocol": "Longest continuous episode per environment from real on-policy saved frames. Read-only FP32 master weights/BF16 compute. Only cached visual inputs require gradients. Loss on final frame only, no optimizer and no SR evaluation. Zeroing old visual embeddings preserves text and token layout; this is an off-manifold sensitivity intervention.",
        "limitations": "Local gradients and two trajectories do not establish a universal memory horizon or future architectural ceiling. Gate-only half-lives bound direct state decay and exclude delta overwrites and inter-layer nonlinear routes. Longer-sequence learning benefits require matched training experiments.",
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
