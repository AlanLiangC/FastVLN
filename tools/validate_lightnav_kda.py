"""Real-weight checks: native grounding, recurrence, probability replay and gradients."""

import argparse
import json
from pathlib import Path

import torch
import yaml

from streamnav.data.schema import HabitatEpisodeSource
from streamnav.envs.habitat_client import HabitatClient
from streamnav.models.lightnav_kda.policy import LightNavKDAPolicy
from streamnav.training.lightnav_trainer import cache_copy
from streamnav.utils.seed import seed_everything


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/lightnav_kda.yaml")
    parser.add_argument("--output", default="runtime/reports/lightnav_kda_validation_20261011")
    args = parser.parse_args()
    cfg = yaml.safe_load(Path(args.config).read_text())
    cfg["device"] = "cuda:0"
    cfg["habitat"]["gpu_device_id"] = 0
    seed_everything(cfg["seed"])
    policy = LightNavKDAPolicy(cfg).eval()
    env = HabitatClient(cfg["habitat"])
    source = HabitatEpisodeSource(cfg["data"]["sources"][0]["manifest"], seed=cfg["seed"])
    try:
        episode = source.sample_episode()
        observation = env.reset(episode)
        state = policy.start_episode(episode.uid, episode.goal_text)
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            feature, frame = policy.frame_features(observation["rgb"], episode.goal_text)
            native = policy.native.model.language_model
            ids = feature.ids
            mask = ids == policy.native.config.video_token_id
            embeddings = native.embed_tokens(ids)
            embeddings = embeddings.masked_scatter(
                mask.unsqueeze(-1), feature.visual.to(embeddings.dtype)
            )
            with torch.autocast(
                "cuda", enabled=policy.language_compute_dtype != torch.float32, dtype=torch.bfloat16
            ):
                expected = native(
                    inputs_embeds=embeddings,
                    position_ids=feature.positions,
                    visual_pos_masks=mask,
                    deepstack_visual_embeds=list(feature.deepstack),
                    use_cache=False,
                ).last_hidden_state[:, -1]
            context = policy.begin(feature, state)
            hidden_error = (expected.float() - context.hidden.float()).abs().max().item()
            native_logits = policy.output_projection(expected, policy.apos_rows)[0]
            native_point = native_logits.argmax().item()
            native_probability_error = (
                (native_logits.softmax(-1) - policy.point_logits(context, "apos").softmax(-1))
                .abs()
                .max()
                .item()
            )
            step = policy.step(feature, state)
            tokens = (step.apos, step.opos, step.action)
            replay = policy.step(feature, state, tokens=tokens)
            replay_error = (step.log_prob - replay.log_prob).abs().item()
            teacher_feature, _ = policy.frame_features(
                observation["rgb"], episode.goal_text, history=[frame] * 8
            )
            teacher_points = policy.teacher_points(teacher_feature, state, step.apos)
            teacher_valid = teacher_feature.positions.shape[-1] == teacher_feature.ids.shape[
                -1
            ] and all(
                bool(torch.isfinite(p).all()) and abs(p.sum().item() - 1) < 1e-4
                for p in teacher_points
            )
            snapshot = cache_copy(step.state, policy.device)
            before = [
                s.recurrent.clone() if s.recurrent is not None else None for s in snapshot.kda_cache
            ]
            following = policy.step(feature, snapshot, tokens=tokens)
            cache_unchanged = all(
                torch.equal(a, s.recurrent)
                for a, s in zip(before, snapshot.kda_cache, strict=True)
                if a is not None
            )
            sizes = [
                sum(s.recurrent.numel() for s in entry.kda_cache if s.recurrent is not None)
                for entry in (step.state, following.state)
            ]
            for memory in policy.memories.values():
                memory.residual_scale.fill_(0.02)
            remembered = policy.step(feature, snapshot, tokens=tokens)
            reset = policy.step(feature, state, tokens=tokens)
            history_effect = (remembered.apos_logits - reset.apos_logits).abs().max().item()
        with torch.autocast("cuda", dtype=torch.bfloat16):
            differentiable = policy.step(feature, state, tokens=tokens)
            loss = -differentiable.log_prob + differentiable.value.square()
        loss.backward()
        frozen_grad = any(
            p.grad is not None for p in policy.native.parameters() if not p.requires_grad
        )
        nonfinite = any(
            not torch.isfinite(p.grad).all() for p in policy._all_trainable if p.grad is not None
        )
        gradients = {"lora": 0.0, "memory": 0.0, "action": 0.0}
        for name, p in policy.named_parameters():
            if p.grad is not None:
                group = (
                    "memory"
                    if name.startswith("memories")
                    else ("action" if name.startswith("action_") else "lora")
                )
                if group in gradients:
                    gradients[group] += p.grad.float().norm().item()
        result = {
            "parameters": policy.parameter_summary(),
            "train_episode": episode.uid,
            "native_hidden_error_max": hidden_error,
            "native_point_argmax": native_point,
            "native_point_probability_error_max": native_probability_error,
            "replay_log_prob_error": replay_error,
            "multi_frame_teacher_valid": teacher_valid,
            "cache_snapshot_unchanged": cache_unchanged,
            "cache_elements": sizes,
            "history_effect_logits_max": history_effect,
            "frozen_base_has_gradient": frozen_grad,
            "nonfinite_gradients": nonfinite,
            "gradient_norm_sums": gradients,
            "peak_vram_bytes": torch.cuda.max_memory_allocated(),
        }
        result["passed"] = (
            native_probability_error < 0.01
            and replay_error < 0.01
            and teacher_valid
            and cache_unchanged
            and sizes[0] == sizes[1]
            and history_effect > 0
            and not frozen_grad
            and not nonfinite
            and all(x > 0 for x in gradients.values())
        )
        output = Path(args.output)
        output.mkdir(parents=True, exist_ok=True)
        (output / "gpu_model_validation.json").write_text(json.dumps(result, indent=2) + "\n")
        print(json.dumps(result, indent=2), flush=True)
        if not result["passed"]:
            raise RuntimeError("Real-model validation failed; do not launch production training")
    finally:
        env.close()


if __name__ == "__main__":
    torch.set_num_threads(4)
    main()
