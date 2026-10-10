"""Measure preprocessing, original GDN parity, conversion error and memory.

All comparisons use unchanged weights. Sensitivity and transfer diagnostics
are not navigation success, grounding accuracy or a performance ceiling.
"""

import argparse
import json
from pathlib import Path

import torch
from audit_architecture_capacity import configuration, set_memory_timescales, write_json
from torch.nn import functional as F
from transformers import AutoImageProcessor, Qwen3_5ForConditionalGeneration

from streamnav.contracts.state import LayerState
from streamnav.models.qwen35_kda.backbone import Qwen35KDABackbone
from streamnav.models.vision.preprocessing import patchify
from streamnav.training.checkpoint import load_policy


def difference(actual, expected):
    actual, expected = actual.float(), expected.float()
    return {
        "relative_l2": ((actual - expected).norm() / expected.norm().clamp_min(1e-12)).item(),
        "cosine_mean": F.cosine_similarity(actual, expected, dim=-1).mean().item(),
        "max_absolute": (actual - expected).abs().max().item(),
        "actual_rms": actual.square().mean().sqrt().item(),
        "reference_rms": expected.square().mean().sqrt().item(),
    }


def preprocessing(source, rgb):
    processor = AutoImageProcessor.from_pretrained(source, local_files_only=True)
    # Our normalized zero padding corresponds to raw pixel value 127.5.
    padded = F.pad(rgb.permute(2, 0, 1).float(), (0, 0, 9, 9), value=127.5)
    expected = processor(images=[padded], do_resize=False, return_tensors="pt")
    actual, grid = patchify(rgb, [270, 480])
    torch.testing.assert_close(actual, expected.pixel_values, atol=1e-6, rtol=1e-6)
    torch.testing.assert_close(grid, expected.image_grid_thw, atol=0, rtol=0)
    return {
        "grid_equal": True,
        "grid": grid.tolist(),
        "max_absolute": (actual - expected.pixel_values).abs().max().item(),
        "protocol": "Same padded sensor RGB; official processor do_resize=False. Tests normalization, temporal repeat and patch order, not official smart-resize equivalence.",
    }


@torch.no_grad()
def conversion(args, data, source):
    original = (
        Qwen3_5ForConditionalGeneration.from_pretrained(
            source, dtype=torch.float32, attn_implementation="sdpa", local_files_only=True
        )
        .to(args.device)
        .eval()
    )
    converted = Qwen35KDABackbone.from_converted(
        "checkpoints/qwen35_0p8b_kda",
        device=args.device,
        dtype=torch.float32,
        image_size=[270, 480],
        goal_conditioning="nav_query",
        kda_output_norm=True,
        gradient_checkpointing=False,
    ).eval()
    captures, hooks = {}, []
    for i, layer in enumerate(original.model.language_model.layers):
        mixer = layer.self_attn if layer.layer_type == "full_attention" else layer.linear_attn

        def capture(module, inputs, kwargs, index=i):
            captures[index] = {"x": kwargs.get("hidden_states", inputs[0] if inputs else None)}
            if "position_embeddings" in kwargs:
                captures[index].update(
                    position_embeddings=kwargs["position_embeddings"],
                    attention_mask=kwargs.get("attention_mask"),
                )

        hooks.append(mixer.register_forward_pre_hook(capture, with_kwargs=True))
    rows = []
    try:
        for clip in data["clips"][::2]:
            instruction = clip["goal"]
            prompt = converted.tokenizer.apply_chat_template(
                [
                    {
                        "role": "system",
                        "content": "You are a fast object navigation policy. Navigate to the requested object.",
                    },
                    {"role": "user", "content": instruction},
                ],
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False,
            )
            ids = converted.tokenizer(prompt, return_tensors="pt").input_ids.to(args.device)
            visual = clip["visual"][0:1].to(args.device)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                frame_tokens = converted.encode_visual_tokens(visual, [instruction])
                tokens = torch.cat((converted.embeddings(ids), frame_tokens), dim=1)
                placeholder = torch.tensor(
                    [
                        [converted.config.vision_start_token_id]
                        + [converted.config.image_token_id] * visual.shape[1]
                        + [converted.config.vision_end_token_id, converted.tokenizer.eos_token_id]
                    ],
                    device=args.device,
                )
                position_ids, _ = original.model.get_rope_index(
                    torch.cat((ids, placeholder), dim=1),
                    mm_token_type_ids=(
                        torch.cat((ids, placeholder), dim=1) == converted.config.image_token_id
                    ).int(),
                    image_grid_thw=torch.tensor([[1, 18, 30]], device=args.device),
                )
                native = original.model.language_model(
                    inputs_embeds=tokens, position_ids=position_ids, use_cache=False
                ).last_hidden_state
                capture_copy = dict(captures)
                converted_hidden, _ = converted.recurrent_forward(tokens, mode="chunk")
                normalized = difference(converted_hidden[:, -1], native[:, -1])
                for layer in converted.layers:
                    if hasattr(layer.mixer, "output_norm"):
                        layer.mixer.output_norm = False
                raw_hidden, _ = converted.recurrent_forward(tokens, mode="chunk")
                raw = difference(raw_hidden[:, -1], native[:, -1])
                for layer in converted.layers:
                    if hasattr(layer.mixer, "output_norm"):
                        layer.mixer.output_norm = True
                gdn, attention = [], []
                for i, layer in enumerate(converted.layers):
                    native_layer = original.model.language_model.layers[i]
                    captured = capture_copy[i]
                    x = captured["x"]
                    if hasattr(layer.mixer, "original"):
                        reference = native_layer.linear_attn(x)
                        actual, _ = layer.mixer(x, LayerState(None, None), mode="chunk")
                        split = x.shape[1] // 2
                        first, state = layer.mixer(
                            x[:, :split], LayerState(None, None), mode="chunk"
                        )
                        second, _ = layer.mixer(x[:, split:], state, mode="chunk")
                        with torch.autocast("cuda", enabled=False):
                            reference_fp32 = native_layer.linear_attn(x.float())
                            actual_fp32, _ = layer.mixer(
                                x.float(), LayerState(None, None), mode="chunk"
                            )
                            first_fp32, fp32_state = layer.mixer(
                                x[:, :split].float(), LayerState(None, None), mode="chunk"
                            )
                            second_fp32, _ = layer.mixer(
                                x[:, split:].float(), fp32_state, mode="chunk"
                            )
                        gdn.append(
                            {
                                "layer": i,
                                "whole": difference(actual, reference),
                                "split": difference(torch.cat((first, second), 1), reference),
                                "whole_fp32": difference(actual_fp32, reference_fp32),
                                "split_fp32": difference(
                                    torch.cat((first_fp32, second_fp32), 1), reference_fp32
                                ),
                            }
                        )
                    else:
                        reference = native_layer.self_attn(
                            x,
                            position_embeddings=captured["position_embeddings"],
                            attention_mask=captured["attention_mask"],
                        )[0]
                        actual, _ = layer.mixer(x, LayerState(None, None), mode="chunk")
                        cos, sin = captured["position_embeddings"]
                        no_rope = native_layer.self_attn(
                            x,
                            position_embeddings=(torch.ones_like(cos), torch.zeros_like(sin)),
                            attention_mask=captured["attention_mask"],
                        )[0]
                        attention.append(
                            {
                                "layer": i,
                                "kda_vs_native": difference(actual, reference),
                                "removing_rope_only_vs_native": difference(no_rope, reference),
                            }
                        )
            row = {
                "goal": instruction,
                "normalized_conversion_readout": normalized,
                "raw_conversion_readout": raw,
                "gdn": gdn,
                "attention": attention,
            }
            rows.append(row)
            print(
                json.dumps(
                    {
                        "goal": instruction,
                        "normalized_conversion_readout": normalized,
                        "raw_conversion_readout": raw,
                    }
                ),
                flush=True,
            )
        same_vision = all(
            torch.equal(value, original.model.visual.state_dict()[name])
            for name, value in converted.vision.state_dict().items()
        )
        current = load_policy(
            configuration(args.checkpoint, args.device), preserve_master_weights=True
        ).eval()
        current_same_vision = all(
            torch.equal(value, original.model.visual.state_dict()[name])
            for name, value in current.backbone.vision.state_dict().items()
        )
        return {
            "protocol": "Original pretrained Qwen and untouched converted initialization, identical prompt/image/NAV input embeddings. Native language layers use official multimodal RoPE positions; converted layers use recurrence. Readout difference is transfer error, not task accuracy. GDN uses identical real native hidden inputs, including zero-state whole and stateful split sequence.",
            "converted_vision_weights_equal_original": same_vision,
            "current_vision_weights_equal_original": current_same_vision,
            "records": rows,
        }
    finally:
        for hook in hooks:
            hook.remove()


@torch.no_grad()
def memory(args, data):
    policy = load_policy(
        configuration(args.checkpoint, args.device), preserve_master_weights=True
    ).eval()
    clips = data["clips"]
    common_visual = torch.cat([c["visual"] for c in clips]).to(args.device)
    rgb = clips[0]["rgb"][0]
    with torch.autocast("cuda", dtype=torch.bfloat16):
        black = policy.backbone.encode_vision(torch.zeros_like(rgb).unsqueeze(0))
    initial_visual = torch.cat((common_visual[0:1], black))
    initial_biases = {
        i: layer.mixer.decay_proj.bias.clone()
        for i, layer in enumerate(policy.backbone.layers)
        if hasattr(layer.mixer, "decay_proj")
    }
    result = []
    for variant in ("current", "long_memory"):
        if variant == "long_memory":
            set_memory_timescales(policy.backbone)
        retained, pooled, hooks = [], [], []
        hooks.append(
            policy.pooling.register_forward_hook(lambda m, x, y: pooled.append(y.detach()))
        )
        for index, layer in enumerate(policy.backbone.layers):
            if not hasattr(layer.mixer, "decay_proj"):
                continue
            mixer = layer.mixer

            def gate_hook(module, inputs, output, index=index, mixer=mixer):
                if output.shape[1] != 138:
                    return
                decay = F.softplus(output.float()).view(
                    output.shape[0], 138, mixer.num_heads, mixer.head_dim
                )
                retained.append(
                    {
                        "layer": index,
                        "direct_frame_retention_mean": (-decay.sum(1)).exp().mean().item(),
                        "direct_frame_retention_max": (-decay.sum(1)).exp().max().item(),
                    }
                )

            hooks.append(mixer.decay_proj.register_forward_hook(gate_hook))
        try:
            with torch.autocast("cuda", dtype=torch.bfloat16):
                states = [policy.start_episode(str(i), clips[0]["goal"]) for i in range(2)]
                _, _, states = policy.forward_batch(
                    torch.stack((rgb, rgb)), states, visual_embeddings=initial_visual
                )
                initial_gates = list(retained)
                records = []
                for lag in range(1, 101):
                    pooled.clear()
                    visual = (
                        common_visual[(lag - 1) % common_visual.shape[0]]
                        .unsqueeze(0)
                        .expand(2, -1, -1)
                    )
                    logits, _, states = policy.forward_batch(
                        torch.stack((rgb, rgb)), states, visual_embeddings=visual
                    )
                    if lag in (1, 2, 5, 10, 20, 50, 100):
                        layer_differences = []
                        for index, (a, b) in enumerate(
                            zip(states[0].kda_cache, states[1].kda_cache, strict=True)
                        ):
                            relative = (
                                (a.recurrent - b.recurrent).norm()
                                / a.recurrent.norm().clamp_min(1e-12)
                            ).item()
                            layer_differences.append(
                                {
                                    "layer": index,
                                    "type": "kda" if index in initial_biases else "gdn",
                                    "relative_l2": relative,
                                }
                            )
                        records.append(
                            {
                                "common_tail_frames": lag,
                                "hidden_difference": difference(pooled[0][1], pooled[0][0]),
                                "probability_tv": (
                                    (logits.softmax(-1)[1] - logits.softmax(-1)[0]).abs().sum() / 2
                                ).item(),
                                "state_differences": layer_differences,
                            }
                        )
            result.append(
                {"variant": variant, "first_frame_actual_gates": initial_gates, "records": records}
            )
            print(json.dumps({"memory_variant": variant, "last": records[-1]}), flush=True)
        finally:
            for hook in hooks:
                hook.remove()
            for i, bias in initial_biases.items():
                policy.backbone.layers[i].mixer.decay_proj.bias.copy_(bias)
    return {
        "protocol": "Two same-goal memories differ only in the first real/black visual frame, then receive 100 identical cached-real-frame inputs. Tail cycles fixed teacher clips; not a valid navigation episode. Tests historical sensitivity, not useful recall. Long-memory changes only KDA bias offsets to nominal 5/20/100-frame head half-lives, with unchanged learned weights.",
        "records": result,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--data", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="cuda:3")
    parser.add_argument("--mode", choices=["transfer", "memory"], required=True)
    parser.add_argument("--disable-tf32", action="store_true")
    args = parser.parse_args()
    torch.cuda.set_device(args.device)
    if args.disable_tf32:
        torch.backends.cudnn.allow_tf32 = False
        torch.backends.cuda.matmul.allow_tf32 = False
    data = torch.load(args.data, map_location="cpu", weights_only=False)
    source = json.loads((Path(args.checkpoint) / "kda_layout.json").read_text())["source"]
    if args.mode == "transfer":
        report = {
            "preprocessing": preprocessing(source, data["clips"][0]["rgb"][0]),
            "conversion": conversion(args, data, source),
        }
    else:
        report = memory(args, data)
    report["tf32_disabled"] = args.disable_tf32
    write_json(args.output, report)


if __name__ == "__main__":
    main()
