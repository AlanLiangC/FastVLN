"""Qwen3.5 point CoT with frame-local native mixers and persistent KDA memory."""

import json
from collections import OrderedDict
from contextlib import contextmanager, nullcontext
from dataclasses import dataclass, replace
from hashlib import sha256
from pathlib import Path
from typing import Any, cast

import torch
from torch import nn
from torch.distributions import Categorical
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint

from streamnav.contracts.state import LayerState, StreamingState
from streamnav.models.point_cot.adapters import KDAMemory, LoRALinear
from streamnav.models.point_cot.delta import StreamingGatedDeltaNet
from streamnav.models.point_cot.inference_graphs import InferenceGraphs
from streamnav.models.point_cot.object_hint import ObjectHintToken
from streamnav.models.point_cot.planning import FutureActionPlan
from streamnav.models.policy.perception import decode_point_logits


@dataclass
class FrameFeatures:
    ids: torch.Tensor
    positions: torch.Tensor
    visual: torch.Tensor
    deepstack: tuple[torch.Tensor, ...]
    hint_token: int = 0
    has_hint_slot: bool = False

    def cpu(self):
        return FrameFeatures(
            self.ids.cpu(),
            self.positions.cpu(),
            self.visual.cpu(),
            tuple(x.cpu() for x in self.deepstack),
            self.hint_token,
            self.has_hint_slot,
        )


@dataclass
class FrameContext:
    hidden: torch.Tensor
    memory: tuple[LayerState, ...]
    local: tuple[LayerState, ...]
    next_position: int | torch.Tensor
    key_mask: torch.Tensor | None = None


@dataclass
class MemoryPacking:
    # Native attention batches independent frames; KDA reads their valid tokens
    # in chronological order. Padding must never enter the recurrent state.
    indices: torch.Tensor
    cu_seqlens: torch.Tensor | None = None
    cu_seqlens_cpu: torch.Tensor | None = None
    independent: bool = False
    lengths: tuple[int, ...] | None = None


@dataclass
class TokenStep:
    apos: torch.Tensor
    opos: torch.Tensor
    action: torch.Tensor
    apos_logits: torch.Tensor
    opos_logits: torch.Tensor
    action_logits: torch.Tensor
    value: torch.Tensor
    state: StreamingState
    action_plan_logits: torch.Tensor | None = None

    @property
    def log_prob(self):
        return sum(
            Categorical(logits=logits).log_prob(token)
            for logits, token in (
                (self.apos_logits, self.apos),
                (self.opos_logits, self.opos),
                (self.action_logits, self.action),
            )
        )


class Qwen35PointCoTPolicy(nn.Module):
    apos_ids: torch.Tensor
    opos_ids: torch.Tensor

    def __init__(self, config):
        super().__init__()
        from transformers import AutoProcessor, Qwen3_5ForConditionalGeneration

        self.settings = config["model"]
        self.base_path = str(Path(self.settings["checkpoint"]).resolve())
        device = torch.device(config["device"])
        self.native: Any = Qwen3_5ForConditionalGeneration.from_pretrained(
            self.base_path, dtype=torch.bfloat16, attn_implementation="sdpa"
        )
        self.native = cast(Any, self.native).to(device).requires_grad_(False).eval()
        self.vision_compute_dtype = getattr(
            torch, self.settings.get("vision_compute_dtype", "bfloat16")
        )
        if self.vision_compute_dtype not in (torch.float32, torch.bfloat16):
            raise ValueError("Unsupported frozen vision dtype")
        self.native.model.visual.to(dtype=self.vision_compute_dtype)
        text = self.native.model.language_model
        layout = self.native.config.text_config
        if layout.hidden_size != 1024 or len(text.layers) != 24:
            raise ValueError("This recipe requires the official Qwen3.5-0.8B multimodal base")
        self.processor = AutoProcessor.from_pretrained(self.base_path)
        self.frame_prompts: OrderedDict[str, str] = OrderedDict()
        self.frame_templates: OrderedDict[tuple, tuple] = OrderedDict()
        self.frame_template_hits = self.frame_template_misses = 0
        # Bound visual token count even for unusually tall/wide source images.
        # Navigation's 448x256 view remains unchanged; there is no crop or pad.
        self.processor.image_processor.size["longest_edge"] = self.settings.get(
            "max_image_pixels", 256 * 448
        )
        self.language_compute_dtype = getattr(torch, self.settings["language_compute_dtype"])
        if self.language_compute_dtype not in (torch.float32, torch.bfloat16):
            raise ValueError("Unsupported language dtype")
        self.float32_matmul_precision = self.settings.get("float32_matmul_precision", "high")
        torch.set_float32_matmul_precision(self.float32_matmul_precision)
        torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = False
        if self.language_compute_dtype == torch.float32:
            text.float()
            self.native.lm_head.weight = text.embed_tokens.weight
        rank, alpha = self.settings["lora_rank"], self.settings["lora_alpha"]
        for layer in text.layers:
            mixer = layer.self_attn if layer.layer_type == "full_attention" else layer.linear_attn
            names = (
                ("q_proj", "k_proj", "v_proj", "o_proj")
                if layer.layer_type == "full_attention"
                else ("in_proj_qkv", "in_proj_z", "in_proj_a", "in_proj_b", "out_proj")
            )
            for module, projections in (
                (mixer, names),
                (layer.mlp, ("gate_proj", "up_proj", "down_proj")),
            ):
                for name in projections:
                    setattr(module, name, LoRALinear(getattr(module, name), rank, alpha))
        self.memories = nn.ModuleDict(
            {
                str(i): KDAMemory(layer.self_attn, self.settings["gate_rank"])
                for i, layer in enumerate(text.layers)
                if layer.layer_type == "full_attention"
            }
        )
        self.point_decoding = self.settings.get("point_decoding", "flat")
        if self.point_decoding not in ("flat", "hierarchical"):
            raise ValueError("Unsupported point decoding")
        self.apos_decoding = self.settings.get("apos_decoding", self.point_decoding)
        self.opos_decoding = self.settings.get("opos_decoding", self.point_decoding)
        if any(
            mode not in ("flat", "hierarchical")
            for mode in (self.apos_decoding, self.opos_decoding)
        ):
            raise ValueError("Unsupported channel point decoding")
        self.future_action_steps = self.settings.get("future_action_steps", 1)
        if not 1 <= self.future_action_steps <= 8:
            raise ValueError("Action horizon must be in [1, 8]")
        self.action_token_strings = tuple(
            f"<nav_{name}>" for name in ("stop", "forward", "left", "right", "up", "down")
        )
        strings = {
            "apos": [f"<apos_{i}>" for i in range(1300)],
            "opos": [f"<opos_{i}>" for i in range(1297)],
        }
        self.tokenizer.add_special_tokens(
            {
                "additional_special_tokens": strings["apos"]
                + strings["opos"]
                + list(self.action_token_strings)
            }
        )
        for name, tokens in strings.items():
            self.register_buffer(
                name + "_ids",
                torch.tensor(self.tokenizer.convert_tokens_to_ids(tokens), device=device),
                persistent=False,
            )
        # Small tied input/output tables; no full-vocabulary projection or resize.
        # Coordinates are defined on the un-cropped original image, 48 x 27.
        words = ["not visible"] + [
            f"({int((i % 48 + 0.5) / 48 * 1000)}, {int((i // 48 + 0.5) / 27 * 1000)})"
            for i in range(1296)
        ]

        def rows(texts):
            return torch.stack(
                [
                    text.embed_tokens.weight[self.tokenizer.encode(w, add_special_tokens=False)]
                    .float()
                    .mean(0)
                    for w in texts
                ]
            ).detach()

        coordinates = rows(words)
        self.apos_rows = nn.Parameter(
            torch.cat((coordinates, rows(["turn left", "turn right", "stop"])))
        )
        self.opos_rows = nn.Parameter(coordinates.clone())
        self.action_rows = nn.Parameter(
            rows(["stop", "move forward", "turn left", "turn right", "look up", "look down"])
        )
        self.action_bias = nn.Parameter(torch.zeros(6, device=device))
        self.critic = nn.Linear(layout.hidden_size, 1, device=device, dtype=torch.float32)
        nn.init.zeros_(self.critic.weight)
        nn.init.zeros_(self.critic.bias)
        self.future_plan = (
            FutureActionPlan(layout.hidden_size, self.future_action_steps - 1, device=device)
            if self.future_action_steps > 1
            else None
        )
        self.object_hint = (
            ObjectHintToken(rows(["no external object information"])[0])
            if self.settings.get("object_hint_token", False)
            else None
        )
        self.supports_object_hints = self.object_hint is not None
        self.loaded_checkpoint = self.base_path
        self.inference_compute_dtype = torch.bfloat16
        self.batch_chat_body = False
        self.perception = True
        self.perception_config = {"point_decoding": self.point_decoding}
        self._all_trainable = tuple(p for p in self.parameters() if p.requires_grad)
        self.inference_graphs = InferenceGraphs()

    @property
    def backbone(self):
        return self

    @property
    def device(self):
        return self.native.device

    @property
    def tokenizer(self):
        return self.processor.tokenizer

    @torch.no_grad()
    def frame_features(self, rgb, instruction):
        from PIL import Image

        image = Image.fromarray(rgb.cpu().numpy())
        return self.frame_features_batch([image], [instruction])[0], rgb

    @torch.no_grad()
    def frame_features_batch(self, images, instructions):
        from PIL import Image

        if len(images) != len(instructions) or not images:
            raise ValueError("One instruction is required per independent image")
        # Same chat prompt as navigation, without cropping. Vision is frozen;
        # one encoder call handles all independent images and variable grids.
        height, width = self.settings["image_size"]
        cache_templates = self.settings.get("cache_frame_templates", False)
        prepared, prompts = [], []
        for image, instruction in zip(images, instructions, strict=True):
            image = image.copy()
            image.thumbnail((width, height), Image.Resampling.BILINEAR)
            prepared.append(image)
            if cache_templates and instruction in self.frame_prompts:
                prompts.append(self.frame_prompts[instruction])
                self.frame_prompts.move_to_end(instruction)
                continue
            messages = [
                {
                    "role": "system",
                    "content": [
                        {
                            "type": "text",
                            "text": "You are a visual navigation assistant. Predict a feasible affordance waypoint, the visible task object position, then one navigation action.",
                        }
                    ],
                },
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": image},
                        {"type": "text", "text": instruction},
                    ],
                },
            ]
            prompts.append(
                self.processor.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True, enable_thinking=False
                )
            )
            if cache_templates:
                self.frame_prompts[instruction] = prompts[-1]
                if len(self.frame_prompts) > 256:
                    self.frame_prompts.popitem(last=False)
        if cache_templates:
            # RGB pixels are always encoded afresh. Only text IDs and spatial
            # position indices depend on instruction/grid rather than pixels.
            inputs = self.processor.image_processor(images=prepared, return_tensors="pt")
            grids = inputs.image_grid_thw.tolist()
            templates = []
            for image, instruction, prompt, grid in zip(
                prepared, instructions, prompts, grids, strict=True
            ):
                key = (instruction, tuple(grid))
                if key not in self.frame_templates:
                    encoded = self.processor(text=[prompt], images=[image], return_tensors="pt")
                    positions, _ = self.native.model.get_rope_index(
                        input_ids=encoded.input_ids,
                        image_grid_thw=encoded.image_grid_thw,
                        mm_token_type_ids=encoded.mm_token_type_ids,
                        attention_mask=encoded.attention_mask,
                    )
                    self.frame_templates[key] = (
                        encoded.input_ids.to(self.device),
                        positions.to(self.device),
                        int((encoded.input_ids == self.native.config.image_token_id).sum()),
                    )
                    self.frame_template_misses += 1
                    if len(self.frame_templates) > 256:
                        self.frame_templates.popitem(last=False)
                else:
                    self.frame_template_hits += 1
                    self.frame_templates.move_to_end(key)
                templates.append(self.frame_templates[key])
            inputs = inputs.to(self.device)
            visual = self.native.model.visual(
                inputs.pixel_values.to(self.vision_compute_dtype),
                grid_thw=inputs.image_grid_thw,
                return_dict=True,
            ).pooler_output
            chunks = visual.split([template[2] for template in templates])
            return [
                FrameFeatures(ids, positions, chunk, ())
                for (ids, positions, _), chunk in zip(templates, chunks, strict=True)
            ]
        inputs = self.processor(
            text=prompts, images=prepared, padding=True, return_tensors="pt"
        ).to(self.device)
        positions, _ = self.native.model.get_rope_index(
            input_ids=inputs.input_ids,
            image_grid_thw=inputs.image_grid_thw,
            mm_token_type_ids=inputs.mm_token_type_ids,
            attention_mask=inputs.attention_mask,
        )
        visual = self.native.model.visual(
            inputs.pixel_values.to(self.vision_compute_dtype),
            grid_thw=inputs.image_grid_thw,
            return_dict=True,
        ).pooler_output
        lengths = (inputs.input_ids == self.native.config.image_token_id).sum(-1).tolist()
        chunks = visual.split(lengths)
        features = []
        for i, chunk in enumerate(chunks):
            valid = inputs.attention_mask[i].bool()
            features.append(
                FrameFeatures(
                    inputs.input_ids[i : i + 1, valid].contiguous(),
                    positions[:, i : i + 1, valid].contiguous(),
                    chunk,
                    (),
                )
            )
        return features

    def condition_features(self, features, hint_tokens=None):
        if self.object_hint is None:
            if hint_tokens is not None and any(hint_tokens):
                raise ValueError("This checkpoint has no object-hint input")
            return features
        if hint_tokens is not None and len(hint_tokens) != len(features):
            raise ValueError("One hint token is required per frame")
        conditioned = []
        for i, feature in enumerate(features):
            token = feature.hint_token if hint_tokens is None else int(hint_tokens[i])
            if not 0 <= token <= 1296:
                raise ValueError("Invalid object hint token")
            if feature.has_hint_slot:
                conditioned.append(replace(feature, hint_token=token))
            else:
                # A single soft token after the frame/text prefix. Its embedding
                # is substituted at prefill/replay time so gradients stay live.
                conditioned.append(
                    FrameFeatures(
                        torch.cat((feature.ids, feature.ids.new_zeros((1, 1))), -1),
                        torch.cat(
                            (
                                feature.positions,
                                feature.positions.new_full(
                                    (3, 1, 1), int(feature.positions.max()) + 1
                                ),
                            ),
                            -1,
                        ),
                        feature.visual,
                        feature.deepstack,
                        token,
                        True,
                    )
                )
        return conditioned

    def embed_hints(self, x, features):
        if self.object_hint is None:
            return x
        rows = torch.arange(len(features), device=x.device)
        positions = torch.tensor([f.ids.shape[-1] - 1 for f in features], device=x.device)
        tokens = torch.tensor([f.hint_token for f in features], device=x.device)
        return x.index_put((rows, positions), self.object_hint(tokens, self.opos_rows).to(x.dtype))

    def start_episode(self, episode_id, instruction):
        return StreamingState(
            tuple(LayerState(None, None) for _ in self.native.model.language_model.layers),
            episode_id,
            sha256(instruction.encode()).hexdigest(),
            0,
            instruction,
        )

    def _layer(
        self,
        index,
        x,
        positions,
        previous,
        local,
        visual_mask,
        deepstack,
        native_only,
        packing=None,
        attention_mask=None,
    ):
        from transformers.integrations.sdpa_attention import sdpa_attention_forward
        from transformers.models.qwen3_5.modeling_qwen3_5 import apply_rotary_pos_emb

        text = self.native.model.language_model
        layer = text.layers[index]
        residual, norm = x, layer.input_layernorm(x)
        b, length, _ = norm.shape
        recurrent = previous.recurrent
        if layer.layer_type == "linear_attention":
            # Native convolution + delta state belongs to this decision only.
            # In packed training rows are independent frames, including padding.
            if packing is not None and packing.independent:
                valid = norm.reshape(1, b * length, -1).index_select(1, packing.indices)
                y, next_local = StreamingGatedDeltaNet(layer.linear_attn)(
                    valid,
                    local,
                    mode="chunk",
                    cu=packing.cu_seqlens,
                    cu_cpu=packing.cu_seqlens_cpu,
                    lengths=packing.lengths,
                )
                y = (
                    y.new_zeros(b * length, y.shape[-1])
                    .index_copy(0, packing.indices, y.squeeze(0))
                    .view(b, length, -1)
                )
            else:
                # Use the differentiable chunk equations during sampling too.
                # Switching kernels for one-token decoding adds recurrent drift
                # when PPO reconstructs the same decisions with gradients.
                y, next_local = StreamingGatedDeltaNet(layer.linear_attn)(norm, local, mode="chunk")
            keys, values = next_local.conv, next_local.recurrent
        else:
            attention = layer.self_attn
            dim = attention.head_dim
            q, gate = attention.q_proj(norm).view(b, length, -1, 2 * dim).chunk(2, -1)
            q = attention.q_norm(q)
            k = attention.k_norm(attention.k_proj(norm).view(b, length, -1, dim))
            v = attention.v_proj(norm).view(b, length, -1, dim)
            cos, sin = text.rotary_emb(norm, positions)
            qr, kr = apply_rotary_pos_emb(q.transpose(1, 2), k.transpose(1, 2), cos, sin)
            keys = kr if local.conv is None else torch.cat((local.conv, kr), dim=-2)
            values = (
                v.transpose(1, 2)
                if local.recurrent is None
                else torch.cat((local.recurrent, v.transpose(1, 2)), dim=-2)
            )
            y, _ = sdpa_attention_forward(
                attention,
                qr,
                keys,
                values,
                attention_mask=attention_mask,
                dropout=0.0,
                scaling=attention.scaling,
            )
            y = y.reshape(b, length, -1) * gate.reshape(b, length, -1).sigmoid()
            if not native_only:
                memory = cast(KDAMemory, self.memories[str(index)])
                # Cross-frame KDA uses unrotated Q/K. Native spatial RoPE resets
                # with each frame and must not create artificial temporal phases.
                inputs = (q, k, v, norm)
                if packing is not None:
                    inputs = tuple(
                        t.reshape(1, b * length, *t.shape[2:]).index_select(1, packing.indices)
                        for t in inputs
                    )
                extra, recurrent = memory(
                    *inputs,
                    recurrent,
                    mode="chunk",
                    cu_seqlens=packing.cu_seqlens if packing else None,
                    cu_seqlens_cpu=packing.cu_seqlens_cpu if packing else None,
                )
                if packing is not None:
                    extra = (
                        extra.new_zeros(b * length, extra.shape[-1])
                        .index_copy(0, packing.indices, extra.squeeze(0))
                        .view(b, length, -1)
                    )
                    if not packing.independent:
                        recurrent = recurrent[-1:]
                y = y + memory.residual_scale.tanh().to(y.dtype) * extra
            y = attention.o_proj(y)
        x = residual + y
        x = x + layer.mlp(layer.post_attention_layernorm(x))
        return x, recurrent, keys, values

    def _forward(self, *args, **kwargs):
        context = (
            torch.autocast(self.device.type, enabled=False)
            if self.language_compute_dtype == torch.float32
            else nullcontext()
        )
        with context:
            if (
                self.settings.get("inference_cuda_graphs", False)
                and self.language_compute_dtype == torch.float32
                and not torch.is_grad_enabled()
                and kwargs.get("key_mask") is not None
            ):
                tag = tuple(m.enabled for m in self.modules() if isinstance(m, LoRALinear))
                return self.inference_graphs(self._forward_impl, args, kwargs, tag=tag)
            return self._forward_impl(*args, **kwargs)

    def _forward_impl(
        self,
        embeddings,
        positions,
        memory,
        local,
        visual_mask=None,
        deepstack=(),
        native_only=False,
        packing=None,
        hidden_indices=None,
        attention_mask=None,
        key_mask=None,
    ):
        x, next_memory, next_local = embeddings, [], []
        for i, (previous, kv) in enumerate(zip(memory, local, strict=True)):
            feature = deepstack[i] if i < len(deepstack) else None
            args = (
                i,
                x,
                positions,
                previous,
                kv,
                visual_mask,
                feature,
                native_only,
                packing,
                attention_mask,
            )
            if torch.is_grad_enabled() and self.settings.get("gradient_checkpointing", True):
                x, recurrent, keys, values = checkpoint(
                    self._layer, *args, use_reentrant=False, preserve_rng_state=False
                )
            else:
                x, recurrent, keys, values = self._layer(*args)
            next_memory.append(LayerState(None, recurrent))
            next_local.append(
                LayerState(keys, values)
                if packing is None or packing.independent
                else LayerState(None, None)
            )
        hidden = (
            x[:, -1]
            if hidden_indices is None
            else x[
                torch.arange(x.shape[0], device=x.device)[:, None]
                if hidden_indices.ndim == 2
                else torch.arange(x.shape[0], device=x.device),
                hidden_indices,
            ]
        )
        return FrameContext(
            self.native.model.language_model.norm(hidden),
            tuple(next_memory),
            tuple(next_local),
            positions.amax(dim=(0, 2)) + 1
            if key_mask is not None
            else int(positions.max()) + 1
            if packing is None
            else 0,
            key_mask,
        )

    def begin(self, feature, state, *, native_only=False):
        feature = self.condition_features([feature])[0]
        ids = feature.ids.to(self.device)
        x = self.native.model.language_model.embed_tokens(ids)
        mask = ids == self.native.config.image_token_id
        x = x.masked_scatter(mask.unsqueeze(-1), feature.visual.to(self.device, x.dtype))
        x = self.embed_hints(x, [feature])
        empty = tuple(LayerState(None, None) for _ in state.kda_cache)
        return self._forward(
            x,
            feature.positions.to(self.device),
            state.kda_cache,
            empty,
            mask,
            feature.deepstack,
            native_only,
        )

    def append(self, context, embedding, *, native_only=False):
        # Match the language residual/RoPE/QKV dtype after frozen visual encoding.
        embedding = embedding.to(self.native.model.language_model.embed_tokens.weight.dtype)
        positions = torch.full(
            (3, 1, 1), context.next_position, device=self.device, dtype=torch.long
        )
        return self._forward(
            embedding.view(1, 1, -1),
            positions,
            context.memory,
            context.local,
            native_only=native_only,
        )

    def begin_batch(self, features, states):
        """Prefill independent frames together without padding in either recurrence."""
        features = self.condition_features(features)
        if not features or len(features) != len(states):
            raise ValueError("One state is required per independent frame")
        lengths = tuple(f.ids.shape[-1] for f in features)
        width, count = max(lengths), len(features)
        ids = torch.zeros((count, width), device=self.device, dtype=torch.long)
        positions = torch.zeros((3, count, width), device=self.device, dtype=torch.long)
        key_mask = torch.zeros((count, width), device=self.device, dtype=torch.bool)
        indices: list[int] = []
        offsets = [0]
        for i, (feature, length) in enumerate(zip(features, lengths, strict=True)):
            ids[i, :length] = feature.ids[0].to(self.device)
            positions[:, i, :length] = feature.positions[:, 0].to(self.device)
            key_mask[i, :length] = True
            indices.extend(range(i * width, i * width + length))
            offsets.append(offsets[-1] + length)
        text = self.native.model.language_model
        x = text.embed_tokens(ids)
        visual_mask = ids == self.native.config.image_token_id
        x = x.masked_scatter(
            visual_mask.unsqueeze(-1),
            torch.cat([f.visual.to(self.device, x.dtype) for f in features]),
        )
        x = self.embed_hints(x, features)
        memory = []
        for i in range(len(states[0].kda_cache)):
            values = [s.kda_cache[i].recurrent for s in states]
            exemplar = next((v for v in values if v is not None), None)
            initial = (
                torch.cat(
                    [
                        v.to(self.device)
                        if v is not None
                        else torch.zeros_like(exemplar, device=self.device)
                        for v in values
                    ]
                )
                if exemplar is not None
                else None
            )
            memory.append(LayerState(None, initial))
        cu_cpu = torch.tensor(offsets, dtype=torch.long)
        packing = MemoryPacking(
            torch.tensor(indices, device=self.device, dtype=torch.long),
            cu_cpu.to(self.device),
            cu_cpu,
            True,
            lengths,
        )
        causal = torch.ones((width, width), device=self.device, dtype=torch.bool).tril()
        return self._forward(
            x,
            positions,
            tuple(memory),
            tuple(LayerState(None, None) for _ in memory),
            visual_mask,
            packing=packing,
            hidden_indices=torch.tensor(lengths, device=self.device) - 1,
            attention_mask=causal[None, None] & key_mask[:, None, None, :],
            key_mask=key_mask,
        )

    def append_batch(self, context, embeddings):
        count = embeddings.shape[0]
        positions = context.next_position.view(1, count, 1).expand(3, -1, -1)
        key_mask = torch.cat(
            (context.key_mask, torch.ones((count, 1), device=self.device, dtype=torch.bool)), -1
        )
        return self._forward(
            embeddings[:, None].to(self.native.model.language_model.embed_tokens.weight.dtype),
            positions,
            context.memory,
            context.local,
            attention_mask=key_mask[:, None, None, :],
            key_mask=key_mask,
        )

    def step_batch(self, features, states, *, tokens=None, deterministic=False):
        """Autoregressive APOS/OPOS/action for independent cached environments."""
        count = len(features)
        forced = tokens if tokens is not None else [None] * count
        greedy = [deterministic] * count if isinstance(deterministic, bool) else deterministic
        if len(forced) != count or len(greedy) != count:
            raise ValueError("Token overrides and decoding modes must match the batch")

        def choose(logits, channel):
            predicted = (
                decode_point_logits(
                    logits,
                    hierarchical=(self.apos_decoding if channel == 0 else self.opos_decoding)
                    == "hierarchical",
                )
                if channel < 2
                else logits.argmax(-1)
            )
            sampled = predicted if all(greedy) else Categorical(logits=logits).sample()
            selected = torch.where(
                torch.tensor(greedy, device=self.device, dtype=torch.bool), predicted, sampled
            )
            for i, override in enumerate(forced):
                if override is not None and override[channel] is not None:
                    selected[i] = override[channel]
            return selected

        context = self.begin_batch(features, states)
        value = self.read_value(context.hidden)
        apos_logits = self.output_projection(context.hidden, self.apos_rows)
        apos = choose(apos_logits, 0)
        context = self.append_batch(context, self.apos_rows[apos])
        opos_logits = self.output_projection(context.hidden, self.opos_rows)
        opos = choose(opos_logits, 1)
        context = self.append_batch(context, self.opos_rows[opos])
        action_logits = self.output_projection(context.hidden, self.action_rows, self.action_bias)
        actions = choose(action_logits, 2)
        context = self.append_batch(context, self.action_rows[actions])
        plans = self.plan_logits(action_logits, context.hidden)
        return [
            TokenStep(
                apos[i],
                opos[i],
                actions[i],
                apos_logits[i],
                opos_logits[i],
                action_logits[i],
                value[i],
                states[i].with_cache(
                    tuple(
                        LayerState(None, s.recurrent[i : i + 1]) if s.recurrent is not None else s
                        for s in context.memory
                    )
                ),
                plans[i] if plans is not None else None,
            )
            for i in range(count)
        ]

    def navigation_features(self, rgb, states, hint_tokens=None):
        from PIL import Image

        features = self.frame_features_batch(
            [Image.fromarray(frame.cpu().numpy()) for frame in rgb],
            [s.instruction for s in states],
        )
        return self.condition_features(features, hint_tokens)

    def point_logits(self, context, name):
        rows = getattr(self, name + "_rows")
        return self.output_projection(context.hidden, rows).squeeze(0)

    def output_projection(self, hidden, rows, bias=None):
        # Quantizing logit outputs to BF16 exaggerates tiny body differences
        # between token-at-a-time sampling and batched PPO probability replay.
        with torch.autocast(self.device.type, enabled=False):
            return F.linear(hidden.float(), rows, bias)

    def read_value(self, hidden):
        with torch.autocast(self.device.type, enabled=False):
            return self.critic(hidden.float()).squeeze(-1)

    def plan_logits(self, current_logits, action_hidden):
        if self.future_plan is None:
            return None
        with torch.autocast(self.device.type, enabled=False):
            future = self.output_projection(
                self.future_plan(action_hidden), self.action_rows, self.action_bias
            )
        return torch.cat((current_logits.unsqueeze(-2), future), dim=-2)

    def replay_fragment(self, features, state, tokens, *, resets=None):
        """Teacher-force recorded tokens with frame-local attention and causal KDA.

        Returns batched logits/values and only the final frame's cache. Point
        tokens remain policy samples from the rollout. Episode boundaries reset
        KDA through cu_seqlens, while native attention never crosses a frame.
        """
        features = self.condition_features(features)
        resets = resets or {}
        lengths = [f.ids.shape[-1] + 3 for f in features]
        width, count = max(lengths), len(features)
        ids = torch.zeros((count, width), device=self.device, dtype=torch.long)
        positions = torch.zeros((3, count, width), device=self.device, dtype=torch.long)
        hidden_indices: list[list[int]] = []
        valid_indices: list[int] = []
        offsets = [0]
        for i, (feature, length) in enumerate(zip(features, lengths, strict=True)):
            n = length - 3
            ids[i, :n] = feature.ids.to(self.device)
            ids[i, n : n + 2] = torch.stack(
                (self.apos_ids[tokens[i][0]], self.opos_ids[tokens[i][1]])
            )
            positions[:, i, :n] = feature.positions[:, 0].to(self.device)
            positions[:, i, n:length] = torch.arange(
                int(feature.positions.max()) + 1,
                int(feature.positions.max()) + 4,
                device=self.device,
            )
            hidden_indices.append(
                [n - 1, n, n + 1, n + 2] if self.future_plan is not None else [n - 1, n, n + 1]
            )
            valid_indices.extend(range(i * width, i * width + length))
            offsets.append(offsets[-1] + length)
        text = self.native.model.language_model
        x = text.embed_tokens(ids.clamp_max(text.embed_tokens.num_embeddings - 1))
        x = self.embed_hints(x, features)
        rows = torch.arange(count, device=self.device)
        ns = torch.tensor([n - 3 for n in lengths], device=self.device)
        x = x.index_put((rows, ns), self.apos_rows[torch.stack([t[0] for t in tokens])].to(x.dtype))
        x = x.index_put(
            (rows, ns + 1), self.opos_rows[torch.stack([t[1] for t in tokens])].to(x.dtype)
        )
        visual_mask = ids == self.native.config.image_token_id
        visual = torch.cat([f.visual.to(self.device, x.dtype) for f in features])
        x = x.masked_scatter(visual_mask.unsqueeze(-1), visual)
        action_indices = torch.tensor([n - 1 for n in lengths], device=self.device)
        rows = torch.arange(count, device=self.device)
        actions = torch.stack([t[2] for t in tokens])
        x = x.index_put((rows, action_indices), self.action_rows[actions].to(x.dtype))
        deepstack = tuple(
            torch.cat([f.deepstack[j].to(self.device) for f in features])
            for j in range(len(features[0].deepstack))
        )
        boundaries = [0, *sorted(i for i in resets if i > 0), count]
        cu_cpu = (
            torch.tensor([offsets[i] for i in boundaries], dtype=torch.long)
            if len(boundaries) > 2
            else None
        )
        packing = MemoryPacking(
            torch.tensor(valid_indices, device=self.device, dtype=torch.long),
            cu_cpu.to(self.device) if cu_cpu is not None else None,
            cu_cpu,
        )
        memory = []
        segments = len(boundaries) - 1
        for i, previous in enumerate(state.kda_cache):
            initial = previous.recurrent
            if str(i) in self.memories and (segments > 1 or 0 in resets):
                m = cast(KDAMemory, self.memories[str(i)])
                initial = torch.zeros(
                    (segments, m.heads, m.head_dim, m.head_dim),
                    device=self.device,
                    dtype=torch.float32,
                )
                if previous.recurrent is not None and 0 not in resets:
                    initial = torch.cat((previous.recurrent, initial[1:]), dim=0)
            memory.append(LayerState(None, initial))
        context = self._forward(
            x,
            positions,
            tuple(memory),
            tuple(LayerState(None, None) for _ in memory),
            visual_mask,
            deepstack,
            packing=packing,
            hidden_indices=torch.tensor(hidden_indices, device=self.device),
        )
        apos_logits = self.output_projection(context.hidden[:, 0], self.apos_rows)
        opos_logits = self.output_projection(context.hidden[:, 1], self.opos_rows)
        action_logits = self.output_projection(
            context.hidden[:, 2], self.action_rows, self.action_bias
        )
        value = self.read_value(context.hidden[:, 0])
        final = state
        if resets:
            final = self.start_episode(*resets[max(resets)])
            final_steps = count - max(resets)
        else:
            final_steps = state.step_index + count
        from dataclasses import replace

        final = replace(final, kda_cache=context.memory, step_index=final_steps)
        return TokenStep(
            torch.stack([t[0] for t in tokens]),
            torch.stack([t[1] for t in tokens]),
            actions,
            apos_logits,
            opos_logits,
            action_logits,
            value,
            final,
            self.plan_logits(action_logits, context.hidden[:, 3])
            if self.future_plan is not None
            else None,
        )

    def action_logits(self, context):
        return self.output_projection(context.hidden, self.action_rows, self.action_bias).squeeze(0)

    def step(self, feature, state, *, tokens=None, deterministic=False):
        context = self.begin(feature, state)
        value = self.read_value(context.hidden).squeeze()
        apos_logits = self.point_logits(context, "apos")
        apos = (
            tokens[0]
            if tokens is not None and tokens[0] is not None
            else (
                decode_point_logits(apos_logits, hierarchical=self.apos_decoding == "hierarchical")
                if deterministic
                else Categorical(logits=apos_logits).sample()
            )
        )
        context = self.append(context, self.apos_rows[apos])
        opos_logits = self.point_logits(context, "opos")
        opos = (
            tokens[1]
            if tokens is not None and tokens[1] is not None
            else (
                decode_point_logits(opos_logits, hierarchical=self.opos_decoding == "hierarchical")
                if deterministic
                else Categorical(logits=opos_logits).sample()
            )
        )
        context = self.append(context, self.opos_rows[opos])
        logits = self.action_logits(context)
        action = (
            tokens[2]
            if tokens is not None and tokens[2] is not None
            else (logits.argmax() if deterministic else Categorical(logits=logits).sample())
        )
        context = self.append(context, self.action_rows[action])
        return TokenStep(
            apos,
            opos,
            action,
            apos_logits,
            opos_logits,
            logits,
            value,
            state.with_cache(context.memory),
            self.plan_logits(logits.unsqueeze(0), context.hidden).squeeze(0)
            if self.future_plan is not None
            else None,
        )

    @contextmanager
    def native_weights(self):
        adapters = [m for m in self.modules() if isinstance(m, LoRALinear)]
        previous = [m.enabled for m in adapters]
        try:
            for m in adapters:
                m.enabled = False
            yield
        finally:
            for m, enabled in zip(adapters, previous, strict=True):
                m.enabled = enabled

    def forward_batch(self, rgb, states, *, return_perception=False, hint_tokens=None):
        steps = self.step_batch(
            self.navigation_features(rgb, states, hint_tokens), states, deterministic=True
        )
        result = (
            torch.stack([s.action_logits for s in steps]),
            torch.stack([s.value for s in steps]),
            [s.state for s in steps],
        )
        if return_perception:
            return (
                *result,
                {
                    "apos": torch.stack([s.apos_logits for s in steps]),
                    "opos": torch.stack([s.opos_logits for s in steps]),
                    "apos_tokens": torch.stack([s.apos for s in steps]),
                    "opos_tokens": torch.stack([s.opos for s in steps]),
                    "point_cot": True,
                    "object_hint_tokens": hint_tokens
                    if hint_tokens is not None
                    else [0] * len(states),
                    **(
                        {"action_plan_logits": torch.stack([s.action_plan_logits for s in steps])}
                        if self.future_plan is not None
                        else {}
                    ),
                },
            )
        return result

    def train(self, mode=True):
        super().train(mode)
        # No dropout in frozen teacher/vision or PPO probability replay.
        self.native.eval()
        return self

    def trainable_state(self):
        return {
            name: p.detach().cpu().contiguous()
            for name, p in self.named_parameters()
            if p.requires_grad
        }

    def parameter_summary(self):
        return {
            "total": sum(p.numel() for p in self.parameters()),
            "trainable": sum(p.numel() for p in self.parameters() if p.requires_grad),
            "memory_layers": len(self.memories),
            "native_local_attention_layers": 6,
            "native_local_gdn_layers": 18,
            "frozen_base": self.base_path,
            "language_compute_dtype": str(self.language_compute_dtype),
            "vision_compute_dtype": str(self.vision_compute_dtype),
            "point_decoding": self.point_decoding,
            "apos_decoding": self.apos_decoding,
            "opos_decoding": self.opos_decoding,
            "future_action_steps": self.future_action_steps,
            "object_hint_token": self.supports_object_hints,
            "action_token_strings": self.action_token_strings,
            "source_revision": json.loads(Path(self.base_path, "source_revision.json").read_text()),
        }
