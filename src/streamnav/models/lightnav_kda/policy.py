"""Current-frame LightNav with autoregressive APOS -> OPOS -> action tokens.

The original Qwen3-VL vision, DeepStack, spatial RoPE, MLPs, token embeddings
and LM head remain intact. Native attention is limited to a single decision;
27 of 36 layers additionally read/write fixed-size KDA memory. Only LoRA,
low-rank memory gates, six action rows and the critic are trainable.
"""

import json
import sys
from contextlib import contextmanager, nullcontext
from dataclasses import dataclass
from functools import partial
from hashlib import sha256
from pathlib import Path
from typing import Any, cast

import torch
from torch import nn
from torch.distributions import Categorical
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint

from streamnav.contracts.state import LayerState, StreamingState
from streamnav.models.lightnav_kda.adapters import KDAMemory, LoRALinear


def video_position_ids(base, **kwargs):
    """Align timestamp-separated video spans with per-tubelet RoPE grids.

    Transformers 5.3 consumes one grid row per contiguous vision span, while
    the processor returns one row with T>1 per video. Keep the original grid
    for the ViT; expand only the position-id call, never the image features.
    """
    grid = kwargs.get("video_grid_thw")
    types = kwargs.get("mm_token_type_ids")
    if grid is not None and types is not None:
        mask = types == 2
        starts = mask & ~F.pad(mask[:, :-1], (1, 0), value=False)
        spans = int(starts.sum())
        if spans != len(grid):
            if spans != int(grid[:, 0].sum()):
                raise ValueError("Video spans do not match temporal tubelet counts")
            expanded = grid.repeat_interleave(grid[:, 0], dim=0).clone()
            expanded[:, 0] = 1
            kwargs["video_grid_thw"] = expanded
    return base(**kwargs)


@dataclass
class FrameFeatures:
    ids: torch.Tensor
    positions: torch.Tensor
    visual: torch.Tensor
    deepstack: tuple[torch.Tensor, ...]

    def cpu(self):
        return FrameFeatures(
            self.ids.cpu(),
            self.positions.cpu(),
            self.visual.cpu(),
            tuple(x.cpu() for x in self.deepstack),
        )


@dataclass
class FrameContext:
    hidden: torch.Tensor
    memory: tuple[LayerState, ...]
    local: tuple[LayerState, ...]
    next_position: int


@dataclass
class MemoryPacking:
    # Native attention batches independent frames; KDA reads their valid tokens
    # in chronological order. Padding must never enter the recurrent state.
    indices: torch.Tensor
    cu_seqlens: torch.Tensor | None = None
    cu_seqlens_cpu: torch.Tensor | None = None


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


class LightNavKDAPolicy(nn.Module):
    apos_ids: torch.Tensor
    opos_ids: torch.Tensor
    apos_rows: torch.Tensor
    opos_rows: torch.Tensor

    def __init__(self, config):
        super().__init__()
        from transformers import Qwen3VLForConditionalGeneration

        # Keep reduction accumulators in FP32 when batching the frozen BF16
        # projections. Only the small output heads themselves run in FP32.
        torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = False
        self.settings = config["model"]
        self.base_path = str(Path(self.settings["checkpoint"]).resolve())
        device = torch.device(config["device"])
        self.native: Any = Qwen3VLForConditionalGeneration.from_pretrained(
            self.base_path, dtype=torch.bfloat16, attn_implementation="sdpa"
        )
        self.native = cast(Any, self.native).to(device).requires_grad_(False)
        self.native.eval()
        text: Any = self.native.model.language_model
        self.base_embedding_head_shared = torch.equal(
            text.embed_tokens.weight, self.native.lm_head.weight
        )
        if self.base_embedding_head_shared:
            # The release serializes both identical tensors. Recover the tied
            # frozen table after checking equality, saving about 0.75 GiB.
            self.native.lm_head.weight = text.embed_tokens.weight
            torch.cuda.empty_cache()
        for layer in text.layers:
            for name in ("q_proj", "k_proj", "v_proj", "o_proj"):
                base = getattr(layer.self_attn, name)
                setattr(
                    layer.self_attn,
                    name,
                    LoRALinear(base, self.settings["lora_rank"], self.settings["lora_alpha"]),
                )
        precision = self.settings.get("language_compute_dtype", "float32")
        if precision not in ("float32", "bfloat16"):
            raise ValueError(f"Unsupported language compute dtype: {precision}")
        self.language_compute_dtype = getattr(torch, precision)
        self.float32_matmul_precision = self.settings.get("float32_matmul_precision", "high")
        torch.set_float32_matmul_precision(self.float32_matmul_precision)
        if self.language_compute_dtype == torch.float32:
            text.float()
            if self.base_embedding_head_shared:
                self.native.lm_head.weight = text.embed_tokens.weight
        interval = self.settings["local_only_every"]
        self.memories = nn.ModuleDict(
            {
                str(i): KDAMemory(layer.self_attn, self.settings["gate_rank"])
                for i, layer in enumerate(text.layers)
                if (i + 1) % interval
            }
        )
        self._prepare_processor()
        self.register_buffer("apos_ids", self._token_family("apos", 1300), persistent=False)
        self.register_buffer("opos_ids", self._token_family("opos", 1297), persistent=False)
        for name in ("apos", "opos"):
            self.register_buffer(
                name + "_rows",
                self.native.lm_head.weight.index_select(0, getattr(self, name + "_ids")).float(),
                persistent=False,
            )
        embedding = text.embed_tokens.weight
        words = ("stop", "move forward", "turn left", "turn right", "look up", "look down")
        rows = [
            embedding[self.tokenizer.encode(w, add_special_tokens=False)].float().mean(0)
            for w in words
        ]
        # Reuse pretrained coarse motion semantics only for initialization. No
        # RVQ code is generated or decoded by the student at runtime.
        import numpy as np

        bundle = Path(self.base_path) / "action_tokenizer"
        manifest = json.loads((bundle / "manifest.json").read_text())
        coarse = np.load(bundle / "codebook_l0.npy") / np.load(bundle / "jacobian_weights.npy")
        first = coarse.reshape(256, 10, 3)[:, 0]
        prototypes = np.array(
            [[0.0, 0.0, 0.0], [0.15, 0.0, 0.0], [0.0, 0.0, np.pi / 6], [0.0, 0.0, -np.pi / 6]]
        )
        self.action_initialization_codes = []
        for i, prototype in enumerate(prototypes):
            code = (
                int(manifest["stop"]["l0"])
                if i == 0
                else int((((first - prototype) / np.array([0.15, 0.1, 0.3])) ** 2).sum(-1).argmin())
            )
            token = self.tokenizer.convert_tokens_to_ids(f"<act_l0_{code}>")
            rows[i] = embedding[token].float().clone()
            self.action_initialization_codes.append(code)
        self.action_token_strings = tuple(
            f"<nav_{name}>" for name in ("stop", "forward", "left", "right", "up", "down")
        )
        self.tokenizer.add_special_tokens(
            {"additional_special_tokens": list(self.action_token_strings)}
        )
        self.action_token_ids = tuple(
            self.tokenizer.convert_tokens_to_ids(t) for t in self.action_token_strings
        )
        self.action_rows = nn.Parameter(torch.stack(rows))
        self.action_bias = nn.Parameter(
            torch.tensor([0.0, 0.0, 0.0, 0.0, -2.0, -2.0], device=device)
        )
        self.critic = nn.Linear(embedding.shape[1], 1, device=device, dtype=torch.float32)
        nn.init.zeros_(self.critic.weight)
        nn.init.zeros_(self.critic.bias)
        self.loaded_checkpoint = self.base_path
        self.inference_compute_dtype = torch.bfloat16
        self.batch_chat_body = False
        self.perception = True  # Evaluation compatibility: LM predictions, no auxiliary head.
        self.perception_config = {"point_decoding": "flat"}
        self._template_cache = {}
        self._all_trainable = tuple(p for p in self.parameters() if p.requires_grad)

    @property
    def backbone(self):
        # Property rather than a registered alias avoids duplicate state-dict weights.
        return self

    @property
    def device(self):
        return self.native.device

    @property
    def tokenizer(self):
        return self.processor.tokenizer

    def _token_family(self, name, size):
        ids = [self.tokenizer.convert_tokens_to_ids(f"<{name}_{i}>") for i in range(size)]
        if len(set(ids)) != size or self.tokenizer.unk_token_id in ids:
            raise ValueError(f"Checkpoint lacks the pretrained {name} token family")
        return torch.tensor(ids, device=self.device)

    def _prepare_processor(self):
        third_party = str(Path(__file__).resolve().parents[4] / "third_party/LightNav-0/src")
        if third_party not in sys.path:
            sys.path.insert(0, third_party)
        from lightnav.data_processor import Qwen3VLDataProcessor
        from lightnav.inference.model import _position_id_func_from_config
        from lightnav.processing import VLNQwen3VLProcessor

        self.processor = VLNQwen3VLProcessor.from_pretrained(self.base_path, padding_side="left")
        self.data_processor = Qwen3VLDataProcessor(
            self.processor,
            model_max_length=8192,
            video_fps=4,
            video_pool_enable=False,
            video_pool_spatial=1,
            video_pool_mode="avg",
            video_pool_stage="post_vit",
            video_size=tuple(self.settings["video_size"]),
            position_id_func=partial(
                video_position_ids, _position_id_func_from_config(self.base_path)
            ),
        )

    def _sample(self, frames, instruction):
        from lightnav.prompts import UNIFIED_TRAJ_PROMPT_TEMPLATE, to_rvq_prompt

        template = to_rvq_prompt(UNIFIED_TRAJ_PROMPT_TEMPLATE)
        # Keep the trained point-generation prefix byte-for-byte. The final
        # decoder uses the six new action rows instead of the RVQ vocabulary.
        # The ViT uses two-frame tubelets. A current-frame input repeats the same
        # RGB; it never reads a previous camera frame or a privileged sensor.
        if len(frames) == 1:
            frames = frames * 2
        video = torch.stack(frames)
        return {
            "video_segments": [
                {
                    "video": video,
                    "frame_indices": list(range(len(frames))),
                    "total_frames": len(frames),
                    "pool_spatial": 1,
                    "pool_mode": "avg",
                }
            ],
            "conversations": [
                {"from": "human", "value": template.format(videos="<video>", task=instruction)},
                {"from": "gpt", "value": "placeholder"},
            ],
            "video_fps": 4,
            "slowfast_abs_frame_indices": True,
            "_skip_normalize": True,
        }

    @torch.no_grad()
    def frame_features(self, rgb, instruction, *, history=None):
        from lightnav.inference.frame_preprocessing import rgb_frame_to_model_tensor

        frame = rgb_frame_to_model_tensor(rgb.cpu().numpy(), tuple(self.settings["video_size"]))
        frames = [frame] if history is None else [*history, frame]
        if len(frames) % 2:
            frames = [frames[0], *frames]
        inputs = self.data_processor.process_sample(
            self._sample(frames, instruction),
            add_generation_prompt=True,
            validate_video_shapes=False,
        )
        boundary = (inputs["labels"] != -100).nonzero(as_tuple=False)[0].item()
        ids = inputs["input_ids"][:boundary].unsqueeze(0).to(self.device)
        grid = inputs["video_grid_thw"].to(self.device)
        pixels = inputs["pixel_values_videos"].to(self.device, torch.bfloat16)
        vision = self.native.model.visual(pixels, grid_thw=grid, return_dict=True)
        positions = inputs["position_ids"][:, :boundary].unsqueeze(1).to(self.device)
        return FrameFeatures(
            ids, positions, vision.pooler_output, tuple(vision.deepstack_features)
        ), frame

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
    ):
        from transformers.integrations.sdpa_attention import sdpa_attention_forward
        from transformers.models.qwen3_vl.modeling_qwen3_vl import apply_rotary_pos_emb

        text: Any = self.native.model.language_model
        layer, attention = text.layers[index], text.layers[index].self_attn
        residual, norm = x, layer.input_layernorm(x)
        b, length, _ = norm.shape
        dim = attention.head_dim
        q = attention.q_norm(attention.q_proj(norm).view(b, length, -1, dim)).transpose(1, 2)
        k = attention.k_norm(attention.k_proj(norm).view(b, length, -1, dim)).transpose(1, 2)
        v = attention.v_proj(norm).view(b, length, -1, dim).transpose(1, 2)
        cos, sin = text.rotary_emb(norm, positions)
        q, k = apply_rotary_pos_emb(q, k, cos, sin)
        keys = k if local.conv is None else torch.cat((local.conv, k), dim=-2)
        values = v if local.recurrent is None else torch.cat((local.recurrent, v), dim=-2)
        y, _ = sdpa_attention_forward(
            attention, q, keys, values, attention_mask=None, dropout=0.0, scaling=attention.scaling
        )
        y = y.reshape(b, length, -1)
        recurrent = previous.recurrent
        if str(index) in self.memories and not native_only:
            memory = cast(KDAMemory, self.memories[str(index)])
            inputs = (q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2), norm)
            if packing is not None:
                inputs = tuple(
                    tensor.reshape(1, b * length, *tensor.shape[2:]).index_select(
                        1, packing.indices
                    )
                    for tensor in inputs
                )
            extra, recurrent = memory(
                *inputs,
                recurrent,
                mode=(
                    "fused"
                    if length == 1
                    and packing is None
                    and not torch.is_grad_enabled()
                    and self.settings.get("recurrent_decode", True)
                    else "chunk"
                ),
                cu_seqlens=packing.cu_seqlens if packing is not None else None,
                cu_seqlens_cpu=packing.cu_seqlens_cpu if packing is not None else None,
            )
            if packing is not None:
                extra = (
                    extra.new_zeros((b * length, extra.shape[-1]))
                    .index_copy(0, packing.indices, extra.squeeze(0))
                    .view(b, length, -1)
                )
                recurrent = recurrent[-1:]
            y = y + memory.residual_scale.tanh().to(y.dtype) * extra
        x = residual + attention.o_proj(y)
        x = x + layer.mlp(layer.post_attention_layernorm(x))
        if deepstack is not None:
            x = text._deepstack_process(x, visual_mask, deepstack)
        return x, recurrent, keys, values

    def _forward(self, *args, **kwargs):
        context = (
            torch.autocast(self.device.type, enabled=False)
            if self.language_compute_dtype == torch.float32
            else nullcontext()
        )
        with context:
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
    ):
        x, next_memory, next_local = embeddings, [], []
        for i, (previous, kv) in enumerate(zip(memory, local, strict=True)):
            feature = deepstack[i] if i < len(deepstack) else None
            args = (i, x, positions, previous, kv, visual_mask, feature, native_only, packing)
            if torch.is_grad_enabled() and self.settings.get("gradient_checkpointing", True):
                x, recurrent, keys, values = checkpoint(
                    self._layer, *args, use_reentrant=False, preserve_rng_state=False
                )
            else:
                x, recurrent, keys, values = self._layer(*args)
            next_memory.append(LayerState(None, recurrent))
            next_local.append(
                LayerState(keys, values) if packing is None else LayerState(None, None)
            )
        hidden = (
            x[:, -1]
            if hidden_indices is None
            else x[torch.arange(x.shape[0], device=x.device)[:, None], hidden_indices]
        )
        return FrameContext(
            self.native.model.language_model.norm(hidden),
            tuple(next_memory),
            tuple(next_local),
            int(positions.max()) + 1 if packing is None else 0,
        )

    def begin(self, feature, state, *, native_only=False):
        ids = feature.ids.to(self.device)
        x = self.native.model.language_model.embed_tokens(ids)
        mask = ids == self.native.config.video_token_id
        x = x.masked_scatter(mask.unsqueeze(-1), feature.visual.to(self.device, x.dtype))
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
        # Match the language residual/RoPE/QKV dtype; vision stays BF16.
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

    def replay_fragment(self, features, state, tokens, *, resets=None):
        """Teacher-force recorded tokens with frame-local attention and causal KDA.

        Returns batched logits/values and only the final frame's cache. Point
        tokens remain policy samples from the rollout. Episode boundaries reset
        KDA through cu_seqlens, while native attention never crosses a frame.
        """
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
            hidden_indices.append([n - 1, n, n + 1])
            valid_indices.extend(range(i * width, i * width + length))
            offsets.append(offsets[-1] + length)
        text = self.native.model.language_model
        x = text.embed_tokens(ids)
        visual_mask = ids == self.native.config.video_token_id
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
                apos_logits.argmax() if deterministic else Categorical(logits=apos_logits).sample()
            )
        )
        context = self.append(
            context, self.native.model.language_model.embed_tokens(self.apos_ids[apos])
        )
        opos_logits = self.point_logits(context, "opos")
        opos = (
            tokens[1]
            if tokens is not None and tokens[1] is not None
            else (
                opos_logits.argmax() if deterministic else Categorical(logits=opos_logits).sample()
            )
        )
        context = self.append(
            context, self.native.model.language_model.embed_tokens(self.opos_ids[opos])
        )
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

    @torch.no_grad()
    def teacher_points(self, feature, state, apos):
        with self.native_weights():
            context = self.begin(
                feature, self.start_episode(state.episode_id, state.instruction), native_only=True
            )
            a = self.point_logits(context, "apos")
            context = self.append(
                context,
                self.native.model.language_model.embed_tokens(self.apos_ids[apos]),
                native_only=True,
            )
            return a.softmax(-1), self.point_logits(context, "opos").softmax(-1)

    def forward_batch(self, rgb, states, *, return_perception=False):
        steps = [
            self.step(self.frame_features(frame, state.instruction)[0], state, deterministic=True)
            for frame, state in zip(rgb, states, strict=True)
        ]
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
                    "point_cot": True,
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
            "frozen_embedding_head_shared": self.base_embedding_head_shared,
            "memory_layers": len(self.memories),
            "native_local_attention_layers": 36,
            "frozen_base": self.base_path,
            "language_compute_dtype": str(self.language_compute_dtype),
            "recurrent_decode": self.settings.get("recurrent_decode", True),
            "float32_matmul_precision": self.float32_matmul_precision,
            "action_token_strings": self.action_token_strings,
            "action_initialization_codes": self.action_initialization_codes,
            "source_revision": json.loads(Path(self.base_path, "source_revision.json").read_text()),
        }
