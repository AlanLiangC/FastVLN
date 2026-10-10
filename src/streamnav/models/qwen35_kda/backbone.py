import json
from contextlib import contextmanager, nullcontext
from pathlib import Path

import torch
from safetensors.torch import load_file
from torch import nn
from torch.utils.checkpoint import checkpoint

from streamnav.contracts.state import LayerState
from streamnav.errors import KDACompatibilityError
from streamnav.models.qwen35_kda.kda_adapter import KDAAdapter, StreamingGatedDeltaNet
from streamnav.models.vision.preprocessing import patchify


class RecurrentDecoderLayer(nn.Module):
    def __init__(self, layer, kda_output_norm=False):
        super().__init__()
        self.input_layernorm = layer.input_layernorm
        self.post_attention_layernorm = layer.post_attention_layernorm
        self.mlp = layer.mlp
        self.mixer = (
            KDAAdapter(layer.self_attn, output_norm=kda_output_norm)
            if layer.layer_type == "full_attention"
            else StreamingGatedDeltaNet(layer.linear_attn)
        )

    def forward(self, x, conv, recurrent, mode="auto", cu=None, cu_cpu=None, lengths=None):
        y, state = self.mixer(
            self.input_layernorm(x), LayerState(conv, recurrent), mode, cu, cu_cpu, lengths
        )
        x = x + y
        x = x + self.mlp(self.post_attention_layernorm(x))
        return x, state.conv, state.recurrent


class Qwen35KDABackbone(nn.Module):
    def __init__(
        self,
        model,
        tokenizer,
        image_size=224,
        gradient_checkpointing=True,
        inference_mode="auto",
        goal_conditioning="episode",
        kda_output_norm=False,
    ):
        super().__init__()
        self.config = model.config
        self.vision = model.visual
        self.embeddings = model.language_model.embed_tokens
        self.layers = nn.ModuleList(
            RecurrentDecoderLayer(layer, kda_output_norm=kda_output_norm)
            for layer in model.language_model.layers
        )
        self.kda_output_norm = kda_output_norm
        self.norm = model.language_model.norm
        self.tokenizer = tokenizer
        self.image_size = image_size
        self.gradient_checkpointing = gradient_checkpointing
        if inference_mode not in ("auto", "chunk", "recurrent"):
            raise ValueError("inference_mode must be auto, chunk or recurrent")
        self.inference_mode = inference_mode
        if goal_conditioning not in ("episode", "nav_query", "chat_query"):
            raise ValueError("goal_conditioning must be episode, nav_query or chat_query")
        self.goal_conditioning = goal_conditioning
        self._goal_ids_cache: dict[str, tuple[int, ...]] = {}
        self._chat_ids_cache: dict[str, tuple[tuple[int, ...], ...]] = {}
        self._assistant_close_ids: tuple[int, ...] | None = None
        self._replay_embeddings: dict[tuple[int, ...], torch.Tensor] | None = None
        hidden = self.config.text_config.hidden_size
        # Retain checkpoint compatibility; chat_query reads the assistant prefix.
        self.nav_token = nn.Parameter(
            torch.empty(1, 1, hidden), requires_grad=goal_conditioning != "chat_query"
        )
        nn.init.normal_(self.nav_token, std=0.02)

    @property
    def device(self):
        return self.nav_token.device

    def recurrent_forward(self, tokens, cache=None, mode="auto", lengths=None):
        if mode == "auto" and not torch.is_grad_enabled():
            mode = self.inference_mode
        if cache is None:
            cache = tuple(LayerState(None, None) for _ in self.layers)
        if len(cache) != len(self.layers):
            raise KDACompatibilityError("Cache depth differs from converted model")
        cu, cu_cpu = None, None
        if lengths is not None:
            if tokens.shape[0] != 1 or sum(lengths) != tokens.shape[1] or min(lengths) < 1:
                raise ValueError("Variable chat batches require positive lengths and flat tokens")
            cu_cpu = torch.tensor([0, *lengths], dtype=torch.int32).cumsum(0).to(torch.int32)
            cu = cu_cpu.to(tokens.device)
        x = tokens
        updated = []
        for layer, state in zip(self.layers, cache, strict=True):
            if self.gradient_checkpointing and torch.is_grad_enabled():
                x, conv, recurrent = checkpoint(
                    layer,
                    x,
                    state.conv,
                    state.recurrent,
                    mode,
                    cu,
                    cu_cpu,
                    lengths,
                    use_reentrant=False,
                )
            else:
                x, conv, recurrent = layer(
                    x, state.conv, state.recurrent, mode, cu, cu_cpu, lengths
                )
            updated.append(LayerState(conv, recurrent))
        return self.norm(x), tuple(updated)

    def prefill(self, instruction):
        if self.goal_conditioning == "chat_query":
            system, _, _ = self.chat_token_ids(instruction)
            return self.recurrent_forward(self.token_embeddings(system).unsqueeze(0))[1]
        prompt = self.tokenizer.apply_chat_template(
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
        ids = self.tokenizer(prompt, return_tensors="pt").input_ids.to(self.device)
        return self.recurrent_forward(self.embeddings(ids))[1]

    def chat_token_ids(self, instruction):
        """Split the official multimodal chat without padding or GPU ID caches."""
        cached = self._chat_ids_cache.get(instruction)
        if cached is None:
            self.goal_token_ids(instruction)  # Reject empty goals consistently.
            prompt = self.tokenizer.apply_chat_template(
                [
                    {
                        "role": "system",
                        "content": "You are a fast object navigation policy. Navigate to the requested object.",
                    },
                    {
                        "role": "user",
                        "content": [{"type": "image"}, {"type": "text", "text": instruction}],
                    },
                ],
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False,
            )
            ids = tuple(self.tokenizer(prompt, add_special_tokens=False).input_ids)
            positions = [i for i, token in enumerate(ids) if token == self.config.image_token_id]
            if len(positions) != 1:
                raise ValueError("Chat frame must contain exactly one image placeholder")
            image = positions[0]
            message_start = self.tokenizer.convert_tokens_to_ids("<|im_start|>")
            starts = [i for i, token in enumerate(ids[:image]) if token == message_start]
            if len(starts) != 2:
                raise ValueError("Chat frame must begin with a system message and a user message")
            user = starts[1]
            cached = (ids[:user], ids[user:image], ids[image + 1 :])
            if len(self._chat_ids_cache) >= 256:
                self._chat_ids_cache.pop(next(iter(self._chat_ids_cache)))
            self._chat_ids_cache[instruction] = cached
        return cached

    def goal_token_ids(self, instruction):
        ids = self._goal_ids_cache.get(instruction)
        if ids is None:
            ids = tuple(self.tokenizer(instruction, add_special_tokens=False).input_ids)
            if not ids:
                raise ValueError("Goal instructions must contain at least one token")
            if len(self._goal_ids_cache) >= 256:
                self._goal_ids_cache.pop(next(iter(self._goal_ids_cache)))
            self._goal_ids_cache[instruction] = ids
        return ids

    @contextmanager
    def reuse_token_embeddings(self):
        """Share differentiable lookups during exactly one replay forward.

        Weights cannot change within that forward. Clearing on exit lets every
        subsequent minibatch read new weights and build a new autograd graph.
        """
        if self._replay_embeddings is not None:
            raise RuntimeError("Token embedding reuse contexts cannot be nested")
        self._replay_embeddings = {}
        try:
            yield
        finally:
            self._replay_embeddings = None

    def token_embeddings(self, ids):
        cache = self._replay_embeddings
        if cache is None:
            return self.embeddings(torch.tensor(ids, device=self.device))
        if ids not in cache:
            cache[ids] = self.embeddings(torch.tensor(ids, device=self.device))
        return cache[ids]

    def encode_vision(self, rgb):
        pixels, grid = patchify(rgb.to(self.device), self.image_size)
        visual = self.vision(pixels.to(self.nav_token.dtype), grid_thw=grid).pooler_output
        return visual.reshape(grid.shape[0], -1, self.config.text_config.hidden_size)

    def encode_rgb(self, rgb, instructions=None):
        return self.encode_visual_tokens(self.encode_vision(rgb), instructions)

    def encode_visual_tokens(self, visual, instructions=None):
        if self.goal_conditioning == "chat_query":
            raise ValueError(
                "chat_query frames require encode_chat_visual_tokens and episode state"
            )
        if visual.ndim != 3 or visual.shape[-1] != self.config.text_config.hidden_size:
            raise ValueError("Visual embeddings must be batch x tokens x hidden_size")
        # Explicit NAV readout token; image boundary embeddings preserve pretrained conventions.
        markers = self.token_embeddings(
            (self.config.vision_start_token_id, self.config.vision_end_token_id)
        )
        b = visual.shape[0]
        query = self.nav_token.expand(b, -1, -1)
        if self.goal_conditioning == "nav_query":
            if instructions is None or len(instructions) != b:
                raise ValueError("nav_query requires one goal instruction per RGB frame")
            # A fixed-size text query supplies the goal at every action and every
            # truncated-BPTT segment. Do not rely solely on a detached episode
            # prefix surviving thousands of recurrent visual-token updates.
            # Pool each goal separately: batch padding must not alter its query.
            # Cache only immutable CPU token IDs. Embeddings must be read anew
            # on every forward so optimizer updates and autograd remain valid.
            ids = [self.goal_token_ids(instruction) for instruction in instructions]
            goals = torch.stack([self.token_embeddings(row).mean(0) for row in ids])
            query = query + goals[:, None, :]
        return torch.cat(
            (
                markers[0:1].expand(b, -1, -1),
                visual,
                markers[1:2].expand(b, -1, -1),
                query,
            ),
            dim=1,
        )

    def encode_chat_visual_tokens(self, visual, instructions, first_frames):
        if visual.ndim != 3 or visual.shape[-1] != self.config.text_config.hidden_size:
            raise ValueError("Visual embeddings must be batch x tokens x hidden_size")
        if len(instructions) != visual.shape[0] or len(first_frames) != visual.shape[0]:
            raise ValueError("chat_query requires one goal and episode position per frame")
        # The previous readout is an empty assistant turn, with no action text.
        # Close it before the next user observation. First frames follow the
        # system prefill directly and match the single-frame diagnostic prompt.
        if self._assistant_close_ids is None:
            self._assistant_close_ids = tuple(
                self.tokenizer("<|im_end|>\n", add_special_tokens=False).input_ids
            )
        close = self._assistant_close_ids
        sequences = []
        for frame, instruction, first in zip(visual, instructions, first_frames, strict=True):
            _, before, after = self.chat_token_ids(instruction)
            prefix = before if first else close + before
            sequences.append(
                torch.cat((self.token_embeddings(prefix), frame, self.token_embeddings(after)))
            )
        return sequences

    @classmethod
    def from_converted(cls, path, device="cuda:0", dtype=torch.bfloat16, **kwargs):
        from transformers import AutoTokenizer, Qwen3_5Config, Qwen3_5Model

        path = Path(path)
        saved_config = path / "resolved_config.yaml"
        if saved_config.exists():
            import yaml

            saved_model = yaml.safe_load(saved_config.read_text())["model"]
            for key, default in (("goal_conditioning", "episode"), ("kda_output_norm", False)):
                kwargs.setdefault(key, saved_model.get(key, default))
        if not (path / "kda_layout.json").is_file():
            raise KDACompatibilityError(
                f"Not a converted checkpoint: {path}; run tools/convert_qwen35_to_kda.py"
            )
        layout = json.loads((path / "kda_layout.json").read_text())
        if layout["format_version"] != 1:
            raise KDACompatibilityError("Unsupported KDA checkpoint version")
        config = Qwen3_5Config.from_pretrained(path)
        config._attn_implementation = "sdpa"
        device = torch.device(device)
        guard = torch.cuda.device(device) if device.type == "cuda" else nullcontext()
        with guard, torch.device("meta"):
            model = Qwen3_5Model(config)
            backbone = cls(model, AutoTokenizer.from_pretrained(path / "tokenizer"), **kwargs)
        weights = load_file(str(path / "model.safetensors"), device="cpu")
        backbone.load_state_dict(weights, strict=True, assign=True)
        # This is the only nonpersistent buffer retained by our backbone. Rebuild
        # from its configuration; never leave an uninitialized meta buffer behind.
        rotary = backbone.vision.rotary_pos_emb
        backbone.vision.rotary_pos_emb = type(rotary)(rotary.dim, rotary.theta)
        if any(t.is_meta for t in backbone.buffers()):
            raise KDACompatibilityError("Unexpected uninitialized nonpersistent buffer")
        return backbone.to(device=device, dtype=dtype)
