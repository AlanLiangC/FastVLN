from hashlib import sha256

import torch
from torch import nn

from streamnav.contracts.action import NavigationAction
from streamnav.contracts.perception import perception_config
from streamnav.contracts.state import PolicyOutput, StreamingState
from streamnav.models.policy.action_distribution import ObjectNavActionDistribution
from streamnav.models.policy.actor_critic import NavigationActorCritic, NavigationPooling
from streamnav.models.policy.perception import NavigationPerception
from streamnav.models.qwen35_kda.cache import stack_caches, unstack_cache


class StreamingObjectNavPolicy(nn.Module):
    loaded_checkpoint: str

    def __init__(
        self,
        backbone,
        value_hidden_dim=512,
        action_dim=6,
        critic_type="linear",
        critic_gain=1.0,
        perception=None,
    ):
        super().__init__()
        self.backbone = backbone
        self.pooling = NavigationPooling()
        self.actor_critic = NavigationActorCritic(
            backbone.config.text_config.hidden_size,
            value_hidden_dim,
            action_dim,
            critic_type,
            critic_gain,
        )
        self.actor_critic.to(device=backbone.device, dtype=backbone.nav_token.dtype)
        self.perception_config = perception_config(perception)
        self.perception = (
            NavigationPerception(
                backbone.config.text_config.hidden_size,
                action_dim,
                self.perception_config["embedding_dim"],
            ).to(device=backbone.device, dtype=backbone.nav_token.dtype)
            if self.perception_config["enabled"]
            else None
        )
        self.distribution = ObjectNavActionDistribution()
        self.loaded_checkpoint = ""
        self.batch_chat_body = False

    def start_episode(self, episode_id, instruction):
        cache = self.backbone.prefill(instruction)
        return StreamingState(
            cache, episode_id, sha256(instruction.encode()).hexdigest(), 0, instruction
        )

    reset = start_episode

    def readout(self, hidden):
        logits, values = self.actor_critic(hidden)
        predictions = None
        if self.perception is not None:
            predictions, residual = self.perception(hidden)
            logits = logits.float() + residual
        return logits.float(), values.float(), predictions

    def forward_batch(self, rgb, states, visual_embeddings=None, *, return_perception=False):
        if getattr(self.backbone, "goal_conditioning", "episode") == "chat_query":
            return self._forward_chat_batch(rgb, states, visual_embeddings, return_perception)
        instructions = (
            [s.instruction for s in states]
            if getattr(self.backbone, "goal_conditioning", "episode") == "nav_query"
            else None
        )
        tokens = (
            self.backbone.encode_rgb(rgb, instructions=instructions)
            if visual_embeddings is None
            else self.backbone.encode_visual_tokens(visual_embeddings, instructions=instructions)
        )
        hidden, cache = self.backbone.recurrent_forward(tokens, stack_caches(states))
        logits, values, predictions = self.readout(self.pooling(hidden))
        caches = unstack_cache(cache, len(states))
        output = (
            logits.float(),
            values.float(),
            [s.with_cache(c) for s, c in zip(states, caches, strict=True)],
        )
        return (*output, predictions) if return_perception else output

    def _forward_chat_batch(self, rgb, states, visual_embeddings, return_perception=False):
        visual = (
            self.backbone.encode_vision(rgb) if visual_embeddings is None else visual_embeddings
        )
        tokens = self.backbone.encode_chat_visual_tokens(
            visual, [s.instruction for s in states], [s.step_index == 0 for s in states]
        )
        if self.batch_chat_body:
            lengths = tuple(t.shape[0] for t in tokens)
            hidden, cache = self.backbone.recurrent_forward(
                torch.cat(tokens).unsqueeze(0), stack_caches(states), lengths=lengths
            )
            ends = torch.tensor(lengths, device=hidden.device).cumsum(0) - 1
            logits, values, predictions = self.readout(hidden[0, ends])
            caches = unstack_cache(cache, len(states))
            output = (
                logits.float(),
                values.float(),
                [s.with_cache(c) for s, c in zip(states, caches, strict=True)],
            )
            return (*output, predictions) if return_perception else output
        # No padding: KDA/GDN have no padding mask. Always use batch width one
        # for this variable-length body, even when two goals have equal length.
        # Shuffled replay can pair that environment with a different length;
        # switching between B=1/B=2 changes BF16 kernels and behavior log-probs.
        # Vision and heads retain the collector/replay batch width.
        readouts, updated = [], []
        for sequence, state in zip(tokens, states, strict=True):
            hidden, cache = self.backbone.recurrent_forward(sequence.unsqueeze(0), state.kda_cache)
            readouts.append(self.pooling(hidden)[0])
            updated.append(state.with_cache(cache))
        logits, values, predictions = self.readout(torch.stack(readouts))
        output = (logits, values, updated)
        return (*output, predictions) if return_perception else output

    def forward_step(self, rgb, state):
        logits, value, states = self.forward_batch(
            rgb.unsqueeze(0) if rgb.ndim == 3 else rgb, [state]
        )
        return PolicyOutput(logits[0], value[0], states[0])

    @torch.no_grad()
    def act(self, rgb, state, deterministic=True):
        output = self.forward_step(rgb, state)
        action = (
            output.logits.argmax(-1)
            if deterministic
            else self.distribution.build(output.logits).sample()
        )
        return NavigationAction(action.item()), output
