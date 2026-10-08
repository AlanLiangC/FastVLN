from hashlib import sha256

import torch
from torch import nn

from streamnav.contracts.action import NavigationAction
from streamnav.contracts.state import PolicyOutput, StreamingState
from streamnav.models.policy.action_distribution import ObjectNavActionDistribution
from streamnav.models.policy.actor_critic import NavigationActorCritic, NavigationPooling
from streamnav.models.qwen35_kda.cache import stack_caches, unstack_cache


class StreamingObjectNavPolicy(nn.Module):
    loaded_checkpoint: str

    def __init__(
        self, backbone, value_hidden_dim=512, action_dim=6, critic_type="linear", critic_gain=1.0
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
        self.distribution = ObjectNavActionDistribution()
        self.loaded_checkpoint = ""

    def start_episode(self, episode_id, instruction):
        cache = self.backbone.prefill(instruction)
        return StreamingState(
            cache, episode_id, sha256(instruction.encode()).hexdigest(), 0, instruction
        )

    reset = start_episode

    def forward_batch(self, rgb, states, visual_embeddings=None):
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
        logits, values = self.actor_critic(self.pooling(hidden))
        caches = unstack_cache(cache, len(states))
        return (
            logits.float(),
            values.float(),
            [s.with_cache(c) for s, c in zip(states, caches, strict=True)],
        )

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
