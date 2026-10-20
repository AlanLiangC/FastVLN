from dataclasses import dataclass, replace

import torch


@dataclass(frozen=True)
class LayerState:
    conv: torch.Tensor | None
    recurrent: torch.Tensor | None


@dataclass(frozen=True)
class StreamingState:
    kda_cache: tuple[LayerState, ...]
    episode_id: str
    instruction_hash: str
    step_index: int
    instruction: str
    next_action_probabilities: torch.Tensor | None = None

    def with_cache(self, cache, *, next_action_probabilities=None):
        return replace(
            self,
            kda_cache=cache,
            step_index=self.step_index + 1,
            next_action_probabilities=next_action_probabilities,
        )

    def after_action(self, collision=False):
        return replace(self, next_action_probabilities=None) if collision else self


@dataclass
class PolicyOutput:
    logits: torch.Tensor
    value: torch.Tensor
    state: StreamingState
