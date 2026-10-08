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

    def with_cache(self, cache):
        return replace(self, kda_cache=cache, step_index=self.step_index + 1)


@dataclass
class PolicyOutput:
    logits: torch.Tensor
    value: torch.Tensor
    state: StreamingState
