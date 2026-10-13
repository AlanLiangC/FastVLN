from dataclasses import replace

import torch

from streamnav.contracts.state import LayerState, StreamingState


def clone_state(state: StreamingState, detach: bool = True) -> StreamingState:
    def copy(tensor):
        if tensor is None:
            return None
        return (tensor.detach() if detach else tensor).clone()

    return replace(
        state,
        kda_cache=tuple(LayerState(copy(s.conv), copy(s.recurrent)) for s in state.kda_cache),
    )


def state_bytes(state: StreamingState) -> int:
    return sum(
        t.numel() * t.element_size()
        for s in state.kda_cache
        for t in (s.conv, s.recurrent)
        if t is not None
    )


def stack_caches(states: list[StreamingState]) -> tuple[LayerState, ...]:
    def stack(items):
        if all(t is None for t in items):
            return None
        if any(t is None for t in items):
            raise ValueError("Cannot batch initialized and empty caches")
        return torch.cat(items, dim=0)

    return tuple(
        LayerState(
            stack([s.kda_cache[i].conv for s in states]),
            stack([s.kda_cache[i].recurrent for s in states]),
        )
        for i in range(len(states[0].kda_cache))
    )


def unstack_cache(cache, batch_size):
    return [
        tuple(
            LayerState(
                s.conv[i : i + 1] if s.conv is not None else None,
                s.recurrent[i : i + 1] if s.recurrent is not None else None,
            )
            for s in cache
        )
        for i in range(batch_size)
    ]
