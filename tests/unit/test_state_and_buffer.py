import pytest
import torch

from streamnav.contracts.state import LayerState, StreamingState
from streamnav.models.qwen35_kda.cache import clone_state, stack_caches, state_bytes, unstack_cache
from streamnav.training.rollout_buffer import RecurrentRolloutBuffer


def state():
    tensor = torch.randn(1, 2, 3, 3, requires_grad=True) * 2
    return StreamingState((LayerState(None, tensor),), "A", "hash", 5, "chair")


def test_cache_detach_and_clone_ownership():
    original = state()
    copied = clone_state(original)
    assert copied.kda_cache[0].recurrent.grad_fn is None
    assert copied.kda_cache[0].recurrent.data_ptr() != original.kda_cache[0].recurrent.data_ptr()
    before = original.kda_cache[0].recurrent.detach().clone()
    copied.kda_cache[0].recurrent.zero_()
    torch.testing.assert_close(original.kda_cache[0].recurrent.detach(), before)


def test_cache_batch_roundtrip():
    a, b = state(), state()
    cache = stack_caches([a, b])
    parts = unstack_cache(cache, 2)
    torch.testing.assert_close(parts[0][0].recurrent, a.kda_cache[0].recurrent)
    torch.testing.assert_close(parts[1][0].recurrent, b.kda_cache[0].recurrent)
    assert state_bytes(a) == 72


def test_contiguous_sequences_cover_rollout_exactly():
    buffer = RecurrentRolloutBuffer(8, 3, (8, 8, 3), 4, 0.8)
    covered = []
    for batch in buffer.sequence_batches(2):
        for seq in batch:
            assert seq.stop - seq.start == 4
            covered.extend((t, seq.env) for t in range(seq.start, seq.stop))
    assert sorted(covered) == [(t, e) for t in range(8) for e in range(3)]
    with pytest.raises(ValueError):
        RecurrentRolloutBuffer(7, 3, (8, 8, 3), 4, 0.8)
