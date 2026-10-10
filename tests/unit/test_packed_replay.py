from types import SimpleNamespace

import pytest
import torch

from streamnav.contracts.state import LayerState
from streamnav.models.policy.streaming_policy import StreamingObjectNavPolicy
from streamnav.training.rollout import replay_sequences
from streamnav.training.rollout_buffer import RecurrentRolloutBuffer


class TinyChatBody(torch.nn.Module):
    """Tokenwise recurrence makes reset, readout and gradient checks exact."""

    def __init__(self):
        super().__init__()
        self.embeddings = torch.nn.Embedding(16, 4, dtype=torch.float64)
        self.nav_token = torch.nn.Parameter(torch.zeros(1, 1, 4, dtype=torch.float64))
        self.vision = torch.nn.Linear(4, 4).requires_grad_(False)
        self.config = SimpleNamespace(text_config=SimpleNamespace(hidden_size=4))
        self.goal_conditioning = "chat_query"

    @property
    def device(self):
        return torch.device("cpu")

    def prefill(self, instruction):
        return self.recurrent_forward(self.embeddings(torch.tensor([[0]])))[1]

    def encode_chat_visual_tokens(self, visual, instructions, first_frames):
        return [
            torch.cat(
                (
                    self.embeddings(torch.tensor([1 if first else 2])),
                    frame,
                    self.embeddings(torch.tensor([3 if goal == "chair" else 4])),
                )
            )
            for frame, goal, first in zip(visual, instructions, first_frames, strict=True)
        ]

    def recurrent_forward(self, tokens, cache=None):
        state = torch.zeros_like(tokens[:, 0]) if cache is None else cache[0].recurrent
        outputs = []
        for token in tokens.unbind(1):
            state = 0.8 * state + token
            outputs.append(state.tanh())
        return torch.stack(outputs, dim=1), (LayerState(None, state),)


def make_rollout():
    torch.manual_seed(41)
    policy = StreamingObjectNavPolicy(TinyChatBody(), critic_gain=0.3)
    buffer = RecurrentRolloutBuffer(7, 2, (1, 1, 3), 7, 0.0)
    buffer.visual_embeddings = torch.randn(7, 2, 3, 4, dtype=torch.float64, requires_grad=True)
    states = [policy.start_episode("a", "chair"), policy.start_episode("b", "bed")]
    # A continuing episode and a fresh episode share the minibatch.
    states[0] = states[0].with_cache(states[0].kda_cache)
    buffer.save_boundary(0, states)
    buffer.resets = {(0, 2): ("c", "bed"), (1, 3): ("d", "chair"), (0, 6): ("e", "chair")}
    return policy, buffer, next(buffer.sequence_batches(2, shuffle=False))


@pytest.mark.parametrize("pack_frames", [2, 3, 7, 100])
def test_packing_preserves_mixed_resets_readouts_and_full_bptt(pack_frames):
    policy, buffer, sequences = make_rollout()
    expected = replay_sequences(policy, buffer, sequences)
    (expected[0].square().sum() + expected[1].square().sum()).backward()
    gradients = {
        name: p.grad.clone() for name, p in policy.named_parameters() if p.grad is not None
    }
    visual_gradient = buffer.visual_embeddings.grad.clone()
    policy.zero_grad(set_to_none=True)
    buffer.visual_embeddings.grad = None
    actual = replay_sequences(policy, buffer, sequences, pack_frames=pack_frames)
    for a, b in zip(actual, expected, strict=True):
        torch.testing.assert_close(a, b, atol=0, rtol=0)
    (actual[0].square().sum() + actual[1].square().sum()).backward()
    for name, parameter in policy.named_parameters():
        if name in gradients:
            torch.testing.assert_close(parameter.grad, gradients[name], atol=1e-12, rtol=1e-12)
    torch.testing.assert_close(
        buffer.visual_embeddings.grad, visual_gradient, atol=1e-12, rtol=1e-12
    )


def test_packed_readout_cannot_use_future_frames_or_previous_episode():
    policy, buffer, sequences = make_rollout()
    logits, _ = replay_sequences(policy, buffer, sequences, pack_frames=7)
    logits[4, 0].sum().backward()
    gradient = buffer.visual_embeddings.grad
    assert gradient[2:5, 0].abs().sum() > 0
    assert gradient[:2].abs().sum() == 0
    assert gradient[5:].abs().sum() == 0
    assert gradient[:, 1].abs().sum() == 0


def test_packing_requires_supported_inputs():
    policy, buffer, sequences = make_rollout()
    with pytest.raises(ValueError, match="positive"):
        replay_sequences(policy, buffer, sequences, pack_frames=0)
    buffer.visual_embeddings = None
    with pytest.raises(ValueError, match="requires chat_query"):
        replay_sequences(policy, buffer, sequences, pack_frames=2)
