import os
from pathlib import Path

import pytest
import torch

from streamnav.models.policy.streaming_policy import StreamingObjectNavPolicy
from streamnav.models.qwen35_kda.backbone import Qwen35KDABackbone
from streamnav.models.qwen35_kda.cache import clone_state, state_bytes

CHECKPOINT = os.environ.get(
    "STREAMNAV_CHAT_CHECKPOINT", "checkpoints/qwen35_0p8b_kda_calibrated_20261008"
)
pytestmark = [
    pytest.mark.gpu,
    pytest.mark.skipif(
        not torch.cuda.is_available() or not Path(CHECKPOINT).exists(),
        reason="Converted chat checkpoint and CUDA required",
    ),
]


def test_variable_chat_batch_preserves_independent_histories_and_gradients():
    torch.manual_seed(42)
    body = Qwen35KDABackbone.from_converted(
        CHECKPOINT,
        dtype=torch.float32,
        goal_conditioning="chat_query",
        kda_output_norm=True,
        inference_mode="chunk",
    )
    body.vision.requires_grad_(False)
    policy = StreamingObjectNavPolicy(body, critic_gain=0.3).eval()
    frames = torch.empty(3, 1, 1, 3, dtype=torch.uint8)
    visual = torch.randn(3, 135, 1024, device=body.device, dtype=torch.bfloat16)
    goals = ["Find a chair.", "Find a very large dining room table.", "Find a bed."]
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        initial = [policy.start_episode(str(i), goal) for i, goal in enumerate(goals)]
        reference, _, states = policy.forward_batch(frames, initial, visual)
        policy.batch_chat_body = True
        actual, _, batched_states = policy.forward_batch(frames, initial, visual)
        torch.testing.assert_close(actual, reference, atol=0.03, rtol=0.03)
        assert [state_bytes(s) for s in batched_states] == [state_bytes(s) for s in states]
        altered = visual.clone()
        altered[1] = 0
        changed, _, changed_states = policy.forward_batch(frames, initial, altered)
        torch.testing.assert_close(changed[[0, 2]], actual[[0, 2]], atol=0, rtol=0)
        for i in (0, 2):
            for a, b in zip(changed_states[i].kda_cache, batched_states[i].kda_cache, strict=True):
                if a.conv is not None:
                    torch.testing.assert_close(a.conv, b.conv, atol=0, rtol=0)
                torch.testing.assert_close(a.recurrent, b.recurrent, atol=0, rtol=0)
        mixed = [states[0], policy.start_episode("new", goals[1]), states[2]]
        policy.batch_chat_body = False
        reference, _, _ = policy.forward_batch(frames, mixed, visual)
        policy.batch_chat_body = True
        actual, _, continued = policy.forward_batch(frames, mixed, visual)
        torch.testing.assert_close(actual, reference, atol=0.04, rtol=0.04)
        assert [s.step_index for s in continued] == [2, 1, 2]
        order = [2, 0, 1]
        shuffled, _, _ = policy.forward_batch(
            frames[order], [mixed[i] for i in order], visual[order]
        )
        torch.testing.assert_close(shuffled, actual[order], atol=0.02, rtol=0.02)
    with torch.autocast("cuda", dtype=torch.bfloat16), body.reuse_token_embeddings():
        logits, values, next_states = policy.forward_batch(
            frames, [clone_state(s) for s in mixed], visual
        )
        next_logits, _, _ = policy.forward_batch(frames, next_states, visual)
        (logits.square().sum() + values.square().sum() + next_logits.square().sum()).backward()
    gradients = [p.grad for p in body.parameters() if p.grad is not None]
    assert gradients and all(torch.isfinite(g).all() for g in gradients)
    assert body.embeddings.weight.grad[list(body.goal_token_ids(goals[1]))].abs().sum() > 0
