import os
from dataclasses import replace
from pathlib import Path

import pytest
import torch

from streamnav.models.policy.streaming_policy import StreamingObjectNavPolicy
from streamnav.models.qwen35_kda.backbone import Qwen35KDABackbone
from streamnav.models.qwen35_kda.cache import clone_state, state_bytes

CHECKPOINT = os.environ.get("STREAMNAV_CHECKPOINT", "checkpoints/qwen35_0p8b_kda")
pytestmark = [
    pytest.mark.gpu,
    pytest.mark.skipif(
        not torch.cuda.is_available() or not Path(CHECKPOINT).exists(),
        reason="Converted checkpoint and CUDA required",
    ),
]


def test_fresh_episode_isolation_and_checkpoint_load():
    policy = StreamingObjectNavPolicy(
        Qwen35KDABackbone.from_converted(CHECKPOINT, image_size=[270, 480])
    ).eval()
    rgb = torch.zeros(270, 480, 3, dtype=torch.uint8)
    with torch.no_grad():
        state_a = policy.start_episode("A", "Find a chair.")
        policy.forward_step(rgb, state_a)
        state_b = policy.reset("B", "Find a bed.")
        after_reset = policy.forward_step(rgb, state_b)
        fresh_b = policy.forward_step(rgb, policy.start_episode("B", "Find a bed."))
    torch.testing.assert_close(after_reset.logits, fresh_b.logits, atol=0, rtol=0)
    assert after_reset.state.episode_id == "B" and after_reset.state.step_index == 1
    assert state_a.step_index == 0


def test_per_step_goal_survives_detached_identical_memory_and_has_gradient():
    policy = StreamingObjectNavPolicy(
        Qwen35KDABackbone.from_converted(
            CHECKPOINT,
            dtype=torch.float32,
            image_size=64,
            inference_mode="chunk",
            goal_conditioning="nav_query",
            kda_output_norm=True,
        )
    ).eval()
    policy.backbone.vision.requires_grad_(False)
    rgb = torch.zeros(2, 64, 64, 3, dtype=torch.uint8)
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        original = policy.start_episode("A", "Find a chair.")
        states = [clone_state(original), replace(clone_state(original), instruction="Find a bed.")]
        sizes = []
        for _ in range(4):
            logits, _, states = policy.forward_batch(rgb, states)
            assert (logits[0] - logits[1]).abs().max().item() > 1e-5
            sizes.append(state_bytes(states[0]))
        assert len(set(sizes)) == 1
        cached, _, _ = policy.forward_batch(rgb, [clone_state(s) for s in states])
        assert len(policy.backbone._goal_ids_cache) == 2
        policy.backbone._goal_ids_cache.clear()
        uncached, _, _ = policy.forward_batch(rgb, [clone_state(s) for s in states])
        torch.testing.assert_close(cached, uncached, atol=0, rtol=0)
    # Simulate a TBPTT boundary. Text embeddings still get action-loss gradients
    # even though episode prefill and the entire incoming memory are detached.
    with torch.autocast("cuda", dtype=torch.bfloat16):
        logits, _, _ = policy.forward_batch(rgb, [clone_state(s) for s in states])
        logits[:, 1].sum().backward()
    ids = policy.backbone.tokenizer("Find a bed.", add_special_tokens=False).input_ids
    gradient = policy.backbone.embeddings.weight.grad[ids]
    assert torch.isfinite(gradient).all() and gradient.abs().sum().item() > 0
