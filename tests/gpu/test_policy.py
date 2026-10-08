import os
from pathlib import Path

import pytest
import torch

from streamnav.models.policy.streaming_policy import StreamingObjectNavPolicy
from streamnav.models.qwen35_kda.backbone import Qwen35KDABackbone

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
