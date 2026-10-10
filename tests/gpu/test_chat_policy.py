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
        reason="Calibrated checkpoint and CUDA required",
    ),
]


def test_chat_streaming_matches_prompt_and_replays_mixed_lengths_with_gradients():
    torch.manual_seed(9021)
    body = Qwen35KDABackbone.from_converted(
        CHECKPOINT,
        dtype=torch.float32,
        image_size=[270, 480],
        goal_conditioning="chat_query",
        kda_output_norm=True,
        inference_mode="chunk",
    )
    body.vision.requires_grad_(False)
    policy = StreamingObjectNavPolicy(body, critic_type="linear", critic_gain=0.0).eval()
    goals = ["Find a chair.", "Find a large dining room table.", "Find a bed."]
    frames = torch.randint(0, 256, (3, 270, 480, 3), dtype=torch.uint8)
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        visual = body.encode_vision(frames)
        initial = [policy.start_episode(str(i), goal) for i, goal in enumerate(goals)]
        system, before, after = body.chat_token_ids(goals[0])
        prompt = body.tokenizer.apply_chat_template(
            [
                {
                    "role": "system",
                    "content": "You are a fast object navigation policy. Navigate to the requested object.",
                },
                {
                    "role": "user",
                    "content": [{"type": "image"}, {"type": "text", "text": goals[0]}],
                },
            ],
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
        ids = body.tokenizer(prompt, add_special_tokens=False).input_ids
        assert list(system + before + (body.config.image_token_id,) + after) == ids
        whole = torch.cat(
            (body.token_embeddings(system + before), visual[0], body.token_embeddings(after))
        ).unsqueeze(0)
        hidden, _ = body.recurrent_forward(whole)
        expected, _ = policy.actor_critic(hidden[:, -1])
        actual, _, _ = policy.forward_batch(frames[:1], initial[:1], visual[:1])
        # Whole-prompt and segmented FLA chunks have small BF16 roundoff.
        torch.testing.assert_close(actual, expected.float(), atol=0.02, rtol=0.03)

        head_inputs = []
        hook = policy.actor_critic.register_forward_pre_hook(
            lambda _, args: head_inputs.append(args[0].clone())
        )
        logits, _, states = policy.forward_batch(frames, initial, visual)
        # Chair and bed have equal token lengths; table differs. Shuffling the
        # replay pairing must not change chair's BF16 body batch/kernel width.
        policy.forward_batch(frames[:2], initial[:2], visual[:2])
        torch.testing.assert_close(head_inputs[0][0], head_inputs[1][0], atol=0, rtol=0)
        hook.remove()
        assert not torch.equal(logits[0], logits[2])
        sizes = [state_bytes(s) for s in states]
        order = [2, 0, 1]
        reordered, _, reordered_states = policy.forward_batch(
            frames[order], [clone_state(states[i]) for i in order], visual[order]
        )
        reference, _, continued = policy.forward_batch(frames, states, visual)
        torch.testing.assert_close(reordered, reference[order], atol=0, rtol=0)
        assert [s.episode_id for s in reordered_states] == [str(i) for i in order]
        assert [s.step_index for s in continued] == [2, 2, 2]
        assert [state_bytes(s) for s in continued] == sizes
        # An episode start in the same batch must not close a nonexistent turn.
        mixed = [
            clone_state(continued[0]),
            policy.reset("new", goals[1]),
            clone_state(continued[2]),
        ]
        reference, _, _ = policy.forward_batch(frames, mixed, visual)
        fresh, _, _ = policy.forward_batch(
            frames[1:2], [policy.reset("new", goals[1])], visual[1:2]
        )
        torch.testing.assert_close(reference[1:2], fresh, atol=0.005, rtol=0.005)

    # Detached TBPTT boundary + mixed reset/ongoing goals: differentiable replay
    # must reproduce collection and route gradients to the current goal tokens.
    with torch.autocast("cuda", dtype=torch.bfloat16), body.reuse_token_embeddings():
        replay, _, states = policy.forward_batch(frames, [clone_state(s) for s in mixed], visual)
        torch.testing.assert_close(replay, reference, atol=0, rtol=0)
        next_logits, _, _ = policy.forward_batch(frames, states, visual)
        (replay[:, 1].sum() + next_logits[:, 2].sum()).backward()
    assert body._replay_embeddings is None
    gradients = [p.grad for p in body.parameters() if p.grad is not None]
    assert gradients and all(torch.isfinite(g).all() for g in gradients)
    assert body.nav_token.grad is None and not body.nav_token.requires_grad
    ids = body.goal_token_ids(goals[1])
    assert body.embeddings.weight.grad[list(ids)].abs().sum() > 0
