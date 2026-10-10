from types import SimpleNamespace

import pytest
import torch

from streamnav.models.qwen35_kda.backbone import Qwen35KDABackbone


def small_backbone():
    backbone = Qwen35KDABackbone.__new__(Qwen35KDABackbone)
    torch.nn.Module.__init__(backbone)
    backbone.config = SimpleNamespace(
        text_config=SimpleNamespace(hidden_size=8),
        vision_start_token_id=1,
        vision_end_token_id=2,
    )
    backbone.embeddings = torch.nn.Embedding(32, 8)
    backbone.nav_token = torch.nn.Parameter(torch.randn(1, 1, 8))
    backbone.goal_conditioning = "nav_query"
    backbone._replay_embeddings = None
    backbone._goal_ids_cache = {"chair": (3, 4), "bed": (3, 5, 5)}
    return backbone


def test_reuse_shares_lookup_graph_and_preserves_forward_and_gradients():
    torch.manual_seed(7)
    backbone = small_backbone()
    visual = torch.randn(2, 3, 8)
    calls = []
    hook = backbone.embeddings.register_forward_hook(lambda *args: calls.append(1))

    def forward():
        return [backbone.encode_visual_tokens(visual, ["chair", "bed"]) for _ in range(20)]

    uncached = forward()
    assert len(calls) == 60
    coefficients = [torch.randn_like(x) for x in uncached]
    sum((x * c).sum() for x, c in zip(uncached, coefficients, strict=True)).backward()
    expected_gradient = backbone.embeddings.weight.grad.clone()
    backbone.zero_grad(set_to_none=True)
    calls.clear()
    with backbone.reuse_token_embeddings():
        cached = forward()
    assert len(calls) == 3
    assert backbone._replay_embeddings is None
    for actual, expected in zip(cached, uncached, strict=True):
        torch.testing.assert_close(actual, expected, atol=0, rtol=0)
    sum((x * c).sum() for x, c in zip(cached, coefficients, strict=True)).backward()
    torch.testing.assert_close(
        backbone.embeddings.weight.grad, expected_gradient, atol=2e-6, rtol=2e-6
    )
    hook.remove()


def test_next_replay_reads_updated_weights_and_builds_fresh_graph():
    backbone = small_backbone()
    optimizer = torch.optim.SGD(backbone.parameters(), lr=0.1)
    ids = (3, 5)
    with backbone.reuse_token_embeddings():
        before = backbone.token_embeddings(ids)
        before.sum().backward()
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    with backbone.reuse_token_embeddings():
        after = backbone.token_embeddings(ids)
        torch.testing.assert_close(after, before.detach() - 0.1)
        after.sum().backward()
    assert backbone.embeddings.weight.grad[ids[0]].sum() == 8


def test_failed_or_nested_replay_cannot_leave_a_stale_embedding_graph():
    backbone = small_backbone()
    with pytest.raises(RuntimeError, match="nested"):
        with backbone.reuse_token_embeddings():
            backbone.token_embeddings((3,))
            with backbone.reuse_token_embeddings():
                pass
    assert backbone._replay_embeddings is None
    with pytest.raises(ValueError, match="injected"):
        with backbone.reuse_token_embeddings():
            backbone.token_embeddings((3,))
            raise ValueError("injected replay failure")
    assert backbone._replay_embeddings is None
