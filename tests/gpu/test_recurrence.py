from types import SimpleNamespace

import pytest
import torch
from torch import nn

from streamnav.contracts.state import LayerState
from streamnav.models.qwen35_kda.kda_adapter import KDAAdapter, StreamingGatedDeltaNet

pytestmark = [
    pytest.mark.gpu,
    pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required"),
]


def small_kda():
    attn = nn.Module()
    attn.head_dim = 32
    attn.config = SimpleNamespace(num_attention_heads=2, num_key_value_heads=1, hidden_size=64)
    attn.q_proj, attn.k_proj, attn.v_proj, attn.o_proj = (
        nn.Linear(64, 128),
        nn.Linear(64, 32),
        nn.Linear(64, 32),
        nn.Linear(64, 64),
    )
    attn.q_norm, attn.k_norm = nn.Identity(), nn.Identity()
    return KDAAdapter(attn).cuda().bfloat16()


@pytest.mark.parametrize("kind", ["kda", "gdn"])
def test_chunk_recurrent_reference_and_initial_state_gradient(kind):
    torch.manual_seed(19)
    if kind == "kda":
        layer = small_kda()
    else:
        from transformers.models.qwen3_5.modeling_qwen3_5 import (
            Qwen3_5GatedDeltaNet,
            Qwen3_5TextConfig,
        )

        cfg = Qwen3_5TextConfig(
            hidden_size=64,
            num_hidden_layers=1,
            layer_types=["linear_attention"],
            linear_key_head_dim=32,
            linear_value_head_dim=32,
            linear_num_key_heads=2,
            linear_num_value_heads=2,
        )
        layer = StreamingGatedDeltaNet(Qwen3_5GatedDeltaNet(cfg, 0)).cuda().bfloat16()
    x = torch.randn(1, 9, 64, device="cuda", dtype=torch.bfloat16)
    with torch.no_grad():
        whole, whole_state = layer(x, LayerState(None, None), mode="chunk")
        reference, _ = layer(x, LayerState(None, None), mode="reference")
        state = LayerState(None, None)
        chunks = []
        for frame in x.split(1, dim=1):
            out, state = layer(frame, state, mode="recurrent")
            chunks.append(out)
        torch.testing.assert_close(torch.cat(chunks, 1), whole, atol=0.012, rtol=0.04)
        torch.testing.assert_close(whole, reference, atol=0.012, rtol=0.04)
        torch.testing.assert_close(state.recurrent, whole_state.recurrent, atol=0.015, rtol=0.05)
        first, cache = layer(x[:, :4], LayerState(None, None), mode="chunk")
        second, _ = layer(x[:, 4:], cache, mode="chunk")
        torch.testing.assert_close(torch.cat([first, second], 1), whole, atol=0.012, rtol=0.04)
    initial = torch.randn_like(state.recurrent, requires_grad=True)
    out, new = layer(x[:, :3], LayerState(state.conv, initial), mode="chunk")
    (out.float().square().mean() + new.recurrent.square().mean()).backward()
    assert initial.grad is not None and initial.grad.abs().sum().item() > 0


def test_kda_cache_bounded_for_500_steps():
    layer = small_kda()
    x = torch.randn(1, 1, 64, device="cuda", dtype=torch.bfloat16)
    state, sizes = LayerState(None, None), []
    with torch.no_grad():
        for _ in range(500):
            _, state = layer(x, state, mode="recurrent")
            sizes.append(state.recurrent.numel() * state.recurrent.element_size())
    assert len(set(sizes)) == 1
