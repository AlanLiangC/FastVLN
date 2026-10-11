"""Frozen pretrained projections with FP32 low-rank trainable increments."""

import math

import torch
from torch import nn
from torch.nn import functional as F


class LoRALinear(nn.Module):
    def __init__(self, base, rank=16, alpha=16):
        super().__init__()
        self.base = base.requires_grad_(False)
        self.a = nn.Parameter(torch.empty(rank, base.in_features, device=base.weight.device))
        self.b = nn.Parameter(torch.zeros(base.out_features, rank, device=base.weight.device))
        nn.init.kaiming_uniform_(self.a, a=math.sqrt(5))
        self.scale, self.enabled = alpha / rank, True

    @property
    def weight(self):
        return self.base.weight

    def forward(self, x):
        output = self.base(x)
        if self.enabled:
            output = output + F.linear(F.linear(x, self.a), self.b).to(output.dtype) * self.scale
        return output


class LowRankGate(nn.Module):
    def __init__(self, hidden, output, rank, bias, device):
        super().__init__()
        self.a = nn.Parameter(torch.empty(rank, hidden, device=device))
        self.b = nn.Parameter(torch.zeros(output, rank, device=device))
        self.bias = nn.Parameter(torch.full((output,), float(bias), device=device))
        nn.init.kaiming_uniform_(self.a, a=math.sqrt(5))

    def forward(self, x):
        return F.linear(F.linear(x, self.a), self.b, self.bias)


class KDAMemory(nn.Module):
    """Add long-term memory while preserving the original within-frame attention.

    Q/K/V/O are shared with the native attention and its LoRA increments. The
    zero-initialized residual scale preserves the native single-frame function.
    No historical KV tokens are retained between decisions.
    """

    def __init__(self, attention, gate_rank=16):
        super().__init__()
        cfg = attention.config
        self.heads, self.kv_heads = cfg.num_attention_heads, cfg.num_key_value_heads
        self.head_dim = attention.head_dim
        device = attention.q_proj.weight.device
        channels = self.heads * self.head_dim
        self.decay = LowRankGate(cfg.hidden_size, channels, gate_rank, -7.0, device)
        self.beta = LowRankGate(cfg.hidden_size, self.heads, gate_rank, -1.0, device)
        self.output_gate = LowRankGate(cfg.hidden_size, channels, gate_rank, 2.0, device)
        self.residual_scale = nn.Parameter(torch.zeros((), device=device))

    def forward(self, q, k, v, x, state, mode="chunk", *, cu_seqlens=None, cu_seqlens_cpu=None):
        from fla.ops.kda import chunk_kda, fused_recurrent_kda

        repeat = self.heads // self.kv_heads
        k, v = (y.repeat_interleave(repeat, dim=2).contiguous() for y in (k, v))
        g = -F.softplus(self.decay(x).float()).view(*x.shape[:2], self.heads, self.head_dim)
        beta = self.beta(x).sigmoid()
        kernel = chunk_kda if mode == "chunk" else fused_recurrent_kda
        y, state = kernel(
            q=q.contiguous(),
            k=k,
            v=v,
            g=g,
            beta=beta,
            initial_state=state,
            output_final_state=True,
            use_qk_l2norm_in_kernel=True,
            cu_seqlens=cu_seqlens,
            cu_seqlens_cpu=cu_seqlens_cpu,
        )
        y = F.rms_norm(y, (self.head_dim,), eps=1e-6)
        y = y * self.output_gate(x).view_as(y).sigmoid()
        return y.reshape(*x.shape[:2], -1), state
