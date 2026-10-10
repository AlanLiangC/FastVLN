"""Functional KDA and original-weight GDN, with differentiable, bounded state.

Qwen's stock multi-token GDN path in transformers 5.3 discards the initial
recurrent state. We retain its weights/equations but explicitly carry both
convolution history and matrix state across arbitrary observation chunks.
"""

import torch
from torch import nn
from torch.nn import functional as F

from streamnav.contracts.state import LayerState


def delta_reference(q, k, v, g, beta, initial_state=None):
    """Small differentiable reference, used in numerical tests only."""
    q, k = F.normalize(q.float(), dim=-1), F.normalize(k.float(), dim=-1)
    q = q * q.shape[-1] ** -0.5
    s = initial_state
    if s is None:
        s = q.new_zeros(q.shape[0], q.shape[2], q.shape[3], v.shape[-1])
    outputs = []
    for t in range(q.shape[1]):
        decay = g[:, t].float().exp()
        if decay.ndim == 2:
            decay = decay.unsqueeze(-1)
        s = s * decay.unsqueeze(-1)
        residual = v[:, t].float() - (s * k[:, t, :, :, None]).sum(-2)
        residual = residual * beta[:, t, :, None].float()
        s = s + k[:, t, :, :, None] * residual.unsqueeze(-2)
        outputs.append((s * q[:, t, :, :, None]).sum(-2))
    return torch.stack(outputs, 1).to(v.dtype), s


class KDAAdapter(nn.Module):
    """Replace full attention, preserving Q/K/V/output projections and output gate.

    RoPE is removed; order is represented by the recurrence. Channel-wise decay
    and delta update gates are newly initialized, not pretrained KDA weights.
    """

    def __init__(self, attention, output_norm=False):
        super().__init__()
        self.head_dim = attention.head_dim
        self.output_norm = output_norm
        self.num_heads = attention.config.num_attention_heads
        self.num_kv_heads = attention.config.num_key_value_heads
        self.q_proj: nn.Module = attention.q_proj
        self.k_proj: nn.Module = attention.k_proj
        self.v_proj: nn.Module = attention.v_proj
        self.o_proj: nn.Module = attention.o_proj
        self.q_norm: nn.Module = attention.q_norm
        self.k_norm: nn.Module = attention.k_norm
        hidden = attention.config.hidden_size
        self.decay_proj = nn.Linear(hidden, self.num_heads * self.head_dim, bias=True)
        self.beta_proj = nn.Linear(hidden, self.num_heads, bias=True)
        nn.init.zeros_(self.decay_proj.weight)
        nn.init.constant_(self.decay_proj.bias, -4.0)
        nn.init.zeros_(self.beta_proj.weight)
        nn.init.zeros_(self.beta_proj.bias)

    def forward(self, x, state: LayerState, mode="auto", cu=None, cu_cpu=None, lengths=None):
        from fla.ops.kda import chunk_kda, fused_recurrent_kda

        b, t, _ = x.shape
        q, gate = self.q_proj(x).view(b, t, self.num_heads, 2 * self.head_dim).chunk(2, -1)
        q = self.q_norm(q).contiguous()
        k = self.k_norm(self.k_proj(x).view(b, t, self.num_kv_heads, self.head_dim))
        v = self.v_proj(x).view(b, t, self.num_kv_heads, self.head_dim)
        repeats = self.num_heads // self.num_kv_heads
        k, v = (y.repeat_interleave(repeats, dim=2).contiguous() for y in (k, v))
        g = -F.softplus(self.decay_proj(x).float()).view(b, t, self.num_heads, self.head_dim)
        beta = self.beta_proj(x).sigmoid()
        if mode == "reference":
            if lengths is not None:
                raise ValueError("Variable-length batches require FLA kernels")
            o, recurrent = delta_reference(q, k, v, g, beta, state.recurrent)
        else:
            chunk = mode == "chunk" or (mode == "auto" and torch.is_grad_enabled())
            kernel = chunk_kda if chunk else fused_recurrent_kda
            o, recurrent = kernel(
                q=q,
                k=k,
                v=v,
                g=g,
                beta=beta,
                initial_state=state.recurrent,
                output_final_state=True,
                use_qk_l2norm_in_kernel=True,
                cu_seqlens=cu,
                **({"cu_seqlens_cpu": cu_cpu} if chunk else {}),
            )
        if self.output_norm:
            # FLA KimiDeltaAttention normalizes each value head before its
            # sigmoid output gate. Raw delta outputs differ in scale from
            # softmax attention, so copied O weights need this normalization.
            # A separate affine scale is redundant with the trainable O matrix.
            o = F.rms_norm(o, (self.head_dim,), eps=1e-5)
        out = self.o_proj((o * gate.sigmoid()).reshape(b, t, -1))
        return out, LayerState(None, recurrent)


class StreamingGatedDeltaNet(nn.Module):
    def __init__(self, original):
        super().__init__()
        self.original = original

    def forward(self, x, state: LayerState, mode="auto", cu=None, cu_cpu=None, lengths=None):
        from fla.ops.gated_delta_rule import (
            chunk_gated_delta_rule,
            fused_recurrent_gated_delta_rule,
        )

        m = self.original
        b, t, _ = x.shape
        qkv = m.in_proj_qkv(x).transpose(1, 2)
        history = state.conv
        if history is None:
            history = qkv.new_zeros(
                b if lengths is None else len(lengths), m.conv_dim, m.conv_kernel_size - 1
            )
        if lengths is None:
            joined = torch.cat((history, qkv), dim=-1)
            conv = joined[:, :, -(m.conv_kernel_size - 1) :].contiguous()
            qkv = F.silu(F.conv1d(joined, m.conv1d.weight, m.conv1d.bias, groups=m.conv_dim))
        else:
            # Convolution histories are independent. Right padding is confined
            # to this causal convolution; trim it before the recurrent kernels
            # and take final history from each sequence's actual final tokens.
            blocks = qkv.split(lengths, dim=-1)
            joined_blocks = [
                torch.cat((history[i : i + 1], block), dim=-1) for i, block in enumerate(blocks)
            ]
            conv = torch.cat([block[:, :, -(m.conv_kernel_size - 1) :] for block in joined_blocks])
            joined = torch.cat(
                [
                    F.pad(block, (0, max(lengths) - length))
                    for block, length in zip(joined_blocks, lengths, strict=True)
                ]
            )
            convolved = F.silu(F.conv1d(joined, m.conv1d.weight, m.conv1d.bias, groups=m.conv_dim))
            qkv = torch.cat(
                [convolved[i : i + 1, :, :length] for i, length in enumerate(lengths)], dim=-1
            )
        qkv = qkv.transpose(1, 2)
        q, k, v = qkv.split((m.key_dim, m.key_dim, m.value_dim), dim=-1)
        q, k = (y.reshape(b, t, m.num_k_heads, m.head_k_dim).contiguous() for y in (q, k))
        v = v.reshape(b, t, m.num_v_heads, m.head_v_dim).contiguous()
        repeats = m.num_v_heads // m.num_k_heads
        if repeats > 1:
            q, k = (y.repeat_interleave(repeats, 2) for y in (q, k))
        beta = m.in_proj_b(x).sigmoid()
        g = -m.A_log.float().exp() * F.softplus(m.in_proj_a(x).float() + m.dt_bias.float())
        if mode == "reference":
            if lengths is not None:
                raise ValueError("Variable-length batches require FLA kernels")
            o, recurrent = delta_reference(q, k, v, g, beta, state.recurrent)
        else:
            chunk = mode == "chunk" or (mode == "auto" and torch.is_grad_enabled())
            kernel = chunk_gated_delta_rule if chunk else fused_recurrent_gated_delta_rule
            o, recurrent = kernel(
                q=q,
                k=k,
                v=v,
                g=g,
                beta=beta,
                initial_state=state.recurrent,
                output_final_state=True,
                use_qk_l2norm_in_kernel=True,
                cu_seqlens=cu,
                **({"cu_seqlens_cpu": cu_cpu} if chunk else {}),
            )
        z = m.in_proj_z(x).reshape(-1, m.head_v_dim)
        o = m.norm(o.reshape(-1, m.head_v_dim), z).reshape(b, t, m.value_dim)
        return m.out_proj(o), LayerState(conv, recurrent)
