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
