from __future__ import annotations

from typing import NamedTuple

import torch
from torch import nn


class AttentionModulation(NamedTuple):
    shift: torch.Tensor
    scale: torch.Tensor


class TemporalCrossAttentionSpec(NamedTuple):
    hidden_dim: int
    n_heads: int
    dropout: float
    gate_init: float


class NormalizedTemporalCrossAttention(nn.Module):
    def __init__(self, spec: TemporalCrossAttentionSpec) -> None:
        super().__init__()
        self.norm_query = nn.LayerNorm(
            spec.hidden_dim,
            elementwise_affine=False,
            eps=1e-6,
        )
        self.norm_context = nn.LayerNorm(
            spec.hidden_dim,
            elementwise_affine=False,
            eps=1e-6,
        )
        self.attention = nn.MultiheadAttention(
            spec.hidden_dim,
            spec.n_heads,
            dropout=spec.dropout,
            batch_first=True,
        )
        self.gate = nn.Parameter(
            torch.full(
                (spec.hidden_dim,),
                spec.gate_init,
            )
        )

    def forward(
        self,
        body_tokens: torch.Tensor,
        context_tokens: torch.Tensor,
        modulation: AttentionModulation,
    ) -> torch.Tensor:
        shift, scale = modulation
        if shift.ndim == 2:
            shift = shift.unsqueeze(1)
            scale = scale.unsqueeze(1)
        query = self.norm_query(body_tokens) * (1.0 + scale) + shift
        context = self.norm_context(context_tokens)
        output, _ = self.attention(
            query,
            context,
            context,
            need_weights=False,
        )
        return body_tokens + output * self.gate
