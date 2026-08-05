from __future__ import annotations

from typing import NamedTuple

import torch
from torch import nn


class AttentionModulation(NamedTuple):
    """adaLN modulation for one attention branch.

    ``gate`` scales the branch's residual contribution. It comes from the
    block's adaLN head (not a standalone Parameter) so the conditioning
    strength can vary with the flow time t: in flow matching the condition
    should dominate near t=0 (pure noise) and fade near t=1 (nearly clean).
    """

    shift: torch.Tensor
    scale: torch.Tensor
    gate: torch.Tensor
    key_padding_mask: torch.Tensor | None = None


class TemporalCrossAttentionSpec(NamedTuple):
    hidden_dim: int
    n_heads: int
    dropout: float


def modulate(
    tokens: torch.Tensor,
    shift: torch.Tensor,
    scale: torch.Tensor,
) -> torch.Tensor:
    if shift.ndim == 2:
        shift = shift.unsqueeze(1)
        scale = scale.unsqueeze(1)
    return tokens * (1.0 + scale) + shift


def gate_residual(residual: torch.Tensor, gate: torch.Tensor) -> torch.Tensor:
    return residual * (gate.unsqueeze(1) if gate.ndim == 2 else gate)


class NormalizedTemporalCrossAttention(nn.Module):
    def __init__(self, spec: TemporalCrossAttentionSpec) -> None:
        super().__init__()
        self.norm_query = nn.LayerNorm(
            spec.hidden_dim,
            elementwise_affine=False,
            eps=1e-6,
        )
        # AFFINE on purpose. The context is a concatenation of two token kinds
        # (the per-frame EE window and the strided lookahead preview) that the
        # caller distinguishes by giving them separate projections and separate
        # positional tables. A non-affine LayerNorm here strips the per-channel
        # scale those projections learn, collapsing the two kinds toward one
        # subspace -- measured on emft_ep0355_dit0805: a coherent 5cm shift of
        # both hands moved the prediction LESS (0.0107 RMS) than shifting only
        # the window (0.0207), i.e. the window and preview responses had become
        # anti-correlated. The affine gains give the model an axis-wise handle
        # that normalization cannot erase.
        self.norm_context = nn.LayerNorm(spec.hidden_dim, eps=1e-6)
        self.attention = nn.MultiheadAttention(
            spec.hidden_dim,
            spec.n_heads,
            dropout=spec.dropout,
            batch_first=True,
        )

    def forward(
        self,
        body_tokens: torch.Tensor,
        context_tokens: torch.Tensor,
        modulation: AttentionModulation,
    ) -> torch.Tensor:
        query = modulate(
            self.norm_query(body_tokens),
            modulation.shift,
            modulation.scale,
        )
        context = self.norm_context(context_tokens)
        output, _ = self.attention(
            query,
            context,
            context,
            need_weights=False,
            key_padding_mask=modulation.key_padding_mask,
        )
        return body_tokens + gate_residual(output, modulation.gate)
