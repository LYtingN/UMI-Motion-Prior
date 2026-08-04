from __future__ import annotations

from typing import NamedTuple

import torch
from torch import nn

from Prior_Recon.Masked_Flow.model.temporal_dit_cross_attention import (
    AttentionModulation,
    NormalizedTemporalCrossAttention,
    TemporalCrossAttentionSpec,
)


class TemporalDiTSpec(NamedTuple):
    hidden_dim: int
    n_heads: int
    ffn_mult: int
    dropout: float
    n_layers: int
    cross_attention_gate_init: float = 0.1


class TemporalDiTConfigError(ValueError):
    def __init__(
        self,
        field: str,
        actual: tuple[int, ...] | float,
    ) -> None:
        self.field = field
        self.actual = actual
        super().__init__(field, actual)

    def __str__(self) -> str:
        return f"Invalid temporal DiT {self.field}: {self.actual}."


def _modulate(
    tokens: torch.Tensor,
    shift: torch.Tensor,
    scale: torch.Tensor,
) -> torch.Tensor:
    if shift.ndim == 2:
        shift = shift.unsqueeze(1)
        scale = scale.unsqueeze(1)
    return tokens * (1.0 + scale) + shift


def _gate(residual: torch.Tensor, gate: torch.Tensor) -> torch.Tensor:
    return residual * (gate.unsqueeze(1) if gate.ndim == 2 else gate)


class TemporalDiTBlock(nn.Module):
    def __init__(self, spec: TemporalDiTSpec) -> None:
        super().__init__()
        self.norm_attention = nn.LayerNorm(
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
        self.context_attention = NormalizedTemporalCrossAttention(
            TemporalCrossAttentionSpec(
                hidden_dim=spec.hidden_dim,
                n_heads=spec.n_heads,
                dropout=spec.dropout,
                gate_init=spec.cross_attention_gate_init,
            )
        )
        self.norm_mlp = nn.LayerNorm(
            spec.hidden_dim,
            elementwise_affine=False,
            eps=1e-6,
        )
        self.mlp = nn.Sequential(
            nn.Linear(spec.hidden_dim, spec.hidden_dim * spec.ffn_mult),
            nn.GELU(approximate="tanh"),
            nn.Dropout(spec.dropout),
            nn.Linear(spec.hidden_dim * spec.ffn_mult, spec.hidden_dim),
        )
        self.ada_ln = nn.Sequential(
            nn.SiLU(),
            nn.Linear(spec.hidden_dim, 6 * spec.hidden_dim),
        )
        nn.init.zeros_(self.ada_ln[-1].weight)
        nn.init.zeros_(self.ada_ln[-1].bias)

    def forward(
        self,
        body_tokens: torch.Tensor,
        context_tokens: torch.Tensor,
        condition: torch.Tensor,
    ) -> torch.Tensor:
        (
            attention_shift,
            attention_scale,
            attention_gate,
            mlp_shift,
            mlp_scale,
            mlp_gate,
        ) = self.ada_ln(condition).chunk(6, dim=-1)
        body_query = _modulate(
            self.norm_attention(body_tokens),
            attention_shift,
            attention_scale,
        )
        attention_output, _ = self.attention(
            body_query,
            body_query,
            body_query,
            need_weights=False,
        )
        body_tokens = body_tokens + _gate(attention_output, attention_gate)
        body_tokens = self.context_attention(
            body_tokens,
            context_tokens,
            AttentionModulation(
                shift=attention_shift,
                scale=attention_scale,
            ),
        )
        mlp_input = _modulate(
            self.norm_mlp(body_tokens),
            mlp_shift,
            mlp_scale,
        )
        return body_tokens + _gate(self.mlp(mlp_input), mlp_gate)


class TemporalDiTTransformer(nn.Module):
    def __init__(self, spec: TemporalDiTSpec) -> None:
        super().__init__()
        if spec.hidden_dim < 1:
            raise TemporalDiTConfigError("hidden_dim", spec.hidden_dim)
        if spec.n_heads < 1:
            raise TemporalDiTConfigError("n_heads", spec.n_heads)
        if spec.ffn_mult < 1:
            raise TemporalDiTConfigError("ffn_mult", spec.ffn_mult)
        if not 0.0 <= spec.dropout < 1.0:
            raise TemporalDiTConfigError("dropout", spec.dropout)
        if spec.n_layers < 1:
            raise TemporalDiTConfigError("n_layers", spec.n_layers)
        if spec.hidden_dim % spec.n_heads != 0:
            raise TemporalDiTConfigError("n_heads", spec.n_heads)
        if not 0.0 < spec.cross_attention_gate_init <= 1.0:
            raise TemporalDiTConfigError(
                "cross_attention_gate_init",
                spec.cross_attention_gate_init,
            )
        self.hidden_dim = spec.hidden_dim
        self.blocks = nn.ModuleList(
            TemporalDiTBlock(spec) for _ in range(spec.n_layers)
        )
        self.final_norm = nn.LayerNorm(
            spec.hidden_dim,
            elementwise_affine=False,
            eps=1e-6,
        )
        self.final_ada_ln = nn.Sequential(
            nn.SiLU(),
            nn.Linear(spec.hidden_dim, 2 * spec.hidden_dim),
        )
        nn.init.zeros_(self.final_ada_ln[-1].weight)
        nn.init.zeros_(self.final_ada_ln[-1].bias)

    def _validate(
        self,
        body_tokens: torch.Tensor,
        context_tokens: torch.Tensor,
        condition: torch.Tensor,
    ) -> None:
        if body_tokens.ndim != 3:
            raise TemporalDiTConfigError("body_shape", tuple(body_tokens.shape))
        if context_tokens.ndim != 3:
            raise TemporalDiTConfigError(
                "context_shape",
                tuple(context_tokens.shape),
            )
        batch, frames, hidden = body_tokens.shape
        if context_tokens.shape[0] != batch:
            raise TemporalDiTConfigError(
                "context_batch",
                tuple(context_tokens.shape),
            )
        if hidden != self.hidden_dim or context_tokens.shape[-1] != hidden:
            raise TemporalDiTConfigError(
                "hidden_dim",
                tuple(context_tokens.shape),
            )
        valid_shared = condition.shape == (batch, hidden)
        valid_per_frame = condition.shape == (batch, frames, hidden)
        if not (valid_shared or valid_per_frame):
            raise TemporalDiTConfigError(
                "condition_shape",
                tuple(condition.shape),
            )

    def forward(
        self,
        body_tokens: torch.Tensor,
        context_tokens: torch.Tensor,
        condition: torch.Tensor,
    ) -> torch.Tensor:
        self._validate(body_tokens, context_tokens, condition)
        hidden = body_tokens
        for block in self.blocks:
            hidden = block(hidden, context_tokens, condition)
        shift, scale = self.final_ada_ln(condition).chunk(2, dim=-1)
        return _modulate(self.final_norm(hidden), shift, scale)
