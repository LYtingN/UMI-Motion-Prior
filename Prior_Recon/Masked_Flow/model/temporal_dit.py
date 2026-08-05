from __future__ import annotations

from typing import NamedTuple

import torch
from torch import nn

from Prior_Recon.Masked_Flow.model.temporal_dit_cross_attention import (
    AttentionModulation,
    NormalizedTemporalCrossAttention,
    TemporalCrossAttentionSpec,
    gate_residual,
    modulate,
)

# Number of hidden-sized chunks the per-block adaLN head emits:
# (shift, scale, gate) x (self-attention, cross-attention, mlp).
N_ADA_LN_CHUNKS = 9
# Index of the first gate chunk within those 9 -- gates are chunks 2, 5, 8.
_GATE_CHUNKS = (2, 5, 8)


class TemporalDiTSpec(NamedTuple):
    hidden_dim: int
    n_heads: int
    ffn_mult: int
    dropout: float
    n_layers: int
    # Initial value of the adaLN GATE bias, i.e. how strongly every residual
    # branch (self-attention, cross-attention, mlp) contributes at step 0.
    #
    # Canonical DiT uses 0.0: every block is the identity at init. That is safe
    # but, when the network also has an ungated pointwise bypass into the output
    # head (here: feat_proj(x_in) + ee_frame_proj(s_ee) + mask_proj + pos_emb
    # reaching out_proj through final_norm alone), it creates a race the bypass
    # wins. With gates AND the output head both zero-init, a branch's weights
    # get gradient proportional to its gate, which is ~0, so out_proj fits a
    # pointwise map first and the gates never catch up. Measured on
    # emft_ep0355_dit0805 (gate_init 0.0 for self-attn/mlp): block 0 rewrote
    # 112% of the residual stream and blocks 1-11 contributed 2-6% (self-attn),
    # 4-9% (mlp) and 0.5-1.2% (cross-attention) each -- 12 layers behaving like
    # a per-frame MLP, which cannot enforce continuity across a pinned-history
    # boundary (seam velocity ratio 4.81 vs 1.06 for ground truth).
    #
    # A small positive bias starts every branch "on" so its weights receive
    # gradient from the first step the output head is non-zero. 0.1 matches the
    # gate magnitude that run actually converged to (|attn_gate| 0.05-0.11), so
    # it is a warm start rather than a scale change. 0.0 restores canonical DiT.
    residual_gate_init: float = 0.1


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


def _init_ada_ln_head(head: nn.Linear, gate_init: float, hidden_dim: int) -> None:
    """Zero the modulation head, then warm only the gate chunks.

    Zero weight + zero shift/scale bias keeps the classic DiT property that the
    modulation is data-independent at init (scale=0 => LayerNorm passthrough).
    The gate bias is set to ``gate_init`` so the branch is live from step 0.
    """
    nn.init.zeros_(head.weight)
    nn.init.zeros_(head.bias)
    if gate_init == 0.0:
        return
    with torch.no_grad():
        for chunk in _GATE_CHUNKS:
            start = chunk * hidden_dim
            head.bias[start : start + hidden_dim].fill_(gate_init)


class TemporalDiTBlock(nn.Module):
    def __init__(self, spec: TemporalDiTSpec) -> None:
        super().__init__()
        self.hidden_dim = spec.hidden_dim
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
            nn.Linear(spec.hidden_dim, N_ADA_LN_CHUNKS * spec.hidden_dim),
        )
        _init_ada_ln_head(
            self.ada_ln[-1],
            spec.residual_gate_init,
            spec.hidden_dim,
        )

    def forward(
        self,
        body_tokens: torch.Tensor,
        context_tokens: torch.Tensor,
        condition: torch.Tensor,
        context_key_padding_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        (
            attention_shift,
            attention_scale,
            attention_gate,
            context_shift,
            context_scale,
            context_gate,
            mlp_shift,
            mlp_scale,
            mlp_gate,
        ) = self.ada_ln(condition).chunk(N_ADA_LN_CHUNKS, dim=-1)
        body_query = modulate(
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
        body_tokens = body_tokens + gate_residual(attention_output, attention_gate)
        # Cross-attention gets its OWN shift/scale/gate. Sharing the self
        # attention's was wrong: those are fitted for norm_attention() of the
        # stream BEFORE the self-attention residual, while the cross-attention
        # normalizes the stream AFTER it -- two different tensors (block 0's
        # self-attention moves the stream by ~112%), so one pair had to serve
        # both. PixArt/MMDiT likewise keep the cross-attention modulation
        # separate.
        body_tokens = self.context_attention(
            body_tokens,
            context_tokens,
            AttentionModulation(
                shift=context_shift,
                scale=context_scale,
                gate=context_gate,
                key_padding_mask=context_key_padding_mask,
            ),
        )
        mlp_input = modulate(
            self.norm_mlp(body_tokens),
            mlp_shift,
            mlp_scale,
        )
        return body_tokens + gate_residual(self.mlp(mlp_input), mlp_gate)


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
        # 0.0 IS allowed: it is canonical DiT's identity-at-init. The previous
        # `0.0 < gate_init` bound made the one correct value unreachable from
        # config, which is why the cross-attention shipped with a 0.1 static
        # gate that injected random attention into all 12 blocks at step 0.
        if not 0.0 <= spec.residual_gate_init <= 1.0:
            raise TemporalDiTConfigError(
                "residual_gate_init",
                spec.residual_gate_init,
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
        context_key_padding_mask: torch.Tensor | None,
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
        if context_key_padding_mask is None:
            return
        if context_key_padding_mask.shape != context_tokens.shape[:2]:
            raise TemporalDiTConfigError(
                "context_key_padding_mask_shape",
                tuple(context_key_padding_mask.shape),
            )
        # An all-masked row makes softmax divide by zero -> NaN for that sample.
        # The caller keeps the per-frame EE tokens unmasked precisely so this
        # cannot happen; fail loudly rather than emit NaN if that ever changes.
        #
        # CPU only, on purpose. Reading a predicate off a device tensor forces a
        # host sync, and this runs on EVERY forward -- ode_steps x num_primitives
        # times per sampled segment (32 x 4 = 128 for the delta73 configs), each
        # one draining the queue in the middle of the ODE loop. The invariant is
        # structural (see _context_key_padding_mask) and is covered by a CPU
        # test, so the guard pays for itself where it is free and stays out of
        # the GPU train/deploy path.
        if context_key_padding_mask.device.type != "cpu":
            return
        if bool(context_key_padding_mask.all(dim=-1).any()):
            raise TemporalDiTConfigError(
                "context_key_padding_mask_all_masked",
                tuple(context_key_padding_mask.shape),
            )

    def forward(
        self,
        body_tokens: torch.Tensor,
        context_tokens: torch.Tensor,
        condition: torch.Tensor,
        context_key_padding_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        self._validate(
            body_tokens,
            context_tokens,
            condition,
            context_key_padding_mask,
        )
        hidden = body_tokens
        for block in self.blocks:
            hidden = block(
                hidden,
                context_tokens,
                condition,
                context_key_padding_mask,
            )
        shift, scale = self.final_ada_ln(condition).chunk(2, dim=-1)
        return modulate(self.final_norm(hidden), shift, scale)
