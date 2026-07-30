from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass(frozen=True)
class TemporalPyramidSpec:
    hidden_dim: int
    n_heads: int
    ffn_mult: int
    dropout: float
    fine_layers: int
    coarse_layers: int
    refine_layers: int
    downsample_factor: int


class TemporalPyramidConfigError(ValueError):
    def __init__(self, field: str, value: str | int) -> None:
        self.field = field
        self.value = value
        super().__init__(field, value)

    def __str__(self) -> str:
        return f"Invalid temporal pyramid {self.field}={self.value}."


def _transformer_stage(
    spec: TemporalPyramidSpec,
    n_layers: int,
) -> nn.TransformerEncoder:
    layer = nn.TransformerEncoderLayer(
        d_model=spec.hidden_dim,
        nhead=spec.n_heads,
        dim_feedforward=spec.hidden_dim * spec.ffn_mult,
        dropout=spec.dropout,
        batch_first=True,
        norm_first=True,
        activation="gelu",
    )
    return nn.TransformerEncoder(layer, num_layers=n_layers)


class TemporalPyramidTransformer(nn.Module):
    def __init__(self, spec: TemporalPyramidSpec) -> None:
        super().__init__()
        if min(spec.fine_layers, spec.coarse_layers, spec.refine_layers) < 1:
            raise TemporalPyramidConfigError("stage_layers", 0)
        if spec.downsample_factor < 2:
            raise TemporalPyramidConfigError(
                "downsample_factor",
                spec.downsample_factor,
            )

        self.fine_encoder = _transformer_stage(
            spec,
            spec.fine_layers,
        )
        self.downsample_factor = spec.downsample_factor
        self.downsample = nn.Conv1d(
            spec.hidden_dim,
            spec.hidden_dim,
            kernel_size=spec.downsample_factor,
            stride=spec.downsample_factor,
        )
        self.coarse_encoder = _transformer_stage(
            spec,
            spec.coarse_layers,
        )
        self.skip_fusion = nn.Sequential(
            nn.Linear(spec.hidden_dim * 2, spec.hidden_dim),
            nn.GELU(),
            nn.LayerNorm(spec.hidden_dim),
        )
        self.refine_encoder = _transformer_stage(
            spec,
            spec.refine_layers,
        )

    def _downsample_body(self, body_tokens: torch.Tensor) -> torch.Tensor:
        frame_count = body_tokens.shape[1]
        if frame_count < 1:
            raise TemporalPyramidConfigError("frame_count", frame_count)

        body_channels = body_tokens.transpose(1, 2)
        pad_frames = (-frame_count) % self.downsample_factor
        if pad_frames:
            body_channels = F.pad(
                body_channels,
                (0, pad_frames),
                mode="replicate",
            )
        return self.downsample(body_channels).transpose(1, 2)

    def _upsample_body(
        self,
        coarse_body: torch.Tensor,
        frame_count: int,
    ) -> torch.Tensor:
        coarse_channels = F.interpolate(
            coarse_body.transpose(1, 2),
            scale_factor=self.downsample_factor,
            mode="linear",
            align_corners=False,
        )
        return coarse_channels[:, :, :frame_count].transpose(1, 2)

    def forward(
        self,
        body_tokens: torch.Tensor,
        context_tokens: torch.Tensor,
    ) -> torch.Tensor:
        frame_count = body_tokens.shape[1]
        context_count = context_tokens.shape[1]

        fine_all = self.fine_encoder(
            torch.cat((context_tokens, body_tokens), dim=1)
        )
        fine_context = fine_all[:, :context_count]
        fine_body = fine_all[:, -frame_count:]

        coarse_body = self._downsample_body(fine_body)
        coarse_all = self.coarse_encoder(
            torch.cat((fine_context, coarse_body), dim=1)
        )
        coarse_body = coarse_all[:, -coarse_body.shape[1]:]
        coarse_up = self._upsample_body(coarse_body, frame_count)

        fused_body = self.skip_fusion(torch.cat((fine_body, coarse_up), dim=-1))
        refined_all = self.refine_encoder(
            torch.cat((fine_context, fused_body), dim=1)
        )
        return refined_all[:, -frame_count:]
