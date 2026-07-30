"""Config for the two-stage (root -> body) cascade masked-flow prior.

Kept OUT of config.py on purpose: the single-stage EEMaskedFlowConfig and its
YAMLs stay untouched. TwoStageMaskedFlowConfig subclasses it, so everything
that consumes the base config (dataset, body-stage model, body-stage loss,
trainer infrastructure) works unchanged; the cascade-specific knobs live in
the extra ``root_stage`` block.

Stage split (per-denoising-step cascade, ardy-style):
  stage 1 (root)  : s_ee window + EE lookahead + pinned root history
                    -> root+contact channels ([0:11] + abs root [69:73])
  stage 2 (body)  : the existing full-dim model, with the root columns pinned
                    (obs_mask) to stage 1's detached pred_x1 at the SAME noise
                    level -- training does exactly what sampling does, so the
                    body stage is natively trained on imperfect root estimates.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from Prior_Recon.Masked_Flow.config import EEMaskedFlowConfig


@dataclass
class RootStageLossConfig:
    """Loss weights for the root stage (15-dim root+contact feature space).

    All channels here ARE root channels, so there is no root-vs-body gradient
    competition to balance -- the reason this stage exists. fps-scaled terms
    (root_consistency, root_vel) carry the same x900 squared-magnitude caveat
    as the single-stage losses: keep their weights small.
    """

    flow_weight: float = 1.0
    recon_weight: float = 0.25
    velocity_weight: float = 0.15
    # Dedicated supervision of the contact channels (local dims 5:7); GT is
    # binary {0,1} and downstream foot_lock thresholds it.
    contact_pred_weight: float = 0.5
    # [C] BCE contact supervision (0/1 push for stable thresholding) + positive-
    # class upweight; mirrors the single-stage EEMaskedFlowLossConfig switches.
    contact_pred_bce: bool = False
    contact_pred_pos_weight: float = 1.0
    # abs_root dual-encoding self-consistency (delta vs abs channels).
    root_consistency_weight: float = 0.05
    # Finite-difference velocity of the decoded root state channels (m/s).
    root_vel_weight: float = 0.02
    # Absolute height anchor (metre scale) against dc drift.
    root_height_weight: float = 0.3
    # Full-horizon drift anchors on the LAST frame's abs channels (the exact
    # counterparts of the single-stage drift_yaw/drift_xy, which are
    # gradient-free in the body stage because its root columns are pinned).
    drift_yaw_weight: float = 0.0
    drift_xy_weight: float = 0.0


@dataclass
class RootStageConfig:
    """Architecture + loss of the stage-1 root model.

    The root model is a second EEMaskedFlowTransformer instance over the
    root-channel subset; it shares the parent's motion/primitive/lookahead
    settings and s_ee condition layout, only the backbone size differs.
    """

    hidden_dim: int = 256
    n_layers: int = 6
    n_heads: int = 8
    dropout: float = 0.1
    time_emb_dim: int = 128
    transformer_ffn_mult: int = 4
    out_proj_hidden_mult: int = 2
    temporal_backbone: str = "flat"
    hierarchy_fine_layers: int = 2
    hierarchy_coarse_layers: int = 2
    hierarchy_refine_layers: int = 2
    hierarchy_downsample_factor: int = 2
    # Multiplier of the root-stage loss inside the joint total.
    stage_weight: float = 1.0
    loss: RootStageLossConfig = field(default_factory=RootStageLossConfig)


@dataclass
class TwoStageMaskedFlowConfig(EEMaskedFlowConfig):
    """EEMaskedFlowConfig + the stage-1 root model block.

    All inherited architecture fields describe the stage-2 body model.
    ``root_stage.temporal_backbone`` independently selects the stage-1 root
    backbone. Dataset, trainer, primitive, and lookahead settings stay shared.
    """

    root_stage: RootStageConfig = field(default_factory=RootStageConfig)
    # Root preview for the body stage: per primitive window (10 frames) the
    # body model sees the stage-1 root of the REMAINING segment frames
    # (segment_len - primitive_len, 24 @ defaults) as soft preview tokens,
    # downsampled by this stride -- same convention as the EE lookahead. The
    # in-window 10 frames enter through the pinned root columns, so together
    # the body stage sees the full 34-frame root trajectory.
    root_look_stride: int = 2

    @property
    def root_look_len(self) -> int:
        prim = self.primitive
        return max(int(prim.segment_len) - int(prim.primitive_len), 0)

    @property
    def n_root_look_tokens(self) -> int:
        if self.root_look_len <= 0:
            return 0
        stride = max(int(self.root_look_stride), 1)
        return (self.root_look_len + stride - 1) // stride
