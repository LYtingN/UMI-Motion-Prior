from __future__ import annotations

from dataclasses import dataclass

from Prior_Recon.Masked_Flow.config import (
    MotionRepConfig,
    PrimitiveConfig,
)


@dataclass
class RootModelConfig:
    n_total_dof: int
    ee_feat_dim: int
    hidden_dim: int
    n_layers: int
    n_heads: int
    dropout: float
    time_emb_dim: int
    transformer_ffn_mult: int
    out_proj_hidden_mult: int
    lookahead_len: int
    lookahead_stride: int
    n_lookahead_tokens: int
    ode_steps: int
    per_frame_noise: bool
    use_ee_vel: bool
    motion: MotionRepConfig
    primitive: PrimitiveConfig
    temporal_backbone: str = "flat"
    hierarchy_fine_layers: int = 2
    hierarchy_coarse_layers: int = 2
    hierarchy_refine_layers: int = 2
    hierarchy_downsample_factor: int = 2
    ee_state_dim: int = 0
    abs_root_channels: bool = False
    use_ee_pos: bool = True
    use_ee_height_anchor: bool = False
    use_ee_anchor: bool = False


def derive_root_model_cfg(cfg, n_total_dof: int) -> RootModelConfig:
    root_stage = cfg.root_stage
    return RootModelConfig(
        n_total_dof=n_total_dof,
        ee_feat_dim=cfg.ee_feat_dim,
        hidden_dim=root_stage.hidden_dim,
        n_layers=root_stage.n_layers,
        n_heads=root_stage.n_heads,
        dropout=root_stage.dropout,
        time_emb_dim=root_stage.time_emb_dim,
        transformer_ffn_mult=root_stage.transformer_ffn_mult,
        out_proj_hidden_mult=root_stage.out_proj_hidden_mult,
        temporal_backbone=str(
            getattr(root_stage, "temporal_backbone", "flat")
        ),
        hierarchy_fine_layers=int(
            getattr(root_stage, "hierarchy_fine_layers", 2)
        ),
        hierarchy_coarse_layers=int(
            getattr(root_stage, "hierarchy_coarse_layers", 2)
        ),
        hierarchy_refine_layers=int(
            getattr(root_stage, "hierarchy_refine_layers", 2)
        ),
        hierarchy_downsample_factor=int(
            getattr(root_stage, "hierarchy_downsample_factor", 2)
        ),
        lookahead_len=int(getattr(cfg, "lookahead_len", 0)),
        lookahead_stride=int(getattr(cfg, "lookahead_stride", 1)),
        n_lookahead_tokens=int(getattr(cfg, "n_lookahead_tokens", 0)),
        ode_steps=int(cfg.ode_steps),
        per_frame_noise=bool(getattr(cfg, "per_frame_noise", False)),
        use_ee_vel=bool(getattr(cfg, "use_ee_vel", False)),
        motion=cfg.motion,
        primitive=getattr(cfg, "primitive", PrimitiveConfig()),
        abs_root_channels=bool(getattr(cfg, "abs_root_channels", False)),
        use_ee_pos=bool(getattr(cfg, "use_ee_pos", True)),
        use_ee_height_anchor=bool(
            getattr(cfg, "use_ee_height_anchor", False)
        ),
        use_ee_anchor=bool(getattr(cfg, "use_ee_anchor", False)),
    )
