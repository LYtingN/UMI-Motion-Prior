"""YAML -> TwoStageMaskedFlowConfig 转换工具（两级 root->body 级联专用）。

与 load_config.py 隔离：单级配置加载路径不变，这里只是在其基础上多解析一个
``root_stage`` 段。基础字段（skeleton/motion/primitive/train/loss + 顶层模型
字段）沿用 load_config 的 builder，语义与单级 YAML 完全一致。

Usage:
    from Prior_Recon.Masked_Flow.configs.load_config_two_stage import (
        two_stage_config_from_yaml,
    )
    cfg = two_stage_config_from_yaml("configs/twostage_delta73_small.yaml")
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from Prior_Recon.Masked_Flow.config_two_stage import (
    RootStageConfig,
    RootStageLossConfig,
    TwoStageMaskedFlowConfig,
)
from Prior_Recon.Masked_Flow.configs.load_config import (
    _build_loss,
    _build_motion,
    _build_primitive,
    _build_skeleton,
    _build_train,
)

_TOP_KEYS = {
    "hidden_dim", "n_layers", "n_heads", "dropout", "time_emb_dim",
    "ode_steps", "transformer_ffn_mult", "out_proj_hidden_mult",
    "temporal_backbone", "hierarchy_fine_layers",
    "hierarchy_coarse_layers", "hierarchy_refine_layers",
    "hierarchy_downsample_factor",
    "use_ee_pos", "use_ee_height_anchor", "use_ee_vel",
    "use_logit_normal_t", "logit_normal_sigma",
    "per_frame_noise", "ee_state_dim",
    "lookahead_len", "lookahead_stride",
    "abs_root_channels",
    "history_perturb_prob", "history_perturb_joint_std",
    "history_perturb_tilt_std", "use_ee_anchor",
    "ee_cond_segment_anchor", "segment_geometry_loss",
}


def two_stage_config_from_yaml(path: str | Path) -> TwoStageMaskedFlowConfig:
    """Load a TwoStageMaskedFlowConfig from a YAML file.

    Missing keys fall back to dataclass defaults; a ``root_stage`` section is
    required so a single-stage YAML cannot be silently mistrained as two-stage.
    """
    with open(path, encoding="utf-8") as f:
        d: dict[str, Any] = yaml.safe_load(f) or {}

    if "root_stage" not in d:
        raise ValueError(
            f"{path} has no 'root_stage' section; this loader is for two-stage "
            "cascade configs. Use load_config.config_from_yaml for single-stage YAMLs."
        )

    top = {k: d[k] for k in _TOP_KEYS if k in d}
    if "root_look_stride" in d:
        top["root_look_stride"] = int(d["root_look_stride"])
    return TwoStageMaskedFlowConfig(
        skeleton=_build_skeleton(d.get("skeleton") or {}),
        motion=_build_motion(d.get("motion") or {}),
        primitive=_build_primitive(d.get("primitive") or {}),
        train=_build_train(d.get("train") or {}),
        loss=_build_loss(d.get("loss") or {}),
        root_stage=_build_root_stage(d["root_stage"] or {}),
        **top,
    )


def _build_root_stage(d: dict) -> RootStageConfig:
    loss_d = d.get("loss") or {}
    loss = RootStageLossConfig(
        flow_weight=float(loss_d.get("flow_weight", 1.0)),
        recon_weight=float(loss_d.get("recon_weight", 0.25)),
        velocity_weight=float(loss_d.get("velocity_weight", 0.15)),
        contact_pred_weight=float(loss_d.get("contact_pred_weight", 0.5)),
        contact_pred_bce=bool(loss_d.get("contact_pred_bce", False)),
        contact_pred_pos_weight=float(loss_d.get("contact_pred_pos_weight", 1.0)),
        root_consistency_weight=float(loss_d.get("root_consistency_weight", 0.05)),
        root_vel_weight=float(loss_d.get("root_vel_weight", 0.02)),
        root_height_weight=float(loss_d.get("root_height_weight", 0.3)),
        drift_yaw_weight=float(loss_d.get("drift_yaw_weight", 0.0)),
        drift_xy_weight=float(loss_d.get("drift_xy_weight", 0.0)),
    )
    return RootStageConfig(
        hidden_dim=int(d.get("hidden_dim", 256)),
        n_layers=int(d.get("n_layers", 6)),
        n_heads=int(d.get("n_heads", 8)),
        dropout=float(d.get("dropout", 0.1)),
        time_emb_dim=int(d.get("time_emb_dim", 128)),
        transformer_ffn_mult=int(d.get("transformer_ffn_mult", 4)),
        out_proj_hidden_mult=int(d.get("out_proj_hidden_mult", 2)),
        temporal_backbone=str(d.get("temporal_backbone", "flat")),
        hierarchy_fine_layers=int(d.get("hierarchy_fine_layers", 2)),
        hierarchy_coarse_layers=int(d.get("hierarchy_coarse_layers", 2)),
        hierarchy_refine_layers=int(d.get("hierarchy_refine_layers", 2)),
        hierarchy_downsample_factor=int(
            d.get("hierarchy_downsample_factor", 2)
        ),
        stage_weight=float(d.get("stage_weight", 1.0)),
        loss=loss,
    )
