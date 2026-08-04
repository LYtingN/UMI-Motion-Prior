"""YAML → EEMaskedFlowConfig 转换工具。

Usage:
    from Prior_Recon.Masked_Flow.configs.load_config import config_from_yaml
    cfg = config_from_yaml("configs/delta69_small.yaml")
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from Prior_Recon.Masked_Flow.config import (
    EEMaskedFlowConfig,
    EEMaskedFlowLossConfig,
    MotionRepConfig,
    PrimitiveConfig,
    SkeletonConfig,
    TrainConfig,
)


def config_from_yaml(path: str | Path) -> EEMaskedFlowConfig:
    """Load an EEMaskedFlowConfig from a YAML file.

    Missing keys fall back to dataclass defaults, so a YAML only needs to
    specify the fields that differ from the defaults.
    """
    with open(path, encoding="utf-8") as f:
        d: dict[str, Any] = yaml.safe_load(f) or {}
    return _build_config(d)


def _build_config(d: dict[str, Any]) -> EEMaskedFlowConfig:
    skeleton  = _build_skeleton(d.get("skeleton") or {})
    motion    = _build_motion(d.get("motion") or {})
    primitive = _build_primitive(d.get("primitive") or {})
    train     = _build_train(d.get("train") or {})
    loss      = _build_loss(d.get("loss") or {})

    top_keys = {
        "hidden_dim", "n_layers", "n_heads", "dropout", "time_emb_dim",
        "ode_steps", "transformer_ffn_mult", "out_proj_hidden_mult",
        "temporal_backbone", "dit_cross_attention_gate_init",
        "hierarchy_fine_layers",
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
    top = {k: d[k] for k in top_keys if k in d}

    return EEMaskedFlowConfig(
        skeleton=skeleton,
        motion=motion,
        primitive=primitive,
        train=train,
        loss=loss,
        **top,
    )


def _build_skeleton(d: dict) -> SkeletonConfig:
    n = int(d.get("n_total_joints", 69))
    upper = d.get("upper_body_global_indices")
    if upper is None:
        upper = list(range(n))
    return SkeletonConfig(
        name=str(d.get("name", "g1_delta69")),
        n_total_joints=n,
        upper_body_global_indices=list(upper),
        wrist_local_indices=list(d.get("wrist_local_indices") or []),
    )


def _build_motion(d: dict) -> MotionRepConfig:
    return MotionRepConfig(
        joint_repr=str(d.get("joint_repr", "dof")),
        seq_len=int(d.get("seq_len", 10)),
        fps=int(d.get("fps", 30)),
        window_stride=int(d.get("window_stride", 8)),
        normalize=bool(d.get("normalize", False)),
    )


def _build_primitive(d: dict) -> PrimitiveConfig:
    return PrimitiveConfig(
        enabled=bool(d.get("enabled", True)),
        history_len=int(d.get("history_len", 2)),
        future_len=int(d.get("future_len", 8)),
        num_primitives=int(d.get("num_primitives", 4)),
        segment_unrolls=int(d.get("segment_unrolls", 1)),
        segment_stride=int(d.get("segment_stride", 8)),
        rollout_start_ratio=float(d.get("rollout_start_ratio", 0.1)),
        rollout_end_ratio=float(d.get("rollout_end_ratio", 0.5)),
        rollout_max_prob=float(d.get("rollout_max_prob", 1.0)),
        val_rollout=bool(d.get("val_rollout", True)),
    )


def _build_train(d: dict) -> TrainConfig:
    return TrainConfig(
        lr=float(d.get("lr", 2e-4)),
        weight_decay=float(d.get("weight_decay", 1e-4)),
        batch_size=int(d.get("batch_size", 256)),
        max_microbatch_size=int(d.get("max_microbatch_size", min(int(d.get("batch_size", 256)), 32))),
        num_workers=int(d.get("num_workers", 0)),
        gpu_memory_fraction=float(d.get("gpu_memory_fraction", 0.85)),
        n_epochs=int(d.get("n_epochs", 500)),
        grad_clip=float(d.get("grad_clip", 1.0)),
        ema_decay=float(d.get("ema_decay", 0.999)),
        warmup_epochs=int(d.get("warmup_epochs", 5)),
        val_split=float(d.get("val_split", 0.1)),
        log_interval=int(d.get("log_interval", 50)),
        ckpt_dir=str(d.get("ckpt_dir", "checkpoints/Prior_Recon/masked_flow_delta69")),
        amp=str(d.get("amp", "none")),
    )


def _build_loss(d: dict) -> EEMaskedFlowLossConfig:
    return EEMaskedFlowLossConfig(
        flow_weight=float(d.get("flow_weight", 1.0)),
        recon_weight=float(d.get("recon_weight", 0.25)),
        velocity_weight=float(d.get("velocity_weight", 0.1)),
        leg_joint_weight=float(d.get("leg_joint_weight", 1.0)),
        root_weight=float(d.get("root_weight", 1.0)),
        accel_weight=float(d.get("accel_weight", 0.0)),
        contact_weight=float(d.get("contact_weight", 0.0)),
        body_trans_weight=float(d.get("body_trans_weight", 0.0)),
        body_rot_weight=float(d.get("body_rot_weight", 0.0)),
        ee_pos_weight=float(d.get("ee_pos_weight", 0.0)),
        ee_rot_weight=float(d.get("ee_rot_weight", 0.0)),
        ee_cond_pos_weight=float(d.get("ee_cond_pos_weight", 0.0)),
        ee_cond_rot_weight=float(d.get("ee_cond_rot_weight", 0.0)),
        anchor_pos_weight=float(d.get("anchor_pos_weight", 0.0)),
        anchor_rot_weight=float(d.get("anchor_rot_weight", 0.0)),
        dof_pos_weight=float(d.get("dof_pos_weight", 0.0)),
        dof_vel_weight=float(d.get("dof_vel_weight", 0.0)),
        foot_contact_weight=float(d.get("foot_contact_weight", 0.0)),
        foot_skate_weight=float(d.get("foot_skate_weight", 0.0)),
        foot_sole_pos_weight=float(d.get("foot_sole_pos_weight", 0.0)),
        foot_sole_skate_weight=float(d.get("foot_sole_skate_weight", 0.0)),
        boundary_foot_skate_weight=float(d.get("boundary_foot_skate_weight", 0.0)),
        boundary_foot_skate_topk_ratio=float(
            d.get("boundary_foot_skate_topk_ratio", 0.25)
        ),
        self_skate_weight=float(d.get("self_skate_weight", 0.0)),
        foot_rot_weight=float(d.get("foot_rot_weight", 0.0)),
        contact_pred_weight=float(d.get("contact_pred_weight", 0.0)),
        contact_pred_bce=bool(d.get("contact_pred_bce", False)),
        contact_pred_pos_weight=float(d.get("contact_pred_pos_weight", 1.0)),
        root_consistency_weight=float(d.get("root_consistency_weight", 0.0)),
        root_vel_weight=float(d.get("root_vel_weight", 0.0)),
        root_height_weight=float(d.get("root_height_weight", 0.0)),
        drift_yaw_weight=float(d.get("drift_yaw_weight", 0.0)),
        drift_xy_weight=float(d.get("drift_xy_weight", 0.0)),
        root_curriculum_start_mult=float(d.get("root_curriculum_start_mult", 1.0)),
        root_curriculum_end_ratio=float(d.get("root_curriculum_end_ratio", 0.0)),
        ee_curriculum_start_ratio=float(d.get("ee_curriculum_start_ratio", 0.0)),
        ee_curriculum_end_ratio=float(d.get("ee_curriculum_end_ratio", 0.0)),
        skate_curriculum_start_ratio=float(d.get("skate_curriculum_start_ratio", 0.0)),
        skate_curriculum_end_ratio=float(d.get("skate_curriculum_end_ratio", 0.0)),
        smooth_weight=float(d.get("smooth_weight", 0.0)),
        seam_accel_weight=float(d.get("seam_accel_weight", 0.0)),
        seam_vel_weight=float(d.get("seam_vel_weight", 0.0)),
        quantize_rot_weight=float(d.get("quantize_rot_weight", 0.0)),
        quantize_trans_weight=float(d.get("quantize_trans_weight", 0.0)),
    )
