from Prior_Recon.Masked_Flow.config import (
    EEMaskedFlowConfig,
    EEMaskedFlowLossConfig,
    MotionRepConfig,
    PrimitiveConfig,
    SkeletonConfig,
    TrainConfig,
)

def g1sonic_delta_masked_flow_config() -> EEMaskedFlowConfig:
    """
    s_full : (B, T, 69)  -- heading-aligned delta features (all 29 DOFs,
                            wrists included)
    s_ee   : (B, T, 36)  -- dual-hand EE: left+right palm pos(3)+relative rot6d(6)
                            + velocity(18)
    """
    return EEMaskedFlowConfig(
        skeleton=SkeletonConfig(
            name="g1_delta69",
            n_total_joints=69,
            # Treat every feature dim as a "joint" with repr_dim=1.
            # wrist_local_indices=[] -> all 69 dims are body (no EE mask split)
            upper_body_global_indices=list(range(69)),
            wrist_local_indices=[],
        ),
        motion=MotionRepConfig(
            joint_repr="dof",    # joint_repr_dim=1 -> n_total_dof=69
            seq_len=10,
            fps=30,
            window_stride=8,
            normalize=False,     # normalisation done inside G1DeltaFeatDataset
        ),
        primitive=PrimitiveConfig(
            enabled=True,
            history_len=2,
            future_len=8,
            num_primitives=4,
            segment_stride=8,
            rollout_start_ratio=0.1,
            rollout_end_ratio=0.5,
            rollout_max_prob=1.0,
            val_rollout=True,
        ),
        train=TrainConfig(
            lr=2e-4,
            weight_decay=1e-4,
            batch_size=256,
            max_microbatch_size=32,
            num_workers=0,
            gpu_memory_fraction=0.85,
            n_epochs=500,
            grad_clip=1.0,
            ema_decay=0.999,
            warmup_epochs=5,
            val_split=0.1,
            log_interval=50,
            ckpt_dir="checkpoints/Prior_Recon/masked_flow_delta69",
        ),
        hidden_dim=768,
        n_layers=12,
        n_heads=12,
        dropout=0.1,
        time_emb_dim=256,
        ode_steps=32,
        # EE conditioning: palm pos(3)+relative rot6d(6) x 2 = 18D, +vel(18) = 36D
        use_ee_pos=True,
        use_ee_height_anchor=False,
        use_ee_vel=True,
        use_logit_normal_t=True,
        logit_normal_sigma=1.0,
        loss=EEMaskedFlowLossConfig(
            flow_weight=1.0,
            recon_weight=0.15,
            velocity_weight=0.15,
            leg_joint_weight=2.0,
            root_weight=1.5,
            accel_weight=0.0,
            contact_weight=0.0, # 0.1
            body_trans_weight=0.05,
            body_rot_weight=1e-2,
            ee_pos_weight=0.2,
            ee_rot_weight=0.05,
            ee_cond_pos_weight=0.5,
            ee_cond_rot_weight=0.1,
            dof_pos_weight=0.03,
            dof_vel_weight=0.00,
            foot_contact_weight=0.01,
            drift_yaw_weight=0.05,
            drift_xy_weight=0.5,
            root_curriculum_start_mult=2.0,
            root_curriculum_end_ratio=0.35,
            ee_curriculum_start_ratio=0.25,
            ee_curriculum_end_ratio=0.65,
            smooth_weight=0.0,
            quantize_rot_weight=0.0,
            quantize_trans_weight=0.0,
        ),
    )
