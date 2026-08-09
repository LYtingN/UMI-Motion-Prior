from __future__ import annotations

from dataclasses import dataclass, field
from typing import List


@dataclass
class SkeletonConfig:
    name: str = "smpl22"
    n_total_joints: int = 22
    upper_body_global_indices: List[int] = field(
        default_factory=lambda: [0, 3, 6, 9, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21]
    )
    wrist_local_indices: List[int] = field(default_factory=lambda: [12, 13])

    @property
    def n_upper_body_joints(self) -> int:
        return len(self.upper_body_global_indices)


@dataclass
class MotionRepConfig:
    joint_repr: str = "rot6d"
    joint_repr_dim: int = 6
    include_positions: bool = False
    seq_len: int = 16
    fps: int = 30
    window_stride: int = 16
    normalize: bool = True

    def __post_init__(self) -> None:
        if self.joint_repr == "pos_rot6d":
            self.joint_repr_dim = 9
        elif self.joint_repr == "rot6d":
            self.joint_repr_dim = 6
        elif self.joint_repr == "dof":
            self.joint_repr_dim = 1


@dataclass
class TrainConfig:
    lr: float = 1e-4
    weight_decay: float = 1e-4
    batch_size: int = 64
    max_microbatch_size: int = 32
    num_workers: int = 0
    gpu_memory_fraction: float = 0.85
    n_epochs: int = 200
    grad_clip: float = 1.0
    ema_decay: float = 0.999
    warmup_epochs: int = 5
    val_split: float = 0.1
    log_interval: int = 50
    ckpt_dir: str = "checkpoints/Prior_Recon/masked_flow"
    amp: str = "none"


@dataclass
class PrimitiveConfig:
    enabled: bool = False
    history_len: int = 2
    future_len: int = 8
    num_primitives: int = 4
    segment_unrolls: int = 1
    segment_stride: int = 8
    rollout_start_ratio: float = 0.1
    rollout_end_ratio: float = 0.5
    rollout_max_prob: float = 1.0
    val_rollout: bool = True

    @property
    def primitive_len(self) -> int:
        return self.history_len + self.future_len

    @property
    def segment_len(self) -> int:
        return self.history_len + self.num_primitives * self.future_len

    @property
    def segment_step(self) -> int:
        return self.num_primitives * self.future_len

    @property
    def unroll_len(self) -> int:
        segment_unrolls = int(getattr(self, "segment_unrolls", 1))
        return self.history_len + segment_unrolls * self.segment_step


@dataclass
class DataConfig:
    # Backward-compatibility shim for older checkpoints that pickle the full
    # config object. The current code path does not use this config anymore,
    # but torch.load still needs the class to exist during unpickling.
    data_root: str = "data/mocap"
    n_episodes: int = 300
    episode_len: int = 300


@dataclass
class EEMaskedFlowLossConfig:
    flow_weight: float = 1.0
    recon_weight: float = 0.25
    velocity_weight: float = 0.1
    leg_joint_weight: float = 1.0
    accel_weight: float = 0.0
    root_weight: float = 1.0
    contact_weight: float = 0.0
    body_trans_weight: float = 0.0
    body_rot_weight: float = 0.0
    ee_pos_weight: float = 0.0
    ee_rot_weight: float = 0.0
    ee_cond_pos_weight: float = 0.0
    ee_cond_rot_weight: float = 0.0
    # Layer C (absolute EE anchor): supervise the predicted absolute hand
    # ORIENTATION against the heading-frame anchor the condition now carries
    # (anchor_rot, a 1-cos geodesic surrogate, rad-scale), so the model must
    # match the absolute initial hand orientation rather than only shape relative
    # to its own unobserved first frame. Requires use_ee_anchor.
    #
    # anchor_pos_weight is RESERVED, not implemented: the position anchor needs
    # the pelvis world xy stored per frame (the delta features discard absolute
    # root xy, and palm keypoints are in mocap-world), which requires a dataset
    # regen. Setting it > 0 raises in MaskedFlowMatchingLoss rather than silently
    # doing nothing. Height is already handled by use_ee_height_anchor.
    anchor_pos_weight: float = 0.0
    anchor_rot_weight: float = 0.0
    dof_pos_weight: float = 0.0
    dof_vel_weight: float = 0.0
    foot_contact_weight: float = 0.0
    # Foot-skate: during GT contact frames, predicted foot world velocity (FK,
    # scaled to m/s) must match the GT foot velocity (~0 in stance). The only
    # term that couples root translation with leg articulation frame-to-frame.
    foot_skate_weight: float = 0.0
    # Layer A (foot sole geometry): supervise the N heel/toe sole points per foot
    # instead of only the ankle-roll origin, plus a contact-masked geodesic
    # orientation loss. The single ankle origin sits near the pitch/roll axes so
    # a rotation of the foot barely moves it -- the origin-only foot_contact /
    # foot_skate terms leave a rotational null space (heel-up / toe-drag). These
    # terms close it. All gated by the SAME skate curriculum as foot_skate.
    #   foot_sole_pos_weight   : contact-masked sole-point world position (m)
    #   foot_sole_skate_weight : stance sole-point world velocity toward zero
    #                            (m/s; fps^2 amplified, so keep small)
    #   foot_rot_weight        : contact-masked geodesic foot-orientation (rad^2)
    foot_sole_pos_weight: float = 0.0
    foot_sole_skate_weight: float = 0.0
    boundary_foot_skate_weight: float = 0.0
    boundary_foot_skate_topk_ratio: float = 0.25
    foot_rot_weight: float = 0.0
    # [B] Predicted-contact self-gated anti-skate. Every other anti-skate term
    # masks stance with the GT contact, so at inference (where post-processing
    # reads the model's OWN predicted contact) the model was never trained to
    # keep the foot still exactly where IT predicts contact. This term masks the
    # predicted sole velocity->0 with the model's own (detached) predicted
    # contact, tying the predicted channel to the foot motion. fps^2 m/s scale
    # like foot_sole_skate -> keep small. Gated by the skate curriculum.
    self_skate_weight: float = 0.0
    # Dedicated supervision of the contact channels (5:7) so downstream
    # contact-driven post-processing gets a reliable signal.
    contact_pred_weight: float = 0.0
    # [C] Supervise the contact channel with BCE (in [0,1] probability space)
    # instead of MSE. MSE's optimum under noisy {0,1} labels is the conditional
    # mean (~0.5, thresholds unreliably); BCE drives the channel to the 0/1
    # extremes. Channel stays a probability, so no downstream consumer changes.
    contact_pred_bce: bool = False
    # BCE positive-class weight (contact frames are the minority in loco clips);
    # >1 upweights the stance class so it is not washed out by swing frames.
    contact_pred_pos_weight: float = 1.0
    # abs_root only: dual-encoding self-consistency — per-frame difference of
    # the absolute channels must equal the delta channels (xy + wrapped yaw).
    root_consistency_weight: float = 0.0
    # Per-frame-decoded root state channels (roll/pitch, height, abs xy/yaw):
    # dedicated finite-difference velocity supervision against GT (m/s scale).
    root_vel_weight: float = 0.0
    # Absolute per-frame supervision of the height channel (metre scale).
    # Counterweight to root_vel: velocity-only supervision on height pins the
    # RATE but leaves the absolute level free to drift (dc offset -> the base
    # floats), and the strong FK foot terms are invariant to the
    # root-up/legs-extend direction, so height needs its own absolute anchor.
    root_height_weight: float = 0.0
    drift_yaw_weight: float = 0.0
    drift_xy_weight: float = 0.0
    root_curriculum_start_mult: float = 1.0
    root_curriculum_end_ratio: float = 0.0
    ee_curriculum_start_ratio: float = 0.0
    ee_curriculum_end_ratio: float = 0.0
    # Ramp-in window for the FK anti-skate terms (foot_skate + foot_contact),
    # as training-progress ratios. Start AFTER the EE curriculum completes so
    # the model settles the EE solution before the world-frame foot terms
    # start competing for the same root/leg channels. end <= start = no ramp
    # (always fully on), which keeps older configs unchanged.
    skate_curriculum_start_ratio: float = 0.0
    skate_curriculum_end_ratio: float = 0.0
    smooth_weight: float = 0.0
    # Segment-mode only (requires segment_geometry_loss): second-difference MSE
    # vs GT on the boundary-crossing triples of the stitched segment. Triples
    # spanning two primitives' GENERATED frames exist in no per-primitive
    # window, so cross-seam second-order continuity (the periodic "hitch" at
    # every future_len frames) was structurally unsupervised. Same raw-diff
    # scale as accel_weight; mask-restricted to seam triples (no double count).
    seam_accel_weight: float = 0.0
    # Segment-mode only (requires segment_geometry_loss): FIRST-difference
    # (velocity) MSE vs GT on the boundary-crossing PAIR of the stitched segment
    # -- the direct companion to seam_accel_weight. seam_accel constrains only
    # curvature (2nd order); the visible "hitch" is a first-order velocity STEP
    # at each primitive boundary (the model treats each primitive's first
    # generated frame as a fresh start, so the cross-seam velocity jumps ~1.6x
    # the interior). This term supervises exactly the pair (b-1, b) at every
    # boundary b = history_len + k*future_len so the velocity carried across the
    # seam matches GT. Both endpoints are the previous / next primitive's
    # GENERATED (non-detached) frames on the stitched segment, so the gradient
    # pulls the two sides together -- the one place that can happen. Same
    # raw-diff scale as velocity_weight; mask-restricted to seam pairs.
    seam_vel_weight: float = 0.0
    quantize_rot_weight: float = 0.0
    quantize_trans_weight: float = 0.0


@dataclass
class EEMaskedFlowConfig:
    skeleton: SkeletonConfig = field(default_factory=SkeletonConfig)
    motion: MotionRepConfig = field(default_factory=MotionRepConfig)
    primitive: PrimitiveConfig = field(default_factory=PrimitiveConfig)
    train: TrainConfig = field(
        default_factory=lambda: TrainConfig(
            ckpt_dir="checkpoints/Prior_Recon/masked_flow",
        )
    )
    feat_root: str | None = None

    hidden_dim: int = 512
    n_layers: int = 8
    n_heads: int = 8
    dropout: float = 0.1
    time_emb_dim: int = 128
    ode_steps: int = 32
    transformer_ffn_mult: int = 4
    out_proj_hidden_mult: int = 2
    temporal_backbone: str = "flat"
    # Initial adaLN gate bias shared by all three dit residual branches
    # (self-attention, cross-attention, mlp). 0.0 is canonical DiT's
    # identity-at-init; a small positive value starts the branches "on" so their
    # weights get gradient before the zero-init output head has fitted a
    # pointwise map through the ungated input bypass. See TemporalDiTSpec.
    dit_residual_gate_init: float = 0.1
    # Drop invalid EE preview tokens from the dit cross-attention key set
    # instead of tagging them with a learned "invalid" embedding. Only the dit
    # backbone reads this. Defaults to False so a config that predates the flag
    # keeps its trained behaviour; new dit runs should set it True.
    dit_mask_invalid_lookahead: bool = False
    hierarchy_fine_layers: int = 4
    hierarchy_coarse_layers: int = 4
    hierarchy_refine_layers: int = 4
    hierarchy_downsample_factor: int = 2
    use_ee_pos: bool = True
    use_ee_height_anchor: bool = False
    use_ee_vel: bool = True
    use_logit_normal_t: bool = False
    logit_normal_sigma: float = 1.0
    # Diffusion forcing: sample an independent noise level per frame during
    # training instead of one shared level for the whole sequence. Off by
    # default to preserve legacy training behaviour.
    per_frame_noise: bool = False
    # EE-as-state: append this many per-frame EE pose dims (2 hands x
    # [pos3 + rot6d] = 18) to the motion feature and hard-pin them through
    # obs_mask at train and sample time. 0 keeps the legacy delta69 layout
    # where EE enters only through the soft s_ee condition tokens.
    ee_state_dim: int = 0
    # EE lookahead: condition each primitive on this many EE frames BEYOND its
    # window end (soft preview tokens), so the body can anticipate where the
    # hands are heading (e.g. start stepping before the arm runs out of reach).
    # 0 disables the preview path entirely (legacy behaviour).
    lookahead_len: int = 0
    # Absolute root channels (drift fix): append 4 per-frame channels to the
    # delta69 motion feature -- xy_rel (2, position relative to the segment's
    # frame 0, in the heading-aligned anchor frame) and yaw_rel as (cos, sin)
    # of the accumulated turn since frame 0. Reconstruction then READS root
    # xy/yaw per frame instead of integrating per-frame deltas, so single-frame
    # errors stay local (no cumsum amplification, no yaw-rotates-all-later-xy
    # drift). GT values are the exact within-window cumsum of the delta
    # channels, so both layouts describe identical motion. False keeps the
    # legacy pure-delta69 layout and integration-based reconstruction.
    abs_root_channels: bool = False
    # Temporal downsampling of the preview: one token per `lookahead_stride`
    # frames. The preview conveys intent, not per-frame targets.
    lookahead_stride: int = 2
    # Layer B (history-mismatch augmentation): during training, inject a small
    # constant pose offset into the PINNED history prefix so the model sees
    # histories that disagree with the EE condition anchor -- the deployment
    # reality (standing-init seed, or real executed state re-seeded every segment
    # in closed loop) that pure GT / self-rollout training never produces.
    # Without it the model has only ever seen history whose first frame already
    # matches the reference, so a standing/real-state start carries a constant
    # offset it cannot correct. Off by default (prob=0 -> legacy behaviour).
    #   history_perturb_prob       : per-segment probability of perturbing
    #   history_perturb_joint_std  : rad std of the per-joint pose offset (11:40)
    #   history_perturb_tilt_std   : rad std of the root roll/pitch offset (0:4)
    history_perturb_prob: float = 0.0
    history_perturb_joint_std: float = 0.0
    history_perturb_tilt_std: float = 0.0
    # Layer C (absolute EE anchor): condition on the initial hand ORIENTATION in
    # the first-frame pelvis heading frame, so the model observes the absolute
    # initial hand rotation instead of only shape relative to its own unobserved
    # first frame. This removes the constant-rotation-offset failure of standing/
    # real-state starts: with the relative-only condition, left-multiplying the
    # whole hand trajectory by any fixed rotation leaves R_0^-1 R_t unchanged, so
    # the absolute start orientation is unobservable. Adds EE_ANCHOR_DIM (=12,
    # 2 hands x rot6d) dims to s_ee. Rotation only -- position x/y is not
    # recoverable from the delta features (no stored pelvis world xy) and height
    # is already covered by use_ee_height_anchor. Computed identically offline
    # (recon) and online (bridge, from executed-state FK). Off by default.
    use_ee_anchor: bool = False
    # Segment-anchored ee_cond (loss-side only, no model I/O change): rebase the
    # ee_cond pos/rot comparison to the GT SEGMENT frame-0 hand pose instead of
    # each primitive's own first frame. The legacy per-primitive rebase reduced
    # the strongest hand term to relative SHAPE about an anchor that drifts with
    # the rollout history; with abs_root the FK of every primitive is already in
    # the segment heading frame, so one shared GT anchor makes ee_cond supervise
    # the ABSOLUTE in-segment hand pose (the s_ee target already carries it).
    # Deploy-legal: the anchor parameterizes the loss only, inference never sees
    # it. Requires abs_root_channels. Off keeps the legacy relative behaviour.
    ee_cond_segment_anchor: bool = False
    # Stitch the four primitives' (non-detached) pred_x1 back into the full
    # segment and compute the geometry/FK losses ONCE on it (per-primitive calls
    # then skip geometry). Buys cross-seam gradients -- the autoregressive
    # history is detached, so without this no gradient ever crosses a primitive
    # boundary -- plus full-horizon drift/EE supervision. Loss aggregation only;
    # the forward pass (and its detached conditioning) is unchanged.
    segment_geometry_loss: bool = False
    loss: EEMaskedFlowLossConfig = field(default_factory=EEMaskedFlowLossConfig)

    @property
    def n_motion_dof(self) -> int:
        """Body-motion feature dims (delta69 [+4 abs-root]), excluding pinned EE state dims."""
        base = self.skeleton.n_upper_body_joints * self.motion.joint_repr_dim
        if getattr(self, "abs_root_channels", False):
            base += 4  # xy_rel(2) + yaw_rel cos/sin(2)
        return base

    @property
    def n_total_dof(self) -> int:
        return self.n_motion_dof + getattr(self, "ee_state_dim", 0)

    @property
    def n_lookahead_tokens(self) -> int:
        look_len = int(getattr(self, "lookahead_len", 0))
        if look_len <= 0:
            return 0
        stride = max(int(getattr(self, "lookahead_stride", 1)), 1)
        return (look_len + stride - 1) // stride

    @property
    def n_ee_joints(self) -> int:
        return len(self.skeleton.wrist_local_indices)

    @property
    def n_body_joints(self) -> int:
        return self.skeleton.n_upper_body_joints - self.n_ee_joints

    # Absolute-anchor block width (Layer C): 2 hands x rot6d = 12. Rotation only
    # -- see use_ee_anchor. Position x/y is intentionally excluded (not
    # recoverable from the delta features without storing pelvis world pos; the
    # height is already handled by use_ee_height_anchor).
    EE_ANCHOR_DIM = 12

    @property
    def ee_feat_dim(self) -> int:
        # Per-frame s_ee layout (offsets consumed by loss / lookahead code):
        #   [0:18]        relative pose (2 hands x [pos3 + rot6d]); read as [:18]
        #   [.. : ..+2]   height anchor        (use_ee_height_anchor)
        #   [.. : ..+12]  absolute rot anchor  (use_ee_anchor, Layer C) -- static,
        #                 broadcast across frames; read via ee_anchor_offset
        #   [last 18]     velocity             (use_ee_vel); read as [-18:]
        # The anchor sits BEFORE velocity on purpose so the [:18] and [-18:]
        # slices existing code relies on stay valid regardless of use_ee_anchor.
        base = 18 if self.use_ee_pos else self.n_ee_joints * self.motion.joint_repr_dim
        if self.use_ee_height_anchor:
            base += 2
        if getattr(self, "use_ee_anchor", False):
            base += self.EE_ANCHOR_DIM
        if self.use_ee_vel:
            base += 18
        return base

    @property
    def ee_anchor_offset(self) -> int:
        """Start index of the 12D absolute-anchor block inside a s_ee frame."""
        off = 18 if self.use_ee_pos else self.n_ee_joints * self.motion.joint_repr_dim
        if self.use_ee_height_anchor:
            off += 2
        return off
