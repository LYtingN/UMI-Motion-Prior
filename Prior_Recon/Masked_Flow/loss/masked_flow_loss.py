from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from Prior_Recon.Masked_Flow.config import EEMaskedFlowConfig
from Prior_Recon.Masked_Flow.loss.g1_kinematics import (
    G129DeltaForwardKinematics,
    euler_xyz_to_matrix,
    geodesic_cos_loss,
    matrix_to_rot6d,
)
from Prior_Recon.Masked_Flow.loss.foot_skate import (
    boundary_sole_skate_loss,
    project_contact_probability,
)


def _rot6d_to_matrix(rot6d: torch.Tensor) -> torch.Tensor:
    """Inverse of matrix_to_rot6d: recover R by Gram-Schmidt on its first two columns."""
    mat32 = rot6d.reshape(*rot6d.shape[:-1], 3, 2)
    b1 = F.normalize(mat32[..., 0], dim=-1)
    proj = (b1 * mat32[..., 1]).sum(dim=-1, keepdim=True)
    b2 = F.normalize(mat32[..., 1] - proj * b1, dim=-1)
    b3 = torch.cross(b1, b2, dim=-1)
    return torch.stack([b1, b2, b3], dim=-1)

_DELTA_YAW_DIM = 4
_CONTACT_SLICE = slice(5, 7)
_DELTA_TRANS_SLICE = slice(7, 10)
_HEIGHT_DIM = 10
_DOF_SLICE = slice(11, 40)
_LEFT_LEG_DELTA_SLICE = slice(40, 46)
_RIGHT_LEG_DELTA_SLICE = slice(46, 52)
_TWO_PI = 2.0 * math.pi

# Loss names produced by _geometry_losses; also the fields swapped out by
# merge_segment_geometry when the geometry suite moves to segment level.
_GEOMETRY_KEYS = (
    "body_trans",
    "body_rot",
    "ee_pos",
    "ee_rot",
    "ee_cond_pos",
    "ee_cond_rot",
    "anchor_pos",
    "anchor_rot",
    "dof_pos",
    "dof_vel",
    "foot_contact",
    "foot_skate",
    "foot_sole_pos",
    "foot_sole_skate",
    "boundary_foot_skate",
    "self_skate",
    "foot_rot",
    "contact_pred",
    "root_consistency",
    "root_vel",
    "root_height",
    "drift_yaw",
    "drift_xy",
    "smooth",
    "seam_accel",
    "seam_vel",
    "quantize_rot",
    "quantize_trans",
)


@dataclass
class MaskedFlowLossOutput:
    total: torch.Tensor
    flow: torch.Tensor
    recon: torch.Tensor
    velocity: torch.Tensor
    accel: torch.Tensor
    contact: torch.Tensor = None  # type: ignore[assignment]
    body_trans: torch.Tensor = None  # type: ignore[assignment]
    body_rot: torch.Tensor = None  # type: ignore[assignment]
    ee_pos: torch.Tensor = None  # type: ignore[assignment]
    ee_rot: torch.Tensor = None  # type: ignore[assignment]
    ee_cond_pos: torch.Tensor = None  # type: ignore[assignment]
    ee_cond_rot: torch.Tensor = None  # type: ignore[assignment]
    anchor_pos: torch.Tensor = None  # type: ignore[assignment]
    anchor_rot: torch.Tensor = None  # type: ignore[assignment]
    dof_pos: torch.Tensor = None  # type: ignore[assignment]
    dof_vel: torch.Tensor = None  # type: ignore[assignment]
    foot_contact: torch.Tensor = None  # type: ignore[assignment]
    foot_skate: torch.Tensor = None  # type: ignore[assignment]
    foot_sole_pos: torch.Tensor = None  # type: ignore[assignment]
    foot_sole_skate: torch.Tensor = None  # type: ignore[assignment]
    boundary_foot_skate: torch.Tensor = None  # type: ignore[assignment]
    self_skate: torch.Tensor = None  # type: ignore[assignment]
    foot_rot: torch.Tensor = None  # type: ignore[assignment]
    contact_pred: torch.Tensor = None  # type: ignore[assignment]
    root_consistency: torch.Tensor = None  # type: ignore[assignment]
    root_vel: torch.Tensor = None  # type: ignore[assignment]
    root_height: torch.Tensor = None  # type: ignore[assignment]
    drift_yaw: torch.Tensor = None  # type: ignore[assignment]
    drift_xy: torch.Tensor = None  # type: ignore[assignment]
    smooth: torch.Tensor = None  # type: ignore[assignment]
    seam_accel: torch.Tensor = None  # type: ignore[assignment]
    seam_vel: torch.Tensor = None  # type: ignore[assignment]
    quantize_rot: torch.Tensor = None  # type: ignore[assignment]
    quantize_trans: torch.Tensor = None  # type: ignore[assignment]

    _OPTIONAL_FIELDS = (
        "contact",
        "body_trans",
        "body_rot",
        "ee_pos",
        "ee_rot",
        "ee_cond_pos",
        "ee_cond_rot",
        "anchor_pos",
        "anchor_rot",
        "dof_pos",
        "dof_vel",
        "foot_contact",
        "foot_skate",
        "foot_sole_pos",
        "foot_sole_skate",
        "boundary_foot_skate",
        "self_skate",
        "foot_rot",
        "contact_pred",
        "root_consistency",
        "root_vel",
        "root_height",
        "drift_yaw",
        "drift_xy",
        "smooth",
        "seam_accel",
        "seam_vel",
        "quantize_rot",
        "quantize_trans",
    )

    def __post_init__(self):
        for name in self._OPTIONAL_FIELDS:
            if getattr(self, name) is None:
                setattr(self, name, self.total.new_zeros(()))

    def as_dict(
        self,
        include_quantize_rot: bool = True,
        include_quantize_trans: bool = True,
    ) -> dict[str, float]:
        values = {
            "loss/total": self.total.item(),
            "loss/flow": self.flow.item(),
            "loss/recon": self.recon.item(),
            "loss/velocity": self.velocity.item(),
            "loss/accel": self.accel.item(),
            "loss/contact": self.contact.item(),
            "loss/body_trans": self.body_trans.item(),
            "loss/body_rot": self.body_rot.item(),
            "loss/ee_pos": self.ee_pos.item(),
            "loss/ee_rot": self.ee_rot.item(),
            "loss/ee_cond_pos": self.ee_cond_pos.item(),
            "loss/ee_cond_rot": self.ee_cond_rot.item(),
            "loss/anchor_pos": self.anchor_pos.item(),
            "loss/anchor_rot": self.anchor_rot.item(),
            "loss/dof_pos": self.dof_pos.item(),
            "loss/dof_vel": self.dof_vel.item(),
            "loss/foot_contact": self.foot_contact.item(),
            "loss/foot_skate": self.foot_skate.item(),
            "loss/foot_sole_pos": self.foot_sole_pos.item(),
            "loss/foot_sole_skate": self.foot_sole_skate.item(),
            "loss/boundary_foot_skate": self.boundary_foot_skate.item(),
            "loss/self_skate": self.self_skate.item(),
            "loss/foot_rot": self.foot_rot.item(),
            "loss/contact_pred": self.contact_pred.item(),
            "loss/root_consistency": self.root_consistency.item(),
            "loss/root_vel": self.root_vel.item(),
            "loss/root_height": self.root_height.item(),
            "loss/drift_yaw": self.drift_yaw.item(),
            "loss/drift_xy": self.drift_xy.item(),
            "loss/smooth": self.smooth.item(),
            "loss/seam_accel": self.seam_accel.item(),
            "loss/seam_vel": self.seam_vel.item(),
        }
        if include_quantize_rot:
            values["loss/quantize_rot"] = self.quantize_rot.item()
        if include_quantize_trans:
            values["loss/quantize_trans"] = self.quantize_trans.item()
        return values


def mean_loss_outputs(losses: list[MaskedFlowLossOutput]) -> MaskedFlowLossOutput:
    if not losses:
        raise ValueError("Cannot average an empty list of MaskedFlowLossOutput.")
    return MaskedFlowLossOutput(
        total=torch.stack([loss.total for loss in losses]).mean(),
        flow=torch.stack([loss.flow for loss in losses]).mean(),
        recon=torch.stack([loss.recon for loss in losses]).mean(),
        velocity=torch.stack([loss.velocity for loss in losses]).mean(),
        accel=torch.stack([loss.accel for loss in losses]).mean(),
        contact=torch.stack([loss.contact for loss in losses]).mean(),
        body_trans=torch.stack([loss.body_trans for loss in losses]).mean(),
        body_rot=torch.stack([loss.body_rot for loss in losses]).mean(),
        ee_pos=torch.stack([loss.ee_pos for loss in losses]).mean(),
        ee_rot=torch.stack([loss.ee_rot for loss in losses]).mean(),
        ee_cond_pos=torch.stack([loss.ee_cond_pos for loss in losses]).mean(),
        ee_cond_rot=torch.stack([loss.ee_cond_rot for loss in losses]).mean(),
        anchor_pos=torch.stack([loss.anchor_pos for loss in losses]).mean(),
        anchor_rot=torch.stack([loss.anchor_rot for loss in losses]).mean(),
        dof_pos=torch.stack([loss.dof_pos for loss in losses]).mean(),
        dof_vel=torch.stack([loss.dof_vel for loss in losses]).mean(),
        foot_contact=torch.stack([loss.foot_contact for loss in losses]).mean(),
        foot_skate=torch.stack([loss.foot_skate for loss in losses]).mean(),
        foot_sole_pos=torch.stack([loss.foot_sole_pos for loss in losses]).mean(),
        foot_sole_skate=torch.stack([loss.foot_sole_skate for loss in losses]).mean(),
        boundary_foot_skate=torch.stack(
            [loss.boundary_foot_skate for loss in losses]
        ).mean(),
        self_skate=torch.stack([loss.self_skate for loss in losses]).mean(),
        foot_rot=torch.stack([loss.foot_rot for loss in losses]).mean(),
        contact_pred=torch.stack([loss.contact_pred for loss in losses]).mean(),
        root_consistency=torch.stack([loss.root_consistency for loss in losses]).mean(),
        root_vel=torch.stack([loss.root_vel for loss in losses]).mean(),
        root_height=torch.stack([loss.root_height for loss in losses]).mean(),
        drift_yaw=torch.stack([loss.drift_yaw for loss in losses]).mean(),
        drift_xy=torch.stack([loss.drift_xy for loss in losses]).mean(),
        smooth=torch.stack([loss.smooth for loss in losses]).mean(),
        seam_accel=torch.stack([loss.seam_accel for loss in losses]).mean(),
        seam_vel=torch.stack([loss.seam_vel for loss in losses]).mean(),
        quantize_rot=torch.stack([loss.quantize_rot for loss in losses]).mean(),
        quantize_trans=torch.stack([loss.quantize_trans for loss in losses]).mean(),
    )


def merge_segment_geometry(
    prim: MaskedFlowLossOutput,
    geom: dict[str, torch.Tensor],
) -> MaskedFlowLossOutput:
    """Combine primitive-averaged base losses with segment-level geometry terms.

    ``prim`` must come from forward(include_geometry=False) calls (its geometry
    fields are zeros, so nothing is double counted); the result carries
    total = prim.total + sum(geom) with the geometry fields replaced by the
    segment-level values.
    """
    unknown = set(geom) - set(_GEOMETRY_KEYS)
    if unknown:
        raise ValueError(f"Unexpected segment geometry keys: {sorted(unknown)}")
    return MaskedFlowLossOutput(
        total=prim.total + sum(geom.values()),
        flow=prim.flow,
        recon=prim.recon,
        velocity=prim.velocity,
        accel=prim.accel,
        contact=prim.contact,
        **geom,
    )


def _frame_unknown_mask(obs_mask: torch.Tensor) -> torch.Tensor:
    """Collapse per-dim observation mask to a per-frame unknown mask."""
    return (1.0 - obs_mask).amax(dim=-1, keepdim=True)


def _transition_mask(frame_mask: torch.Tensor) -> torch.Tensor:
    """Mask for finite differences: trainable when EITHER endpoint is unknown.

    A pinned endpoint (history prefix / rollout carry) is a constant, so the
    difference still has a well-defined gradient into the unknown endpoint and
    anchors it to the observed boundary. Requiring BOTH endpoints unknown
    (the previous `f[1:] * f[:-1]`) silently dropped the history->future seam —
    exactly where autoregressive stitching jumps. With a GT history the seam
    term reduces to extra recon pressure on the first generated frame; with a
    rollout history it becomes a genuine "continue smoothly from the drifted
    state" constraint.
    """
    if frame_mask.shape[1] < 2:
        return frame_mask[:, :0]
    return torch.maximum(frame_mask[:, 1:], frame_mask[:, :-1])


def _expand_mask(mask: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    while mask.ndim < target.ndim:
        mask = mask.unsqueeze(-1)
    return mask.expand_as(target)


def _masked_mse(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    mask_full = _expand_mask(mask, pred).to(dtype=pred.dtype)
    denom = mask_full.sum().clamp_min(1.0)
    return (((pred - target) ** 2) * mask_full).sum() / denom


def _masked_l1(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    mask_full = _expand_mask(mask, pred).to(dtype=pred.dtype)
    denom = mask_full.sum().clamp_min(1.0)
    return ((pred - target).abs() * mask_full).sum() / denom


def _masked_bce(
    pred: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
    pos_weight: float = 1.0,
    eps: float = 1e-6,
) -> torch.Tensor:
    """Masked binary cross-entropy in PROBABILITY space (pred, target in [0,1]).

    Unlike MSE, whose optimum under noisy {0,1} labels is the conditional mean
    (the "mushy" ~0.3-0.7 channel that thresholds unreliably), BCE drives the
    prediction to the 0/1 extremes, so a fixed 0.5 threshold downstream becomes
    stable.

    The channel MUST stay a [0,1] probability, not a logit: downstream consumers
    threshold it at 0.5 (foot_lock / body_qp) and the flow target + recon MSE are
    computed on the same channel as binary {0,1}. So BCE is taken on the
    probability directly, with the log arguments clamped to [eps, 1-eps] for
    numerical safety.

    ``torch.clamp`` has ZERO gradient in its saturated region, so a raw flow
    output that leaves [0,1] on the WRONG side (measured: contact preds reach
    ~[-0.03, 1.03]) would get no gradient from this term -- gradient-dead exactly
    where the channel is most wrong, leaving only the weak recon MSE to correct
    it. A quadratic barrier on the RAW pred restores a restoring gradient outside
    [0,1] (it is identically 0 inside, so in-range BCE is undistorted).
    """
    p = project_contact_probability(pred).clamp(eps, 1.0 - eps)
    t = target.clamp(0.0, 1.0)
    bce = -(pos_weight * t * torch.log(p) + (1.0 - t) * torch.log1p(-p))
    # Out-of-range barrier: 0 for pred in [0,1], quadratic (with gradient) beyond
    # -- reintroduces the pull the clamp would otherwise kill.
    oob = torch.relu(-pred) ** 2 + torch.relu(pred - 1.0) ** 2
    loss = bce + oob
    mask_full = _expand_mask(mask, pred).to(dtype=pred.dtype)
    denom = mask_full.sum().clamp_min(1.0)
    return (loss * mask_full).sum() / denom


def _quantize_tensor(tensor: torch.Tensor, bits: int = 8) -> torch.Tensor:
    if bits == 8:
        scale = 127.0 / tensor.abs().amax().clamp_min(1e-8)
        return (tensor * scale).round() / scale
    if bits == 16:
        return tensor.half().float()
    return tensor


def _wrap_angle_diff(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    diff = torch.remainder(pred - target + math.pi, _TWO_PI) - math.pi
    return diff


def _delta69_to_clip_state(
    feat: torch.Tensor,
    yaw_offset: torch.Tensor | None = None,
    abs_root: bool = False,
) -> dict[str, torch.Tensor]:
    """Reconstruct clip-local root state from delta69 features.

    Primitive slices keep delta translations in the parent segment frame, so
    geometry/FK losses need the primitive's yaw offset inside that segment.

    With ``abs_root`` (73-dim features) root xy/yaw are READ from the absolute
    channels (segment-anchored, so primitive slices carry their in-segment
    offset -- consistent between pred and target) instead of integrated.
    """
    expect = 73 if abs_root else 69
    if feat.ndim != 3 or feat.shape[-1] != expect:
        raise ValueError(
            f"Expected delta tensor with shape (B, T, {expect}), got {tuple(feat.shape)}"
        )

    batch, seq_len, _ = feat.shape
    roll = torch.atan2(feat[..., 0], feat[..., 1] + 1.0)
    pitch = torch.atan2(feat[..., 2], feat[..., 3] + 1.0)

    yaw = torch.zeros(batch, seq_len, device=feat.device, dtype=feat.dtype)
    if yaw_offset is not None:
        yaw = yaw + yaw_offset.to(device=feat.device, dtype=feat.dtype).view(batch, 1)
    if abs_root:
        yaw = yaw + torch.atan2(feat[..., 72], feat[..., 71])
    elif seq_len > 1:
        yaw[:, 1:] = yaw[:, :1] + torch.cumsum(feat[:, :-1, _DELTA_YAW_DIM], dim=1)

    root_pos = torch.zeros(batch, seq_len, 3, device=feat.device, dtype=feat.dtype)
    root_pos[..., 2] = feat[..., _HEIGHT_DIM]
    if abs_root:
        # Segment-anchored absolute xy: read, don't integrate. Note the
        # yaw_offset handling differs from xy on purpose: yaw_offset is an
        # angle shift, while the xy channels already contain the in-segment
        # translation offset for primitive slices.
        root_pos[..., :2] = feat[..., 69:71]
    elif seq_len > 1:
        # Dataset heading-aligns delta_trans by the window-start yaw, so XY is
        # already in the clip-local fixed frame. Do not rotate again by yaw[t].
        delta_xy = feat[:, :-1, _DELTA_TRANS_SLICE][..., :2]
        root_pos[:, 1:, :2] = torch.cumsum(delta_xy, dim=1)

    root_rot_mat = euler_xyz_to_matrix(roll, pitch, yaw)
    dof_pos = feat[..., _DOF_SLICE]
    contact = project_contact_probability(feat[..., _CONTACT_SLICE])

    return {
        "root_pos": root_pos,
        "root_rot_mat": root_rot_mat,
        "root_rot_6d": matrix_to_rot6d(root_rot_mat),
        "yaw": yaw,
        "dof_pos": dof_pos,
        "contact": contact,
    }


class MaskedFlowMatchingLoss(nn.Module):
    """Unified delta69 loss stack used by Prior_Recon training."""

    def __init__(self, cfg: EEMaskedFlowConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.fk = G129DeltaForwardKinematics(fps=cfg.motion.fps)
        self.training_progress = 0.0
        self.abs_root = bool(getattr(cfg, "abs_root_channels", False))

        # anchor_pos is reserved but not implemented (needs a dataset regen with
        # per-frame pelvis world xy). Fail loudly instead of training silently
        # without the term the user asked for.
        if getattr(cfg.loss, "anchor_pos_weight", 0.0) > 0.0:
            raise NotImplementedError(
                "loss.anchor_pos_weight > 0 is not implemented: the position "
                "anchor needs pelvis world xy stored per frame (dataset regen). "
                "Use anchor_rot_weight for the orientation anchor; height is "
                "covered by use_ee_height_anchor."
            )
        if getattr(cfg.loss, "anchor_rot_weight", 0.0) > 0.0 and not getattr(
            cfg, "use_ee_anchor", False
        ):
            raise ValueError(
                "loss.anchor_rot_weight > 0 requires use_ee_anchor=True (the "
                "anchor condition channels the loss reads from)."
            )
        if getattr(cfg, "ee_cond_segment_anchor", False) and not self.abs_root:
            raise ValueError(
                "ee_cond_segment_anchor=True requires abs_root_channels=True: "
                "without the absolute root channels a k>0 primitive's FK is "
                "expressed in its own local frame, so a shared segment-frame-0 "
                "anchor has no consistent meaning."
            )
        if getattr(cfg.loss, "seam_accel_weight", 0.0) > 0.0 and not getattr(
            cfg, "segment_geometry_loss", False
        ):
            raise ValueError(
                "loss.seam_accel_weight > 0 requires segment_geometry_loss=True: "
                "the seam-accel term is defined on the stitched segment (it "
                "supervises the boundary-crossing triples no per-primitive "
                "window ever contains)."
            )
        if getattr(cfg.loss, "seam_vel_weight", 0.0) > 0.0 and not getattr(
            cfg, "segment_geometry_loss", False
        ):
            raise ValueError(
                "loss.seam_vel_weight > 0 requires segment_geometry_loss=True: "
                "the seam-vel term is defined on the stitched segment (it "
                "supervises the boundary-crossing velocity pair no per-primitive "
                "window ever contains)."
            )
        if getattr(cfg.loss, "boundary_foot_skate_weight", 0.0) > 0.0 and not getattr(
            cfg, "segment_geometry_loss", False
        ):
            raise ValueError(
                "loss.boundary_foot_skate_weight > 0 requires "
                "segment_geometry_loss=True."
            )

        w = torch.ones(cfg.n_total_dof)

        leg_w = getattr(cfg.loss, "leg_joint_weight", 1.0)
        if leg_w != 1.0:
            w[11:23] = leg_w
            w[40:52] = leg_w

        root_w = getattr(cfg.loss, "root_weight", 1.0)
        if root_w != 1.0:
            w[0:5] = root_w
            w[_DELTA_TRANS_SLICE] = root_w
            w[_HEIGHT_DIM] = root_w
            if self.abs_root:
                # xy_rel + yaw_rel cos/sin are root channels too.
                w[69:73] = root_w

        self.register_buffer("joint_weight", w)

    def set_training_progress(self, progress: float) -> None:
        self.training_progress = min(max(float(progress), 0.0), 1.0)

    def _linear_ramp(self, start: float, end: float) -> float:
        if end <= start:
            return 1.0
        return min(max((self.training_progress - start) / (end - start), 0.0), 1.0)

    def _ee_curriculum_scale(self) -> float:
        lc = self.cfg.loss
        start = float(getattr(lc, "ee_curriculum_start_ratio", 0.0))
        end = float(getattr(lc, "ee_curriculum_end_ratio", 0.0))
        return self._linear_ramp(start, end)

    def _skate_curriculum_scale(self) -> float:
        """Ramp-in for the FK anti-skate terms (foot_skate + foot_contact).

        Scheduled AFTER the EE curriculum completes: both loss families pull
        on the same root/leg channels through FK, and letting the skate terms
        run at full strength from step 0 lets them win the early capacity
        race before the EE losses even ramp in (observed as EE accuracy
        regressing when the skate suite was added).
        """
        lc = self.cfg.loss
        start = float(getattr(lc, "skate_curriculum_start_ratio", 0.0))
        end = float(getattr(lc, "skate_curriculum_end_ratio", 0.0))
        return self._linear_ramp(start, end)

    def _root_curriculum_mult(self) -> float:
        lc = self.cfg.loss
        start_mult = float(getattr(lc, "root_curriculum_start_mult", 1.0))
        end_ratio = float(getattr(lc, "root_curriculum_end_ratio", 0.0))
        if start_mult <= 1.0 or end_ratio <= 0.0:
            return 1.0
        progress = min(self.training_progress / end_ratio, 1.0)
        return start_mult + progress * (1.0 - start_mult)

    @staticmethod
    def _extract_s_ee_condition(
        s_ee: torch.Tensor,
        rebase: bool = True,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if s_ee.shape[-1] < 18:
            raise ValueError(f"Expected s_ee last dim >= 18, got {s_ee.shape[-1]}.")
        ee_base = s_ee[..., :18].reshape(*s_ee.shape[:2], 2, 9)
        pos = ee_base[..., :3]
        rot6d = ee_base[..., 3:9]
        if not rebase:
            # segment_anchored_v1 values are ALREADY relative to the segment's
            # first frame (pos: palm offset from the segment-frame-0 palm;
            # rot6d: R_0^{-1} R_t). Return them untouched so the caller can
            # compare a pred rebased to the same GT segment anchor -- the
            # non-zero offset a k>0 primitive slice carries is exactly the
            # absolute in-segment hand target, not noise to subtract away.
            return pos, rot6d
        # legacy: the target may carry a non-zero segment offset at the
        # primitive's first frame, but the pred side is FK pose relative to that frame.
        # Rebase the target to the primitive first frame so the loss enforces relative
        # shape regardless of whether s_ee is segment- or primitive-anchored.
        pos_rel = pos - pos[:, :1]
        rot_mat = _rot6d_to_matrix(rot6d)
        rot_mat0_inv = rot_mat[:, :1].transpose(-1, -2)
        rot_rel_mat = torch.matmul(rot_mat0_inv, rot_mat)
        rot6d_rel = matrix_to_rot6d(rot_rel_mat)
        return pos_rel, rot6d_rel

    @staticmethod
    def _relative_hand_fk(
        hand_pos: torch.Tensor,
        hand_rot_mat: torch.Tensor,
        anchor_pos: torch.Tensor | None = None,
        anchor_rot_mat: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Hand pose relative to an anchor: the sequence's own first frame
        (legacy, anchor=None) or a shared external anchor such as the GT
        segment-frame-0 hand pose ((B, 2, 3) / (B, 2, 3, 3))."""
        if anchor_pos is None:
            anchor_pos = hand_pos[:, 0]
            anchor_rot_mat = hand_rot_mat[:, 0]
        pos_rel = hand_pos - anchor_pos.unsqueeze(1)
        rot_rel = torch.matmul(
            anchor_rot_mat.unsqueeze(1).transpose(-1, -2), hand_rot_mat
        )
        return pos_rel, matrix_to_rot6d(rot_rel)

    @torch.no_grad()
    def gt_hand_anchor(
        self, gt_frame0: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """FK hand pose of the GT SEGMENT frame 0, in the segment heading frame.

        ``gt_frame0`` is the (B, 1, D) GT feature row of segment frame 0
        (primitive 0's first pinned frame, UNPERTURBED -- same convention as the
        Layer B target). Returns ((B, 2, 3) pos, (B, 2, 3, 3) rot) used as the
        shared ee_cond anchor: with abs_root every primitive's FK already lives
        in the segment heading frame, so rebasing pred hands to this single GT
        anchor puts them in exactly the coordinates the segment-anchored s_ee
        target carries. Loss-side only -- inference never consumes it.
        """
        n_body = int(getattr(self.cfg, "n_motion_dof", 69) or 69)
        state = _delta69_to_clip_state(
            gt_frame0[..., :n_body], yaw_offset=None, abs_root=self.abs_root
        )
        fk = self.fk(
            root_pos=state["root_pos"],
            root_rot_mat=state["root_rot_mat"],
            dof_pos=state["dof_pos"],
        )
        return fk["hand_translation"][:, 0], fk["hand_rotation_mat"][:, 0]

    def _extract_anchor_rot_mat(self, s_ee: torch.Tensor) -> torch.Tensor:
        """Absolute-anchor rotation matrices (B, 2, 3, 3) from the s_ee anchor block.

        The anchor is a static 12D (2 hands x rot6d) block broadcast across
        frames, so any frame carries it -- read frame 0. Layout offset is
        cfg.ee_anchor_offset (after the relative pose + optional height anchor,
        before velocity). Requires use_ee_anchor.
        """
        off = int(self.cfg.ee_anchor_offset)
        anchor6d = s_ee[:, 0, off : off + 12].reshape(-1, 2, 6)
        return _rot6d_to_matrix(anchor6d)

    def _weighted_masked_mse(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        weight = self.joint_weight.to(pred.device)
        root_mult = self._root_curriculum_mult()
        if root_mult != 1.0:
            weight = weight.clone()
            weight[:5] *= root_mult
            weight[_DELTA_TRANS_SLICE] *= root_mult
            weight[_HEIGHT_DIM] *= root_mult
        denom = mask.sum().clamp_min(1.0)
        return (((pred - target) ** 2) * mask * weight).sum() / denom

    def _base_losses(
        self,
        pred_v: torch.Tensor,
        pred_x1: torch.Tensor,
        target_x1: torch.Tensor,
        target_v: torch.Tensor,
        obs_mask: torch.Tensor,
    ) -> MaskedFlowLossOutput:
        lc = self.cfg.loss
        unknown_mask = 1.0 - obs_mask
        # Seam-inclusive: a difference is trainable when either endpoint is
        # generated (see _transition_mask). Fully-pinned pairs (both endpoints
        # observed, e.g. the EE-state columns) stay excluded so they don't
        # dilute the denominator.
        vel_mask = torch.maximum(unknown_mask[:, 1:], unknown_mask[:, :-1])

        l_flow = lc.flow_weight * self._weighted_masked_mse(pred_v, target_v, unknown_mask)
        l_recon = lc.recon_weight * self._weighted_masked_mse(pred_x1, target_x1, unknown_mask)

        pred_vel = pred_x1[:, 1:] - pred_x1[:, :-1]
        target_vel = target_x1[:, 1:] - target_x1[:, :-1]
        l_vel = lc.velocity_weight * self._weighted_masked_mse(pred_vel, target_vel, vel_mask)

        if lc.accel_weight > 0 and pred_x1.shape[1] > 2:
            pred_acc = pred_vel[:, 1:] - pred_vel[:, :-1]
            target_acc = target_vel[:, 1:] - target_vel[:, :-1]
            acc_mask = torch.maximum(
                unknown_mask[:, 2:],
                torch.maximum(unknown_mask[:, 1:-1], unknown_mask[:, :-2]),
            )
            l_acc = lc.accel_weight * self._weighted_masked_mse(pred_acc, target_acc, acc_mask)
        else:
            l_acc = pred_x1.new_zeros(())

        total = l_flow + l_recon + l_vel + l_acc
        return MaskedFlowLossOutput(
            total=total,
            flow=l_flow,
            recon=l_recon,
            velocity=l_vel,
            accel=l_acc,
        )

    def _contact_regularizer(
        self,
        pred_x1: torch.Tensor,
        target_x1: torch.Tensor,
        frame_mask: torch.Tensor,
    ) -> torch.Tensor:
        cw = getattr(self.cfg.loss, "contact_weight", 0.0)
        if cw <= 0.0:
            return pred_x1.new_zeros(())

        contact_l = target_x1[..., 5:6].clamp(0.0, 1.0) * frame_mask
        contact_r = target_x1[..., 6:7].clamp(0.0, 1.0) * frame_mask
        delta_left = pred_x1[..., _LEFT_LEG_DELTA_SLICE]
        delta_right = pred_x1[..., _RIGHT_LEG_DELTA_SLICE]

        left_loss = _masked_mse(delta_left, torch.zeros_like(delta_left), contact_l)
        right_loss = _masked_mse(delta_right, torch.zeros_like(delta_right), contact_r)
        return cw * (left_loss + right_loss)

    def _geometry_losses(
        self,
        pred_x1: torch.Tensor,
        target_x1: torch.Tensor,
        obs_mask: torch.Tensor,
        s_ee: torch.Tensor | None = None,
        yaw_offset: torch.Tensor | None = None,
        anchor_active: bool = True,
        ee_anchor: tuple[torch.Tensor, torch.Tensor] | None = None,
        include_boundary_foot_skate: bool = False,
    ) -> dict[str, torch.Tensor]:
        lc = self.cfg.loss
        zero = pred_x1.new_zeros(())
        losses = {name: zero for name in _GEOMETRY_KEYS}

        needs_geometry = any(
            getattr(lc, name, 0.0) > 0.0
            for name in (
                "body_trans_weight",
                "body_rot_weight",
                "ee_pos_weight",
                "ee_rot_weight",
                "ee_cond_pos_weight",
                "ee_cond_rot_weight",
                "anchor_rot_weight",
                "dof_pos_weight",
                "dof_vel_weight",
                "foot_contact_weight",
                "foot_skate_weight",
                "foot_sole_pos_weight",
                "foot_sole_skate_weight",
                "boundary_foot_skate_weight",
                "self_skate_weight",
                "foot_rot_weight",
                "contact_pred_weight",
                "root_consistency_weight",
                "root_vel_weight",
                "root_height_weight",
                "drift_yaw_weight",
                "drift_xy_weight",
                "smooth_weight",
                "quantize_rot_weight",
                "quantize_trans_weight",
            )
        )
        if not needs_geometry:
            return losses

        frame_mask = _frame_unknown_mask(obs_mask)
        trans_mask = _transition_mask(frame_mask)
        last_mask = frame_mask[:, -1, 0]
        root_mult = self._root_curriculum_mult()

        # EE-as-state layouts append pinned EE columns after the delta69 body
        # features; geometry decoding and smoothness only apply to the body.
        n_body = int(getattr(self.cfg, "n_motion_dof", 69) or 69)
        pred_body = pred_x1[..., :n_body]
        target_body = target_x1[..., :n_body]

        # abs-root: yaw_rel channels already carry the in-segment angle of
        # primitive slices; adding the trainer's yaw_offset again would double
        # count it, so drop the offset (xy channels likewise carry theirs).
        state_yaw_offset = None if self.abs_root else yaw_offset
        pred_state = _delta69_to_clip_state(
            pred_body, yaw_offset=state_yaw_offset, abs_root=self.abs_root
        )
        with torch.no_grad():
            target_state = _delta69_to_clip_state(
                target_body, yaw_offset=state_yaw_offset, abs_root=self.abs_root
            )

        if lc.smooth_weight > 0.0 and pred_x1.shape[1] > 1:
            pred_diff = pred_body[:, 1:] - pred_body[:, :-1]
            losses["smooth"] = lc.smooth_weight * _masked_l1(
                pred_diff,
                torch.zeros_like(pred_diff),
                trans_mask,
            )

        if getattr(lc, "contact_pred_weight", 0.0) > 0.0:
            # Dedicated contact-channel supervision (GT is binary {0,1}); the
            # shared flow/recon MSE alone leaves the channels too mushy for
            # contact-driven post-processing to threshold reliably.
            #
            # [C] contact_pred_bce: MSE's optimum under {0,1} labels is the
            # conditional mean (mushy ~0.5), so switch to BCE which pushes the
            # channel to the 0/1 extremes -> stable thresholding downstream.
            # pos_weight counteracts the stance/swing imbalance (contact frames
            # are the minority in loco clips). Kept in [0,1] probability space so
            # every consumer that reads the channel as a probability is unchanged.
            if getattr(lc, "contact_pred_bce", False):
                losses["contact_pred"] = lc.contact_pred_weight * _masked_bce(
                    pred_body[..., _CONTACT_SLICE],
                    target_body[..., _CONTACT_SLICE],
                    frame_mask,
                    pos_weight=float(getattr(lc, "contact_pred_pos_weight", 1.0)),
                )
            else:
                losses["contact_pred"] = lc.contact_pred_weight * _masked_mse(
                    pred_body[..., _CONTACT_SLICE],
                    target_body[..., _CONTACT_SLICE],
                    frame_mask,
                )

        fps = float(getattr(self.cfg.motion, "fps", 30.0))
        if (
            self.abs_root
            and getattr(lc, "root_consistency_weight", 0.0) > 0.0
            and pred_x1.shape[1] > 1
        ):
            # Dual-encoding self-consistency: the abs channels' per-frame
            # difference must equal the delta channels. Both sides are
            # predictions, so the gradient ties the two encodings together --
            # the abs read becomes as smooth as the integration path.
            xy_step = (pred_body[:, 1:, 69:71] - pred_body[:, :-1, 69:71]) * fps
            xy_delta = pred_body[:, :-1, _DELTA_TRANS_SLICE][..., :2] * fps
            l_cons = _masked_mse(xy_step, xy_delta, trans_mask)
            yaw_abs = torch.atan2(pred_body[..., 72], pred_body[..., 71])
            yaw_step = _wrap_angle_diff(yaw_abs[:, 1:], yaw_abs[:, :-1]) * fps
            yaw_delta = pred_body[:, :-1, _DELTA_YAW_DIM] * fps
            l_cons = l_cons + _masked_mse(
                yaw_step.unsqueeze(-1), yaw_delta.unsqueeze(-1), trans_mask
            )
            losses["root_consistency"] = (
                lc.root_consistency_weight * root_mult * l_cons
            )

        if getattr(lc, "root_vel_weight", 0.0) > 0.0 and pred_x1.shape[1] > 1:
            # Per-frame-decoded root state channels get their own velocity
            # supervision (m/s scale): the shared velocity loss spreads over
            # all dims, leaving these few channels free to jitter.
            root_dims = [0, 1, 2, 3, _HEIGHT_DIM]
            if self.abs_root:
                root_dims += [69, 70, 71, 72]
            pred_rv = (pred_body[:, 1:, root_dims] - pred_body[:, :-1, root_dims]) * fps
            target_rv = (
                target_body[:, 1:, root_dims] - target_body[:, :-1, root_dims]
            ) * fps
            losses["root_vel"] = lc.root_vel_weight * root_mult * _masked_mse(
                pred_rv, target_rv, trans_mask
            )

        if getattr(lc, "root_height_weight", 0.0) > 0.0:
            # Absolute per-frame height anchor (metre scale). root_vel pins
            # the height RATE only — a dc offset is invisible to it — and the
            # FK foot terms are invariant to root-up/legs-extend, so without
            # this term the absolute level can drift (floating base).
            losses["root_height"] = lc.root_height_weight * root_mult * _masked_mse(
                pred_body[..., _HEIGHT_DIM : _HEIGHT_DIM + 1],
                target_body[..., _HEIGHT_DIM : _HEIGHT_DIM + 1],
                frame_mask,
            )

        if lc.drift_yaw_weight > 0.0:
            yaw_diff = _wrap_angle_diff(pred_state["yaw"][:, -1], target_state["yaw"][:, -1])
            losses["drift_yaw"] = lc.drift_yaw_weight * root_mult * _masked_mse(
                yaw_diff.unsqueeze(-1),
                torch.zeros_like(yaw_diff).unsqueeze(-1),
                last_mask.unsqueeze(-1),
            )

        if lc.drift_xy_weight > 0.0:
            losses["drift_xy"] = lc.drift_xy_weight * root_mult * _masked_mse(
                pred_state["root_pos"][:, -1, :2],
                target_state["root_pos"][:, -1, :2],
                last_mask.unsqueeze(-1),
            )

        if lc.quantize_rot_weight > 0.0:
            quant_pred_rot = _quantize_tensor(pred_state["root_rot_6d"][:, -1], bits=8)
            quant_gt_rot = _quantize_tensor(target_state["root_rot_6d"][:, -1], bits=8)
            losses["quantize_rot"] = lc.quantize_rot_weight * _masked_mse(
                quant_pred_rot,
                quant_gt_rot,
                last_mask.unsqueeze(-1),
            )

        if lc.quantize_trans_weight > 0.0:
            quant_pred_xy = _quantize_tensor(pred_state["root_pos"][..., :2], bits=8)
            quant_gt_xy = _quantize_tensor(target_state["root_pos"][..., :2], bits=8)
            losses["quantize_trans"] = lc.quantize_trans_weight * root_mult * _masked_mse(
                quant_pred_xy,
                quant_gt_xy,
                frame_mask,
            )

        if lc.dof_pos_weight > 0.0:
            losses["dof_pos"] = lc.dof_pos_weight * _masked_mse(
                pred_state["dof_pos"],
                target_state["dof_pos"],
                frame_mask,
            )

        if lc.dof_vel_weight > 0.0 and pred_x1.shape[1] > 1:
            fps = float(getattr(self.cfg.motion, "fps", 30.0))
            pred_dof_vel = (pred_state["dof_pos"][:, 1:] - pred_state["dof_pos"][:, :-1]) * fps
            target_dof_vel = (target_state["dof_pos"][:, 1:] - target_state["dof_pos"][:, :-1]) * fps
            losses["dof_vel"] = lc.dof_vel_weight * _masked_mse(
                pred_dof_vel,
                target_dof_vel,
                trans_mask,
            )

        needs_fk = any(
            getattr(lc, name, 0.0) > 0.0
            for name in (
                "body_trans_weight",
                "body_rot_weight",
                "ee_pos_weight",
                "ee_rot_weight",
                "ee_cond_pos_weight",
                "ee_cond_rot_weight",
                "anchor_rot_weight",
                "foot_contact_weight",
                "foot_skate_weight",
                "foot_sole_pos_weight",
                "foot_sole_skate_weight",
                "self_skate_weight",
                "foot_rot_weight",
            )
        )
        if not needs_fk:
            return losses

        pred_fk = self.fk(
            root_pos=pred_state["root_pos"],
            root_rot_mat=pred_state["root_rot_mat"],
            dof_pos=pred_state["dof_pos"],
        )
        with torch.no_grad():
            target_fk = self.fk(
                root_pos=target_state["root_pos"],
                root_rot_mat=target_state["root_rot_mat"],
                dof_pos=target_state["dof_pos"],
            )

        if lc.body_trans_weight > 0.0:
            losses["body_trans"] = lc.body_trans_weight * root_mult * _masked_mse(
                pred_fk["global_translation"],
                target_fk["global_translation"],
                frame_mask,
            )

        if lc.body_rot_weight > 0.0:
            losses["body_rot"] = lc.body_rot_weight * _masked_mse(
                pred_fk["global_rotation_6d"],
                target_fk["global_rotation_6d"],
                frame_mask,
            )

        ee_scale = self._ee_curriculum_scale()
        if getattr(lc, "ee_pos_weight", 0.0) > 0.0 and ee_scale > 0.0:
            losses["ee_pos"] = lc.ee_pos_weight * ee_scale * root_mult * _masked_mse(
                pred_fk["hand_translation"],
                target_fk["hand_translation"],
                frame_mask,
            )

        if getattr(lc, "ee_rot_weight", 0.0) > 0.0 and ee_scale > 0.0:
            losses["ee_rot"] = lc.ee_rot_weight * ee_scale * _masked_mse(
                pred_fk["hand_rotation_6d"],
                target_fk["hand_rotation_6d"],
                frame_mask,
            )

        needs_ee_condition = (
            s_ee is not None
            and getattr(self.cfg, "use_ee_pos", False)
            and ee_scale > 0.0
            and (
                getattr(lc, "ee_cond_pos_weight", 0.0) > 0.0
                or getattr(lc, "ee_cond_rot_weight", 0.0) > 0.0
            )
        )
        if needs_ee_condition:
            # ee_anchor set (ee_cond_segment_anchor): rebase pred to the GT
            # SEGMENT frame-0 hand pose and use the s_ee values as-is -- both
            # sides then share one stable segment anchor, so the strong ee_cond
            # weights supervise the ABSOLUTE in-segment hand pose instead of
            # shape about a per-primitive anchor that drifts with the rollout
            # history. ee_anchor None keeps the legacy relative comparison.
            cond_pos_rel, cond_rot6d_rel = self._extract_s_ee_condition(
                s_ee, rebase=ee_anchor is None
            )
            pred_pos_rel, pred_rot6d_rel = self._relative_hand_fk(
                pred_fk["hand_translation"],
                pred_fk["hand_rotation_mat"],
                anchor_pos=None if ee_anchor is None else ee_anchor[0],
                anchor_rot_mat=None if ee_anchor is None else ee_anchor[1],
            )
            if getattr(lc, "ee_cond_pos_weight", 0.0) > 0.0:
                losses["ee_cond_pos"] = lc.ee_cond_pos_weight * ee_scale * root_mult * _masked_mse(
                    pred_pos_rel,
                    cond_pos_rel,
                    frame_mask,
                )
            if getattr(lc, "ee_cond_rot_weight", 0.0) > 0.0:
                losses["ee_cond_rot"] = lc.ee_cond_rot_weight * ee_scale * _masked_mse(
                    pred_rot6d_rel,
                    cond_rot6d_rel,
                    frame_mask,
                )

        # Layer C: absolute-hand-orientation anchor loss. The anchor A is the
        # segment-frame-0 absolute hand orientation (heading frame); composing it
        # with the relative condition gives the ABSOLUTE hand orientation target
        # at every frame:  A @ (R_0^{-1} R_t) = R_z(-yaw0) R_t. Supervising the
        # predicted absolute hand orientation against that forces the model to
        # actually CONSUME the anchor channel to place the hands absolutely,
        # rather than only matching relative shape (ee_cond_rot) about its own
        # unobserved first frame -- the exact freedom that let a standing / real-
        # state start carry a constant rotation offset. Only on the primitive
        # whose FK is in the heading frame (yaw_offset == 0, i.e. primitive 0,
        # where the seed mismatch lives); supervised on GENERATED frames only
        # (the pinned history prefix is a given, not a prediction).
        needs_anchor = (
            s_ee is not None
            and getattr(self.cfg, "use_ee_anchor", False)
            and anchor_active
            and ee_scale > 0.0
            and getattr(lc, "anchor_rot_weight", 0.0) > 0.0
        )
        if needs_anchor:
            anchor_mat = self._extract_anchor_rot_mat(s_ee)  # (B, 2, 3, 3)
            _, cond_rot6d_rel = self._extract_s_ee_condition(s_ee)  # (B, T, 2, 6)
            cond_rel_mat = _rot6d_to_matrix(cond_rot6d_rel)  # (B, T, 2, 3, 3)
            # Absolute target per frame = A @ relative(t).
            target_abs = torch.matmul(anchor_mat.unsqueeze(1), cond_rel_mat)
            geo = geodesic_cos_loss(
                pred_fk["hand_rotation_mat"], target_abs
            )  # (B, T, 2)
            anchor_mask = frame_mask.expand(-1, -1, geo.shape[-1])  # generated frames
            denom = anchor_mask.sum().clamp_min(1.0)
            losses["anchor_rot"] = (
                lc.anchor_rot_weight * ee_scale * (geo * anchor_mask).sum() / denom
            )

        skate_scale = self._skate_curriculum_scale()
        if lc.foot_contact_weight > 0.0 and skate_scale > 0.0:
            foot_contact_mask = target_state["contact"].unsqueeze(-1) * frame_mask.unsqueeze(-1)
            losses["foot_contact"] = (
                lc.foot_contact_weight * skate_scale * root_mult * _masked_mse(
                    pred_fk["foot_translation"],
                    target_fk["foot_translation"],
                    foot_contact_mask,
                )
            )

        if (
            getattr(lc, "foot_skate_weight", 0.0) > 0.0
            and skate_scale > 0.0
            and pred_x1.shape[1] > 1
        ):
            # Anti-skate: during stance (both endpoints in GT contact) the
            # predicted foot's world velocity must match GT (~0). World foot
            # velocity = root velocity + leg articulation, so this is the term
            # that forces every centimetre of root motion to be produced by
            # the stance leg instead of dragging the foot.
            #
            # NOTE the m/s scale: the *fps makes the squared error ~fps^2
            # (~900x) larger than the metre-scale terms, so the WEIGHT must be
            # small (delta73_skate_small: 0.05, not 0.5) or this term hijacks
            # the gradient budget from the EE losses. No root_mult here on
            # purpose: this is a leg<->root coupling term, not a root-accuracy
            # term, and the early-training root boost only amplified the
            # hijack.
            pred_fv = (
                pred_fk["foot_translation"][:, 1:] - pred_fk["foot_translation"][:, :-1]
            ) * fps
            target_fv = (
                target_fk["foot_translation"][:, 1:]
                - target_fk["foot_translation"][:, :-1]
            ) * fps
            stance = target_state["contact"][:, 1:] * target_state["contact"][:, :-1]
            skate_mask = stance.unsqueeze(-1) * trans_mask.unsqueeze(-1)
            losses["foot_skate"] = lc.foot_skate_weight * skate_scale * _masked_mse(
                pred_fv,
                target_fv,
                skate_mask,
            )

        # ── Layer A: sole-geometry foot terms ───────────────────────────────
        # contact (B, T, 2) is per-foot; broadcast over the N sole points and 3
        # coords. Same skate curriculum gate as foot_skate/foot_contact.
        contact_bt2 = target_state["contact"]  # (B, T, 2)

        if lc.foot_sole_pos_weight > 0.0 and skate_scale > 0.0:
            # Contact-masked absolute position of every heel/toe sole point. With
            # >=3 non-collinear points this constrains the full foot pose during
            # stance -- unlike the single ankle-origin foot_contact term, which is
            # near-invariant to foot rotation (pitch/roll axes pass through it).
            sole_mask = contact_bt2[..., None, None] * frame_mask[..., None, None]
            losses["foot_sole_pos"] = (
                lc.foot_sole_pos_weight * skate_scale * root_mult * _masked_mse(
                    pred_fk["foot_sole_translation"],
                    target_fk["foot_sole_translation"],
                    sole_mask,
                )
            )

        if (
            lc.foot_sole_skate_weight > 0.0
            and skate_scale > 0.0
            and pred_x1.shape[1] > 1
        ):
            # Per-sole-point stance anti-skate: every heel/toe corner must hold
            # still in world during stance. Penalising the corners (not just the
            # ankle origin) also kills the "pivot in place" motion the origin term
            # is blind to. Same fps^2 m/s scale caveat as foot_skate -> small
            # weight; no root_mult (leg<->root coupling, not root accuracy).
            pred_sv = (
                pred_fk["foot_sole_translation"][:, 1:]
                - pred_fk["foot_sole_translation"][:, :-1]
            ) * fps
            stance = contact_bt2[:, 1:] * contact_bt2[:, :-1]  # (B, T-1, 2)
            sole_skate_mask = (
                stance[..., None, None] * trans_mask[..., None, None]
            )
            losses["foot_sole_skate"] = (
                lc.foot_sole_skate_weight * skate_scale * _masked_mse(
                    pred_sv,
                    torch.zeros_like(pred_sv),
                    sole_skate_mask,
                )
            )

        if (
            include_boundary_foot_skate
            and getattr(lc, "boundary_foot_skate_weight", 0.0) > 0.0
        ):
            losses["boundary_foot_skate"] = (
                lc.boundary_foot_skate_weight
                * skate_scale
                * boundary_sole_skate_loss(
                    pred_fk["foot_sole_translation"],
                    contact_bt2,
                    trans_mask,
                    fps=fps,
                    history_len=self.cfg.primitive.history_len,
                    future_len=self.cfg.primitive.future_len,
                    num_primitives=self.cfg.primitive.num_primitives,
                    topk_ratio=float(
                        getattr(lc, "boundary_foot_skate_topk_ratio", 0.25)
                    ),
                )
            )

        if (
            getattr(lc, "self_skate_weight", 0.0) > 0.0
            and skate_scale > 0.0
            and pred_x1.shape[1] > 1
        ):
            # [B] Predicted-contact self-gated anti-skate. Every OTHER anti-skate
            # term masks stance with the GT contact (target_state["contact"]), so
            # the model is told WHERE to hold the foot still. But at inference
            # there is no GT: recon/bridge foot-lock/body-qp/pyroki read the
            # model's OWN predicted contact channel to decide where to lock. The
            # model was never trained so that "where I PREDICT contact, my own
            # foot holds still" -- the predicted channel is a passive output that
            # never gates the foot. This term closes that loop: mask the predicted
            # sole velocity->0 with the model's own (soft, detached) predicted
            # contact, so a frame the model calls contact is forced to be
            # skate-free. Detached mask => the gradient flows to the FOOT MOTION
            # (root+legs), not into trivially lowering the contact probability;
            # channel accuracy stays owned by contact_pred. Uses sole points
            # (not the ankle origin) for the same pivot-in-place reason as
            # foot_sole_skate; same fps m/s scale -> keep the weight small.
            pred_sv_self = (
                pred_fk["foot_sole_translation"][:, 1:]
                - pred_fk["foot_sole_translation"][:, :-1]
            ) * fps
            pred_contact = project_contact_probability(
                pred_body[..., _CONTACT_SLICE].detach()
            )
            pred_stance = pred_contact[:, 1:] * pred_contact[:, :-1]  # (B, T-1, 2)
            self_skate_mask = (
                pred_stance[..., None, None] * trans_mask[..., None, None]
            )
            losses["self_skate"] = (
                lc.self_skate_weight * skate_scale * _masked_mse(
                    pred_sv_self,
                    torch.zeros_like(pred_sv_self),
                    self_skate_mask,
                )
            )

        if lc.foot_rot_weight > 0.0 and skate_scale > 0.0:
            # Contact-masked geodesic foot-orientation loss. Direct SO(3) penalty
            # on the ankle_roll world rotation; removes the residual rotational
            # freedom the position terms leave when the stance leg is near a
            # kinematic singularity. Uses the 1-cos(theta) surrogate (0 grad at
            # perfect alignment, unlike arccos).
            geo = geodesic_cos_loss(
                pred_fk["foot_rotation_mat"], target_fk["foot_rotation_mat"]
            )  # (B, T, 2), in [0, 2]
            foot_rot_mask = contact_bt2 * frame_mask
            denom = foot_rot_mask.sum().clamp_min(1.0)
            losses["foot_rot"] = (
                lc.foot_rot_weight * skate_scale * (geo * foot_rot_mask).sum() / denom
            )

        return losses

    def segment_geometry(
        self,
        pred_x1: torch.Tensor,
        target_x1: torch.Tensor,
        obs_mask: torch.Tensor,
        s_ee: torch.Tensor | None = None,
        ee_anchor: tuple[torch.Tensor, torch.Tensor] | None = None,
    ) -> dict[str, torch.Tensor]:
        """Geometry losses over the full stitched segment (all primitives).

        yaw_offset is None: with abs_root the stitched features carry their
        in-segment absolute xy/yaw; without it the cumsum integration over the
        stitched frames reconstructs the segment-local frame directly.
        anchor_active=True: the whole stitched sequence lives in the heading
        frame, so the Layer C anchor target is valid on every generated frame
        (the per-primitive prim-0 restriction guarded the yaw_offset path).
        """
        geom = self._geometry_losses(
            pred_x1,
            target_x1,
            obs_mask,
            s_ee=s_ee,
            yaw_offset=None,
            anchor_active=True,
            ee_anchor=ee_anchor,
            include_boundary_foot_skate=True,
        )
        geom["seam_accel"] = self._seam_accel_loss(pred_x1, target_x1, obs_mask)
        geom["seam_vel"] = self._seam_vel_loss(pred_x1, target_x1, obs_mask)
        return geom

    def _seam_vel_loss(
        self,
        pred_x1: torch.Tensor,
        target_x1: torch.Tensor,
        obs_mask: torch.Tensor,
    ) -> torch.Tensor:
        """First-difference (velocity) supervision on the boundary-crossing PAIR
        of the stitched segment -- the first-order companion to
        ``_seam_accel_loss``.

        The visible seam hitch is a first-order velocity STEP: measured on the
        recon, the frame-to-frame joint velocity jumps ~1.6x the interior right
        after each primitive's history prefix, because each primitive denoises
        its 8 generated frames as a fresh start and nothing forces the velocity
        it inherits across the boundary to match the velocity the previous
        primitive was leaving with. The per-primitive velocity loss covers the
        pinned-history seam of the NEXT window, but its history endpoint is a
        DETACHED constant, so its gradient can only move the successor's first
        frame -- it cannot pull the predecessor's last frame toward it. On the
        stitched segment both endpoints of the seam pair (b-1, b) are the two
        primitives' GENERATED (non-detached) frames, so the gradient pulls the
        two sides together. Supervises the pair at every boundary
        b = history_len + k*future_len; mask-restricted so mid-window pairs keep
        their per-primitive-only weighting (no double counting).
        """
        w = getattr(self.cfg.loss, "seam_vel_weight", 0.0)
        T = pred_x1.shape[1]
        if w <= 0.0 or T < 2:
            return pred_x1.new_zeros(())
        prim_cfg = self.cfg.primitive
        # Pair (b-1, b) lives at velocity index b-1 (vel[i] = x[i+1] - x[i]).
        seam_i = []
        for k in range(1, prim_cfg.num_primitives):
            b = prim_cfg.history_len + k * prim_cfg.future_len
            i = b - 1
            if 0 <= i <= T - 2:
                seam_i.append(i)
        if not seam_i:
            return pred_x1.new_zeros(())
        pred_vel = pred_x1[:, 1:] - pred_x1[:, :-1]
        target_vel = target_x1[:, 1:] - target_x1[:, :-1]
        unknown = 1.0 - obs_mask
        vel_mask = torch.maximum(unknown[:, 1:], unknown[:, :-1])
        seam_mask = torch.zeros_like(vel_mask)
        seam_mask[:, seam_i] = 1.0
        return w * self._weighted_masked_mse(pred_vel, target_vel, vel_mask * seam_mask)

    def _seam_accel_loss(
        self,
        pred_x1: torch.Tensor,
        target_x1: torch.Tensor,
        obs_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Second-difference supervision on the boundary-crossing triples of the
        stitched segment.

        The per-primitive accel loss covers every triple a 10-frame window
        contains -- including the pinned-history seam of the NEXT window -- but
        a triple spanning two primitives' GENERATED frames exists in no window,
        so cross-seam second-order continuity was structurally unsupervised
        (first-order crosses the seam via smooth/root_vel/dof_vel at segment
        level). This term closes that gap on exactly those triples: for each
        boundary b = history_len + k*future_len the second differences starting
        at t in {b-2, b-1}. Mask-restricted, so mid-window triples keep their
        per-primitive-only weighting (no double counting).
        """
        w = getattr(self.cfg.loss, "seam_accel_weight", 0.0)
        T = pred_x1.shape[1]
        if w <= 0.0 or T < 3:
            return pred_x1.new_zeros(())
        prim_cfg = self.cfg.primitive
        seam_t = []
        for k in range(1, prim_cfg.num_primitives):
            b = prim_cfg.history_len + k * prim_cfg.future_len
            seam_t += [t for t in (b - 2, b - 1) if 0 <= t <= T - 3]
        if not seam_t:
            return pred_x1.new_zeros(())
        pred_acc = pred_x1[:, 2:] - 2.0 * pred_x1[:, 1:-1] + pred_x1[:, :-2]
        target_acc = target_x1[:, 2:] - 2.0 * target_x1[:, 1:-1] + target_x1[:, :-2]
        unknown = 1.0 - obs_mask
        acc_mask = torch.maximum(
            unknown[:, 2:], torch.maximum(unknown[:, 1:-1], unknown[:, :-2])
        )
        seam_mask = torch.zeros_like(acc_mask)
        seam_mask[:, seam_t] = 1.0
        return w * self._weighted_masked_mse(pred_acc, target_acc, acc_mask * seam_mask)

    def forward(
        self,
        pred_v: torch.Tensor,
        pred_x1: torch.Tensor,
        target_x1: torch.Tensor,
        target_v: torch.Tensor,
        obs_mask: torch.Tensor,
        s_ee: torch.Tensor | None = None,
        yaw_offset: torch.Tensor | None = None,
        anchor_active: bool = True,
        ee_anchor: tuple[torch.Tensor, torch.Tensor] | None = None,
        include_geometry: bool = True,
    ) -> MaskedFlowLossOutput:
        out = self._base_losses(pred_v, pred_x1, target_x1, target_v, obs_mask)
        frame_mask = _frame_unknown_mask(obs_mask)
        l_contact = self._contact_regularizer(pred_x1, target_x1, frame_mask)
        if include_geometry:
            geom = self._geometry_losses(
                pred_x1,
                target_x1,
                obs_mask,
                s_ee=s_ee,
                yaw_offset=yaw_offset,
                anchor_active=anchor_active,
                ee_anchor=ee_anchor,
            )
        else:
            # segment_geometry_loss mode: the geometry suite is computed once on
            # the stitched segment (see segment_geometry + merge_segment_geometry),
            # not per primitive -- moved, not duplicated.
            zero = pred_x1.new_zeros(())
            geom = {name: zero for name in _GEOMETRY_KEYS}
        extra_total = l_contact + sum(geom.values())

        return MaskedFlowLossOutput(
            total=out.total + extra_total,
            flow=out.flow,
            recon=out.recon,
            velocity=out.velocity,
            accel=out.accel,
            contact=l_contact,
            body_trans=geom["body_trans"],
            body_rot=geom["body_rot"],
            ee_pos=geom["ee_pos"],
            ee_rot=geom["ee_rot"],
            ee_cond_pos=geom["ee_cond_pos"],
            ee_cond_rot=geom["ee_cond_rot"],
            anchor_pos=geom["anchor_pos"],
            anchor_rot=geom["anchor_rot"],
            dof_pos=geom["dof_pos"],
            dof_vel=geom["dof_vel"],
            foot_contact=geom["foot_contact"],
            foot_skate=geom["foot_skate"],
            foot_sole_pos=geom["foot_sole_pos"],
            foot_sole_skate=geom["foot_sole_skate"],
            boundary_foot_skate=geom["boundary_foot_skate"],
            self_skate=geom["self_skate"],
            foot_rot=geom["foot_rot"],
            contact_pred=geom["contact_pred"],
            root_consistency=geom["root_consistency"],
            root_vel=geom["root_vel"],
            root_height=geom["root_height"],
            drift_yaw=geom["drift_yaw"],
            drift_xy=geom["drift_xy"],
            smooth=geom["smooth"],
            quantize_rot=geom["quantize_rot"],
            quantize_trans=geom["quantize_trans"],
        )
