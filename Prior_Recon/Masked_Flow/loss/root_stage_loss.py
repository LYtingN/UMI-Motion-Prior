"""Loss for the stage-1 root model of the two-stage cascade.

Operates on the ROOT-CHANNEL SUBSET of the delta69/73 features, in its own
compact layout (see model/two_stage_cascade.root_channel_indices):

    [0:2]   roll  (sin, cos-1)
    [2:4]   pitch (sin, cos-1)
    [4]     delta_yaw
    [5:7]   foot contact (L, R)
    [7:10]  delta_trans
    [10]    height
    [11:13] xy_rel   (abs_root_channels only)
    [13:15] yaw_rel (cos, sin) (abs_root_channels only)

Every channel here is a root/gait channel, so unlike the single-stage loss
there is no root-vs-body weight balancing: flow/recon/velocity supervise the
whole vector, and the few dedicated terms (contact_pred, root_consistency,
root_vel, root_height) mirror their single-stage counterparts on the local
indices. No FK terms -- world-frame geometry is the body stage's job.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn as nn

from Prior_Recon.Masked_Flow.loss.masked_flow_loss import (
    _frame_unknown_mask,
    _masked_bce,
    _masked_mse,
    _transition_mask,
)

_TWO_PI = 2.0 * math.pi

# Local (root-subset) channel layout.
_R_TILT = slice(0, 4)
_R_DYAW = 4
_R_CONTACT = slice(5, 7)
_R_DTRANS = slice(7, 10)
_R_HEIGHT = 10
_R_ABS_XY = slice(11, 13)
_R_ABS_YAW = slice(13, 15)

ROOT_STATE_DIMS = 11
ABS_ROOT_DIMS = 4


@dataclass
class RootStageLossOutput:
    total: torch.Tensor
    flow: torch.Tensor
    recon: torch.Tensor
    velocity: torch.Tensor
    contact_pred: torch.Tensor = None  # type: ignore[assignment]
    root_consistency: torch.Tensor = None  # type: ignore[assignment]
    root_vel: torch.Tensor = None  # type: ignore[assignment]
    root_height: torch.Tensor = None  # type: ignore[assignment]
    drift_yaw: torch.Tensor = None  # type: ignore[assignment]
    drift_xy: torch.Tensor = None  # type: ignore[assignment]

    _OPTIONAL_FIELDS = (
        "contact_pred",
        "root_consistency",
        "root_vel",
        "root_height",
        "drift_yaw",
        "drift_xy",
    )

    def __post_init__(self):
        for name in self._OPTIONAL_FIELDS:
            if getattr(self, name) is None:
                setattr(self, name, self.total.new_zeros(()))

    def as_dict(self) -> dict[str, float]:
        return {
            "root/total": self.total.item(),
            "root/flow": self.flow.item(),
            "root/recon": self.recon.item(),
            "root/velocity": self.velocity.item(),
            "root/contact_pred": self.contact_pred.item(),
            "root/root_consistency": self.root_consistency.item(),
            "root/root_vel": self.root_vel.item(),
            "root/root_height": self.root_height.item(),
            "root/drift_yaw": self.drift_yaw.item(),
            "root/drift_xy": self.drift_xy.item(),
        }


def mean_root_loss_outputs(losses: list[RootStageLossOutput]) -> RootStageLossOutput:
    if not losses:
        raise ValueError("Cannot average an empty list of RootStageLossOutput.")
    return RootStageLossOutput(
        total=torch.stack([loss.total for loss in losses]).mean(),
        flow=torch.stack([loss.flow for loss in losses]).mean(),
        recon=torch.stack([loss.recon for loss in losses]).mean(),
        velocity=torch.stack([loss.velocity for loss in losses]).mean(),
        contact_pred=torch.stack([loss.contact_pred for loss in losses]).mean(),
        root_consistency=torch.stack([loss.root_consistency for loss in losses]).mean(),
        root_vel=torch.stack([loss.root_vel for loss in losses]).mean(),
        root_height=torch.stack([loss.root_height for loss in losses]).mean(),
        drift_yaw=torch.stack([loss.drift_yaw for loss in losses]).mean(),
        drift_xy=torch.stack([loss.drift_xy for loss in losses]).mean(),
    )


def _wrap_angle(diff: torch.Tensor) -> torch.Tensor:
    return torch.remainder(diff + math.pi, _TWO_PI) - math.pi


class RootStageLoss(nn.Module):
    """Flow-matching + root-state loss on the root-channel subset."""

    def __init__(self, cfg) -> None:
        super().__init__()
        self.w = cfg.root_stage.loss
        self.abs_root = bool(getattr(cfg, "abs_root_channels", False))
        self.fps = float(getattr(cfg.motion, "fps", 30.0))
        self.n_dims = ROOT_STATE_DIMS + (ABS_ROOT_DIMS if self.abs_root else 0)

    def forward(
        self,
        pred_v: torch.Tensor,
        pred_x1: torch.Tensor,
        target_x1: torch.Tensor,
        target_v: torch.Tensor,
        obs_mask: torch.Tensor,
    ) -> RootStageLossOutput:
        if pred_x1.shape[-1] != self.n_dims:
            raise ValueError(
                f"RootStageLoss expects {self.n_dims} root channels, got {pred_x1.shape[-1]}."
            )
        w = self.w
        unknown_mask = 1.0 - obs_mask
        frame_mask = _frame_unknown_mask(obs_mask)
        trans_mask = _transition_mask(frame_mask)

        l_flow = w.flow_weight * _masked_mse(pred_v, target_v, unknown_mask)
        l_recon = w.recon_weight * _masked_mse(pred_x1, target_x1, unknown_mask)

        if pred_x1.shape[1] > 1:
            vel_mask = torch.maximum(unknown_mask[:, 1:], unknown_mask[:, :-1])
            pred_vel = pred_x1[:, 1:] - pred_x1[:, :-1]
            target_vel = target_x1[:, 1:] - target_x1[:, :-1]
            l_vel = w.velocity_weight * _masked_mse(pred_vel, target_vel, vel_mask)
        else:
            l_vel = pred_x1.new_zeros(())

        zero = pred_x1.new_zeros(())
        l_contact = zero
        if w.contact_pred_weight > 0.0:
            # [C] BCE contact supervision (see masked_flow_loss._masked_bce):
            # push the channel to 0/1 instead of MSE's mushy conditional mean, so
            # downstream thresholding is stable. Mirror the single-stage switch.
            if getattr(w, "contact_pred_bce", False):
                l_contact = w.contact_pred_weight * _masked_bce(
                    pred_x1[..., _R_CONTACT],
                    target_x1[..., _R_CONTACT],
                    frame_mask,
                    pos_weight=float(getattr(w, "contact_pred_pos_weight", 1.0)),
                )
            else:
                l_contact = w.contact_pred_weight * _masked_mse(
                    pred_x1[..., _R_CONTACT], target_x1[..., _R_CONTACT], frame_mask
                )

        l_cons = zero
        if self.abs_root and w.root_consistency_weight > 0.0 and pred_x1.shape[1] > 1:
            # Dual-encoding self-consistency, same construction as the
            # single-stage term but on local indices: the abs channels'
            # per-frame difference must equal the delta channels. fps-scaled
            # (m/s, rad/s) -> squared magnitude ~x900, weight must stay small.
            xy_step = (pred_x1[:, 1:, _R_ABS_XY] - pred_x1[:, :-1, _R_ABS_XY]) * self.fps
            xy_delta = pred_x1[:, :-1, _R_DTRANS][..., :2] * self.fps
            cons = _masked_mse(xy_step, xy_delta, trans_mask)
            yaw_abs = torch.atan2(pred_x1[..., 14], pred_x1[..., 13])
            yaw_step = _wrap_angle(yaw_abs[:, 1:] - yaw_abs[:, :-1]) * self.fps
            yaw_delta = pred_x1[:, :-1, _R_DYAW] * self.fps
            cons = cons + _masked_mse(
                yaw_step.unsqueeze(-1), yaw_delta.unsqueeze(-1), trans_mask
            )
            l_cons = w.root_consistency_weight * cons

        l_root_vel = zero
        if w.root_vel_weight > 0.0 and pred_x1.shape[1] > 1:
            root_dims = [0, 1, 2, 3, _R_HEIGHT]
            if self.abs_root:
                root_dims += [11, 12, 13, 14]
            pred_rv = (pred_x1[:, 1:, root_dims] - pred_x1[:, :-1, root_dims]) * self.fps
            target_rv = (
                target_x1[:, 1:, root_dims] - target_x1[:, :-1, root_dims]
            ) * self.fps
            l_root_vel = w.root_vel_weight * _masked_mse(pred_rv, target_rv, trans_mask)

        l_height = zero
        if w.root_height_weight > 0.0:
            l_height = w.root_height_weight * _masked_mse(
                pred_x1[..., _R_HEIGHT : _R_HEIGHT + 1],
                target_x1[..., _R_HEIGHT : _R_HEIGHT + 1],
                frame_mask,
            )

        # Full-horizon drift anchors on the last frame's abs channels (the
        # single-stage drift terms' exact counterparts on local indices;
        # abs_root-gated like root_consistency).
        l_dyaw = zero
        l_dxy = zero
        if self.abs_root:
            last_mask = frame_mask[:, -1, 0]
            if w.drift_yaw_weight > 0.0:
                yaw_pred = torch.atan2(pred_x1[..., 14], pred_x1[..., 13])
                yaw_tgt = torch.atan2(target_x1[..., 14], target_x1[..., 13])
                diff = _wrap_angle(yaw_pred[:, -1] - yaw_tgt[:, -1])
                l_dyaw = w.drift_yaw_weight * _masked_mse(
                    diff.unsqueeze(-1),
                    torch.zeros_like(diff).unsqueeze(-1),
                    last_mask.unsqueeze(-1),
                )
            if w.drift_xy_weight > 0.0:
                l_dxy = w.drift_xy_weight * _masked_mse(
                    pred_x1[:, -1, _R_ABS_XY],
                    target_x1[:, -1, _R_ABS_XY],
                    last_mask.unsqueeze(-1),
                )

        total = (
            l_flow + l_recon + l_vel + l_contact + l_cons + l_root_vel + l_height
            + l_dyaw + l_dxy
        )
        return RootStageLossOutput(
            total=total,
            flow=l_flow,
            recon=l_recon,
            velocity=l_vel,
            contact_pred=l_contact,
            root_consistency=l_cons,
            root_vel=l_root_vel,
            root_height=l_height,
            drift_yaw=l_dyaw,
            drift_xy=l_dxy,
        )
