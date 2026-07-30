from __future__ import annotations

import xml.etree.ElementTree as ET
from pathlib import Path

import torch
import torch.nn as nn

from Prior_Recon.Masked_Flow.utils.assets import default_g1_mjcf_xml_path


def _normalize(v: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    return v / v.norm(dim=-1, keepdim=True).clamp_min(eps)


def _quat_wxyz_to_matrix(quat: torch.Tensor) -> torch.Tensor:
    quat = _normalize(quat)
    w, x, y, z = quat.unbind(dim=-1)
    ww, xx, yy, zz = w * w, x * x, y * y, z * z
    wx, wy, wz = w * x, w * y, w * z
    xy, xz, yz = x * y, x * z, y * z
    return torch.stack(
        [
            torch.stack([ww + xx - yy - zz, 2.0 * (xy - wz), 2.0 * (xz + wy)], dim=-1),
            torch.stack([2.0 * (xy + wz), ww - xx + yy - zz, 2.0 * (yz - wx)], dim=-1),
            torch.stack([2.0 * (xz - wy), 2.0 * (yz + wx), ww - xx - yy + zz], dim=-1),
        ],
        dim=-2,
    )


def _axis_angle_to_matrix(axis_angle: torch.Tensor) -> torch.Tensor:
    angle = axis_angle.norm(dim=-1, keepdim=True)
    axis = axis_angle / angle.clamp_min(1e-8)
    x, y, z = axis.unbind(dim=-1)
    cos = torch.cos(angle)[..., 0]
    sin = torch.sin(angle)[..., 0]
    one_minus_cos = 1.0 - cos

    return torch.stack(
        [
            torch.stack(
                [
                    cos + x * x * one_minus_cos,
                    x * y * one_minus_cos - z * sin,
                    x * z * one_minus_cos + y * sin,
                ],
                dim=-1,
            ),
            torch.stack(
                [
                    y * x * one_minus_cos + z * sin,
                    cos + y * y * one_minus_cos,
                    y * z * one_minus_cos - x * sin,
                ],
                dim=-1,
            ),
            torch.stack(
                [
                    z * x * one_minus_cos - y * sin,
                    z * y * one_minus_cos + x * sin,
                    cos + z * z * one_minus_cos,
                ],
                dim=-1,
            ),
        ],
        dim=-2,
    )


def euler_xyz_to_matrix(roll: torch.Tensor, pitch: torch.Tensor, yaw: torch.Tensor) -> torch.Tensor:
    cr, sr = torch.cos(roll), torch.sin(roll)
    cp, sp = torch.cos(pitch), torch.sin(pitch)
    cy, sy = torch.cos(yaw), torch.sin(yaw)

    return torch.stack(
        [
            torch.stack([cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr], dim=-1),
            torch.stack([sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr], dim=-1),
            torch.stack([-sp, cp * sr, cp * cr], dim=-1),
        ],
        dim=-2,
    )


def matrix_to_rot6d(matrix: torch.Tensor) -> torch.Tensor:
    return matrix[..., :, :2].reshape(*matrix.shape[:-2], 6)


def _rel_trace(r_pred: torch.Tensor, r_target: torch.Tensor) -> torch.Tensor:
    rel = torch.matmul(r_pred.transpose(-1, -2), r_target)
    return rel[..., 0, 0] + rel[..., 1, 1] + rel[..., 2, 2]


def geodesic_cos_loss(r_pred: torch.Tensor, r_target: torch.Tensor) -> torch.Tensor:
    """SO(3) orientation error for LOSSES, batched over (..., 3, 3).

    Returns ``(3 - trace(R_pred^T R_target)) / 2 = 1 - cos(theta)`` in [0, 2].
    Monotonic in the true geodesic angle, but unlike ``arccos`` it is exactly 0
    with 0 gradient at the identity (arccos has an unbounded derivative near
    cos=1, so squaring it would push AWAY from a perfectly aligned pose). This is
    the standard geodesic surrogate; use ``geodesic_angle`` for interpretable
    metric reporting, this for optimisation.
    """
    return (3.0 - _rel_trace(r_pred, r_target)) * 0.5


def geodesic_angle(r_pred: torch.Tensor, r_target: torch.Tensor) -> torch.Tensor:
    """Geodesic angle (rad) between two rotation matrices, for METRIC reporting.

    theta = arccos((trace(R_pred^T R_target) - 1) / 2), clamped. Interpretable
    (radians/degrees) but the arccos gradient blows up near identity, so prefer
    ``geodesic_cos_loss`` inside a differentiable objective.
    """
    cos = ((_rel_trace(r_pred, r_target) - 1.0) * 0.5).clamp(-1.0 + 1e-6, 1.0 - 1e-6)
    return torch.arccos(cos)


class G129DeltaForwardKinematics(nn.Module):
    """Torch FK for the G1 29DOF tree used by delta69 losses."""

    _FOOT_NAMES = ("left_ankle_roll_link", "right_ankle_roll_link")
    _HAND_NAMES = ("left_wrist_yaw_link", "right_wrist_yaw_link")
    _HAND_OFFSETS = (
        (0.0415, 0.0030, 0.0),
        (0.0415, -0.0030, 0.0),
    )

    def __init__(self, fps: float = 30.0):
        super().__init__()
        self.fps = float(fps)
        skeleton = self._parse_mjcf(default_g1_mjcf_xml_path())
        self.body_names = skeleton["body_names"]
        self.foot_ids = [self.body_names.index(name) for name in self._FOOT_NAMES]
        self.hand_ids = [self.body_names.index(name) for name in self._HAND_NAMES]

        self.register_buffer("parent_indices", torch.tensor(skeleton["parent_indices"], dtype=torch.long))
        self.register_buffer("local_translations", torch.tensor(skeleton["local_translations"], dtype=torch.float32))
        self.register_buffer("local_rot_mats", _quat_wxyz_to_matrix(torch.tensor(skeleton["local_quats"], dtype=torch.float32)))
        self.register_buffer("body_dof_indices", torch.tensor(skeleton["body_dof_indices"], dtype=torch.long))
        self.register_buffer("dof_axes", torch.tensor(skeleton["dof_axes"], dtype=torch.float32))
        self.register_buffer("hand_offsets", torch.tensor(self._HAND_OFFSETS, dtype=torch.float32))
        self.register_buffer("_identity3", torch.eye(3, dtype=torch.float32))

        # Sole contact points (Layer A): each ankle_roll_link carries N>=1
        # primitive geoms (no mesh) whose local `pos` are the heel/toe corners of
        # the foot. The single ankle-roll body origin lies near the pitch/roll
        # axes, so it is nearly invariant to foot rotation -- the loss on it alone
        # leaves a rotational null space (heel-up/toe-down). These N non-collinear
        # sole points make foot orientation observable to the FK losses. Falls
        # back to the body origin (N=1) when the XML has no primitive foot geoms.
        sole_local = torch.tensor(
            skeleton["foot_sole_local"], dtype=torch.float32
        )  # (2, N, 3)
        self.n_sole_points = int(sole_local.shape[1])
        self.register_buffer("foot_sole_local", sole_local)

    @staticmethod
    def _parse_mjcf(xml_path: Path) -> dict[str, list]:
        tree = ET.parse(xml_path)
        root = tree.getroot()
        worldbody = root.find("worldbody")
        if worldbody is None:
            raise ValueError(f"Invalid MJCF: no worldbody in {xml_path}")

        root_body = worldbody.find("body")
        if root_body is None:
            raise ValueError(f"Invalid MJCF: no root body in {xml_path}")

        actuator = root.find("actuator")
        if actuator is None:
            raise ValueError(f"Invalid MJCF: no actuator block in {xml_path}")

        dof_joint_names = [motor.attrib["joint"] for motor in actuator.findall("motor")]
        joint_name_to_idx = {name: idx for idx, name in enumerate(dof_joint_names)}

        body_names: list[str] = []
        parent_indices: list[int] = []
        local_translations: list[list[float]] = []
        local_quats: list[list[float]] = []
        body_dof_indices: list[int] = []
        dof_axes: list[list[float]] = []
        # Local sole-point positions per foot body (heel/toe corners), read from
        # the ankle_roll_link primitive geoms (those with a `pos` and no mesh).
        foot_sole_by_name: dict[str, list[list[float]]] = {}

        def _parse_body(xml_body: ET.Element, parent_idx: int) -> None:
            body_idx = len(body_names)
            name = xml_body.attrib.get("name", f"body_{body_idx}")
            body_names.append(name)
            parent_indices.append(parent_idx)
            local_translations.append(
                [float(v) for v in xml_body.attrib.get("pos", "0 0 0").split()]
            )
            local_quats.append(
                [float(v) for v in xml_body.attrib.get("quat", "1 0 0 0").split()]
            )

            if name in G129DeltaForwardKinematics._FOOT_NAMES:
                sole_pts = [
                    [float(v) for v in geom.attrib["pos"].split()]
                    for geom in xml_body.findall("geom")
                    if "mesh" not in geom.attrib and "pos" in geom.attrib
                ]
                if sole_pts:
                    foot_sole_by_name[name] = sole_pts

            revolute_joint = None
            for joint in xml_body.findall("joint"):
                if joint.attrib.get("type", "hinge") != "free":
                    revolute_joint = joint
                    break

            if revolute_joint is None:
                body_dof_indices.append(-1)
                dof_axes.append([0.0, 0.0, 0.0])
            else:
                joint_name = revolute_joint.attrib.get("name")
                if joint_name not in joint_name_to_idx:
                    raise KeyError(f"Joint {joint_name} missing from actuator order in {xml_path}")
                body_dof_indices.append(joint_name_to_idx[joint_name])
                dof_axes.append([float(v) for v in revolute_joint.attrib.get("axis", "0 0 1").split()])

            for child in xml_body.findall("body"):
                _parse_body(child, body_idx)

        _parse_body(root_body, -1)

        # Assemble (2, N, 3) sole points in _FOOT_NAMES order. All feet must
        # expose the same count; if a foot has none (e.g. a mesh-only XML), fall
        # back to a single origin point [0,0,0] for every foot so downstream FK
        # degrades to the legacy single-point behaviour instead of crashing.
        sole_lists = [foot_sole_by_name.get(n) for n in G129DeltaForwardKinematics._FOOT_NAMES]
        counts = {len(s) for s in sole_lists if s}
        if len(counts) > 1:
            raise ValueError(
                f"Inconsistent foot sole-point counts across feet in {xml_path}: {counts}"
            )
        if not counts:
            foot_sole_local = [[[0.0, 0.0, 0.0]] for _ in G129DeltaForwardKinematics._FOOT_NAMES]
        else:
            n_pts = counts.pop()
            foot_sole_local = [
                s if s else [[0.0, 0.0, 0.0]] * n_pts for s in sole_lists
            ]

        return {
            "body_names": body_names,
            "parent_indices": parent_indices,
            "local_translations": local_translations,
            "local_quats": local_quats,
            "body_dof_indices": body_dof_indices,
            "dof_axes": dof_axes,
            "foot_sole_local": foot_sole_local,
        }

    def forward(
        self,
        root_pos: torch.Tensor,
        root_rot_mat: torch.Tensor,
        dof_pos: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        if root_pos.ndim != 3 or root_rot_mat.ndim != 4 or dof_pos.ndim != 3:
            raise ValueError(
                "Expected shapes root_pos=(B,T,3), root_rot_mat=(B,T,3,3), dof_pos=(B,T,29), "
                f"got {tuple(root_pos.shape)}, {tuple(root_rot_mat.shape)}, {tuple(dof_pos.shape)}"
            )

        batch, seq_len = root_pos.shape[:2]
        num_bodies = self.parent_indices.shape[0]

        body_pos = [root_pos]
        body_rot = [root_rot_mat]
        identity = self._identity3.to(dtype=dof_pos.dtype).view(1, 1, 3, 3)

        for body_idx in range(1, num_bodies):
            parent_idx = int(self.parent_indices[body_idx].item())
            parent_pos = body_pos[parent_idx]
            parent_rot = body_rot[parent_idx]

            offset = self.local_translations[body_idx].view(1, 1, 3, 1).to(dof_pos)
            local_rest = self.local_rot_mats[body_idx].view(1, 1, 3, 3).to(dof_pos)
            dof_idx = int(self.body_dof_indices[body_idx].item())

            if dof_idx >= 0:
                axis = self.dof_axes[body_idx].view(1, 1, 3).to(dof_pos)
                joint_rot = _axis_angle_to_matrix(axis * dof_pos[..., dof_idx : dof_idx + 1])
            else:
                joint_rot = identity.expand(batch, seq_len, -1, -1)

            local_rot = torch.matmul(local_rest, joint_rot)
            world_rot = torch.matmul(parent_rot, local_rot)
            world_pos = parent_pos + torch.matmul(parent_rot, offset).squeeze(-1)

            body_pos.append(world_pos)
            body_rot.append(world_rot)

        global_pos = torch.stack(body_pos, dim=2)
        global_rot_mat = torch.stack(body_rot, dim=2)
        hand_rot_mat = global_rot_mat[:, :, self.hand_ids]
        hand_offsets = self.hand_offsets.view(1, 1, -1, 3, 1).to(global_pos)
        hand_pos = global_pos[:, :, self.hand_ids] + torch.matmul(hand_rot_mat, hand_offsets).squeeze(-1)

        # Sole contact points: p_world = p_foot + R_foot @ p_sole_local, for each
        # foot's N heel/toe corners. (B, T, 2, N, 3). R_foot is the ankle_roll
        # world rotation. These are what the Layer A foot losses supervise; the
        # multi-point spread makes foot rotation observable (single ankle origin
        # does not). foot_rotation_* expose R_foot for the geodesic orientation
        # loss.
        foot_rot_mat = global_rot_mat[:, :, self.foot_ids]  # (B, T, 2, 3, 3)
        foot_origin = global_pos[:, :, self.foot_ids]  # (B, T, 2, 3)
        sole_local = self.foot_sole_local.view(1, 1, 2, self.n_sole_points, 3, 1).to(global_pos)
        # (B,T,2,1,3,3) @ (1,1,2,N,3,1) -> (B,T,2,N,3,1)
        sole_world = foot_origin.unsqueeze(-2) + torch.matmul(
            foot_rot_mat.unsqueeze(-3), sole_local
        ).squeeze(-1)

        dof_vel = torch.zeros_like(dof_pos)
        if seq_len > 1:
            scale = self.fps
            dof_vel[:, 1:] = (dof_pos[:, 1:] - dof_pos[:, :-1]) * scale
            dof_vel[:, 0] = dof_vel[:, 1]

        return {
            "global_translation": global_pos,
            "global_rotation_mat": global_rot_mat,
            "global_rotation_6d": matrix_to_rot6d(global_rot_mat),
            "dof_pos": dof_pos,
            "dof_vel": dof_vel,
            "foot_translation": foot_origin,
            "foot_sole_translation": sole_world,
            "foot_rotation_mat": foot_rot_mat,
            "foot_rotation_6d": matrix_to_rot6d(foot_rot_mat),
            "hand_translation": hand_pos,
            "hand_rotation_mat": hand_rot_mat,
            "hand_rotation_6d": matrix_to_rot6d(hand_rot_mat),
        }
