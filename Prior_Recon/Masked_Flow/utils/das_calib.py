#!/usr/bin/env python3
"""DAS-Gripper <-> recon-palm calibration (single source of truth).

The DAS_Controller_V3 gripper is welded onto each G1 wrist by
``assets/build_g1_with_das.py``; the merged model is
``assets/g1_29dof_with_das_controller.xml``. This module exposes the two rigid
transforms that relate the three frames that matter downstream, derived from
the SAME mount primitives the XML builder uses (so they cannot drift), and
cross-checked against the generated XML via FK in ``_selftest`` below.

Frames
------
``wrist``  : ``*_wrist_yaw_link`` -- the G1 body recon FKs and the deploy
             ``HandPoseFK`` returns (raw, no palm offset).
``vio``    : ``*_das_base_link`` -- the DAS ``base_link``, i.e. the VIO / policy
             EE frame. This is the frame the UMI ``eef_pose`` stream lives in
             AND the frame the Diffusion Policy relative action pivots about.
``palm``   : ``wrist + _HAND_OFFSETS`` -- recon's hand keypoint anchor
             (``recon_delta69._HAND_OFFSETS``, identity rotation vs wrist).

Transforms (constant, body-fixed)
---------------------------------
``T_WRIST_VIO``     (4,4)   wrist -> vio.  Same for both hands (symmetric mount):
                            R = R_y(+15 deg), t ~ (0.1323, 0, 0.0767).
``T_VIO_PALM``      (2,4,4) vio -> palm, per hand [left, right].
                            R = R_y(-15 deg), t ~ (-0.0679, +/-0.003, -0.0976).

Self-consistency (guaranteed by construction, verified to ~1e-16 by FK):
    T_WRIST_VIO @ T_VIO_PALM[i]  ==  [ I | _HAND_OFFSETS[i] ]

so the full-rigid chain collapses to recon's translation-only ``_HAND_OFFSETS``
when the policy motion is identity -- the two 15 deg pitches cancel. This is
what keeps ``test_dp_base_anchor.py`` (identity action -> recon FK) passing.

Why it is NOT a pure translation
--------------------------------
``T_VIO_PALM`` carries a 15 deg pitch. recon relativizes quats via
``quat_wxyz_relative_to_first`` = ``inv(q0) @ q(t)`` (LEFT multiply), while this
mount offset is body-fixed (RIGHT multiply). A right offset does NOT cancel
under a left relativization -- it conjugates: ``rel_palm = R_vp^-1 rel_vio
R_vp`` -- and the palm position ``p_vio + R_vio(t) t_vp`` is time varying. So a
non-identity VIO trajectory needs the FULL rigid transform, not a fixed
translation, to land on the palm frame recon was trained on.
"""
from __future__ import annotations

import math

import numpy as np

# --- mount primitives (mirror assets/build_g1_with_das.py) -----------------
# DAS base_link -> link_Flange (from URDF joint_Flange), reused as the mount
# pitch. Keep in sync with build_g1_with_das.py; _selftest cross-checks the XML.
_PITCH = -0.2618  # rad; DAS base->flange pitch about +y
_T_BASE_FLANGE_T = (-0.067889, -3.8794e-05, -0.097619)

# Flange frame expressed in the wrist frame (the mount choice). Identity
# orientation + x offset -> the gripper extends forward like the removed hand.
# x = old hand/palm mount (0.0415) + 8.5 mm connector plate between the wrist
# link and the DAS flange (pure thickness spacer along wrist +x). Keep in sync
# with build_g1_with_das.FLANGE_MOUNT_X.
_CONNECTOR_THICKNESS = 0.0085
_FLANGE_MOUNT_X = 0.0415 + _CONNECTOR_THICKNESS  # = 0.0500
_FLANGE_MOUNT_RPY = (0.0, 0.0, 0.0)

# recon's hand keypoint anchor (== recon_delta69._HAND_OFFSETS). Defined locally
# so this module stays free of recon_delta69's heavy imports (torch/mujoco);
# _selftest asserts it still equals the recon constant.
_HAND_OFFSETS = np.array(
    [[0.0415, 0.0030, 0.0], [0.0415, -0.0030, 0.0]], dtype=np.float64
)


def _ry(a: float) -> np.ndarray:
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]], dtype=np.float64)


def _rpy(r: float, p: float, y: float) -> np.ndarray:
    cr, sr = math.cos(r), math.sin(r)
    cp, sp = math.cos(p), math.sin(p)
    cy, sy = math.cos(y), math.sin(y)
    Rx = np.array([[1, 0, 0], [0, cr, -sr], [0, sr, cr]], dtype=np.float64)
    Ry = np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]], dtype=np.float64)
    Rz = np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]], dtype=np.float64)
    return Rz @ Ry @ Rx


def _homog(R: np.ndarray, t) -> np.ndarray:
    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = R
    T[:3, 3] = t
    return T


# --- derived transforms (module constants) ---------------------------------
_T_BASE_FLANGE = _homog(_ry(_PITCH), _T_BASE_FLANGE_T)
_T_WRIST_FLANGE = _homog(_rpy(*_FLANGE_MOUNT_RPY), (_FLANGE_MOUNT_X, 0.0, 0.0))

#: wrist -> vio (das_base_link). (4,4). Same for both hands.
T_WRIST_VIO = _T_WRIST_FLANGE @ np.linalg.inv(_T_BASE_FLANGE)

_T_VIO_WRIST = np.linalg.inv(T_WRIST_VIO)

#: vio -> palm, per hand [left, right]. (2,4,4).
T_VIO_PALM = np.stack(
    [_T_VIO_WRIST @ _homog(np.eye(3), _HAND_OFFSETS[i]) for i in range(2)], axis=0
)

#: wrist -> palm, per hand [left, right]. (2,4,4). Rotation is identity by
#: construction; translation == _HAND_OFFSETS. Provided for convenience.
T_WRIST_PALM = np.stack(
    [_homog(np.eye(3), _HAND_OFFSETS[i]) for i in range(2)], axis=0
)


def _mat_from_quat_wxyz(q: np.ndarray) -> np.ndarray:
    """(...,4) wxyz -> (...,3,3)."""
    q = np.asarray(q, dtype=np.float64)
    q = q / np.linalg.norm(q, axis=-1, keepdims=True).clip(1e-12)
    w, x, y, z = q[..., 0], q[..., 1], q[..., 2], q[..., 3]
    m = np.stack(
        [
            1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y),
            2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x),
            2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y),
        ],
        axis=-1,
    ).reshape(q.shape[:-1] + (3, 3))
    return m


def _quat_wxyz_from_mat(R: np.ndarray) -> np.ndarray:
    """(...,3,3) -> (...,4) wxyz. Trace method, batched."""
    R = np.asarray(R, dtype=np.float64)
    m00, m11, m22 = R[..., 0, 0], R[..., 1, 1], R[..., 2, 2]
    tr = m00 + m11 + m22
    w = np.sqrt(np.clip(1.0 + tr, 0.0, None)) / 2.0
    x = np.sqrt(np.clip(1.0 + m00 - m11 - m22, 0.0, None)) / 2.0
    y = np.sqrt(np.clip(1.0 - m00 + m11 - m22, 0.0, None)) / 2.0
    z = np.sqrt(np.clip(1.0 - m00 - m11 + m22, 0.0, None)) / 2.0
    x = np.copysign(x, R[..., 2, 1] - R[..., 1, 2])
    y = np.copysign(y, R[..., 0, 2] - R[..., 2, 0])
    z = np.copysign(z, R[..., 1, 0] - R[..., 0, 1])
    q = np.stack([w, x, y, z], axis=-1)
    return q / np.linalg.norm(q, axis=-1, keepdims=True).clip(1e-12)


def vio_pose_to_palm(
    pos: np.ndarray, quat_wxyz: np.ndarray, hand: int
) -> tuple[np.ndarray, np.ndarray]:
    """Map a VIO (das_base_link) pose trajectory to the recon palm anchor.

    Applies the FULL rigid ``T_VIO_PALM[hand]`` on the RIGHT (body-fixed):
        T_palm(t) = T_vio(t) @ T_VIO_PALM[hand]
    i.e. ``p_palm = p_vio + R_vio @ t_vp`` and ``R_palm = R_vio @ R_vp``.

    Args:
        pos: (...,3) VIO frame positions (world).
        quat_wxyz: (...,4) VIO frame orientations (world), wxyz.
        hand: 0 = left, 1 = right.

    Returns:
        (pos_palm (...,3), quat_palm_wxyz (...,4)).
    """
    pos = np.asarray(pos, dtype=np.float64)
    R_vio = _mat_from_quat_wxyz(quat_wxyz)  # (...,3,3)
    Tvp = T_VIO_PALM[hand]
    R_vp, t_vp = Tvp[:3, :3], Tvp[:3, 3]
    pos_palm = pos + np.einsum("...ij,j->...i", R_vio, t_vp)
    R_palm = np.einsum("...ij,jk->...ik", R_vio, R_vp)
    return pos_palm, _quat_wxyz_from_mat(R_palm)


def wrist_mat_to_vio_mat(wrist_mats: np.ndarray) -> np.ndarray:
    """Raw wrist world frame(s) -> VIO (das_base_link) world frame(s).

    ``T_vio = T_wrist @ T_WRIST_VIO``. Accepts (...,4,4).
    """
    wrist_mats = np.asarray(wrist_mats, dtype=np.float64)
    return np.einsum("...ij,jk->...ik", wrist_mats, T_WRIST_VIO)


def _selftest() -> None:
    """Cross-check the analytic constants against the generated XML via FK,
    and against recon_delta69._HAND_OFFSETS. Requires mujoco + scipy."""
    import mujoco
    from pathlib import Path

    # 1) wrist->palm collapses to _HAND_OFFSETS (rotation identity).
    for i in range(2):
        Twp = T_WRIST_VIO @ T_VIO_PALM[i]
        assert np.allclose(Twp[:3, 3], _HAND_OFFSETS[i], atol=1e-12), Twp[:3, 3]
        assert np.allclose(Twp[:3, :3], np.eye(3), atol=1e-12), Twp[:3, :3]

    # 2) matches recon's own _HAND_OFFSETS constant.
    from Prior_Recon.Masked_Flow.visual.recon_delta69 import (
        _HAND_OFFSETS as RECON_OFF,
    )
    assert np.allclose(_HAND_OFFSETS, np.asarray(RECON_OFF, np.float64), atol=1e-9)

    # 3) matches the generated merged XML via FK, to ~1e-15.
    xml = Path(__file__).resolve().parents[1] / "assets" / (
        "g1_29dof_with_das_controller.xml"
    )
    m = mujoco.MjModel.from_xml_path(str(xml))
    d = mujoco.MjData(m)
    mujoco.mj_forward(m, d)

    def bT(n):
        b = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, n)
        T = np.eye(4)
        T[:3, :3] = d.xmat[b].reshape(3, 3)
        T[:3, 3] = d.xpos[b]
        return T

    def sT(n):
        s = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SITE, n)
        T = np.eye(4)
        T[:3, :3] = d.site_xmat[s].reshape(3, 3)
        T[:3, 3] = d.site_xpos[s]
        return T

    max_err = 0.0
    for i, side in enumerate(("left", "right")):
        Tw, Tb, Tp = bT(f"{side}_wrist_yaw_link"), bT(f"{side}_das_base_link"), sT(
            f"{side}_palm_anchor"
        )
        wv = np.linalg.inv(Tw) @ Tb
        vp = np.linalg.inv(Tb) @ Tp
        max_err = max(
            max_err,
            float(np.abs(wv - T_WRIST_VIO).max()),
            float(np.abs(vp - T_VIO_PALM[i]).max()),
        )
    assert max_err < 1e-12, f"XML/analytic mismatch: {max_err:.2e}"
    print(f"das_calib self-test OK (max XML-vs-analytic err = {max_err:.2e})")
    print("T_WRIST_VIO:\n", T_WRIST_VIO)
    print("T_VIO_PALM[left]:\n", T_VIO_PALM[0])
    print("T_VIO_PALM[right]:\n", T_VIO_PALM[1])


if __name__ == "__main__":
    _selftest()
