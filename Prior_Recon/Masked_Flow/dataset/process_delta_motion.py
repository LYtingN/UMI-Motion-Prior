#!/usr/bin/env python3
"""
process_delta_motion.py
~~~~~~~~~~~~~~~~~~~~~~~
G1 Sonic CSV  →  69D TextOp-style delta motion feature (+ 1 auxiliary yaw).

特征设计  (T, 70) — 保存带辅助yaw通道
------------------------------------------
  0:4   root_tilt_sincos  [4]
        根部 roll / pitch 的零中心化表示:
        [sin(roll), cos(roll)-1, sin(pitch), cos(pitch)-1]
        这是“当前倾斜姿态”，不是增量。
  4     delta_yaw         [1]
        帧间 yaw 增量 (rad)，归一化到 [-π, π]。
        表示“相对上一帧转了多少”。
  5:7   contact_mask      [2]
        左/右脚二值接触标记，由脚部高度和脚速双阈值判断得到。
        1 更接近支撑脚，0 更接近摆动脚。
  7:10  delta_trans_world [3]
        世界坐标系下根部帧间位移 [dx, dy, dz]，单位 m/frame。
        这里只先保存 world-frame 位移，Dataset 再按窗口起点 yaw
        旋转成 heading-local 位移给模型。
  10    height            [1]
        根部绝对高度 z，单位 m；这是绝对量，不是增量。
  11:40 dof               [29]
        当前帧 29 个关节角度 (rad) —— 包含腿、腰、双臂、腕部。
        这块描述当前整个人形姿态。
  40:69 delta_dof         [29]
        29 个关节的帧间角度增量 (rad)，归一化到 [-π, π]。
        这块描述每个关节“这一帧动了多少”。
  69    absolute_yaw      [1]
        辅助通道: 当前帧绝对 yaw (rad)。Dataset 在 __getitem__ 中用它
        将 delta_trans_world 旋转到航向对齐坐标系，随后剥离，不直接送入模型。

输入: T+1 帧 CSV → 输出 T 帧特征

29关节顺序 (同 process_g1sonic_motion.py):
   0  left_hip_pitch    12 waist_yaw      19 right_shoulder_pitch
   1  left_hip_roll     13 waist_roll     20 right_shoulder_roll
   2  left_hip_yaw      14 waist_pitch    21 right_shoulder_yaw
   3  left_knee         15 left_shoulder_pitch  22 right_elbow
   4  left_ankle_pitch  16 left_shoulder_roll   23 right_wrist_roll
   5  left_ankle_roll   17 left_shoulder_yaw    24 right_wrist_pitch
   6  right_hip_pitch   18 left_elbow            25 right_wrist_yaw
   7  right_hip_roll                             26 left_elbow (→ elbow above)
   8  right_hip_yaw                              ...
   9  right_knee
  10  right_ankle_pitch
  11  right_ankle_roll

Usage
-----
  python process_delta_motion.py \\
      --data-root ~/NYX/g1_sonic_data/balance_data_v1 \\
      --out-dir   ~/NYX/g1_sonic_data/delta_feat \\
      --xml-path  /path/to/g1.xml

  python process_delta_motion.py \\
      --data-root ~/NYX/g1_sonic_data/csv \\
      --match-root ~/NYX/g1_sonic_data/Full_npy/large_selected_npy \\
      --out-dir   ~/NYX/g1_sonic_data/delta_feat \\
      --xml-path  /path/to/g1.xml \\
      --workers 8 --overwrite

输出格式
--------
默认输出为 `.npz` bundle，字段包括:
  feat      : (T, 70)    delta motion feature
  keypoints : (T, 2, 7)  预计算双手 keypoints，单位 m + quat_wxyz

其中 `keypoints` 已与 `feat` 对齐到相同时间长度，供训练时直接读取，
避免再次回查原始 CSV。
"""
from __future__ import annotations

import argparse
import multiprocessing as mp
import tempfile
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Constants — identical to process_g1sonic_motion.py
# ---------------------------------------------------------------------------

_DOF_COLS = [
    "left_hip_pitch_joint_dof",    "left_hip_roll_joint_dof",
    "left_hip_yaw_joint_dof",      "left_knee_joint_dof",
    "left_ankle_pitch_joint_dof",  "left_ankle_roll_joint_dof",
    "right_hip_pitch_joint_dof",   "right_hip_roll_joint_dof",
    "right_hip_yaw_joint_dof",     "right_knee_joint_dof",
    "right_ankle_pitch_joint_dof", "right_ankle_roll_joint_dof",
    "waist_yaw_joint_dof",         "waist_roll_joint_dof",
    "waist_pitch_joint_dof",
    "left_shoulder_pitch_joint_dof",  "left_shoulder_roll_joint_dof",
    "left_shoulder_yaw_joint_dof",    "left_elbow_joint_dof",
    "left_wrist_roll_joint_dof",      "left_wrist_pitch_joint_dof",
    "left_wrist_yaw_joint_dof",
    "right_shoulder_pitch_joint_dof", "right_shoulder_roll_joint_dof",
    "right_shoulder_yaw_joint_dof",   "right_elbow_joint_dof",
    "right_wrist_roll_joint_dof",     "right_wrist_pitch_joint_dof",
    "right_wrist_yaw_joint_dof",
]
_ROOT_POS_COLS = ["root_translateX", "root_translateY", "root_translateZ"]
_ROOT_ROT_COLS = ["root_rotateX",    "root_rotateY",    "root_rotateZ"]
_HAND_KP_COLS = [
    "left_palm_x",  "left_palm_y",  "left_palm_z",
    "left_palm_qw", "left_palm_qx", "left_palm_qy", "left_palm_qz",
    "right_palm_x", "right_palm_y", "right_palm_z",
    "right_palm_qw","right_palm_qx","right_palm_qy","right_palm_qz",
]

# 29 body names in DOF order
_BODY_NAMES = [
    "left_hip_pitch_link",     "left_hip_roll_link",
    "left_hip_yaw_link",       "left_knee_link",
    "left_ankle_pitch_link",   "left_ankle_roll_link",      # 5  ← left foot
    "right_hip_pitch_link",    "right_hip_roll_link",
    "right_hip_yaw_link",      "right_knee_link",
    "right_ankle_pitch_link",  "right_ankle_roll_link",     # 11 ← right foot
    "waist_yaw_link",          "waist_roll_link",
    "torso_link",
    "left_shoulder_pitch_link","left_shoulder_roll_link",
    "left_shoulder_yaw_link",  "left_elbow_link",
    "left_wrist_roll_link",    "left_wrist_pitch_link",
    "left_wrist_yaw_link",
    "right_shoulder_pitch_link","right_shoulder_roll_link",
    "right_shoulder_yaw_link", "right_elbow_link",
    "right_wrist_roll_link",   "right_wrist_pitch_link",
    "right_wrist_yaw_link",
]

_FID_L     = 5   # left_ankle_roll_link index in _BODY_NAMES
_FID_R     = 11  # right_ankle_roll_link index in _BODY_NAMES
FEAT_DIM   = 70  # 69 model dims + 1 auxiliary yaw

# Hand keypoint FK — identical definition to process_delta_teleop_mjstate.py
# and recon_delta69._HAND_OFFSETS, so palm keypoints derived here are
# self-consistent with the model's own keypoint convention.
_HAND_BODY_NAMES = ["left_wrist_yaw_link", "right_wrist_yaw_link"]
_HAND_OFFSETS = np.array(
    [
        [0.0415, 0.0030, 0.0],
        [0.0415, -0.0030, 0.0],
    ],
    dtype=np.float64,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _angle_diff(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Wrapped angle difference (a - b) in [-π, π]."""
    d = a - b
    return (d + np.pi) % (2 * np.pi) - np.pi


def _resolve_mjcf_xml(xml_path: str) -> tuple[str, str | None]:
    """Return an MJCF path whose asset directories are resolvable by MuJoCo.

    Some checked-in XMLs live in `<root>/mjcf/` while meshes live in
    `<root>/meshes/`, but the XML still says `meshdir="meshes"`. MuJoCo then
    looks for `<root>/mjcf/meshes/...` and fails. When that happens, rewrite the
    compiler meshdir to an absolute existing path in a temporary XML.
    """
    xml_file = Path(xml_path).resolve()
    tree = ET.parse(xml_file)
    root = tree.getroot()
    compiler = root.find("compiler")
    if compiler is None:
        return str(xml_file), None

    meshdir = compiler.get("meshdir")
    if not meshdir:
        return str(xml_file), None

    meshdir_path = Path(meshdir)
    if meshdir_path.is_absolute():
        return str(xml_file), None

    candidates = [
        (xml_file.parent / meshdir_path).resolve(),
        (xml_file.parent.parent / meshdir_path).resolve(),
        (xml_file.parent.parent / "meshes").resolve(),
    ]
    existing = next((p for p in candidates if p.exists()), None)
    if existing is None:
        return str(xml_file), None

    declared = (xml_file.parent / meshdir_path).resolve()
    if declared == existing:
        return str(xml_file), None

    compiler.set("meshdir", str(existing))
    with tempfile.NamedTemporaryFile(
        mode="wb", suffix=".xml", delete=False, prefix="resolved_g1_mjcf_"
    ) as f:
        tree.write(f, encoding="utf-8", xml_declaration=False)
        tmp_path = f.name
    return tmp_path, tmp_path


def _fk_feet(
    csv_path: str, xml_path: str, with_hands: bool = False
):
    """
    Minimal MuJoCo FK for root + two ankle joints (+ optional hand keypoints).

    Returns
    -------
    root_pos   : (T, 3)  pelvis world position (m)
    ankle_l    : (T, 3)  left  ankle_roll world position (m)
    ankle_r    : (T, 3)  right ankle_roll world position (m)
    hand_kp    : (T, 2, 7) world palm [pos_m + quat_wxyz]  (only if with_hands)
    """
    import mujoco
    from scipy.spatial.transform import Rotation as R

    df    = pd.read_csv(csv_path)
    T     = len(df)
    D2R   = np.pi / 180.0

    trans = df[_ROOT_POS_COLS].values * 0.01   # cm → m
    euler = df[_ROOT_ROT_COLS].values * D2R    # deg → rad  (XYZ extrinsic)
    dof   = df[_DOF_COLS].values   * D2R       # deg → rad

    resolved_xml_path, temp_xml_path = _resolve_mjcf_xml(xml_path)
    try:
        model  = mujoco.MjModel.from_xml_path(resolved_xml_path)
        data_m = mujoco.MjData(model)
    finally:
        if temp_xml_path is not None:
            try:
                Path(temp_xml_path).unlink(missing_ok=True)
            except Exception:
                pass

    def bid(name: str) -> int:
        return mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)

    pelvis_id   = bid("pelvis")
    ankle_l_id  = bid(_BODY_NAMES[_FID_L])
    ankle_r_id  = bid(_BODY_NAMES[_FID_R])

    required = [(pelvis_id, "pelvis"),
                (ankle_l_id, _BODY_NAMES[_FID_L]),
                (ankle_r_id, _BODY_NAMES[_FID_R])]

    hand_ids: list[int] = []
    if with_hands:
        for name in _HAND_BODY_NAMES:
            hid = bid(name)
            hand_ids.append(hid)
            required.append((hid, name))

    for body_id, name in required:
        if body_id < 0:
            raise RuntimeError(f"Body '{name}' not found in {xml_path}")

    root_pos = np.zeros((T, 3), dtype=np.float32)
    ankle_l  = np.zeros((T, 3), dtype=np.float32)
    ankle_r  = np.zeros((T, 3), dtype=np.float32)
    hand_kp  = np.zeros((T, 2, 7), dtype=np.float32) if with_hands else None

    for t in range(T):
        q_xyzw = R.from_euler("xyz", euler[t]).as_quat()
        q_wxyz = np.roll(q_xyzw, 1)
        data_m.qpos[:3]  = trans[t]
        data_m.qpos[3:7] = q_wxyz
        data_m.qpos[7:]  = dof[t]
        mujoco.mj_forward(model, data_m)
        root_pos[t] = data_m.xpos[pelvis_id]
        ankle_l[t]  = data_m.xpos[ankle_l_id]
        ankle_r[t]  = data_m.xpos[ankle_r_id]
        if with_hands:
            for k, hid in enumerate(hand_ids):
                rot = data_m.xmat[hid].reshape(3, 3)
                hand_kp[t, k, :3] = data_m.xpos[hid] + rot @ _HAND_OFFSETS[k]
                hand_kp[t, k, 3:] = data_m.xquat[hid]

    if with_hands:
        return root_pos, ankle_l, ankle_r, hand_kp
    return root_pos, ankle_l, ankle_r


# Bumped when the contact labelling rule changes; stamped into every npz so
# training can refuse label-dependent losses on stale data. v2 = m/s velocity
# threshold + majority filter (v1 compared m/frame against 0.5 = 15 m/s,
# degenerating to a height-only test). v3 = velocity computed with the TRUE
# source frame rate (v2 hardcoded fps=30 while e.g. the sonic SOMA corpus is
# 50 fps, understating speed by 0.6x -> swing frames up to 0.5 m/s passed the
# 0.3 m/s gate); the fps used is stamped as ``source_fps``. v4 tightens that
# gate to the audited 0.1 m/s threshold.
CONTACT_LABEL_VERSION = 4


def _foot_contacts(
    ankle_pos: np.ndarray,   # (T, 3)
    fps:       float = 30.0,
    vel_thr:   float = 0.1,  # m/s
    ht_thr:    float = 0.08, # m above ground
) -> np.ndarray:
    """
    Binary foot contact for T-1 frames (indices 0..T-2):
      contact[t] = 1 if ankle_height[t] < ht_thr AND ankle_speed[t] < vel_thr

    Speed is in m/s (per-frame displacement * fps). The previous version
    compared the raw per-frame displacement against 0.5, i.e. a 15 m/s bound
    at 30 fps -- never binding -- so contacts degenerated to a pure height
    test and swing-phase frames skimming the ground were mislabelled as
    contact. A majority-of-3 filter removes single-frame flickers so
    downstream contact intervals stay contiguous.

    Returns (T-1,) float array in {0, 1}.
    """
    T = ankle_pos.shape[0]
    ht    = ankle_pos[:, 2]                                            # (T,)
    speed = np.linalg.norm(np.diff(ankle_pos, axis=0), axis=1) * fps   # m/s

    contact = ((ht[:T-1] < ht_thr) & (speed < vel_thr)).astype(np.float32)
    if contact.shape[0] >= 3:
        # flip lone flickers (010 -> 000, 101 -> 111)
        agree = contact[:-2] == contact[2:]
        contact[1:-1] = np.where(agree, contact[:-2], contact[1:-1])
    return contact


def _load_keypoints_abs(df: pd.DataFrame) -> np.ndarray:
    """Load precomputed 2-hand trajectory from CSV.

    Returns:
        (T_all, 2, 7) absolute keypoints in metres + quat_wxyz.
    """
    missing = [col for col in _HAND_KP_COLS if col not in df.columns]
    if missing:
        raise ValueError(
            "CSV missing precomputed keypoint columns. "
            f"First missing columns: {missing[:4]}"
        )
    kp = df[_HAND_KP_COLS].values.astype(np.float32).reshape(-1, 2, 7)
    kp[:, :, :3] *= 0.01
    return kp


# ---------------------------------------------------------------------------
# Core feature extraction
# ---------------------------------------------------------------------------

def _resample_uniform(
    euler_all: np.ndarray,      # (T, 3) roll, pitch, yaw [rad]
    dof_all: np.ndarray,        # (T, 29)
    root_pos: np.ndarray,       # (T, 3)
    ankle_l: np.ndarray,        # (T, 3)
    ankle_r: np.ndarray,        # (T, 3)
    keypoints_abs: np.ndarray,  # (T, 2, 7) pos + quat_wxyz
    src_fps: float,
    tgt_fps: float,
) -> tuple[np.ndarray, ...]:
    """Resample one clip from src_fps to tgt_fps on a uniform time grid.

    Linear signals (positions, joint angles) are lerped; orientations (root
    euler via rotation, hand quats) are slerped. Yaw must NOT be lerped
    channel-wise — it wraps at ±π — hence the rotation round-trip.

    This exists so the 50 fps sonic corpus can be regenerated at the 30 fps
    the whole training/deploy stack assumes (cfg.motion.fps, planner.fps,
    run_bridge's 30->50 retime). Training on raw 50 fps frames while
    deploying them as 30 fps plays every motion back at 0.6x speed.
    """
    from scipy.spatial.transform import Rotation, Slerp

    T = euler_all.shape[0]
    t_src = np.arange(T, dtype=np.float64) / src_fps
    n_out = int(np.floor(t_src[-1] * tgt_fps)) + 1
    t_out = np.arange(n_out, dtype=np.float64) / tgt_fps

    def lerp(x: np.ndarray) -> np.ndarray:
        flat = x.reshape(T, -1)
        out = np.stack(
            [np.interp(t_out, t_src, flat[:, j]) for j in range(flat.shape[1])],
            axis=1,
        )
        return out.reshape((n_out,) + x.shape[1:])

    root_rot = Rotation.from_euler("xyz", euler_all)
    euler_out = Slerp(t_src, root_rot)(t_out).as_euler("xyz")

    kp_pos = lerp(keypoints_abs[..., :3])
    kp_quat = np.empty((n_out, 2, 4), dtype=np.float64)
    for h in range(2):
        q = Rotation.from_quat(np.roll(keypoints_abs[:, h, 3:], -1, axis=-1))
        kp_quat[:, h] = np.roll(Slerp(t_src, q)(t_out).as_quat(), 1, axis=-1)
    keypoints_out = np.concatenate([kp_pos, kp_quat], axis=-1)

    return (
        euler_out,
        lerp(dof_all),
        lerp(root_pos),
        lerp(ankle_l),
        lerp(ankle_r),
        keypoints_out,
    )


def process_file(
    csv_path: str, xml_path: str, *, fps: float, target_fps: float | None = None
) -> tuple[np.ndarray, np.ndarray]:
    """
    Process one CSV file into a delta-feature bundle.

    Layout
    ------
      0:4   root_tilt_sincos  [4]
      4     delta_yaw         [1]
      5:7   contact_mask      [2]
      7:10  delta_trans_world [3]
      10    height            [1]
      11:40 dof               [29]
      40:69 delta_dof         [29]
      69    absolute_yaw      [1]  (auxiliary, not fed to model)

    Returns
    -------
      feat      : (T, 70)
      keypoints : (T, 2, 7) absolute hand keypoints aligned to feat frames
    """
    df    = pd.read_csv(csv_path)
    D2R   = np.pi / 180.0

    euler_all = df[_ROOT_ROT_COLS].values * D2R   # (T_all, 3): roll, pitch, yaw
    dof_all   = df[_DOF_COLS].values   * D2R      # (T_all, 29)

    # Keypoints: prefer precomputed palm columns; otherwise derive them via the
    # same MuJoCo FK used for the feet (wrist_yaw body + _HAND_OFFSETS), so the
    # palm condition is self-consistent with the model's own convention.
    has_palm_cols = all(c in df.columns for c in _HAND_KP_COLS)
    if has_palm_cols:
        keypoints_abs = _load_keypoints_abs(df)       # (T_all, 2, 7)
        root_pos, ankle_l, ankle_r = _fk_feet(csv_path, xml_path)
    else:
        root_pos, ankle_l, ankle_r, keypoints_abs = _fk_feet(
            csv_path, xml_path, with_hands=True
        )
    # root_pos: (T_all, 3)

    eff_fps = fps
    if target_fps and abs(target_fps - fps) > 1e-6:
        (euler_all, dof_all, root_pos, ankle_l, ankle_r,
         keypoints_abs) = _resample_uniform(
            euler_all, dof_all, root_pos, ankle_l, ankle_r, keypoints_abs,
            src_fps=fps, tgt_fps=target_fps,
        )
        eff_fps = target_fps

    return _build_feature(
        euler_all, dof_all, root_pos, ankle_l, ankle_r, keypoints_abs, csv_path,
        fps=eff_fps,
    )


def _build_feature(
    euler_all: np.ndarray,      # (T_all, 3) roll, pitch, yaw [rad]
    dof_all: np.ndarray,        # (T_all, 29) [rad]
    root_pos: np.ndarray,       # (T_all, 3) pelvis world pos [m]
    ankle_l: np.ndarray,        # (T_all, 3) left ankle world pos [m]
    ankle_r: np.ndarray,        # (T_all, 3) right ankle world pos [m]
    keypoints_abs: np.ndarray,  # (T_all, 2, 7) palm pos_m + quat_wxyz
    src_name: str = "",
    *,
    fps: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Assemble the (T, 70) delta feature from FK outputs (backend-agnostic).

    Shared by the MuJoCo (per-file) and torch-GPU (batched) processing paths so
    both produce byte-identical feature layouts.
    """
    T_all = euler_all.shape[0]
    T = T_all - 1   # output frames
    if T < 1:
        raise ValueError(f"CSV too short (need ≥2 rows): {src_name}")

    feat = np.zeros((T, FEAT_DIM), dtype=np.float32)

    roll  = euler_all[:, 0]   # (T_all,)
    pitch = euler_all[:, 1]
    yaw   = euler_all[:, 2]

    # 0:4  root_tilt_sincos — zero-centred (sin, cos-1)
    feat[:, 0] = np.sin(roll[:T])
    feat[:, 1] = np.cos(roll[:T]) - 1.0
    feat[:, 2] = np.sin(pitch[:T])
    feat[:, 3] = np.cos(pitch[:T]) - 1.0

    # 4  delta_yaw
    feat[:, 4] = _angle_diff(yaw[1:], yaw[:T])

    # 5:7  contact_mask (binary AND of vel & height). fps must be the source
    # frame rate: the velocity gate is in m/s (see CONTACT_LABEL_VERSION v3).
    feat[:, 5] = _foot_contacts(ankle_l, fps=fps)   # left
    feat[:, 6] = _foot_contacts(ankle_r, fps=fps)   # right

    # 7:10  delta_trans_world (world-space, heading alignment deferred to Dataset)
    feat[:, 7:10] = root_pos[1:] - root_pos[:T]

    # 10  height (absolute root z)
    feat[:, 10] = root_pos[:T, 2]

    # 11:40  dof (rad)
    feat[:, 11:40] = dof_all[:T]

    # 40:69  delta_dof — wrapped to [-π, π]
    feat[:, 40:69] = _angle_diff(dof_all[1:], dof_all[:T])

    # 69  auxiliary absolute_yaw (used in Dataset for heading alignment)
    feat[:, 69] = yaw[:T]

    return feat, keypoints_abs[:T]


# ---------------------------------------------------------------------------
# Parallel processing
# ---------------------------------------------------------------------------

def _process_one(args: tuple) -> tuple[str, str | None]:
    """Worker: (csv_path, out_path, xml_path, fps, target_fps) → (csv_path, error_or_None)."""
    csv_path, out_path, xml_path, fps, target_fps = args
    eff_fps = target_fps if target_fps else fps
    try:
        feat, keypoints = process_file(
            csv_path, xml_path, fps=fps, target_fps=target_fps
        )
        np.savez_compressed(
            out_path,
            feat=feat.astype(np.float32),
            keypoints=keypoints.astype(np.float32),
            contact_label_version=np.int32(CONTACT_LABEL_VERSION),
            # rate of the frames actually stored (post-resample)
            source_fps=np.float32(eff_fps),
        )
        return csv_path, None
    except Exception as exc:
        return csv_path, str(exc)


# ---------------------------------------------------------------------------
# GPU (torch) batched processing — replaces the per-frame MuJoCo FK CPU loop
# with a single vectorised FK on the GPU. Verified numerically identical to the
# MuJoCo path (max |Δ| ~2e-7 m on root/ankles/hands, i.e. float32 precision).
# ---------------------------------------------------------------------------

def _fk_gpu_single(csv_path: str, fk, device):
    """Torch-GPU FK for one clip. Returns numpy FK outputs matching _fk_feet.

    Returns (euler_all, dof_all, root_pos, ankle_l, ankle_r, keypoints_abs),
    where keypoints_abs is FK-derived when the CSV lacks palm columns, else
    loaded from the CSV (same policy as process_file).
    """
    import torch
    from Prior_Recon.Masked_Flow.loss.g1_kinematics import euler_xyz_to_matrix

    df = pd.read_csv(csv_path)
    D2R = np.pi / 180.0
    euler_all = df[_ROOT_ROT_COLS].values * D2R      # (T,3)
    dof_all = df[_DOF_COLS].values * D2R             # (T,29)
    trans = df[_ROOT_POS_COLS].values * 0.01         # (T,3) cm->m
    T = len(df)

    rp = torch.as_tensor(trans, dtype=torch.float32, device=device).view(1, T, 3)
    rr = euler_xyz_to_matrix(
        torch.as_tensor(euler_all[:, 0], dtype=torch.float32, device=device),
        torch.as_tensor(euler_all[:, 1], dtype=torch.float32, device=device),
        torch.as_tensor(euler_all[:, 2], dtype=torch.float32, device=device),
    ).view(1, T, 3, 3)
    dp = torch.as_tensor(dof_all, dtype=torch.float32, device=device).view(1, T, 29)

    out = fk(rp, rr, dp)
    root_pos = out["global_translation"][0, :, 0].cpu().numpy()      # body 0 = pelvis
    foot = out["foot_translation"][0].cpu().numpy()                  # (T,2,3)
    ankle_l, ankle_r = foot[:, 0], foot[:, 1]

    has_palm_cols = all(c in df.columns for c in _HAND_KP_COLS)
    if has_palm_cols:
        keypoints_abs = _load_keypoints_abs(df)                     # (T,2,7)
    else:
        hand_pos = out["hand_translation"][0].cpu().numpy()          # (T,2,3)
        # wrist-body world quat (wxyz) via rotation matrix -> quat
        hand_rot = out["hand_rotation_mat"][0].cpu().numpy()         # (T,2,3,3)
        from scipy.spatial.transform import Rotation as R
        keypoints_abs = np.zeros((T, 2, 7), dtype=np.float32)
        keypoints_abs[:, :, :3] = hand_pos
        for k in range(2):
            q_xyzw = R.from_matrix(hand_rot[:, k]).as_quat()         # (T,4) xyzw
            keypoints_abs[:, k, 3:] = np.roll(q_xyzw, 1, axis=1)     # -> wxyz

    return euler_all, dof_all, root_pos, ankle_l, ankle_r, keypoints_abs


def process_on_gpu(tasks: list, device: str = "cuda") -> list:
    """Process all tasks on the GPU sequentially (one CUDA context).

    tasks: list of (csv_path, out_path, xml_path, fps, target_fps).
    Returns list of (csv_path, error_or_None).
    """
    import torch
    from Prior_Recon.Masked_Flow.loss.g1_kinematics import (
        G129DeltaForwardKinematics,
    )

    dev = torch.device(device)
    fk = G129DeltaForwardKinematics().to(dev).eval()
    results = []
    with torch.no_grad():
        for i, (csv_path, out_path, _xml, fps, target_fps) in enumerate(tasks):
            try:
                (euler_all, dof_all, root_pos, ankle_l, ankle_r,
                 keypoints_abs) = _fk_gpu_single(csv_path, fk, dev)
                eff_fps = fps
                if target_fps and abs(target_fps - fps) > 1e-6:
                    (euler_all, dof_all, root_pos, ankle_l, ankle_r,
                     keypoints_abs) = _resample_uniform(
                        euler_all, dof_all, root_pos, ankle_l, ankle_r,
                        keypoints_abs, src_fps=fps, tgt_fps=target_fps,
                    )
                    eff_fps = target_fps
                feat, keypoints = _build_feature(
                    euler_all, dof_all, root_pos, ankle_l, ankle_r,
                    keypoints_abs, csv_path, fps=eff_fps,
                )
                np.savez_compressed(
                    out_path,
                    feat=feat.astype(np.float32),
                    keypoints=keypoints.astype(np.float32),
                    contact_label_version=np.int32(CONTACT_LABEL_VERSION),
                    source_fps=np.float32(eff_fps),
                )
                results.append((csv_path, None))
            except Exception as exc:
                results.append((csv_path, str(exc)))
            if (i + 1) % 500 == 0:
                print(f"  [gpu] {i + 1}/{len(tasks)}", flush=True)
    return results


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="CSV → 70D delta motion features")
    parser.add_argument("--data-root",  required=True,
                        help="Root directory containing **/**.csv motion files")
    parser.add_argument(
        "--match-root",
        default=None,
        help="Optional root of selected files (for example large_selected_npy). "
             "When set, only process CSVs whose relative path matches a file under this root.",
    )
    parser.add_argument("--out-dir",    required=True,
                        help="Output root; files saved under "
                             f"<out-dir>/delta_feat_v{CONTACT_LABEL_VERSION}/<relative>.npz")
    parser.add_argument("--xml-path",   required=True,
                        help="Path to MuJoCo g1.xml skeleton file")
    parser.add_argument("--fps",        type=float, required=True,
                        help="TRUE frame rate of the source CSVs (sonic SOMA "
                             "corpus: 50). Required on purpose: the contact "
                             "velocity gate is in m/s, and a silently wrong "
                             "default fps was exactly the v2 label bug.")
    parser.add_argument("--target-fps", type=float, default=None,
                        help="Resample clips to this rate before feature "
                             "extraction (lerp + slerp). Use 30 to match the "
                             "training/deploy stack's assumed frame rate; "
                             "without it, 50 fps frames deployed as 30 fps "
                             "play every motion back at 0.6x speed. "
                             "Default: keep the source rate.")
    parser.add_argument("--workers",    type=int, default=4,
                        help="Number of parallel workers (default: 4, CPU path only)")
    parser.add_argument("--overwrite",  action="store_true",
                        help="Re-process even if output already exists")
    parser.add_argument("--device",     default="cpu",
                        help="'cpu' (MuJoCo per-frame FK + multiprocessing) or "
                             "'cuda[:N]' (batched torch FK on GPU, ~20x faster). "
                             "The two paths are numerically identical (~2e-7 m).")
    args = parser.parse_args()

    data_root  = Path(args.data_root)
    # Versioned subdir: regenerated data never lands in (or mixes with) an
    # older generation's directory.
    out_root   = Path(args.out_dir) / f"delta_feat_v{CONTACT_LABEL_VERSION}"
    out_root.mkdir(parents=True, exist_ok=True)
    match_root = Path(args.match_root) if args.match_root else None

    tasks: list[tuple[str, str, str]] = []
    n_candidates = 0
    n_missing_csv = 0

    if match_root is not None:
        selected_by_rel: dict[Path, Path] = {}
        for selected_path in sorted(match_root.rglob("*.npy")) + sorted(match_root.rglob("*.npz")):
            rel = selected_path.relative_to(match_root).with_suffix(".csv")
            selected_by_rel.setdefault(rel, selected_path)
        selected_files = list(selected_by_rel.values())
        if not selected_files:
            print(f"[WARN] No selector files found under {match_root}")
            return
        for selected_path in selected_files:
            rel = selected_path.relative_to(match_root)
            csv_path = data_root / rel.with_suffix(".csv")
            n_candidates += 1
            if not csv_path.exists():
                n_missing_csv += 1
                continue
            out_path = out_root / rel.with_suffix(".npz")
            out_path.parent.mkdir(parents=True, exist_ok=True)
            if out_path.exists() and not args.overwrite:
                continue
            tasks.append(
                (str(csv_path), str(out_path), args.xml_path, args.fps,
                 args.target_fps)
            )
    else:
        csv_files = sorted(data_root.rglob("*.csv"))
        if not csv_files:
            print(f"[WARN] No CSV files found under {data_root}")
            return
        for csv_path in csv_files:
            rel = csv_path.relative_to(data_root)
            n_candidates += 1
            out_path = out_root / rel.with_suffix(".npz")
            out_path.parent.mkdir(parents=True, exist_ok=True)
            if out_path.exists() and not args.overwrite:
                continue
            tasks.append(
                (str(csv_path), str(out_path), args.xml_path, args.fps,
                 args.target_fps)
            )

    print(
        f"Processing {len(tasks)} / {n_candidates} files  →  {out_root}"
        + (f"  (missing_csv={n_missing_csv})" if match_root is not None else "")
    )

    n_ok = n_err = 0
    if args.device.startswith("cuda"):
        print(f"[device] GPU path (batched torch FK) on {args.device}")
        for path, err in process_on_gpu(tasks, device=args.device):
            if err:
                n_err += 1
                print(f"[ERROR] {path}: {err}")
            else:
                n_ok += 1
    elif args.workers <= 1:
        for t in tasks:
            path, err = _process_one(t)
            if err:
                n_err += 1
                print(f"[ERROR] {path}: {err}")
            else:
                n_ok += 1
                print(f"[OK]    {path}")
    else:
        with mp.Pool(args.workers) as pool:
            for path, err in pool.imap_unordered(_process_one, tasks):
                if err:
                    n_err += 1
                    print(f"[ERROR] {path}: {err}")
                else:
                    n_ok += 1

    print(f"Done. ok={n_ok} err={n_err}")


if __name__ == "__main__":
    main()
