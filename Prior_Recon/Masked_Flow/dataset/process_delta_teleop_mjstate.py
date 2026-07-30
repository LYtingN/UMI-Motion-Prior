#!/usr/bin/env python3
"""
process_delta_teleop_mjstate.py
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Teleop-schema parquet -> TextOp-style delta69 motion feature bundle, using the
embedded ``observation.sim.mujoco_state`` as ground truth.

Why this instead of process_delta_teleop_parquet.py
---------------------------------------------------
The teleop clips shipped with GR00T (e.g. g1_pnpbottle.parquet) do NOT carry an
``observation.root_orientation`` column, so the old reconstruction path (FK +
stance-foot XY integration from ``observation.state``) is not applicable and is
also lossy. They DO carry ``observation.sim.mujoco_state``, which is the raw
MuJoCo physics state:

    mujoco_state[:, 0]       = sim time
    mujoco_state[:, 1:1+nq]  = qpos  = [root_xyz(3), root_quat_wxyz(4), 29 joints]
    mujoco_state[:, 1+nq:..] = qvel  (unused here)

The qpos layout is exactly the 36-D qpos that recon_delta69.py feeds into the
same G1 MJCF, so root pose and joints are exact GT — no FK/stance heuristics.

Hand keypoints (T, 2, 7) are derived by running MuJoCo FK on the wrist-yaw
bodies with the same ``_HAND_OFFSETS`` recon_delta69.py uses, so the palm
condition is self-consistent with the model's own keypoint definition.

Output layout (identical to process_delta_motion.py / data_test/*)
------------------------------------------------------------------
  feat      : (T, 70) float32
      0:4   root_tilt_sincos   4: delta_yaw      5:7  contact_mask [left,right]
      7:10  delta_trans_world  10: height        11:40 dof
      40:69 delta_dof          69: absolute_yaw  (auxiliary)
  keypoints : (T, 2, 7) float32  world-frame [left, right] palm pos_m + quat_wxyz

Clips are resampled from the source rate to --target-fps (default 30, the
Masked_Flow training rate) before deltas are computed.

Usage
-----
  python Prior_Recon/Masked_Flow/dataset/process_delta_teleop_mjstate.py \
      --parquet GR00T-WholeBodyControl/decoupled_wbc/tests/replay_data/g1_pnpbottle.parquet \
      --out-dir Prior_Recon/Masked_Flow/data_test/pnp-bottle
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial.transform import Rotation as R
from scipy.spatial.transform import Slerp

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from Prior_Recon.Masked_Flow.dataset.process_delta_motion import (
    CONTACT_LABEL_VERSION,
    _angle_diff,
    _foot_contacts,
    _resolve_mjcf_xml,
)
from Prior_Recon.Masked_Flow.utils import default_g1_mjcf_xml_path

N_DOF = 29
FEAT_DIM = 70
_NQ = 7 + N_DOF  # 3 root pos + 4 root quat + 29 joints

# Teleop clips recorded from a simulator carry a reset/spawn frame at index 0:
# the robot sits at the default pose/origin for one frame, then teleports to the
# real first pose. The 0->1 transition is nonphysical for a single 1/30 s step
# (root jumps ~1 m, yaw flips ~20 deg, joints snap ~1 rad), and because feat is
# built from adjacent-frame deltas that garbage lands in feat[0] and makes the
# recon twitch for the first frames. We detect and drop those leading reset
# frames from the raw qpos BEFORE FK/resample so nothing downstream sees them.
_RESET_XY_THR = 0.05  # m/frame; >1.5 m/s @30fps is nonphysical for a teleop start
_RESET_YAW_THR = np.deg2rad(5.0)  # rad/frame; >150 deg/s
_RESET_DOF_THR = 0.2  # rad/frame; nonphysical single-step joint snap
_RESET_REL_FACTOR = 20.0  # also require the jump to dwarf the clip's own median
_RESET_MAX_DROP = 5  # safety cap so a genuinely fast clip is never fully eaten


def _drop_reset_frames(qpos: np.ndarray, times: np.ndarray) -> tuple[np.ndarray, np.ndarray, int]:
    """Strip leading simulator-reset frames from a raw qpos/times clip.

    A leading transition is flagged as a reset when its root-XY step, yaw step,
    or max joint step is both above an absolute nonphysical floor AND far larger
    than the clip's own median step (so a uniformly fast clip is not trimmed).
    Returns the trimmed (qpos, times) and the number of frames dropped.
    """
    n_dropped = 0
    while qpos.shape[0] - n_dropped > 2 and n_dropped < _RESET_MAX_DROP:
        q = qpos[n_dropped:]
        dxy = np.linalg.norm(np.diff(q[:, :2], axis=0), axis=1)
        yaw = R.from_quat(np.roll(q[:, 3:7], -1, axis=-1)).as_euler("xyz")[:, 2]
        dyaw = np.abs(_angle_diff(yaw[1:], yaw[:-1]))
        ddof = np.abs(_angle_diff(q[1:, 7:], q[:-1, 7:])).max(axis=1)
        xy_bad = dxy[0] > max(_RESET_XY_THR, _RESET_REL_FACTOR * np.median(dxy))
        yaw_bad = dyaw[0] > max(_RESET_YAW_THR, _RESET_REL_FACTOR * np.median(dyaw))
        dof_bad = ddof[0] > max(_RESET_DOF_THR, _RESET_REL_FACTOR * np.median(ddof))
        if not (xy_bad or yaw_bad or dof_bad):
            break
        n_dropped += 1
    if n_dropped:
        qpos = qpos[n_dropped:]
        times = times[n_dropped:]
    return qpos, times, n_dropped

_HAND_BODY_NAMES = ["left_wrist_yaw_link", "right_wrist_yaw_link"]
# Palm offset in each wrist-yaw body frame, matching recon_delta69._HAND_OFFSETS.
_HAND_OFFSETS = np.array(
    [
        [0.0415, 0.0030, 0.0],
        [0.0415, -0.0030, 0.0],
    ],
    dtype=np.float64,
)
_FOOT_BODY_NAMES = ["left_ankle_roll_link", "right_ankle_roll_link"]


def _stack_col(df: pd.DataFrame, col: str) -> np.ndarray:
    return np.stack([np.asarray(v, dtype=np.float64) for v in df[col].values])


def _extract_qpos(ep_df: pd.DataFrame) -> np.ndarray:
    """Pull the (T, 36) qpos block out of observation.sim.mujoco_state."""
    if "observation.sim.mujoco_state" not in ep_df.columns:
        raise ValueError("parquet missing 'observation.sim.mujoco_state' column")
    state = _stack_col(ep_df, "observation.sim.mujoco_state")
    if "observation.sim.mujoco_state_len" in ep_df.columns:
        state_len = int(ep_df["observation.sim.mujoco_state_len"].iloc[0])
        state = state[:, :state_len]
    if state.shape[1] < 1 + _NQ:
        raise ValueError(
            f"mujoco_state width {state.shape[1]} < 1+{_NQ}; cannot read qpos"
        )
    return state[:, 1 : 1 + _NQ]  # [root_xyz(3), root_quat_wxyz(4), 29 joints]


def _fk_world(
    qpos: np.ndarray,
    xml_path: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """FK over the whole clip.

    Returns
    -------
    hand_pos  : (T, 2, 3) world palm positions (wrist body + _HAND_OFFSETS)
    hand_quat : (T, 2, 4) world palm quats (wxyz), = wrist body orientation
    foot_pos  : (T, 2, 3) world ankle positions (for contact detection)
    """
    import mujoco

    resolved_xml, temp_xml = _resolve_mjcf_xml(xml_path)
    try:
        model = mujoco.MjModel.from_xml_path(resolved_xml)
        data = mujoco.MjData(model)

        hand_ids = []
        for name in _HAND_BODY_NAMES:
            bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)
            if bid < 0:
                raise RuntimeError(f"Body '{name}' not found in {resolved_xml}")
            hand_ids.append(bid)
        foot_ids = []
        for name in _FOOT_BODY_NAMES:
            bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)
            if bid < 0:
                raise RuntimeError(f"Body '{name}' not found in {resolved_xml}")
            foot_ids.append(bid)

        T = qpos.shape[0]
        hand_pos = np.zeros((T, 2, 3), dtype=np.float64)
        hand_quat = np.zeros((T, 2, 4), dtype=np.float64)
        foot_pos = np.zeros((T, 2, 3), dtype=np.float64)
        for t in range(T):
            data.qpos[:_NQ] = qpos[t]
            mujoco.mj_forward(model, data)
            for k, bid in enumerate(hand_ids):
                rot = data.xmat[bid].reshape(3, 3)
                hand_pos[t, k] = data.xpos[bid] + rot @ _HAND_OFFSETS[k]
                hand_quat[t, k] = data.xquat[bid]
            for k, bid in enumerate(foot_ids):
                foot_pos[t, k] = data.xpos[bid]
        return hand_pos, hand_quat, foot_pos
    finally:
        if temp_xml is not None:
            Path(temp_xml).unlink(missing_ok=True)


def _resample(
    times: np.ndarray,
    target_fps: float,
    qpos: np.ndarray,
    hand_pos: np.ndarray,
    hand_quat: np.ndarray,
    foot_pos: np.ndarray,
) -> tuple[np.ndarray, ...]:
    """Linear/Slerp resampling of all absolute signals onto a target_fps grid."""
    t0, t1 = float(times[0]), float(times[-1])
    n_out = int(np.floor((t1 - t0) * target_fps)) + 1
    t_out = t0 + np.arange(n_out) / target_fps

    def lerp(arr: np.ndarray) -> np.ndarray:
        flat = arr.reshape(arr.shape[0], -1)
        out = np.stack(
            [np.interp(t_out, times, flat[:, i]) for i in range(flat.shape[1])], axis=1
        )
        return out.reshape((n_out,) + arr.shape[1:])

    def slerp_wxyz(quat_wxyz: np.ndarray) -> np.ndarray:
        rot = R.from_quat(np.roll(quat_wxyz, -1, axis=-1))
        out_xyzw = Slerp(times, rot)(t_out).as_quat()
        return np.roll(out_xyzw, 1, axis=-1)

    root_pos = lerp(qpos[:, :3])
    root_quat = slerp_wxyz(qpos[:, 3:7])
    joints = lerp(qpos[:, 7:])
    hand_pos_r = lerp(hand_pos)
    hand_quat_r = np.stack([slerp_wxyz(hand_quat[:, h]) for h in range(2)], axis=1)
    foot_pos_r = lerp(foot_pos)
    qpos_r = np.concatenate([root_pos, root_quat, joints], axis=1)
    return qpos_r, hand_pos_r, hand_quat_r, foot_pos_r


def process_episode(
    ep_df: pd.DataFrame,
    xml_path: str,
    target_fps: float | None,
) -> tuple[np.ndarray, np.ndarray, float]:
    """Build (feat[T, 70], keypoints[T, 2, 7], effective_fps) for one episode."""
    if "frame_index" in ep_df.columns:
        ep_df = ep_df.sort_values("frame_index")

    qpos = _extract_qpos(ep_df)
    times = np.asarray(ep_df["timestamp"].values, dtype=np.float64)
    if qpos.shape[0] < 2:
        raise ValueError(f"Episode too short: T={qpos.shape[0]}")

    qpos, times, n_reset = _drop_reset_frames(qpos, times)
    if n_reset:
        print(f"    dropped {n_reset} leading reset frame(s) before feature build")
    if qpos.shape[0] < 2:
        raise ValueError(
            f"Episode too short after dropping {n_reset} reset frame(s): T={qpos.shape[0]}"
        )

    hand_pos, hand_quat, foot_pos = _fk_world(qpos, xml_path)

    if target_fps:
        qpos, hand_pos, hand_quat, foot_pos = _resample(
            times, target_fps, qpos, hand_pos, hand_quat, foot_pos
        )

    root_pos = qpos[:, :3]
    root_rot = R.from_quat(np.roll(qpos[:, 3:7], -1, axis=-1))
    joints = qpos[:, 7:]
    euler = root_rot.as_euler("xyz")
    roll, pitch, yaw = euler[:, 0], euler[:, 1], euler[:, 2]

    # Frame rate the frames actually have after (optional) resampling; the
    # contact velocity gate is in m/s so it must see this, not a fixed 30.
    eff_fps = (
        float(target_fps)
        if target_fps
        else 1.0 / float(np.median(np.diff(times)))
    )

    T = root_pos.shape[0] - 1
    feat = np.zeros((T, FEAT_DIM), dtype=np.float32)
    feat[:, 0] = np.sin(roll[:T])
    feat[:, 1] = np.cos(roll[:T]) - 1.0
    feat[:, 2] = np.sin(pitch[:T])
    feat[:, 3] = np.cos(pitch[:T]) - 1.0
    feat[:, 4] = _angle_diff(yaw[1:], yaw[:T])
    feat[:, 5] = _foot_contacts(foot_pos[:, 0], fps=eff_fps)
    feat[:, 6] = _foot_contacts(foot_pos[:, 1], fps=eff_fps)
    feat[:, 7:10] = root_pos[1:] - root_pos[:T]
    feat[:, 10] = root_pos[:T, 2]
    feat[:, 11:40] = joints[:T]
    feat[:, 40:69] = _angle_diff(joints[1:], joints[:T])
    feat[:, 69] = yaw[:T]

    keypoints = np.concatenate([hand_pos, hand_quat], axis=-1)[:T]
    return feat, keypoints.astype(np.float32), eff_fps


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Teleop parquet (mujoco_state) -> 70D delta motion features"
    )
    parser.add_argument("--parquet", required=True, help="Parquet file or directory")
    parser.add_argument(
        "--out-dir",
        default=None,
        help="Output directory (default: same directory as each parquet)",
    )
    parser.add_argument(
        "--target-fps",
        type=float,
        default=30.0,
        help="Resample to this rate before computing deltas; 0 disables resampling",
    )
    parser.add_argument("--xml-path", default=str(default_g1_mjcf_xml_path()))
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    src = Path(args.parquet).expanduser()
    files = sorted(src.rglob("*.parquet")) if src.is_dir() else [src]
    if not files:
        raise FileNotFoundError(f"No parquet files under {src}")

    target_fps = args.target_fps if args.target_fps > 0 else None
    for parquet_path in files:
        out_dir = Path(args.out_dir).expanduser() if args.out_dir else parquet_path.parent
        out_dir.mkdir(parents=True, exist_ok=True)
        df = pd.read_parquet(parquet_path)
        if "episode_index" not in df.columns:
            raise ValueError(f"{parquet_path}: missing 'episode_index' column")
        for ep in sorted(df["episode_index"].unique()):
            out_path = out_dir / f"ep_{int(ep):05d}.npz"
            if out_path.exists() and not args.overwrite:
                print(f"[SKIP] {out_path} exists (use --overwrite)")
                continue
            feat, keypoints, eff_fps = process_episode(
                df[df["episode_index"] == ep], args.xml_path, target_fps
            )
            np.savez_compressed(
                out_path,
                feat=feat.astype(np.float32),
                keypoints=keypoints.astype(np.float32),
                contact_label_version=np.int32(CONTACT_LABEL_VERSION),
                source_fps=np.float32(eff_fps),
            )
            print(
                f"[OK] {parquet_path.name} ep={int(ep)} -> {out_path}  "
                f"frames={feat.shape[0]}"
                + (f" (resampled to {target_fps:g} fps)" if target_fps else "")
            )
    print("Done.")


if __name__ == "__main__":
    main()
