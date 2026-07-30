#!/usr/bin/env python3
"""
process_delta_motion_parquet.py
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Parquet (joint_pos + 5 robot poses) -> TextOp-style delta motion feature.

Usage
-----
    python process_delta_motion_parquet.py \\
        --data-root /path/to/parquet_dir \\
        --out-dir /path/to/output_dir

Input mapping
-------------
The input parquet clips store:

  robot0 = right_palm
  robot1 = left_palm
  robot2 = pelvis
  robot3 = right_foot
  robot4 = left_foot

Output layout
-------------
  feat      : (T, 70) float32
      0:4   root_tilt_sincos
      4     delta_yaw
      5:7   contact_mask      [left, right]
      7:10  delta_trans_world
      10    height
      11:40 dof
      40:69 delta_dof
      69    absolute_yaw      (auxiliary, not fed to model)

  keypoints : (T, 2, 7) float32
      kp[0] = left_palm   <- robot1
      kp[1] = right_palm  <- robot0
"""
from __future__ import annotations

import argparse
import multiprocessing as mp
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial.transform import Rotation as R

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from Prior_Recon.Masked_Flow.dataset.process_delta_motion import (
    CONTACT_LABEL_VERSION,
    _foot_contacts,
)

N_DOF = 29
FEAT_DIM = 70

_LEFT_PALM_ROBOT_ID = 1
_RIGHT_PALM_ROBOT_ID = 0
_PELVIS_ROBOT_ID = 2
_RIGHT_FOOT_ROBOT_ID = 3
_LEFT_FOOT_ROBOT_ID = 4


def _angle_diff(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    d = a - b
    return (d + np.pi) % (2 * np.pi) - np.pi


def _axis_angle_to_quat_wxyz(aa: np.ndarray) -> np.ndarray:
    """Convert (..., 3) axis-angle to (..., 4) quaternion in wxyz order."""
    theta = np.linalg.norm(aa, axis=-1, keepdims=True)
    small = theta < 1e-8
    half = 0.5 * theta
    sinc_half = np.where(small, 0.5, np.sin(half) / np.where(small, 1.0, theta))
    xyz = aa * sinc_half
    w = np.cos(half)
    return np.concatenate([w, xyz], axis=-1).astype(np.float32)


def _stack_col(df: pd.DataFrame, col: str) -> np.ndarray:
    return np.stack([np.asarray(v, dtype=np.float32) for v in df[col].values])


def _rotvec_to_euler_xyz(rotvec: np.ndarray) -> np.ndarray:
    return R.from_rotvec(rotvec).as_euler("xyz").astype(np.float32)


def process_episode(
    ep_df: pd.DataFrame, *, fps: float
) -> tuple[np.ndarray, np.ndarray]:
    """Build (feat[T, 70], keypoints[T, 2, 7]) for one episode."""
    if "frame_index" in ep_df.columns:
        ep_df = ep_df.sort_values("frame_index")

    dof_all = _stack_col(ep_df, "joint_pos")
    if dof_all.shape[1] != N_DOF:
        raise ValueError(f"joint_pos dim {dof_all.shape[1]} != {N_DOF}")

    T_all = dof_all.shape[0]
    if T_all < 2:
        raise ValueError(f"Episode too short: T={T_all}")

    left_palm_pos = _stack_col(ep_df, f"robot{_LEFT_PALM_ROBOT_ID}_eef_pos")
    right_palm_pos = _stack_col(ep_df, f"robot{_RIGHT_PALM_ROBOT_ID}_eef_pos")
    left_palm_aa = _stack_col(ep_df, f"robot{_LEFT_PALM_ROBOT_ID}_eef_rot_axis_angle")
    right_palm_aa = _stack_col(ep_df, f"robot{_RIGHT_PALM_ROBOT_ID}_eef_rot_axis_angle")

    pelvis_pos = _stack_col(ep_df, f"robot{_PELVIS_ROBOT_ID}_eef_pos")
    pelvis_aa = _stack_col(ep_df, f"robot{_PELVIS_ROBOT_ID}_eef_rot_axis_angle")
    left_foot_pos = _stack_col(ep_df, f"robot{_LEFT_FOOT_ROBOT_ID}_eef_pos")
    right_foot_pos = _stack_col(ep_df, f"robot{_RIGHT_FOOT_ROBOT_ID}_eef_pos")

    hand_pos = np.stack([left_palm_pos, right_palm_pos], axis=1)
    hand_aa = np.stack([left_palm_aa, right_palm_aa], axis=1)
    hand_quat = _axis_angle_to_quat_wxyz(hand_aa)
    keypoints_abs = np.concatenate([hand_pos, hand_quat], axis=-1)

    pelvis_euler = _rotvec_to_euler_xyz(pelvis_aa)
    roll = pelvis_euler[:, 0]
    pitch = pelvis_euler[:, 1]
    yaw = pelvis_euler[:, 2]

    T = T_all - 1
    feat = np.zeros((T, FEAT_DIM), dtype=np.float32)

    feat[:, 0] = np.sin(roll[:T])
    feat[:, 1] = np.cos(roll[:T]) - 1.0
    feat[:, 2] = np.sin(pitch[:T])
    feat[:, 3] = np.cos(pitch[:T]) - 1.0
    feat[:, 4] = _angle_diff(yaw[1:], yaw[:T])
    feat[:, 5] = _foot_contacts(left_foot_pos, fps=fps)
    feat[:, 6] = _foot_contacts(right_foot_pos, fps=fps)
    feat[:, 7:10] = pelvis_pos[1:] - pelvis_pos[:T]
    feat[:, 10] = pelvis_pos[:T, 2]
    feat[:, 11:40] = dof_all[:T]
    feat[:, 40:69] = _angle_diff(dof_all[1:], dof_all[:T])
    feat[:, 69] = yaw[:T]

    return feat, keypoints_abs[:T].astype(np.float32)


def _process_one(args: tuple[str, str, bool, float]) -> tuple[str, str]:
    parquet_path, out_dir, overwrite, fps = args
    try:
        df = pd.read_parquet(parquet_path)
        stem = Path(parquet_path).stem
        out_subdir = Path(out_dir) / stem
        out_subdir.mkdir(parents=True, exist_ok=True)

        if "episode_index" not in df.columns:
            raise ValueError("parquet missing 'episode_index' column")

        episodes = sorted(df["episode_index"].unique())
        n_written = n_skipped = n_errors = 0
        for ep in episodes:
            out_path = out_subdir / f"ep_{int(ep):05d}.npz"
            if out_path.exists() and not overwrite:
                n_skipped += 1
                continue
            ep_df = df[df["episode_index"] == ep]
            try:
                feat, keypoints = process_episode(ep_df, fps=fps)
            except Exception as exc:  # noqa: BLE001
                n_errors += 1
                print(f"[WARN] {parquet_path} ep={int(ep)}: {exc}")
                continue
            np.savez_compressed(
                out_path,
                feat=feat.astype(np.float32),
                keypoints=keypoints.astype(np.float32),
                contact_label_version=np.int32(CONTACT_LABEL_VERSION),
                source_fps=np.float32(fps),
            )
            n_written += 1
        return parquet_path, f"written={n_written} skipped={n_skipped} errors={n_errors}"
    except Exception as exc:  # noqa: BLE001
        return parquet_path, f"ERROR: {exc}"


def main() -> None:
    parser = argparse.ArgumentParser(description="Parquet -> 70D delta motion features")
    parser.add_argument(
        "--data-root",
        required=True,
        help="Directory containing *.parquet files",
    )
    parser.add_argument(
        "--out-dir",
        required=True,
        help="Output directory; one <stem>/ep_XXXXX.npz per episode",
    )
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--fps", type=float, required=True,
        help="TRUE frame rate of the parquet episodes; the contact velocity "
             "gate is in m/s. Required on purpose (no silent default).",
    )
    args = parser.parse_args()

    data_root = Path(args.data_root)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    parquet_files = sorted(data_root.glob("*.parquet"))
    if not parquet_files:
        print(f"[WARN] No parquet files under {data_root}")
        return

    tasks = [(str(p), str(out_dir), args.overwrite, args.fps) for p in parquet_files]
    print(f"Processing {len(tasks)} parquet files -> {out_dir}")

    if args.workers <= 1:
        for t in tasks:
            path, msg = _process_one(t)
            print(f"[OK] {path}: {msg}")
    else:
        with mp.Pool(args.workers) as pool:
            for path, msg in pool.imap_unordered(_process_one, tasks):
                print(f"[OK] {path}: {msg}")

    print("Done.")


if __name__ == "__main__":
    main()
