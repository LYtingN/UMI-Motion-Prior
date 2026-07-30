#!/usr/bin/env python3
"""
process_delta_teleop_parquet.py
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Teleop-schema parquet (observation.state / observation.eef_state) ->
TextOp-style delta motion feature bundle, identical layout to
process_delta_motion.py output.

Input schema (per row)
----------------------
  observation.state (43,):
      [0:22]   12 leg + 3 waist + 7 left-arm joints (G1 qpos order)
      [22:29]  7 left-hand finger joints   (NOT body DOF -- skipped)
      [29:36]  7 right-arm joints
      [36:43]  7 right-hand finger joints  (NOT body DOF -- skipped)
  observation.eef_state (14,):
      [0:7]    left palm  pos(3) + quat_wxyz(4), ROOT-relative
      [7:14]   right palm pos(3) + quat_wxyz(4), ROOT-relative
  observation.root_orientation (4,): root quat wxyz (world)
  timestamp: seconds (source clips are 50 fps)

Root translation has no direct column.
  - Height: FK with the lowest ankle pinned near the floor.
  - XY: stance-foot anchoring -- the in-contact ankle is fixed in world,
    so the root delta is the negative of that ankle's root-relative motion.
    (teleop.smpl_joints pelvis XY is the OPERATOR's SMPL tracking, not robot
    odometry: legs visibly walk ~0.76 m in the first 10 s while SMPL pelvis
    moves ~0.1 m. Do NOT use it as the root trajectory.)

Output layout (same as process_delta_motion.py)
-----------------------------------------------
  feat      : (T, 70) float32
      0:4   root_tilt_sincos   4: delta_yaw      5:7  contact_mask
      7:10  delta_trans_world  10: height        11:40 dof
      40:69 delta_dof          69: absolute_yaw  (auxiliary)
  keypoints : (T, 2, 7) float32  world-frame [left, right] palm pos+quat_wxyz

Clips are resampled from the source rate to --target-fps (default 30,
the Masked_Flow training rate) before deltas are computed.

Usage
-----
  python Prior_Recon/Masked_Flow/dataset/process_delta_teleop_parquet.py \
      --parquet Prior_Recon/Masked_Flow/data/test/squat_pickup/episode_000000.parquet \
      --out-dir Prior_Recon/Masked_Flow/data/test/squat_pickup
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
_STATE_LEFT_BLOCK = slice(0, 22)    # legs + waist + left arm
_STATE_RIGHT_ARM = slice(29, 36)    # right arm (after 7 left-hand finger dims)
_ANKLE_BODY_NAMES = ["left_ankle_roll_link", "right_ankle_roll_link"]
_FLOOR_CLEARANCE = 0.002  # lowest ankle height used to recover root z


def _stack_col(df: pd.DataFrame, col: str) -> np.ndarray:
    return np.stack([np.asarray(v, dtype=np.float64) for v in df[col].values])


def _body_joints(state: np.ndarray) -> np.ndarray:
    """(T, 43) observation.state -> (T, 29) G1 body joints, finger dims dropped."""
    if state.shape[1] != 43:
        raise ValueError(f"observation.state dim {state.shape[1]} != 43")
    return np.concatenate(
        [state[:, _STATE_LEFT_BLOCK], state[:, _STATE_RIGHT_ARM]], axis=1
    )


def _fk_root_z_and_ankles(
    joints: np.ndarray,
    root_quat_wxyz: np.ndarray,
    xml_path: str,
) -> tuple[np.ndarray, np.ndarray]:
    """FK with root at origin; root z pins the lowest ankle at _FLOOR_CLEARANCE.

    Returns
    -------
    root_z      : (T,)    recovered pelvis height
    ankles_rel  : (T,2,3) ankle positions relative to the (zero-translation) root
    """
    import mujoco

    resolved_xml, temp_xml = _resolve_mjcf_xml(xml_path)
    try:
        model = mujoco.MjModel.from_xml_path(resolved_xml)
        data = mujoco.MjData(model)
    finally:
        if temp_xml is not None:
            Path(temp_xml).unlink(missing_ok=True)

    body_ids = []
    for name in _ANKLE_BODY_NAMES:
        bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)
        if bid < 0:
            raise RuntimeError(f"Body '{name}' not found in {xml_path}")
        body_ids.append(bid)

    T = joints.shape[0]
    root_z = np.zeros(T, dtype=np.float64)
    ankles_rel = np.zeros((T, 2, 3), dtype=np.float64)
    for t in range(T):
        data.qpos[:3] = 0.0
        data.qpos[3:7] = root_quat_wxyz[t]
        data.qpos[7 : 7 + N_DOF] = joints[t]
        mujoco.mj_forward(model, data)
        for k, bid in enumerate(body_ids):
            ankles_rel[t, k] = data.xpos[bid]
        root_z[t] = _FLOOR_CLEARANCE - min(ankles_rel[t, 0, 2], ankles_rel[t, 1, 2])
    return root_z, ankles_rel


def _root_xy_from_stance(
    ankles_rel: np.ndarray,
    root_z: np.ndarray,
    ht_thr: float = 0.08,
) -> np.ndarray:
    """Integrate world root XY by anchoring the stance foot.

    ankles_rel is FK output with zero root translation but the true root
    orientation, so a planted (world-fixed) ankle moving by d in the root
    frame means the root moved by -d in world.

    Returns (T, 2) world root XY starting at the origin.
    """
    T = ankles_rel.shape[0]
    ankle_z_world = ankles_rel[:, :, 2] + root_z[:, None]
    droot = np.zeros((T - 1, 2), dtype=np.float64)
    for t in range(T - 1):
        deltas = -(ankles_rel[t + 1, :, :2] - ankles_rel[t, :, :2])
        # The lowest foot is the planted one. Averaging both feet during
        # double support is biased: the rear foot rolls over its toes while
        # still below ht_thr, which cancels real forward motion.
        stance = int(np.argmin(ankle_z_world[t]))
        if ankle_z_world[t, stance] < ht_thr:
            droot[t] = deltas[stance]
    root_xy = np.zeros((T, 2), dtype=np.float64)
    root_xy[1:] = np.cumsum(droot, axis=0)
    return root_xy


def _resample_episode(
    times: np.ndarray,
    target_fps: float,
    root_pos: np.ndarray,
    root_quat_wxyz: np.ndarray,
    joints: np.ndarray,
    kp_pos: np.ndarray,
    kp_quat_wxyz: np.ndarray,
    ankles_world: np.ndarray,
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

    kp_quat_out = np.stack([slerp_wxyz(kp_quat_wxyz[:, h]) for h in range(2)], axis=1)
    return (
        lerp(root_pos),
        slerp_wxyz(root_quat_wxyz),
        lerp(joints),
        lerp(kp_pos),
        kp_quat_out,
        lerp(ankles_world),
    )


def process_episode(
    ep_df: pd.DataFrame,
    xml_path: str,
    target_fps: float | None,
) -> tuple[np.ndarray, np.ndarray, float]:
    """Build (feat[T, 70], keypoints[T, 2, 7], effective_fps) for one episode."""
    if "frame_index" in ep_df.columns:
        ep_df = ep_df.sort_values("frame_index")

    state = _stack_col(ep_df, "observation.state")
    joints = _body_joints(state)
    eef = _stack_col(ep_df, "observation.eef_state").reshape(-1, 2, 7)
    root_quat_wxyz = _stack_col(ep_df, "observation.root_orientation")
    times = np.asarray(ep_df["timestamp"].values, dtype=np.float64)

    T_all = joints.shape[0]
    if T_all < 2:
        raise ValueError(f"Episode too short: T={T_all}")

    root_rot = R.from_quat(np.roll(root_quat_wxyz, -1, axis=-1))
    root_z, ankles_rel = _fk_root_z_and_ankles(joints, root_quat_wxyz, xml_path)

    root_pos = np.zeros((T_all, 3), dtype=np.float64)
    root_pos[:, :2] = _root_xy_from_stance(ankles_rel, root_z)
    root_pos[:, 2] = root_z

    # Root-relative eef -> world keypoints
    kp_pos = np.stack(
        [root_rot.apply(eef[:, h, :3]) for h in range(2)], axis=1
    ) + root_pos[:, None, :]
    kp_quat = np.stack(
        [
            np.roll(
                (root_rot * R.from_quat(np.roll(eef[:, h, 3:], -1, axis=-1))).as_quat(),
                1,
                axis=-1,
            )
            for h in range(2)
        ],
        axis=1,
    )

    # FK ran with zero root translation, so world ankles = relative + root offset
    ankles_world = ankles_rel + root_pos[:, None, :]

    if target_fps:
        (
            root_pos,
            root_quat_wxyz,
            joints,
            kp_pos,
            kp_quat,
            ankles_world,
        ) = _resample_episode(
            times,
            target_fps,
            root_pos,
            root_quat_wxyz,
            joints,
            kp_pos,
            kp_quat,
            ankles_world,
        )
        root_rot = R.from_quat(np.roll(root_quat_wxyz, -1, axis=-1))

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
    feat[:, 5] = _foot_contacts(ankles_world[:, 0], fps=eff_fps)
    feat[:, 6] = _foot_contacts(ankles_world[:, 1], fps=eff_fps)
    feat[:, 7:10] = root_pos[1:] - root_pos[:T]
    feat[:, 10] = root_pos[:T, 2]
    feat[:, 11:40] = joints[:T]
    feat[:, 40:69] = _angle_diff(joints[1:], joints[:T])
    feat[:, 69] = yaw[:T]

    keypoints = np.concatenate([kp_pos, kp_quat], axis=-1)[:T]
    return feat, keypoints.astype(np.float32), eff_fps


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Teleop parquet -> 70D delta motion features"
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
