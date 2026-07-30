#!/usr/bin/env python3
"""
process_umi_ee.py
~~~~~~~~~~~~~~~~~
UMI DAS-Gripper VIO mcap -> EE-only delta bundle that ``recon_delta69.py`` can
reconstruct a full G1 body from.

recon_delta69 conditions the masked-flow model purely on the ``keypoints``
(T, 2, 7) field -- world-frame [left, right] palm pos + quat_wxyz -- and
generates the whole body. When the ``feat`` root/pose channels (dims 0:11 and
the aux yaw dim 69) are all zero, ``_has_missing_root_supervision`` fires and
recon floor-aligns + anchors the body at a standing init instead of reading a
GT root. That is exactly the UMI case: we have two hand trajectories and
nothing else. So this writer emits:

  feat      : (T, 70) float32, ALL ZERO except the contact channels are also
              left zero (they are ignored on the missing-root path). Kept as a
              well-formed 70-wide array so _load_delta_file / _load_delta_clip
              accept it unchanged.
  keypoints : (T, 2, 7) float32, world [left, right] palm pos + quat_wxyz.

Frame handling
--------------
The VIO eef_pose world frame is ALREADY gravity-aligned (z up): VIO seeds its
world with the IMU gravity estimate, so raw z is the true vertical. (Verified:
pnp_bottle's table grasp = gripper close lands at the global raw-z minimum, and
both hands start at equal raw-z.) So we do NOT re-rotate the poses. We only:
  1. resample both hands onto a common --target-fps grid (source ~30 fps);
  2. rigidly translate the pair: XY so frame-0 midpoint sits in front of the
     pelvis, and vertically per --anchor (start-mid -> frame-0 midpoint at the
     G1 torso_link height ~0.81 m; lowest-zero -> trajectory's lowest point
     at z=0). Only the vertical anchor is load-bearing (the model subtracts XY
     and re-centres per segment).

The VIO ``eef_pose`` lives in the DAS ``base_link`` frame, which is NOT recon's
palm anchor: on this gripper ``base_link`` sits ~13 cm behind + below the wrist
with a 15 deg pitch (see ``utils/das_calib``). That mount offset is BODY-FIXED
(right-multiplied), and recon relativizes quats via ``quat_wxyz_relative_to_first``
= ``inv(q0) @ q(t)`` (LEFT-multiplied), so the offset does NOT cancel -- a right
offset conjugates under a left relativization (``rel_palm = R_vp^-1 rel_vio R_vp``)
and the palm position ``p_vio + R_vio(t) @ t_vp`` is time-varying. So we apply
the FULL rigid VIO->palm transform per hand (``--calib das``) before recentring,
landing the trajectory on the exact palm frame recon was trained on. Pass
``--calib none`` to feed raw VIO poses through (old behaviour; only correct if
the sensor frame already coincides with the palm anchor).

Usage
-----
  python Prior_Recon/Masked_Flow/dataset/process_umi_ee.py \
      --mcap dataset/umi_origin/pnp_bottle/DAS-Gripper_..._vio.merged.mcap \
      --out-dir Prior_Recon/Masked_Flow/data_test/umi-pnp-bottle
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from Prior_Recon.Masked_Flow.dataset.process_delta_motion import (
    CONTACT_LABEL_VERSION,
)
from Prior_Recon.Masked_Flow.dataset.umi_mcap import (
    LEFT_ROBOT,
    RIGHT_ROBOT,
    common_time_grid,
    read_umi_mcap,
    resample_pose,
)
from Prior_Recon.Masked_Flow.utils.das_calib import vio_pose_to_palm

# Where we park the first-frame hand midpoint. XY (0.126) is the standing-init
# hand midpoint (FK on recon_delta69._STANDING_INIT_JOINTS puts the wrist palms
# at (0.126, +/-0.224, 0.707)); the vertical baseline is set to the G1
# torso_link height (0.814 m, also from FK on the standing pose) so the start
# anchor sits at chest/torso level rather than the lower resting wrist height.
_STANDING_HAND_MID = np.array([0.126, 0.0, 0.814], dtype=np.float64)


def build_keypoints(
    mcap_path: str | Path,
    target_fps: float = 30.0,
    anchor: str = "start-mid",
    calib: str = "das",
) -> tuple[np.ndarray, np.ndarray, float]:
    """Return (feat[T,70] zeros, keypoints[T,2,7], eff_fps).

    ``anchor`` picks the vertical reference:
      "start-mid"   frame-0 hand midpoint -> G1 torso_link height (z=0.814).
      "lowest-zero" the lowest point of the whole trajectory (either hand,
                    any frame) -> z=0 (floor = hands' lowest reach).
    XY is always parked so frame-0 midpoint sits in front of the pelvis; the
    model subtracts XY per segment, so only the vertical anchor is load-bearing.

    ``calib`` maps the VIO (DAS ``base_link``) pose onto recon's palm anchor:
      "das"  apply the full rigid VIO->palm transform per hand (see das_calib).
      "none" pass raw VIO poses through (only correct if the sensor frame
             already coincides with the palm anchor).
    """
    data = read_umi_mcap(mcap_path, with_video=False)
    if LEFT_ROBOT not in data.eef or RIGHT_ROBOT not in data.eef:
        raise ValueError("mcap missing robot0/robot1 eef_pose streams")

    grid = common_time_grid(
        [data.eef[LEFT_ROBOT].t, data.eef[RIGHT_ROBOT].t], target_fps
    )
    l_pos, l_quat = resample_pose(data.eef[LEFT_ROBOT], grid)
    r_pos, r_quat = resample_pose(data.eef[RIGHT_ROBOT], grid)

    # Map VIO (das_base_link) poses onto recon's palm anchor. This is a
    # body-fixed rigid offset with a 15 deg pitch (das_calib.T_VIO_PALM); it does
    # NOT cancel under recon's left-relativization, so it must be applied here.
    if calib == "das":
        l_pos, l_quat = vio_pose_to_palm(l_pos, l_quat, hand=0)
        r_pos, r_quat = vio_pose_to_palm(r_pos, r_quat, hand=1)
    elif calib != "none":
        raise ValueError(f"unknown calib {calib!r} (expected 'das' or 'none')")

    # NOTE: the VIO eef_pose world frame is ALREADY gravity-aligned (z up) --
    # VIO initialises its world with the IMU gravity estimate, so raw z is the
    # true vertical. Verified: pnp_bottle's table grasp (gripper close) lands at
    # the global raw-z minimum, and both hands start at equal raw-z. So we do
    # NOT rotate the poses; an earlier IMU re-alignment mistakenly turned the
    # horizontal L-R hand spacing into a fake height difference.
    #
    # Recentre so frame-0 midpoint is in front of the pelvis in XY, then set the
    # vertical baseline per ``anchor``.
    mid0 = 0.5 * (l_pos[0] + r_pos[0])
    shift = _STANDING_HAND_MID - mid0            # start-mid: also fixes z
    if anchor == "lowest-zero":
        zmin = min(l_pos[:, 2].min(), r_pos[:, 2].min())
        shift[2] = -zmin                         # lowest point -> z=0
    elif anchor != "start-mid":
        raise ValueError(f"unknown anchor {anchor!r}")
    l_pos = l_pos + shift
    r_pos = r_pos + shift

    T = grid.shape[0]
    keypoints = np.zeros((T, 2, 7), dtype=np.float32)
    keypoints[:, 0, :3] = l_pos
    keypoints[:, 0, 3:] = l_quat
    keypoints[:, 1, :3] = r_pos
    keypoints[:, 1, 3:] = r_quat

    feat = np.zeros((T, 70), dtype=np.float32)
    return feat, keypoints, float(target_fps)


def main() -> None:
    ap = argparse.ArgumentParser(description="UMI mcap -> EE-only recon bundle")
    ap.add_argument("--mcap", required=True, help="*_vio.merged.mcap file or directory")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--target-fps", type=float, default=30.0)
    ap.add_argument(
        "--anchor", choices=["start-mid", "lowest-zero"], default="start-mid",
        help="vertical baseline: frame-0 midpoint->0.814 (start-mid) or "
             "trajectory lowest point->0 (lowest-zero)",
    )
    ap.add_argument(
        "--calib", choices=["das", "none"], default="das",
        help="VIO->palm mapping: 'das' applies the full rigid DAS-gripper "
             "base_link->palm transform per hand (default); 'none' feeds raw "
             "VIO poses through (sensor frame must already be the palm anchor)",
    )
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    src = Path(args.mcap).expanduser()
    files = sorted(src.rglob("*.mcap")) if src.is_dir() else [src]
    if not files:
        raise FileNotFoundError(f"No mcap under {src}")

    out_dir = Path(args.out_dir).expanduser()
    out_dir.mkdir(parents=True, exist_ok=True)
    for i, mcap_path in enumerate(files):
        out_path = out_dir / f"ep_{i:05d}.npz"
        if out_path.exists() and not args.overwrite:
            print(f"[SKIP] {out_path} exists")
            continue
        feat, keypoints, eff_fps = build_keypoints(
            mcap_path, args.target_fps, anchor=args.anchor, calib=args.calib
        )
        # Stamp the current contact-label version so _load_delta_file does NOT
        # run its on-the-fly relabel: that relabel writes contact flags into
        # feat[:, 5:7], which would break _has_missing_root_supervision's
        # feat[:, :11]==0 test and pull the clip off the EE-only recon path.
        np.savez_compressed(
            out_path,
            feat=feat,
            keypoints=keypoints,
            contact_label_version=np.int32(CONTACT_LABEL_VERSION),
            source_fps=np.float32(eff_fps),
        )
        print(
            f"[OK] {mcap_path.name} -> {out_path}  frames={feat.shape[0]} "
            f"(EE-only, resampled to {args.target_fps:g} fps)"
        )
    print("Done.")


if __name__ == "__main__":
    main()
