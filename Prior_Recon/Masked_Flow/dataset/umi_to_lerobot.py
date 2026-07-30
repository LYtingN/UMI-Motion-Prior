#!/usr/bin/env python3
"""
umi_to_lerobot.py
~~~~~~~~~~~~~~~~~
Convert a UMI DAS-Gripper VIO mcap into a LeRobot v2.1 dataset (faithful,
UMI-native schema) with decoded camera video.

One mcap = one episode. Every numeric stream is resampled onto a single
--fps grid over the streams' common time span; the two h264 camera streams are
transcoded to per-robot mp4 at the same fps.

LeRobot layout written under --out-root:
  data/chunk-000/episode_000000.parquet
  videos/chunk-000/observation.images.{left,right}/episode_000000.mp4
  meta/info.json meta/episodes.jsonl meta/tasks.jsonl meta/modality.json

Parquet columns (per frame, UMI-native):
  observation.eef_pose_left   (7)  world pos(3)+quat_wxyz(4)   robot0/master/left
  observation.eef_pose_right  (7)                              robot1/sub/right
  observation.rel_eef_left    (7)  VIO relative_eef_pose (frame "body")
  observation.rel_eef_right   (7)
  observation.gripper         (2)  [left, right] magnetic_encoder width
  observation.imu_left        (6)  ang_vel(3)+lin_acc(3)
  observation.imu_right       (6)
  observation.tactile_left    (2)  [mean, max] pressure of robot0's two pads *
  observation.tactile_right   (2)  robot1's two pads
  timestamp frame_index episode_index index task_index

  * the raw tactile grid is 50x10=500 per pad; storing the full grid per frame
    bloats the parquet, so we summarise each robot's (left+right pad) contact as
    [mean, max] over all cells. Set --full-tactile to store the raw 1000-wide
    vector instead.

Usage
-----
  python Prior_Recon/Masked_Flow/dataset/umi_to_lerobot.py \
      --mcap dataset/umi_origin/pnp_bottle/DAS-Gripper_..._vio.merged.mcap \
      --out-root dataset/umi --task "pick and place the bottle" --fps 30
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from Prior_Recon.Masked_Flow.dataset.umi_mcap import (
    LEFT_ROBOT,
    RIGHT_ROBOT,
    common_time_grid,
    read_umi_mcap,
    resample_pose,
    resample_scalar,
)

_CHUNK = "chunk-000"


def _pose_cols(pos: np.ndarray, quat: np.ndarray) -> np.ndarray:
    return np.concatenate([pos, quat], axis=1).astype(np.float32)


def _summarise_tactile(stream_vals: np.ndarray, grid, src_t) -> np.ndarray:
    """Per-frame [mean, max] over the flattened pressure grid, nearest-neighbour
    in time (tactile is high-dim; interp per cell is wasteful and meaningless)."""
    idx = np.searchsorted(src_t.astype(np.float64), grid, side="left")
    idx = np.clip(idx, 0, len(src_t) - 1)
    picked = stream_vals[idx]
    return np.stack([picked.mean(axis=1), picked.max(axis=1)], axis=1).astype(np.float32)


def _decode_h264_to_mp4(
    packets: list[tuple[int, bytes]],
    grid_ns: np.ndarray,
    out_path: Path,
    width: int,
    height: int,
    fps: float,
    ffmpeg: str,
) -> tuple[int, int, int]:
    """Decode the concatenated h264 elementary stream to RGB frames, pick the
    frame nearest each grid timestamp, and re-encode to a constant-fps mp4.

    Returns ``(n_frames_written, decoded_height, decoded_width)`` -- the decoded
    resolution can differ from the camera_info calibration size, so the caller
    stamps info.json with what was actually written.
    """
    import imageio.v2 as imageio

    raw = b"".join(p[1] for p in packets)
    src_t = np.array([p[0] for p in packets], dtype=np.float64)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_h264 = out_path.with_suffix(".h264")
    tmp_h264.write_bytes(raw)
    try:
        reader = imageio.get_reader(
            str(tmp_h264), format="ffmpeg", input_params=["-f", "h264"]
        )
        frames = [np.asarray(f) for f in reader]
        reader.close()
    finally:
        tmp_h264.unlink(missing_ok=True)

    if not frames:
        raise RuntimeError(f"No frames decoded from {out_path.name}")
    frames = np.stack(frames)  # (Nsrc, H, W, 3)
    # Align decoded frames to packet timestamps (they correspond 1:1 in order),
    # then nearest-neighbour resample onto the grid.
    n = min(len(frames), len(src_t))
    frames, src_t = frames[:n], src_t[:n]
    idx = np.clip(np.searchsorted(src_t, grid_ns, side="left"), 0, n - 1)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    writer = imageio.get_writer(
        str(out_path), format="ffmpeg", fps=fps, codec="libx264",
        macro_block_size=None, ffmpeg_log_level="error",
    )
    try:
        for i in idx:
            writer.append_data(frames[i])
    finally:
        writer.close()
    return len(idx), int(frames.shape[1]), int(frames.shape[2])


def convert(
    mcap_path: Path,
    out_root: Path,
    task: str,
    fps: float,
    episode_index: int,
    with_video: bool,
    full_tactile: bool,
    ffmpeg: str,
) -> dict:
    import pandas as pd

    data = read_umi_mcap(mcap_path, with_video=with_video)
    for robot in (LEFT_ROBOT, RIGHT_ROBOT):
        if robot not in data.eef:
            raise ValueError(f"{mcap_path}: missing {robot} eef_pose")

    grid = common_time_grid(
        [data.eef[LEFT_ROBOT].t, data.eef[RIGHT_ROBOT].t], fps
    )
    T = grid.shape[0]

    lp, lq = resample_pose(data.eef[LEFT_ROBOT], grid)
    rp, rq = resample_pose(data.eef[RIGHT_ROBOT], grid)
    cols: dict[str, np.ndarray] = {
        "observation.eef_pose_left": _pose_cols(lp, lq),
        "observation.eef_pose_right": _pose_cols(rp, rq),
    }
    if LEFT_ROBOT in data.relative_eef and RIGHT_ROBOT in data.relative_eef:
        rlp, rlq = resample_pose(data.relative_eef[LEFT_ROBOT], grid)
        rrp, rrq = resample_pose(data.relative_eef[RIGHT_ROBOT], grid)
        cols["observation.rel_eef_left"] = _pose_cols(rlp, rlq)
        cols["observation.rel_eef_right"] = _pose_cols(rrp, rrq)

    grip_l = resample_scalar(data.gripper[LEFT_ROBOT], grid)
    grip_r = resample_scalar(data.gripper[RIGHT_ROBOT], grid)
    cols["observation.gripper"] = np.stack([grip_l, grip_r], axis=1).astype(np.float32)

    cols["observation.imu_left"] = resample_scalar(data.imu[LEFT_ROBOT], grid).astype(np.float32)
    cols["observation.imu_right"] = resample_scalar(data.imu[RIGHT_ROBOT], grid).astype(np.float32)

    for robot, tag in ((LEFT_ROBOT, "left"), (RIGHT_ROBOT, "right")):
        pads = [data.tactile[k] for k in (f"{robot}/left", f"{robot}/right") if k in data.tactile]
        if not pads:
            continue
        if full_tactile:
            cat = np.concatenate(
                [resample_scalar(p, grid) for p in pads], axis=1
            ).astype(np.float32)
            cols[f"observation.tactile_{tag}"] = cat
        else:
            summ = [_summarise_tactile(p.value, grid, p.t) for p in pads]
            # combine the robot's two pads: mean of means, max of maxes
            means = np.stack([s[:, 0] for s in summ], axis=1).mean(axis=1)
            maxes = np.stack([s[:, 1] for s in summ], axis=1).max(axis=1)
            cols[f"observation.tactile_{tag}"] = np.stack([means, maxes], axis=1).astype(np.float32)

    # ---- write parquet ----
    df = pd.DataFrame({k: list(v) for k, v in cols.items()})
    df["timestamp"] = ((grid - grid[0]) / 1e9).astype(np.float32)
    df["frame_index"] = np.arange(T, dtype=np.int64)
    df["episode_index"] = np.int64(episode_index)
    df["index"] = np.arange(T, dtype=np.int64)
    df["task_index"] = np.int64(0)

    data_dir = out_root / "data" / _CHUNK
    data_dir.mkdir(parents=True, exist_ok=True)
    parquet_path = data_dir / f"episode_{episode_index:06d}.parquet"
    df.to_parquet(parquet_path)

    # ---- video ----
    video_keys: dict[str, dict] = {}
    if with_video:
        robot_to_key = {LEFT_ROBOT: "observation.images.left", RIGHT_ROBOT: "observation.images.right"}
        for robot, vkey in robot_to_key.items():
            packets = data.camera.get(robot)
            if not packets:
                continue
            info = data.camera_info.get(robot, {"width": 640, "height": 480})
            out_mp4 = out_root / "videos" / _CHUNK / vkey / f"episode_{episode_index:06d}.mp4"
            n, vh, vw = _decode_h264_to_mp4(
                packets, grid, out_mp4, info["width"], info["height"], fps, ffmpeg
            )
            video_keys[vkey] = {
                "dtype": "video", "shape": [vh, vw, 3],
                "names": ["height", "width", "channel"],
                "info": {
                    "video.height": vh, "video.width": vw,
                    "video.codec": "h264", "video.pix_fmt": "yuv420p",
                    "video.is_depth_map": False, "video.fps": fps,
                    "video.channels": 3, "has_audio": False,
                },
            }
            print(f"    video {vkey}: {n} frames @ {vw}x{vh} -> {out_mp4}")

    # feature spec (shapes) for info.json
    feat_spec: dict[str, dict] = {}
    for k, v in cols.items():
        dim = v.shape[1] if v.ndim > 1 else 1
        feat_spec[k] = {"dtype": "float32", "shape": [int(dim)], "names": None}
    return {
        "parquet": parquet_path,
        "length": int(T),
        "features": feat_spec,
        "video_keys": video_keys,
        "camera_info": data.camera_info,
    }


def _write_meta(
    out_root: Path, fps: float, results: list[dict], task: str
) -> None:
    meta = out_root / "meta"
    meta.mkdir(parents=True, exist_ok=True)
    total_frames = sum(r["length"] for r in results)
    features = dict(results[0]["features"])
    features.update(results[0]["video_keys"])
    # scalar bookkeeping columns
    for name in ("timestamp",):
        features[name] = {"dtype": "float32", "shape": [1], "names": None}
    for name in ("frame_index", "episode_index", "index", "task_index"):
        features[name] = {"dtype": "int64", "shape": [1], "names": None}

    info = {
        "codebase_version": "v2.1",
        "robot_type": "umi_das_gripper",
        "total_episodes": len(results),
        "total_frames": total_frames,
        "total_tasks": 1,
        "total_videos": sum(len(r["video_keys"]) for r in results),
        "total_chunks": 1,
        "chunks_size": 1000,
        "fps": fps,
        "splits": {"train": f"0:{len(results)}"},
        "data_path": "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet",
        "video_path": "videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4",
        "features": features,
    }
    (meta / "info.json").write_text(json.dumps(info, indent=4))

    with (meta / "episodes.jsonl").open("w") as fh:
        for i, r in enumerate(results):
            fh.write(json.dumps({"episode_index": i, "tasks": [task], "length": r["length"]}) + "\n")
    with (meta / "tasks.jsonl").open("w") as fh:
        fh.write(json.dumps({"task_index": 0, "task": task}) + "\n")

    # modality.json: byte-offset spec so the whole EEF/gripper/imu layout is
    # explicit for downstream loaders (mirrors the sonic washing meta style).
    def _blocks(feat_spec):
        state, off = {}, 0
        for k, spec in feat_spec.items():
            d = spec["shape"][0]
            state[k.replace("observation.", "")] = {"start": off, "end": off + d, "original_key": k}
            off += d
        return state
    modality = {
        "state": _blocks(results[0]["features"]),
        "video": {k: {"original_key": k} for k in results[0]["video_keys"]},
    }
    (meta / "modality.json").write_text(json.dumps(modality, indent=4))


def main() -> None:
    import imageio_ffmpeg

    ap = argparse.ArgumentParser(description="UMI mcap -> LeRobot v2.1 dataset")
    ap.add_argument("--mcap", required=True, help="mcap file or directory (one episode per file)")
    ap.add_argument("--out-root", required=True)
    ap.add_argument("--task", default="pick and place the bottle")
    ap.add_argument("--fps", type=float, default=30.0)
    ap.add_argument("--no-video", action="store_true", help="skip camera decode")
    ap.add_argument("--full-tactile", action="store_true", help="store raw 500-per-pad grid")
    args = ap.parse_args()

    src = Path(args.mcap).expanduser()
    files = sorted(src.rglob("*.mcap")) if src.is_dir() else [src]
    if not files:
        raise FileNotFoundError(f"No mcap under {src}")
    out_root = Path(args.out_root).expanduser()
    ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()

    results = []
    for i, mcap_path in enumerate(files):
        print(f"[{i+1}/{len(files)}] {mcap_path.name}")
        r = convert(
            mcap_path, out_root, args.task, args.fps, i,
            with_video=not args.no_video, full_tactile=args.full_tactile, ffmpeg=ffmpeg,
        )
        print(f"    parquet: {r['parquet']}  frames={r['length']}")
        results.append(r)

    _write_meta(out_root, args.fps, results, args.task)
    print(f"Done. {len(results)} episode(s) -> {out_root}")


if __name__ == "__main__":
    main()
