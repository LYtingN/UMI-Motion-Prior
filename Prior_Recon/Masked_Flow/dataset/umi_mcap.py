#!/usr/bin/env python3
"""
umi_mcap.py
~~~~~~~~~~~
Shared reader for UMI DAS-Gripper Foxglove-protobuf MCAP logs (the
``*_vio.merged.mcap`` produced by the two-device VIO loop-fusion pipeline).

The merged log carries two devices under ``/robot0`` (master, held in the LEFT
hand) and ``/robot1`` (sub, RIGHT hand). Every numeric stream is read verbatim
and returned as ``(timestamps_ns, values)`` so downstream converters can put
them on whatever common grid they need.

Frame note
----------
The VIO world frame is ALREADY gravity-aligned: its +Z is true vertical (VIO
seeds the world with the IMU gravity estimate). Pass ``eef_pose`` through RAW --
do NOT re-rotate it. Proof for the pnp_bottle clip: the right gripper's grasp
(magnetic_encoder minimum, frame 309) lands at the global raw-z minimum of all
747 frames, and both hands start at equal raw-z (frame-0 R-L z-diff ~= -0.007).

``estimate_world_up``/``gravity_align_rotation`` remain here only as a record of
a discarded approach: an earlier converter estimated world-up from the IMU
specific force (rotated into world with the EEF quat) and rotated poses by
~55 deg. That was WRONG -- the IMU frame != the eef_pose frame, so rotating the
accel by the EEF quat mislands gravity onto raw-y and turns the horizontal L-R
hand spacing into a fake ~0.25 m height difference. These helpers MUST NOT be
applied to ``eef_pose``.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation as R
from scipy.spatial.transform import Slerp

# robot0 is the master gripper carried in the left hand; robot1 the sub / right.
LEFT_ROBOT = "robot0"
RIGHT_ROBOT = "robot1"


@dataclass
class PoseStream:
    t: np.ndarray            # (N,) header timestamps, nanoseconds
    pos: np.ndarray          # (N, 3) world position
    quat_wxyz: np.ndarray    # (N, 4) world orientation, wxyz


@dataclass
class ScalarStream:
    t: np.ndarray            # (N,)
    value: np.ndarray        # (N,) or (N, D)


@dataclass
class UMIData:
    eef: dict[str, PoseStream] = field(default_factory=dict)
    relative_eef: dict[str, PoseStream] = field(default_factory=dict)
    gripper: dict[str, ScalarStream] = field(default_factory=dict)
    imu: dict[str, ScalarStream] = field(default_factory=dict)         # (N, 6) [ang_vel(3), lin_acc(3)]
    tactile: dict[str, ScalarStream] = field(default_factory=dict)     # key "robot0/left" -> (N, rows*cols)
    camera: dict[str, list] = field(default_factory=dict)              # key robot -> list[(t_ns, h264_bytes)]
    camera_info: dict[str, dict] = field(default_factory=dict)
    task: str = ""


def read_umi_mcap(path: str | Path, with_video: bool = False) -> UMIData:
    """Read every numeric stream (and optionally the raw h264 packets) once."""
    from mcap.reader import make_reader
    from mcap_protobuf.decoder import DecoderFactory

    path = Path(path).expanduser()
    out = UMIData()

    def _pose(pb) -> tuple[float, list, list]:
        p, o = pb.pose.position, pb.pose.orientation
        return pb.header.timestamp, [p.x, p.y, p.z], [o.w, o.x, o.y, o.z]

    raw: dict[str, list] = {}
    with open(path, "rb") as fh:
        reader = make_reader(fh, decoder_factories=[DecoderFactory()])
        for schema, channel, _msg, pb in reader.iter_decoded_messages():
            topic = channel.topic
            parts = topic.strip("/").split("/")
            robot = parts[0] if parts and parts[0].startswith("robot") else None
            raw.setdefault(topic, [])
            if topic.endswith("/vio/eef_pose") or topic.endswith("/vio/relative_eef_pose"):
                raw[topic].append(_pose(pb))
            elif topic.endswith("/sensor/magnetic_encoder"):
                raw[topic].append((pb.header.timestamp, float(pb.value)))
            elif topic.endswith("/sensor/imu"):
                a, w = pb.linear_acceleration, pb.angular_velocity
                raw[topic].append(
                    (pb.header.timestamp, [w.x, w.y, w.z, a.x, a.y, a.z])
                )
            elif "/sensor/tactile" in topic:
                raw[topic].append(
                    (pb.header.timestamp, np.asarray(pb.pressure_data, dtype=np.float32))
                )
            elif topic.endswith("/sensor/camera0/camera_info") and robot:
                out.camera_info[robot] = {
                    "width": int(pb.width), "height": int(pb.height),
                    "K": list(pb.K), "D": list(pb.D),
                    "distortion_model": pb.distortion_model,
                }
            elif with_video and topic.endswith("/sensor/camera0/compressed") and robot:
                out.camera.setdefault(robot, []).append(
                    (pb.header.timestamp, bytes(pb.data))
                )
            elif topic.endswith("/tasks") or topic == "/task":
                pass

    def _to_pose(rows) -> PoseStream:
        rows = sorted(rows, key=lambda r: r[0])
        t = np.array([r[0] for r in rows], dtype=np.int64)
        pos = np.array([r[1] for r in rows], dtype=np.float64)
        quat = np.array([r[2] for r in rows], dtype=np.float64)
        return PoseStream(t=t, pos=pos, quat_wxyz=quat)

    def _to_scalar(rows) -> ScalarStream:
        rows = sorted(rows, key=lambda r: r[0])
        t = np.array([r[0] for r in rows], dtype=np.int64)
        val = np.array([r[1] for r in rows])
        return ScalarStream(t=t, value=val)

    for topic, rows in raw.items():
        if not rows:
            continue
        robot = topic.strip("/").split("/")[0]
        if topic.endswith("/vio/eef_pose"):
            out.eef[robot] = _to_pose(rows)
        elif topic.endswith("/vio/relative_eef_pose"):
            out.relative_eef[robot] = _to_pose(rows)
        elif topic.endswith("/sensor/magnetic_encoder"):
            out.gripper[robot] = _to_scalar(rows)
        elif topic.endswith("/sensor/imu"):
            out.imu[robot] = _to_scalar(rows)
        elif "/sensor/tactile" in topic:
            side = "left" if topic.endswith("tactile_left") else "right"
            out.tactile[f"{robot}/{side}"] = _to_scalar(rows)

    for robot in out.camera:
        out.camera[robot].sort(key=lambda r: r[0])
    return out


# --------------------------------------------------------------------------- #
# Gravity alignment
# --------------------------------------------------------------------------- #
def estimate_world_up(data: UMIData, robot: str = LEFT_ROBOT) -> np.ndarray:
    """Unit up-vector in the VIO world frame from the IMU specific force.

    At rest the accelerometer reads +g along local up; rotating it into world
    with the (time-synced) EEF orientation and averaging recovers world up.
    """
    eef = data.eef[robot]
    imu = data.imu[robot]
    te = eef.t.astype(np.float64)
    ti = imu.t.astype(np.float64)
    mask = (ti >= te[0]) & (ti <= te[-1])
    rot = Slerp(te, R.from_quat(np.roll(eef.quat_wxyz, -1, axis=1)))(ti[mask])
    acc_world = rot.apply(imu.value[mask, 3:6])  # linear_acceleration
    up = acc_world.mean(axis=0)
    return up / np.linalg.norm(up)


def gravity_align_rotation(up_world: np.ndarray) -> np.ndarray:
    """(3,3) rotation mapping ``up_world`` -> +Z, forward from world +X.

    Rows are the new-frame axes in old-world coordinates, i.e.
    ``pos_new = Rot @ pos_old``. Yaw about the new +Z is fixed by projecting the
    old +X axis onto the ground plane (arbitrary but stable); recon re-anchors
    yaw to the standing pose anyway, so only the vertical direction is
    physically load-bearing here.
    """
    z = np.asarray(up_world, dtype=np.float64)
    z = z / np.linalg.norm(z)
    x0 = np.array([1.0, 0.0, 0.0])
    x = x0 - z * (x0 @ z)
    if np.linalg.norm(x) < 1e-6:
        x0 = np.array([0.0, 1.0, 0.0])
        x = x0 - z * (x0 @ z)
    x /= np.linalg.norm(x)
    y = np.cross(z, x)
    return np.stack([x, y, z], axis=0)


# --------------------------------------------------------------------------- #
# Resampling helpers
# --------------------------------------------------------------------------- #
def common_time_grid(streams_t: list[np.ndarray], fps: float) -> np.ndarray:
    """Uniform ns grid covering the overlap of every stream (so all interpolate)."""
    t0 = max(float(t[0]) for t in streams_t)
    t1 = min(float(t[-1]) for t in streams_t)
    n = int(np.floor((t1 - t0) / 1e9 * fps)) + 1
    return (t0 + np.arange(n) / fps * 1e9).astype(np.float64)


def resample_pose(stream: PoseStream, grid_ns: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    t = stream.t.astype(np.float64)
    pos = np.stack([np.interp(grid_ns, t, stream.pos[:, i]) for i in range(3)], axis=1)
    rot = R.from_quat(np.roll(stream.quat_wxyz, -1, axis=1))
    quat_xyzw = Slerp(t, rot)(grid_ns).as_quat()
    return pos, np.roll(quat_xyzw, 1, axis=1)


def resample_scalar(stream: ScalarStream, grid_ns: np.ndarray) -> np.ndarray:
    t = stream.t.astype(np.float64)
    val = stream.value
    if val.ndim == 1:
        return np.interp(grid_ns, t, val)
    return np.stack([np.interp(grid_ns, t, val[:, i]) for i in range(val.shape[1])], axis=1)
