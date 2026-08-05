from __future__ import annotations

from typing import NamedTuple


class EvaluationProtocol(NamedTuple):
    fps: float = 30.0
    warmup_frames: int = 2
    keyframe_stride: int = 16
    waypoint_stride: int = 16
    floor_height_m: float = 0.0
    foot_contact_height_m: float = 0.05


class PairMetrics(NamedTuple):
    skate_m_s: float
    joint_rot_deg: float
    joint_pos_m: float
    keyframe_body_m: float
    traj_m: float
    waypoint_m: float


class AggregateMetrics(NamedTuple):
    skate_m_s: float
    r_precision_top3_percent: float
    fid: float
    joint_rot_deg: float
    joint_pos_m: float
    keyframe_body_m: float
    traj_m: float
    waypoint_m: float


class ClipResult(NamedTuple):
    clip_id: str
    fps: float
    metrics: PairMetrics
    body_reference_available: bool


class EvaluationSummary(NamedTuple):
    clips: list[ClipResult]
    aggregate: AggregateMetrics
    embedding_backend: str
