from __future__ import annotations

from pathlib import Path
from typing import NamedTuple

import numpy as np
from numpy.typing import NDArray

from Prior_Recon.Masked_Flow.evaluation.errors import EvaluationInputError

FloatArray = NDArray[np.float64]
_QPOS_KEYS = ("qpos", "planned_qpos", "prior_qpos")
_MOTION_SUFFIXES = (".npy", ".npz")


class MotionData(NamedTuple):
    qpos: FloatArray
    keypoints: FloatArray | None
    fps: float
    root_xy_known: bool
    fps_is_explicit: bool
    body_reference_available: bool


def _index_motion_files(path: Path) -> dict[str, Path]:
    resolved = path.expanduser().resolve()
    if resolved.is_file():
        return {resolved.stem: resolved}
    if not resolved.is_dir():
        raise EvaluationInputError(str(resolved), "path does not exist")
    indexed: dict[str, Path] = {}
    for candidate in sorted(resolved.rglob("*")):
        if candidate.suffix not in _MOTION_SUFFIXES:
            continue
        clip_id = candidate.relative_to(resolved).with_suffix("").as_posix()
        if clip_id in indexed:
            raise EvaluationInputError(str(resolved), f"duplicate clip id {clip_id}")
        indexed[clip_id] = candidate
    if not indexed:
        raise EvaluationInputError(str(resolved), "no .npy or .npz motion files found")
    return indexed


def pair_motion_paths(generated: Path, reference: Path) -> list[tuple[str, Path, Path]]:
    generated_files = _index_motion_files(generated)
    reference_files = _index_motion_files(reference)
    if generated.is_file() and reference.is_file():
        generated_path = next(iter(generated_files.values()))
        reference_path = next(iter(reference_files.values()))
        return [(generated_path.stem, generated_path, reference_path)]
    if generated_files.keys() != reference_files.keys():
        missing_generated = sorted(reference_files.keys() - generated_files.keys())
        missing_reference = sorted(generated_files.keys() - reference_files.keys())
        raise EvaluationInputError(
            "motion pairs",
            f"missing generated={missing_generated}, missing reference={missing_reference}",
        )
    return [
        (clip_id, generated_files[clip_id], reference_files[clip_id])
        for clip_id in sorted(generated_files)
    ]


def resolve_pair_fps(
    generated: MotionData,
    reference: MotionData,
    clip_id: str,
) -> float:
    if (
        generated.fps_is_explicit
        and reference.fps_is_explicit
        and not np.isclose(generated.fps, reference.fps, rtol=1e-6, atol=1e-6)
    ):
        raise EvaluationInputError(
            clip_id,
            f"fps mismatch {generated.fps:g} != {reference.fps:g}",
        )
    if reference.fps_is_explicit:
        return float(reference.fps)
    if generated.fps_is_explicit:
        return float(generated.fps)
    return float(reference.fps)


def raw_feat_to_qpos(feat: FloatArray) -> FloatArray:
    if feat.ndim != 2 or feat.shape[1] != 70:
        raise EvaluationInputError("feat", f"expected raw (T, 70), got {feat.shape}")
    if feat.shape[0] < 2 or not np.isfinite(feat).all():
        raise EvaluationInputError("feat", "at least two finite frames are required")

    roll = np.arctan2(feat[:, 0], np.clip(feat[:, 1] + 1.0, -1.0, 1.0))
    pitch = np.arctan2(feat[:, 2], np.clip(feat[:, 3] + 1.0, -1.0, 1.0))
    yaw = feat[:, 69]
    half_roll, half_pitch, half_yaw = roll * 0.5, pitch * 0.5, yaw * 0.5
    cr, sr = np.cos(half_roll), np.sin(half_roll)
    cp, sp = np.cos(half_pitch), np.sin(half_pitch)
    cy, sy = np.cos(half_yaw), np.sin(half_yaw)
    quaternion = np.stack(
        [
            cr * cp * cy + sr * sp * sy,
            sr * cp * cy - cr * sp * sy,
            cr * sp * cy + sr * cp * sy,
            cr * cp * sy - sr * sp * cy,
        ],
        axis=1,
    )
    root_position = np.zeros((feat.shape[0], 3), dtype=np.float64)
    root_position[:, 2] = feat[:, 10]
    root_position[1:, :2] = np.cumsum(feat[:-1, 7:9], axis=0)
    return np.asarray(
        np.concatenate([root_position, quaternion, feat[:, 11:40]], axis=1),
        dtype=np.float64,
    )


def _parse_qpos(qpos: FloatArray, path: Path) -> FloatArray:
    converted = np.asarray(qpos, dtype=np.float64)
    if converted.ndim != 2 or converted.shape[1] != 36:
        raise EvaluationInputError(
            str(path), f"qpos must be (T, 36), got {converted.shape}"
        )
    if converted.shape[0] < 2 or not np.isfinite(converted).all():
        raise EvaluationInputError(
            str(path), "qpos must contain at least two finite frames"
        )
    if np.any(np.linalg.norm(converted[:, 3:7], axis=1) < 1e-8):
        raise EvaluationInputError(str(path), "root quaternions must be non-zero")
    return converted


def load_motion(path: Path, default_fps: float) -> MotionData:
    resolved = path.expanduser().resolve()
    if not np.isfinite(default_fps) or default_fps <= 0.0:
        raise EvaluationInputError("default_fps", "must be positive")
    if resolved.suffix == ".npy":
        qpos = _parse_qpos(np.load(str(resolved)), resolved)
        return MotionData(
            qpos=qpos,
            keypoints=None,
            fps=default_fps,
            root_xy_known=True,
            fps_is_explicit=False,
            body_reference_available=True,
        )
    if resolved.suffix != ".npz":
        raise EvaluationInputError(str(resolved), "only .npy and .npz are supported")

    with np.load(str(resolved)) as artifact:
        qpos_key = next((key for key in _QPOS_KEYS if key in artifact), None)
        root_xy_known = qpos_key is not None
        body_reference_available = True
        if qpos_key is None:
            if "feat" not in artifact:
                raise EvaluationInputError(
                    str(resolved),
                    f"expected one of {_QPOS_KEYS} or feat",
                )
            raw_feat = artifact["feat"].astype(np.float64)
            qpos = raw_feat_to_qpos(raw_feat)
            body_reference_available = not np.allclose(raw_feat, 0.0, atol=1e-8)
        else:
            qpos = _parse_qpos(artifact[qpos_key], resolved)
        keypoints = (
            artifact["keypoints"].astype(np.float64)
            if "keypoints" in artifact
            else None
        )
        fps_is_explicit = "source_fps" in artifact
        fps = float(artifact["source_fps"]) if fps_is_explicit else default_fps

    if not body_reference_available and keypoints is None:
        raise EvaluationInputError(
            str(resolved),
            "all-zero EE-only feat requires keypoints",
        )

    if keypoints is not None and (
        keypoints.ndim != 3
        or keypoints.shape[1:] != (2, 7)
        or keypoints.shape[0] != qpos.shape[0]
        or not np.isfinite(keypoints).all()
    ):
        raise EvaluationInputError(
            str(resolved),
            f"keypoints must be finite (T, 2, 7), got {keypoints.shape}",
        )
    if keypoints is not None and np.any(
        np.linalg.norm(keypoints[..., 3:7], axis=-1) < 1e-8
    ):
        raise EvaluationInputError(
            str(resolved),
            "keypoint quaternions must be non-zero",
        )
    if not np.isfinite(fps) or fps <= 0.0:
        raise EvaluationInputError(str(resolved), "fps must be positive")
    return MotionData(
        qpos=qpos,
        keypoints=keypoints,
        fps=fps,
        root_xy_known=root_xy_known,
        fps_is_explicit=fps_is_explicit,
        body_reference_available=body_reference_available,
    )
