from __future__ import annotations

from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from Prior_Recon.Masked_Flow.evaluation.errors import EvaluationInputError

FloatArray = NDArray[np.float64]


def _resample(sequence: FloatArray, sample_count: int) -> FloatArray:
    source_time = np.linspace(0.0, 1.0, sequence.shape[0])
    target_time = np.linspace(0.0, 1.0, sample_count)
    flattened = sequence.reshape(sequence.shape[0], -1)
    result = np.empty((sample_count, flattened.shape[1]), dtype=np.float64)
    for feature_index in range(flattened.shape[1]):
        result[:, feature_index] = np.interp(
            target_time,
            source_time,
            flattened[:, feature_index],
        )
    return result


def hand_trajectory_descriptor(hand_positions: FloatArray) -> FloatArray:
    centered = hand_positions - np.mean(hand_positions[0], axis=0)[None, None, :]
    return _resample(centered, sample_count=32).reshape(-1)


def motion_descriptor(
    qpos: FloatArray,
    hand_positions: FloatArray,
    fps: float,
) -> FloatArray:
    root_velocity = np.diff(qpos[:, :3], axis=0) * fps
    joints = qpos[:, 7:]
    joint_velocity = np.diff(joints, axis=0) * fps
    hand_velocity = np.diff(hand_positions, axis=0) * fps
    return np.concatenate(
        [
            np.mean(root_velocity, axis=0),
            np.std(root_velocity, axis=0),
            np.mean(np.sin(joints), axis=0),
            np.std(np.sin(joints), axis=0),
            np.mean(np.cos(joints), axis=0),
            np.std(np.cos(joints), axis=0),
            np.mean(joint_velocity, axis=0),
            np.std(joint_velocity, axis=0),
            np.mean(hand_velocity, axis=(0, 1)),
            np.std(hand_velocity, axis=(0, 1)),
        ]
    )


def load_embedding(path: Path) -> FloatArray:
    if not path.is_file():
        raise EvaluationInputError(str(path), "file does not exist")
    embedding = np.asarray(np.load(str(path)), dtype=np.float64)
    if embedding.ndim != 1 or embedding.size == 0:
        raise EvaluationInputError(
            str(path), "embedding must be a one-dimensional vector"
        )
    if not np.isfinite(embedding).all() or np.linalg.norm(embedding) < 1e-12:
        raise EvaluationInputError(str(path), "embedding must be finite and non-zero")
    return embedding


def stack_embeddings(embeddings: list[FloatArray], label: str) -> FloatArray:
    shapes = {embedding.shape for embedding in embeddings}
    if len(shapes) != 1:
        raise EvaluationInputError(label, "all clips must use one shared dimension")
    return np.stack(embeddings)
