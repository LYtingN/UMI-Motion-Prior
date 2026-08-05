from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from numpy.typing import NDArray

from Prior_Recon.Masked_Flow.evaluation.descriptors import load_embedding
from Prior_Recon.Masked_Flow.evaluation.errors import EvaluationInputError
from Prior_Recon.Masked_Flow.evaluation.io import load_motion, raw_feat_to_qpos


def test_raw_feat_to_qpos_integrates_root_and_preserves_joints() -> None:
    # Given: a three-frame feat70 clip with x steps, yaw, height, and joint pose.
    feat = np.zeros((3, 70), dtype=np.float64)
    feat[:, 7] = 0.1
    feat[:, 10] = 0.8
    feat[:, 11:40] = 0.25
    feat[:, 69] = np.deg2rad([0.0, 30.0, 60.0])

    # When: the training representation is decoded to MuJoCo qpos.
    qpos = raw_feat_to_qpos(feat)

    # Then: frame-t stores the pose after integrating preceding frame deltas.
    np.testing.assert_allclose(qpos[:, 0], [0.0, 0.1, 0.2], atol=1e-7)
    np.testing.assert_allclose(qpos[:, 2], 0.8, atol=1e-7)
    np.testing.assert_allclose(qpos[:, 7:], 0.25, atol=1e-7)


def test_raw_feat_to_qpos_rejects_model_facing_73d_window() -> None:
    with pytest.raises(EvaluationInputError, match=r"expected raw \(T, 70\)"):
        raw_feat_to_qpos(np.zeros((4, 73), dtype=np.float64))


def test_load_motion_accepts_generated_npy_and_reference_npz(tmp_path: Path) -> None:
    # Given: the two artifact formats already emitted by Prior-Recon tools.
    qpos = np.zeros((4, 36), dtype=np.float32)
    qpos[:, 3] = 1.0
    generated_path = tmp_path / "generated.npy"
    reference_path = tmp_path / "reference.npz"
    np.save(generated_path, qpos)
    np.savez(reference_path, qpos=qpos, source_fps=np.array(25.0))

    # When: both files cross the evaluation input boundary.
    generated = load_motion(generated_path, default_fps=30.0)
    reference = load_motion(reference_path, default_fps=30.0)

    # Then: qpos is preserved and an artifact fps overrides the CLI default.
    np.testing.assert_array_equal(generated.qpos, qpos)
    assert generated.fps == 30.0
    np.testing.assert_array_equal(reference.qpos, qpos)
    assert reference.fps == 25.0


def test_load_motion_rejects_zero_keypoint_quaternion(tmp_path: Path) -> None:
    qpos = np.zeros((4, 36), dtype=np.float32)
    qpos[:, 3] = 1.0
    keypoints = np.zeros((4, 2, 7), dtype=np.float32)
    artifact_path = tmp_path / "reference.npz"
    np.savez(artifact_path, qpos=qpos, keypoints=keypoints)

    with pytest.raises(EvaluationInputError, match="keypoint quaternions"):
        load_motion(artifact_path, default_fps=30.0)


@pytest.mark.parametrize(
    "embedding",
    [np.zeros((2, 2), dtype=np.float64), np.zeros(4, dtype=np.float64)],
)
def test_load_embedding_rejects_non_vector_or_zero_vector(
    tmp_path: Path,
    embedding: NDArray[np.float64],
) -> None:
    embedding_path = tmp_path / "embedding.npy"
    np.save(embedding_path, embedding)

    with pytest.raises(EvaluationInputError):
        load_embedding(embedding_path)


def test_load_embedding_reports_missing_file_as_input_error(tmp_path: Path) -> None:
    with pytest.raises(EvaluationInputError, match="file does not exist"):
        load_embedding(tmp_path / "missing.npy")
