from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from Prior_Recon.Masked_Flow.evaluation.errors import EvaluationInputError

FloatArray = NDArray[np.float64]


def mean_finite(values: list[float]) -> float:
    finite = np.asarray(values, dtype=np.float64)
    finite = finite[np.isfinite(finite)]
    return float(np.mean(finite)) if finite.size else float("nan")


def mean_metric(values: list[float], metric_name: str) -> float:
    array = np.asarray(values, dtype=np.float64)
    if not np.isfinite(array).all():
        raise EvaluationInputError(metric_name, "computed a non-finite value")
    return float(np.mean(array))


def mean_foot_skate_m_s(
    sole_positions: FloatArray,
    fps: float,
    floor_height_m: float,
    contact_height_m: float,
) -> float:
    if sole_positions.ndim != 4 or sole_positions.shape[-1] != 3:
        raise EvaluationInputError(
            "sole_positions",
            f"expected (T, feet, sole_points, 3), got {sole_positions.shape}",
        )
    if sole_positions.shape[0] < 2:
        raise EvaluationInputError("sole_positions", "at least two frames are required")
    if not np.isfinite(sole_positions).all():
        raise EvaluationInputError("sole_positions", "finite values are required")
    if not np.isfinite(fps) or fps <= 0.0:
        raise EvaluationInputError("fps", "must be positive")
    if not np.isfinite(floor_height_m) or not np.isfinite(contact_height_m):
        raise EvaluationInputError("foot contact threshold", "must be finite")
    if contact_height_m < 0.0:
        raise EvaluationInputError("contact_height_m", "must be non-negative")

    foot_height = np.min(sole_positions[..., 2], axis=2)
    in_contact = (foot_height[:-1] <= floor_height_m + contact_height_m) & (
        foot_height[1:] <= floor_height_m + contact_height_m
    )
    horizontal_speed = (
        np.linalg.norm(
            sole_positions[1:, ..., :2] - sole_positions[:-1, ..., :2],
            axis=-1,
        )
        * fps
    )
    selected_speed = horizontal_speed[in_contact]
    if selected_speed.size == 0:
        return float("nan")
    return float(np.mean(selected_speed))


def mean_quaternion_error_deg(
    generated_wxyz: FloatArray,
    target_wxyz: FloatArray,
) -> float:
    if generated_wxyz.shape != target_wxyz.shape or generated_wxyz.shape[-1] != 4:
        raise EvaluationInputError(
            "quaternions",
            f"matching (..., 4) arrays required, got {generated_wxyz.shape} and "
            f"{target_wxyz.shape}",
        )
    generated_norm = np.linalg.norm(generated_wxyz, axis=-1, keepdims=True)
    target_norm = np.linalg.norm(target_wxyz, axis=-1, keepdims=True)
    if (
        not np.isfinite(generated_wxyz).all()
        or not np.isfinite(target_wxyz).all()
        or np.any(generated_norm < 1e-12)
        or np.any(target_norm < 1e-12)
    ):
        raise EvaluationInputError("quaternions", "finite non-zero values are required")
    generated = generated_wxyz / generated_norm
    target = target_wxyz / target_norm
    cosine_half_angle = np.clip(
        np.abs(np.sum(generated * target, axis=-1)),
        0.0,
        1.0,
    )
    return float(np.mean(np.degrees(2.0 * np.arccos(cosine_half_angle))))


def r_precision_at_k(
    query_embeddings: FloatArray,
    motion_embeddings: FloatArray,
    k: int,
) -> float:
    if query_embeddings.shape != motion_embeddings.shape or query_embeddings.ndim != 2:
        raise EvaluationInputError(
            "embeddings",
            f"matching (N, D) arrays required, got {query_embeddings.shape} and "
            f"{motion_embeddings.shape}",
        )
    if query_embeddings.shape[0] == 0:
        raise EvaluationInputError("embeddings", "at least one pair is required")
    if k < 1:
        raise EvaluationInputError("k", "must be at least one")
    if (
        not np.isfinite(query_embeddings).all()
        or not np.isfinite(motion_embeddings).all()
        or np.any(np.linalg.norm(query_embeddings, axis=1) < 1e-12)
        or np.any(np.linalg.norm(motion_embeddings, axis=1) < 1e-12)
    ):
        raise EvaluationInputError("embeddings", "finite non-zero rows are required")

    query_norm = query_embeddings / np.linalg.norm(
        query_embeddings,
        axis=1,
        keepdims=True,
    ).clip(min=1e-12)
    motion_norm = motion_embeddings / np.linalg.norm(
        motion_embeddings,
        axis=1,
        keepdims=True,
    ).clip(min=1e-12)
    ranking = np.argsort(-(query_norm @ motion_norm.T), axis=1)
    top_k = ranking[:, : min(k, ranking.shape[1])]
    correct = np.arange(ranking.shape[0])[:, None]
    return float(np.mean(np.any(top_k == correct, axis=1)) * 100.0)


def _positive_semidefinite_sqrt(matrix: FloatArray) -> FloatArray:
    symmetric = (matrix + matrix.T) * 0.5
    eigenvalues, eigenvectors = np.linalg.eigh(symmetric)
    return (eigenvectors * np.sqrt(np.clip(eigenvalues, 0.0, None))) @ eigenvectors.T


def frechet_distance(
    generated_embeddings: FloatArray,
    reference_embeddings: FloatArray,
) -> float:
    if (
        generated_embeddings.ndim != 2
        or reference_embeddings.ndim != 2
        or generated_embeddings.shape[1] != reference_embeddings.shape[1]
    ):
        raise EvaluationInputError(
            "embeddings",
            f"(N, D) arrays with equal D required, got {generated_embeddings.shape} "
            f"and {reference_embeddings.shape}",
        )
    if min(generated_embeddings.shape[0], reference_embeddings.shape[0]) < 2:
        raise EvaluationInputError(
            "embeddings", "FID requires at least two samples per set"
        )
    if (
        not np.isfinite(generated_embeddings).all()
        or not np.isfinite(reference_embeddings).all()
    ):
        raise EvaluationInputError("embeddings", "finite values are required")

    generated_mean = np.mean(generated_embeddings, axis=0)
    reference_mean = np.mean(reference_embeddings, axis=0)
    generated_cov = np.atleast_2d(np.cov(generated_embeddings, rowvar=False))
    reference_cov = np.atleast_2d(np.cov(reference_embeddings, rowvar=False))
    reference_sqrt = _positive_semidefinite_sqrt(reference_cov)
    covariance_product_sqrt = _positive_semidefinite_sqrt(
        reference_sqrt @ generated_cov @ reference_sqrt
    )
    mean_distance = float(np.sum((generated_mean - reference_mean) ** 2))
    covariance_distance = float(
        np.trace(generated_cov + reference_cov - 2.0 * covariance_product_sqrt)
    )
    return max(mean_distance + covariance_distance, 0.0)
