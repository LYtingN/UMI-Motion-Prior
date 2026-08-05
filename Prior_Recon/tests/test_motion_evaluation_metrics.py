from __future__ import annotations

import numpy as np
import pytest

from Prior_Recon.Masked_Flow.evaluation.errors import EvaluationInputError
from Prior_Recon.Masked_Flow.evaluation.metrics import (
    frechet_distance,
    mean_foot_skate_m_s,
    mean_quaternion_error_deg,
    r_precision_at_k,
)


def test_foot_skate_reports_horizontal_contact_speed() -> None:
    # Given: two feet remain on the floor and translate 0.1 m per frame.
    sole_positions = np.zeros((3, 2, 1, 3), dtype=np.float64)
    sole_positions[:, :, 0, 0] = np.array([0.0, 0.1, 0.2])[:, None]

    # When: skate is measured at 10 fps.
    skate = mean_foot_skate_m_s(
        sole_positions,
        fps=10.0,
        floor_height_m=0.0,
        contact_height_m=0.05,
    )

    # Then: the reported mean contact speed is 1 m/s.
    assert skate == 1.0


@pytest.mark.parametrize("fps", [np.nan, np.inf])
def test_foot_skate_rejects_non_finite_fps(fps: float) -> None:
    sole_positions = np.zeros((2, 2, 1, 3), dtype=np.float64)

    with pytest.raises(EvaluationInputError, match="fps"):
        mean_foot_skate_m_s(sole_positions, fps, 0.0, 0.05)


def test_foot_skate_rejects_non_finite_positions() -> None:
    sole_positions = np.zeros((2, 2, 1, 3), dtype=np.float64)
    sole_positions[0, 0, 0, 0] = np.nan

    with pytest.raises(EvaluationInputError, match="sole_positions"):
        mean_foot_skate_m_s(sole_positions, 30.0, 0.0, 0.05)


def test_quaternion_error_uses_shortest_geodesic_angle() -> None:
    # Given: identity targets and predictions rotated 90 degrees about z.
    target = np.array([[1.0, 0.0, 0.0, 0.0]], dtype=np.float64)
    half_angle = np.deg2rad(45.0)
    generated = np.array(
        [[np.cos(half_angle), 0.0, 0.0, np.sin(half_angle)]],
        dtype=np.float64,
    )

    # When: orientation error is measured.
    error = mean_quaternion_error_deg(generated, target)

    # Then: the result is the SO(3) geodesic angle, independent of sign.
    assert np.isclose(error, 90.0)
    assert np.isclose(mean_quaternion_error_deg(-generated, target), 90.0)


def test_r_precision_counts_the_matching_item_in_top_k() -> None:
    # Given: three normalized query/motion pairs with matching rows.
    embeddings = np.eye(3, dtype=np.float64)

    # When: retrieval precision is evaluated at one and three.
    top1 = r_precision_at_k(embeddings, embeddings, k=1)
    top3 = r_precision_at_k(embeddings, embeddings, k=3)

    # Then: every query retrieves its paired motion.
    assert top1 == 100.0
    assert top3 == 100.0


def test_frechet_distance_detects_a_distribution_mean_shift() -> None:
    # Given: equal-covariance embeddings shifted by one unit on one axis.
    reference = np.array([[-1.0, 0.0], [1.0, 0.0]], dtype=np.float64)
    generated = reference + np.array([1.0, 0.0], dtype=np.float64)

    # When: FID is computed in that evaluator space.
    distance = frechet_distance(generated, reference)

    # Then: only the squared mean shift remains.
    assert np.isclose(distance, 1.0)
    assert np.isclose(frechet_distance(reference, reference), 0.0)
