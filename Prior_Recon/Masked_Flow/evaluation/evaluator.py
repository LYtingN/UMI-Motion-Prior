from __future__ import annotations

from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from Prior_Recon.Masked_Flow.evaluation.descriptors import (
    hand_trajectory_descriptor,
    load_embedding,
    motion_descriptor,
    stack_embeddings,
)
from Prior_Recon.Masked_Flow.evaluation.errors import EvaluationInputError
from Prior_Recon.Masked_Flow.evaluation.io import load_motion, resolve_pair_fps
from Prior_Recon.Masked_Flow.evaluation.kinematics import G1Kinematics
from Prior_Recon.Masked_Flow.evaluation.metrics import (
    frechet_distance,
    mean_finite,
    mean_foot_skate_m_s,
    mean_metric,
    mean_quaternion_error_deg,
    r_precision_at_k,
)
from Prior_Recon.Masked_Flow.evaluation.models import (
    AggregateMetrics,
    ClipResult,
    EvaluationProtocol,
    EvaluationSummary,
    PairMetrics,
)

FloatArray = NDArray[np.float64]
def _mean_position_error(generated: FloatArray, target: FloatArray) -> float:
    return float(np.mean(np.linalg.norm(generated - target, axis=-1)))


def _align_keypoint_xy_to_local_root(
    keypoints: FloatArray,
    reference_hand_positions: FloatArray,
) -> FloatArray:
    aligned = keypoints.copy()
    xy_offset = np.mean(
        aligned[0, :, :2] - reference_hand_positions[0, :, :2],
        axis=0,
    )
    aligned[..., :2] -= xy_offset
    return aligned


def evaluate_motion_pairs(
    pairs: list[tuple[str, Path, Path]],
    protocol: EvaluationProtocol,
    embedding_roots: tuple[Path, Path, Path] | None = None,
) -> EvaluationSummary:
    if (
        not np.isfinite(protocol.fps)
        or protocol.fps <= 0.0
        or not np.isfinite(protocol.floor_height_m)
        or not np.isfinite(protocol.foot_contact_height_m)
        or protocol.foot_contact_height_m < 0.0
        or protocol.warmup_frames < 0
        or min(
            protocol.keyframe_stride,
            protocol.waypoint_stride,
        )
        < 1
    ):
        raise EvaluationInputError("protocol", "invalid warmup or stride")
    kinematics = G1Kinematics()
    clip_results: list[ClipResult] = []
    query_embeddings: list[FloatArray] = []
    generated_embeddings: list[FloatArray] = []
    reference_embeddings: list[FloatArray] = []
    generated_fid_embeddings: list[FloatArray] = []

    for clip_id, generated_path, reference_path in pairs:
        generated = load_motion(generated_path, default_fps=protocol.fps)
        reference = load_motion(reference_path, default_fps=protocol.fps)
        if generated.qpos.shape[0] != reference.qpos.shape[0]:
            raise EvaluationInputError(
                clip_id,
                f"frame mismatch {generated.qpos.shape[0]} != {reference.qpos.shape[0]}",
            )
        if protocol.warmup_frames > generated.qpos.shape[0] - 2:
            raise EvaluationInputError(clip_id, "warmup must leave at least two frames")
        clip_fps = resolve_pair_fps(generated, reference, clip_id)
        generated_fk = kinematics.forward(generated.qpos)
        reference_fk = kinematics.forward(reference.qpos)
        target_hands = (
            reference.keypoints
            if reference.keypoints is not None
            else np.concatenate(
                [reference_fk.hand_positions, reference_fk.hand_quaternions],
                axis=-1,
            )
        )
        if (
            reference.keypoints is not None
            and not reference.root_xy_known
            and reference.body_reference_available
        ):
            target_hands = _align_keypoint_xy_to_local_root(
                target_hands,
                reference_fk.hand_positions,
            )
        evaluation_slice = slice(protocol.warmup_frames, None)
        keyframes = np.arange(
            protocol.warmup_frames,
            generated.qpos.shape[0],
            protocol.keyframe_stride,
        )
        waypoints = np.arange(
            protocol.warmup_frames,
            generated.qpos.shape[0],
            protocol.waypoint_stride,
        )
        body_metrics = (
            (
                _mean_position_error(
                    generated_fk.body_positions[keyframes],
                    reference_fk.body_positions[keyframes],
                ),
                _mean_position_error(
                    generated.qpos[evaluation_slice, :2],
                    reference.qpos[evaluation_slice, :2],
                ),
                _mean_position_error(
                    generated.qpos[waypoints, :2],
                    reference.qpos[waypoints, :2],
                ),
            )
            if reference.body_reference_available
            else (float("nan"), float("nan"), float("nan"))
        )
        metrics = PairMetrics(
            skate_m_s=mean_foot_skate_m_s(
                generated_fk.sole_positions[evaluation_slice],
                fps=clip_fps,
                floor_height_m=protocol.floor_height_m,
                contact_height_m=protocol.foot_contact_height_m,
            ),
            joint_rot_deg=mean_quaternion_error_deg(
                generated_fk.hand_quaternions[evaluation_slice],
                target_hands[evaluation_slice, :, 3:7],
            ),
            joint_pos_m=_mean_position_error(
                generated_fk.hand_positions[evaluation_slice],
                target_hands[evaluation_slice, :, :3],
            ),
            keyframe_body_m=body_metrics[0],
            traj_m=body_metrics[1],
            waypoint_m=body_metrics[2],
        )
        clip_results.append(
            ClipResult(
                clip_id=clip_id,
                fps=clip_fps,
                metrics=metrics,
                body_reference_available=reference.body_reference_available,
            )
        )
        if embedding_roots is None:
            query_embeddings.append(
                hand_trajectory_descriptor(target_hands[evaluation_slice, :, :3])
            )
            generated_embeddings.append(
                hand_trajectory_descriptor(
                    generated_fk.hand_positions[evaluation_slice]
                )
            )
            if reference.body_reference_available:
                reference_embeddings.append(
                    motion_descriptor(
                        reference.qpos[evaluation_slice],
                        reference_fk.hand_positions[evaluation_slice],
                        clip_fps,
                    )
                )
                generated_fid_embeddings.append(
                    motion_descriptor(
                        generated.qpos[evaluation_slice],
                        generated_fk.hand_positions[evaluation_slice],
                        clip_fps,
                    )
                )
        else:
            condition_root, generated_root, reference_root = embedding_roots
            query_embedding = load_embedding(condition_root / f"{clip_id}.npy")
            generated_embedding = load_embedding(generated_root / f"{clip_id}.npy")
            reference_embedding = load_embedding(reference_root / f"{clip_id}.npy")
            if not (
                query_embedding.shape
                == generated_embedding.shape
                == reference_embedding.shape
            ):
                raise EvaluationInputError(
                    clip_id,
                    "condition, generated, and reference embeddings must have "
                    "matching dimensions",
                )
            query_embeddings.append(query_embedding)
            generated_embeddings.append(generated_embedding)
            reference_embeddings.append(reference_embedding)
            generated_fid_embeddings.append(generated_embedding)

    if embedding_roots is None:
        backend = (
            "ee_trajectory_retrieval+g1_kinematic_fid"
            if len(reference_embeddings) >= 2
            else "ee_trajectory_retrieval_only"
        )
    else:
        generated_fid_embeddings = generated_embeddings
        backend = "external_evaluator_embeddings"

    query_array = stack_embeddings(query_embeddings, "condition embeddings")
    generated_array = stack_embeddings(generated_embeddings, "generated embeddings")
    frechet = float("nan")
    if embedding_roots is not None or len(reference_embeddings) >= 2:
        reference_array = stack_embeddings(reference_embeddings, "reference embeddings")
        generated_fid_array = stack_embeddings(
            generated_fid_embeddings,
            "generated FID embeddings",
        )
        frechet = frechet_distance(generated_fid_array, reference_array)
    aggregate = AggregateMetrics(
        skate_m_s=mean_finite([clip.metrics.skate_m_s for clip in clip_results]),
        r_precision_top3_percent=r_precision_at_k(query_array, generated_array, k=3),
        fid=frechet,
        joint_rot_deg=mean_metric(
            [clip.metrics.joint_rot_deg for clip in clip_results], "joint_rot_deg"
        ),
        joint_pos_m=mean_metric(
            [clip.metrics.joint_pos_m for clip in clip_results], "joint_pos_m"
        ),
        keyframe_body_m=mean_finite(
            [clip.metrics.keyframe_body_m for clip in clip_results]
        ),
        traj_m=mean_finite([clip.metrics.traj_m for clip in clip_results]),
        waypoint_m=mean_finite([clip.metrics.waypoint_m for clip in clip_results]),
    )
    return EvaluationSummary(
        clips=clip_results,
        aggregate=aggregate,
        embedding_backend=backend,
    )
