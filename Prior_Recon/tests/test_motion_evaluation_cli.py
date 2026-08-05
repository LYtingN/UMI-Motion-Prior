from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from Prior_Recon.Masked_Flow.evaluation.kinematics import G1Kinematics


def _neutral_qpos(frame_count: int, joint_offset: float) -> NDArray[np.float32]:
    qpos = np.zeros((frame_count, 36), dtype=np.float32)
    qpos[:, 2] = 0.8
    qpos[:, 3] = 1.0
    qpos[:, 7:] = joint_offset
    return qpos


def test_cli_evaluates_paired_motion_directories(tmp_path: Path) -> None:
    # Given: two generated clips that are exactly equal to their references.
    generated_dir = tmp_path / "generated"
    reference_dir = tmp_path / "reference"
    generated_dir.mkdir()
    reference_dir.mkdir()
    for clip_id, joint_offset in (("clip_a", 0.0), ("clip_b", 0.05)):
        qpos = _neutral_qpos(frame_count=20, joint_offset=joint_offset)
        if clip_id == "clip_b":
            qpos[:, 2] = 2.0
        np.save(generated_dir / f"{clip_id}.npy", qpos)
        np.savez(reference_dir / f"{clip_id}.npz", qpos=qpos)
    output_path = tmp_path / "metrics.json"
    environment = os.environ.copy()
    environment["PYTHONPATH"] = str(Path(__file__).resolve().parents[2])

    # When: the public CLI evaluates and aggregates the directories.
    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "Prior_Recon.Masked_Flow.scripts.evaluate_motion_generation",
            "--generated",
            str(generated_dir),
            "--reference",
            str(reference_dir),
            "--output",
            str(output_path),
            "--fps",
            "30",
        ],
        check=False,
        capture_output=True,
        text=True,
        env=environment,
    )

    # Then: every paired-error metric and distribution distance is zero.
    assert completed.returncode == 0, completed.stderr
    report = json.loads(output_path.read_text(encoding="utf-8"))
    assert report["num_clips"] == 2
    assert report["skate_valid_clips"] == 1
    assert report["skate_omitted_clips"] == 1
    assert report["per_clip"][0]["fps"] == 30.0
    assert report["aggregate"]["skate_m_s"] == 0.0
    assert report["aggregate"]["joint_rot_deg"] == 0.0
    assert report["aggregate"]["joint_pos_m"] == 0.0
    assert report["aggregate"]["keyframe_body_m"] == 0.0
    assert report["aggregate"]["traj_m"] == 0.0
    assert report["aggregate"]["waypoint_m"] == 0.0
    assert report["metric_scope"] == "internal_proxy_not_paper_comparable"
    assert report["aggregate"]["proxy_r_precision_top3_percent"] == 100.0
    assert np.isclose(
        report["aggregate"]["proxy_frechet_distance"],
        0.0,
        atol=1e-8,
    )
    assert "r_precision_top3_percent" not in report["aggregate"]
    assert "fid" not in report["aggregate"]


def test_cli_uses_external_evaluator_embeddings_when_supplied(tmp_path: Path) -> None:
    # Given: paired motions and matching frozen-evaluator embeddings.
    generated_dir = tmp_path / "generated"
    reference_dir = tmp_path / "reference"
    embedding_dirs = tuple(
        tmp_path / name
        for name in ("condition", "generated_embedding", "reference_embedding")
    )
    generated_dir.mkdir()
    reference_dir.mkdir()
    for directory in embedding_dirs:
        directory.mkdir()
    for index, clip_id in enumerate(("clip_a", "clip_b")):
        qpos = _neutral_qpos(frame_count=20, joint_offset=index * 0.05)
        np.save(generated_dir / f"{clip_id}.npy", qpos)
        np.savez(reference_dir / f"{clip_id}.npz", qpos=qpos)
        embedding = np.eye(2, dtype=np.float64)[index]
        for directory in embedding_dirs:
            np.save(directory / f"{clip_id}.npy", embedding)
    output_path = tmp_path / "metrics.json"
    environment = os.environ.copy()
    environment["PYTHONPATH"] = str(Path(__file__).resolve().parents[2])

    # When: all three evaluator embedding directories are passed to the CLI.
    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "Prior_Recon.Masked_Flow.scripts.evaluate_motion_generation",
            "--generated",
            str(generated_dir),
            "--reference",
            str(reference_dir),
            "--condition-embeddings",
            str(embedding_dirs[0]),
            "--generated-embeddings",
            str(embedding_dirs[1]),
            "--reference-embeddings",
            str(embedding_dirs[2]),
            "--output",
            str(output_path),
        ],
        check=False,
        capture_output=True,
        text=True,
        env=environment,
    )

    # Then: retrieval/FID use the external backend and preserve perfect pairing.
    assert completed.returncode == 0, completed.stderr
    report = json.loads(output_path.read_text(encoding="utf-8"))
    assert report["embedding_backend"] == "external_evaluator_embeddings"
    assert report["metric_scope"] == "external_evaluator_protocol_user_verified"
    assert report["aggregate"]["r_precision_top3_percent"] == 100.0
    assert np.isclose(report["aggregate"]["fid"], 0.0, atol=1e-8)


def test_cli_rejects_embedding_dimension_mismatch_across_clips(tmp_path: Path) -> None:
    generated_dir = tmp_path / "generated"
    reference_dir = tmp_path / "reference"
    embedding_dirs = tuple(
        tmp_path / name for name in ("condition", "generated_emb", "reference_emb")
    )
    generated_dir.mkdir()
    reference_dir.mkdir()
    for directory in embedding_dirs:
        directory.mkdir()
    for index, clip_id in enumerate(("clip_a", "clip_b")):
        qpos = _neutral_qpos(frame_count=20, joint_offset=index * 0.05)
        np.save(generated_dir / f"{clip_id}.npy", qpos)
        np.save(reference_dir / f"{clip_id}.npy", qpos)
        embedding = np.ones(2 + index, dtype=np.float64)
        for directory in embedding_dirs:
            np.save(directory / f"{clip_id}.npy", embedding)
    environment = os.environ.copy()
    environment["PYTHONPATH"] = str(Path(__file__).resolve().parents[2])

    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "Prior_Recon.Masked_Flow.scripts.evaluate_motion_generation",
            "--generated",
            str(generated_dir),
            "--reference",
            str(reference_dir),
            "--condition-embeddings",
            str(embedding_dirs[0]),
            "--generated-embeddings",
            str(embedding_dirs[1]),
            "--reference-embeddings",
            str(embedding_dirs[2]),
            "--output",
            str(tmp_path / "metrics.json"),
        ],
        check=False,
        capture_output=True,
        text=True,
        env=environment,
    )

    assert completed.returncode != 0
    assert "shared dimension" in completed.stderr


def test_cli_aligns_absolute_keypoints_with_feat_local_root(tmp_path: Path) -> None:
    generated_dir = tmp_path / "generated"
    reference_dir = tmp_path / "reference"
    generated_dir.mkdir()
    reference_dir.mkdir()
    local_qpos = _neutral_qpos(frame_count=20, joint_offset=0.0)
    world_qpos = local_qpos.copy()
    world_qpos[:, :2] = np.array([3.0, -2.0], dtype=np.float32)
    world_fk = G1Kinematics().forward(world_qpos.astype(np.float64))
    keypoints = np.concatenate(
        [world_fk.hand_positions, world_fk.hand_quaternions],
        axis=-1,
    )
    feat = np.zeros((20, 70), dtype=np.float32)
    feat[:, 10] = 0.8
    for clip_id in ("clip_a", "clip_b"):
        np.save(generated_dir / f"{clip_id}.npy", local_qpos)
        np.savez(
            reference_dir / f"{clip_id}.npz",
            feat=feat,
            keypoints=keypoints,
            source_fps=np.array(30.0),
        )
    output_path = tmp_path / "metrics.json"
    environment = os.environ.copy()
    environment["PYTHONPATH"] = str(Path(__file__).resolve().parents[2])

    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "Prior_Recon.Masked_Flow.scripts.evaluate_motion_generation",
            "--generated",
            str(generated_dir),
            "--reference",
            str(reference_dir),
            "--output",
            str(output_path),
        ],
        check=False,
        capture_output=True,
        text=True,
        env=environment,
    )

    assert completed.returncode == 0, completed.stderr
    report = json.loads(output_path.read_text(encoding="utf-8"))
    assert np.isclose(report["aggregate"]["joint_pos_m"], 0.0, atol=1e-7)


def test_cli_rejects_generated_reference_fps_mismatch(tmp_path: Path) -> None:
    qpos = _neutral_qpos(frame_count=20, joint_offset=0.0)
    generated_path = tmp_path / "generated.npz"
    reference_path = tmp_path / "reference.npz"
    np.savez(generated_path, qpos=qpos, source_fps=np.array(20.0))
    np.savez(reference_path, qpos=qpos, source_fps=np.array(30.0))
    environment = os.environ.copy()
    environment["PYTHONPATH"] = str(Path(__file__).resolve().parents[2])

    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "Prior_Recon.Masked_Flow.scripts.evaluate_motion_generation",
            "--generated",
            str(generated_path),
            "--reference",
            str(reference_path),
            "--output",
            str(tmp_path / "metrics.json"),
        ],
        check=False,
        capture_output=True,
        text=True,
        env=environment,
    )

    assert completed.returncode != 0
    assert "fps mismatch" in completed.stderr


def test_cli_excludes_warmup_from_skate_and_proxy_fid(tmp_path: Path) -> None:
    generated_dir = tmp_path / "generated"
    reference_dir = tmp_path / "reference"
    generated_dir.mkdir()
    reference_dir.mkdir()
    for index, clip_id in enumerate(("clip_a", "clip_b")):
        reference_qpos = _neutral_qpos(frame_count=20, joint_offset=index * 0.05)
        generated_qpos = reference_qpos.copy()
        generated_qpos[:2, 0] = np.array([1.0, 0.5], dtype=np.float32)
        np.save(generated_dir / f"{clip_id}.npy", generated_qpos)
        np.save(reference_dir / f"{clip_id}.npy", reference_qpos)
    output_path = tmp_path / "metrics.json"
    environment = os.environ.copy()
    environment["PYTHONPATH"] = str(Path(__file__).resolve().parents[2])

    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "Prior_Recon.Masked_Flow.scripts.evaluate_motion_generation",
            "--generated",
            str(generated_dir),
            "--reference",
            str(reference_dir),
            "--output",
            str(output_path),
            "--warmup-frames",
            "2",
        ],
        check=False,
        capture_output=True,
        text=True,
        env=environment,
    )

    assert completed.returncode == 0, completed.stderr
    report = json.loads(output_path.read_text(encoding="utf-8"))
    assert report["aggregate"]["skate_m_s"] == 0.0
    assert np.isclose(
        report["aggregate"]["proxy_frechet_distance"],
        0.0,
        atol=1e-8,
    )
