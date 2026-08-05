from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np

from Prior_Recon.Masked_Flow.evaluation.kinematics import G1Kinematics


def test_cli_marks_body_metrics_unavailable_for_ee_only_references(
    tmp_path: Path,
) -> None:
    # Given: UMI-style references with valid hands and all-zero body placeholders.
    generated_dir = tmp_path / "generated"
    reference_dir = tmp_path / "reference"
    generated_dir.mkdir()
    reference_dir.mkdir()
    qpos = np.zeros((20, 36), dtype=np.float32)
    qpos[:, 2] = 0.76
    qpos[:, 3] = 1.0
    fk = G1Kinematics().forward(qpos.astype(np.float64))
    keypoints = np.concatenate(
        [fk.hand_positions, fk.hand_quaternions],
        axis=-1,
    )
    for clip_id in ("clip_a", "clip_b"):
        np.save(generated_dir / f"{clip_id}.npy", qpos)
        np.savez(
            reference_dir / f"{clip_id}.npz",
            feat=np.zeros((20, 70), dtype=np.float32),
            keypoints=keypoints,
            source_fps=np.array(30.0),
        )
    output_path = tmp_path / "metrics.json"
    environment = os.environ.copy()
    environment["PYTHONPATH"] = str(Path(__file__).resolve().parents[2])

    # When: the public CLI evaluates the EE-only reference set.
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

    # Then: hand fidelity remains measurable without fabricated body scores.
    assert completed.returncode == 0, completed.stderr
    report = json.loads(output_path.read_text(encoding="utf-8"))
    assert np.isclose(report["aggregate"]["joint_pos_m"], 0.0, atol=1e-7)
    assert np.isclose(report["aggregate"]["joint_rot_deg"], 0.0, atol=1e-7)
    assert report["aggregate"]["keyframe_body_m"] is None
    assert report["aggregate"]["traj_m"] is None
    assert report["aggregate"]["waypoint_m"] is None
    assert report["aggregate"]["proxy_frechet_distance"] is None
    assert report["body_reference_valid_clips"] == 0
    assert report["body_reference_omitted_clips"] == 2
