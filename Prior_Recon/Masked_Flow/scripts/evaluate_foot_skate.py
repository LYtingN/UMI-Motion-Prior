from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parents[3]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from pipeline.Deploy.bridge.online_planner import OnlinePrimitivePlanner, SegmentSeed
from pipeline.Deploy.bridge.sonic.foot_diagnostics import (
    G1FootDiagnostics,
    summarize_foot_skate,
)
from Prior_Recon.Masked_Flow.utils.assets import default_g1_mjcf_xml_path


def _select_files(directory: Path, max_windows: int | None) -> list[Path]:
    files = sorted(directory.glob("*.npz"))
    if max_windows is None or len(files) <= max_windows:
        return files
    indices = np.linspace(0, len(files) - 1, max_windows, dtype=np.int64)
    return [files[index] for index in indices]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--diagnostic-dir", required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--ode-steps", type=int, default=None)
    parser.add_argument("--max-windows", type=int, default=8)
    parser.add_argument("--xml", default=str(default_g1_mjcf_xml_path()))
    parser.add_argument("--stance-p95-limit", type=float, default=0.05)
    parser.add_argument("--boundary-p95-limit", type=float, default=0.08)
    parser.add_argument("--enforce", action="store_true")
    args = parser.parse_args()

    files = _select_files(Path(args.diagnostic_dir), args.max_windows)
    if not files:
        raise SystemExit(f"No diagnostic npz files under {args.diagnostic_dir}")

    planner = OnlinePrimitivePlanner(
        args.ckpt,
        device=args.device,
        ode_steps=args.ode_steps,
        temperature=0.0,
    )
    diagnostics = G1FootDiagnostics(args.xml)
    all_steps: list[np.ndarray] = []
    all_contacts: list[np.ndarray] = []
    boundary_frames: list[int] = []
    frame_offset = 0

    for path in files:
        with np.load(path) as bundle:
            keypoints = bundle["keypoints_conditioned"].astype(np.float32)
            history = bundle["seed_history"].astype(np.float32)
            anchor_xy = bundle["prior_qpos"][0, :2].astype(np.float32)
            anchor_yaw = float(bundle["anchor_yaw"])
        seed = SegmentSeed(
            history_anchor=history,
            anchor_xy=anchor_xy,
            anchor_yaw=anchor_yaw,
        )
        planned = planner.plan_segment(
            seg_idx=0,
            start=0,
            seed=seed,
            kp_window=keypoints,
        )
        foot = diagnostics.evaluate(planned.qpos)
        contact = planned.feat69[:, 5:7] >= 0.5
        contact[0] = False
        all_steps.append(foot.foot_xy_step_m)
        all_contacts.append(contact)
        boundary_frames.extend(
            frame_offset + planner.hist_len + index * planner.future_len
            for index in range(planner.num_primitives)
        )
        frame_offset += planned.window_len

    summary = summarize_foot_skate(
        np.concatenate(all_steps),
        np.concatenate(all_contacts),
        fps=planner.fps,
        boundary_frames=np.asarray(boundary_frames),
        stance_p95_limit_m_s=args.stance_p95_limit,
        boundary_p95_limit_m_s=args.boundary_p95_limit,
    )
    verdict = "PASS" if summary.passed else "FAIL"
    print(f"checkpoint: {args.ckpt}")
    print(f"windows: {len(files)}")
    print(
        f"stance speed: p95={summary.stance_p95_m_s:.4f} m/s "
        f"max={summary.stance_max_m_s:.4f} m/s "
        f"(limit={args.stance_p95_limit:.4f})"
    )
    print(
        f"boundary speed: p95={summary.boundary_p95_m_s:.4f} m/s "
        f"max={summary.boundary_max_m_s:.4f} m/s "
        f"(limit={args.boundary_p95_limit:.4f})"
    )
    print(f"foot-skate acceptance: {verdict}")
    if args.enforce and not summary.passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
