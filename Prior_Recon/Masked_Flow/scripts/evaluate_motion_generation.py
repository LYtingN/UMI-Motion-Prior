from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from Prior_Recon.Masked_Flow.evaluation.evaluator import evaluate_motion_pairs
from Prior_Recon.Masked_Flow.evaluation.io import pair_motion_paths
from Prior_Recon.Masked_Flow.evaluation.models import EvaluationProtocol


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate Masked-Flow motion generation on paired G1 trajectories."
    )
    parser.add_argument("--generated", required=True, type=Path)
    parser.add_argument("--reference", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--fps", type=float, default=30.0)
    parser.add_argument("--warmup-frames", type=int, default=2)
    parser.add_argument("--keyframe-stride", type=int, default=16)
    parser.add_argument("--waypoint-stride", type=int, default=16)
    parser.add_argument("--floor-height", type=float, default=0.0)
    parser.add_argument("--foot-contact-height", type=float, default=0.05)
    parser.add_argument("--condition-embeddings", type=Path)
    parser.add_argument("--generated-embeddings", type=Path)
    parser.add_argument("--reference-embeddings", type=Path)
    return parser.parse_args()


def _json_number(value: float) -> float | None:
    return float(value) if np.isfinite(value) else None


def main() -> None:
    args = _parse_args()
    embedding_values = (
        args.condition_embeddings,
        args.generated_embeddings,
        args.reference_embeddings,
    )
    supplied_count = sum(value is not None for value in embedding_values)
    if supplied_count not in (0, 3):
        raise SystemExit("all three embedding directories must be supplied together")
    embedding_roots = embedding_values if supplied_count == 3 else None
    protocol = EvaluationProtocol(
        fps=args.fps,
        warmup_frames=args.warmup_frames,
        keyframe_stride=args.keyframe_stride,
        waypoint_stride=args.waypoint_stride,
        floor_height_m=args.floor_height,
        foot_contact_height_m=args.foot_contact_height,
    )
    summary = evaluate_motion_pairs(
        pair_motion_paths(args.generated, args.reference),
        protocol=protocol,
        embedding_roots=embedding_roots,
    )
    aggregate_report = {
        key: _json_number(value) for key, value in summary.aggregate._asdict().items()
    }
    uses_external_embeddings = (
        summary.embedding_backend == "external_evaluator_embeddings"
    )
    if not uses_external_embeddings:
        aggregate_report["proxy_r_precision_top3_percent"] = aggregate_report.pop(
            "r_precision_top3_percent"
        )
        aggregate_report["proxy_frechet_distance"] = aggregate_report.pop("fid")
    report = {
        "num_clips": len(summary.clips),
        "skate_valid_clips": sum(
            int(np.isfinite(clip.metrics.skate_m_s)) for clip in summary.clips
        ),
        "skate_omitted_clips": sum(
            int(not np.isfinite(clip.metrics.skate_m_s)) for clip in summary.clips
        ),
        "body_reference_valid_clips": sum(
            int(clip.body_reference_available) for clip in summary.clips
        ),
        "body_reference_omitted_clips": sum(
            int(not clip.body_reference_available) for clip in summary.clips
        ),
        "embedding_backend": summary.embedding_backend,
        "metric_scope": (
            "external_evaluator_protocol_user_verified"
            if uses_external_embeddings
            else "internal_proxy_not_paper_comparable"
        ),
        "protocol": protocol._asdict(),
        "aggregate": aggregate_report,
        "per_clip": [
            {
                "clip_id": clip.clip_id,
                "fps": clip.fps,
                "body_reference_available": clip.body_reference_available,
                **{
                    key: _json_number(value)
                    for key, value in clip.metrics._asdict().items()
                },
            }
            for clip in summary.clips
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(f"clips: {len(summary.clips)}")
    print(f"embedding backend: {summary.embedding_backend}")
    for key, value in aggregate_report.items():
        print(f"{key}: {value}")
    print(f"report: {args.output}")


if __name__ == "__main__":
    main()
