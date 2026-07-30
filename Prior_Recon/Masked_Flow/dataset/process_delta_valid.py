#!/usr/bin/env python3
"""
process_delta_valid.py
~~~~~~~~~~~~~~~~~~~~~~~~
Build a held-out **validation** set of 70D delta motion features.

Selects motions present in `motion_feat_v2` but absent from `delta_feat_v2`
(i.e. never seen during training), samples N per recording session, runs the
same CSV -> feature pipeline as ``process_delta_motion.py`` and writes the
``.npz`` bundles directly under ``<out-dir>/<session_id>/<name>.npz``.

Usage
-----
  python process_delta_valid.py \\
      --csv-root     ~/NYX/g1_sonic_data/csv \\
      --motion-root  ~/NYX/g1_sonic_data/Full_npy/motion_feat_v2 \\
      --delta-root   ~/NYX/g1_sonic_data/delta_feat/delta_feat_v2 \\
      --out-dir      ~/NYX/g1_sonic_data/delta_feat_valid \\
      --xml-path     <repo>/assets/g1_29dof.xml \\
      --per-session 10 --workers 8 [--dry-run]
"""
from __future__ import annotations

import argparse
import multiprocessing as mp
import random
from pathlib import Path

import numpy as np

from process_delta_motion import CONTACT_LABEL_VERSION, process_file


def _rel_keys(root: Path, suffix: str) -> set[str]:
    """Relative paths (POSIX, no extension) of all *suffix* files under root."""
    return {
        p.relative_to(root).with_suffix("").as_posix()
        for p in root.rglob(f"*{suffix}")
    }


def _process_one(args: tuple) -> tuple[str, str | None]:
    csv_path, out_path, xml_path, fps, target_fps = args
    eff_fps = target_fps if target_fps else fps
    try:
        feat, keypoints = process_file(
            csv_path, xml_path, fps=fps, target_fps=target_fps
        )
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            out_path,
            feat=feat.astype(np.float32),
            keypoints=keypoints.astype(np.float32),
            contact_label_version=np.int32(CONTACT_LABEL_VERSION),
            source_fps=np.float32(eff_fps),
        )
        return csv_path, None
    except Exception as exc:  # noqa: BLE001 - report, keep going
        return csv_path, str(exc)


def main() -> None:
    ap = argparse.ArgumentParser(description="Build held-out delta-feature valid set")
    ap.add_argument("--csv-root", required=True)
    ap.add_argument("--motion-root", required=True)
    ap.add_argument("--delta-root", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--xml-path", required=True)
    ap.add_argument("--fps", type=float, required=True,
                    help="TRUE frame rate of the source CSVs (sonic SOMA: 50); "
                         "the contact velocity gate is in m/s.")
    ap.add_argument("--target-fps", type=float, default=None,
                    help="Resample to this rate before feature extraction. "
                         "MUST match whatever the train set was generated "
                         "with (e.g. 30).")
    ap.add_argument("--per-session", type=int, default=10)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--dry-run", action="store_true",
                    help="Only report selection counts; do not process.")
    args = ap.parse_args()

    csv_root = Path(args.csv_root).expanduser()
    motion_root = Path(args.motion_root).expanduser()
    delta_root = Path(args.delta_root).expanduser()
    out_dir = Path(args.out_dir).expanduser()

    motion_keys = _rel_keys(motion_root, ".npy")
    delta_keys = _rel_keys(delta_root, ".npz")
    held_out = sorted(motion_keys - delta_keys)
    print(f"motion={len(motion_keys)}  delta={len(delta_keys)}  "
          f"held_out={len(held_out)}")

    # Group held-out rel keys by session id (first path component).
    by_session: dict[str, list[str]] = {}
    for rel in held_out:
        by_session.setdefault(rel.split("/", 1)[0], []).append(rel)

    rng = random.Random(args.seed)
    selected: list[str] = []
    for session in sorted(by_session):
        pool = by_session[session]
        k = min(args.per_session, len(pool))
        selected.extend(rng.sample(pool, k))
    selected.sort()
    print(f"sessions={len(by_session)}  per_session={args.per_session}  "
          f"selected={len(selected)}")

    tasks: list[tuple[str, str, str, float, float | None]] = []
    n_missing_csv = 0
    for rel in selected:
        csv_path = csv_root / f"{rel}.csv"
        if not csv_path.exists():
            n_missing_csv += 1
            continue
        out_path = out_dir / f"{rel}.npz"
        if out_path.exists() and not args.overwrite:
            continue
        tasks.append(
            (str(csv_path), str(out_path), args.xml_path, args.fps,
             args.target_fps)
        )

    print(f"to_process={len(tasks)}  missing_csv={n_missing_csv}  -> {out_dir}")
    if args.dry_run:
        for t in tasks[:10]:
            print(f"  [sample] {t[0]} -> {t[1]}")
        print("dry-run: nothing written.")
        return

    n_ok = n_err = 0
    if args.workers <= 1:
        results = (_process_one(t) for t in tasks)
        for path, err in results:
            if err:
                n_err += 1
                print(f"[ERROR] {path}: {err}")
            else:
                n_ok += 1
    else:
        with mp.Pool(args.workers) as pool:
            for path, err in pool.imap_unordered(_process_one, tasks):
                if err:
                    n_err += 1
                    print(f"[ERROR] {path}: {err}")
                else:
                    n_ok += 1
    print(f"Done. ok={n_ok}  err={n_err}")


if __name__ == "__main__":
    main()
