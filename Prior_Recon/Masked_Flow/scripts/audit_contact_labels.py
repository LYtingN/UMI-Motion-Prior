"""Audit GT foot-contact labels in delta-feature npz bundles.

Run on the training box (needs torch for the qpos decode import + mujoco):
    python -m Prior_Recon.Masked_Flow.scripts.audit_contact_labels \
        --data-dir <dir with *.npz>

For every clip it reports, per foot:
  * stored labels (feat[:, 5:7]): contact ratio, flicker (transitions/s), and
    the FK foot speed during labelled contact (mean / p95, m/s) -- if the
    labels were clean this speed should be ~0; large values mean swing frames
    were mislabelled as contact (the pre-fix labels used a 15 m/s velocity
    bound, i.e. height-only).
  * candidate relabel from FK ankle positions with the FIXED rule
    (speed < --vel-thr m/s AND height < --ht-thr m, majority-of-3 filtered):
    same stats plus agreement with the stored labels.

If the candidate columns look sane (contact speed ~0, ratio plausible) the
next step is regenerating the dataset with the fixed process_delta_motion.py
and retraining delta73_skate_small.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parents[3]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from Prior_Recon.Masked_Flow.dataset.process_delta_motion import (
    _foot_contacts,
)
from Prior_Recon.Masked_Flow.utils.assets import default_g1_mjcf_xml_path
from Prior_Recon.Masked_Flow.visual.foot_lock import _G1LegIK
from Prior_Recon.Masked_Flow.visual.recon_delta69 import (
    _delta69_raw_clip_to_qpos,
)


def _stats(contact: np.ndarray, speed: np.ndarray, fps: float) -> dict[str, float]:
    """contact: (T-1,) in {0,1}; speed: (T-1,) FK foot speed in m/s."""
    n = contact.shape[0]
    ratio = float(contact.mean()) if n else 0.0
    flips = float(np.abs(np.diff(contact)).sum()) / max(n / fps, 1e-6)
    in_c = contact > 0.5
    if in_c.sum() == 0:
        return {"ratio": ratio, "flips_per_s": flips, "v_mean": 0.0, "v_p95": 0.0}
    v = speed[in_c]
    return {
        "ratio": ratio,
        "flips_per_s": flips,
        "v_mean": float(v.mean()),
        "v_p95": float(np.percentile(v, 95)),
    }


def _confusion(reference: np.ndarray, candidate: np.ndarray) -> dict[str, int]:
    truth = np.asarray(reference) > 0.5
    predicted = np.asarray(candidate) > 0.5
    if truth.shape != predicted.shape:
        raise ValueError(
            f"reference and candidate shapes differ: {truth.shape} vs "
            f"{predicted.shape}"
        )
    return {
        "tp": int(np.sum(truth & predicted)),
        "fp": int(np.sum(~truth & predicted)),
        "fn": int(np.sum(truth & ~predicted)),
        "tn": int(np.sum(~truth & ~predicted)),
    }


def _classification_metrics(counts: dict[str, int]) -> dict[str, float]:
    tp, fp = counts["tp"], counts["fp"]
    fn, tn = counts["fn"], counts["tn"]
    positives = tp + fn
    negatives = fp + tn
    return {
        "precision": tp / max(tp + fp, 1),
        "recall": tp / max(positives, 1),
        "agreement": (tp + tn) / max(positives + negatives, 1),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--xml", default=str(default_g1_mjcf_xml_path()))
    ap.add_argument(
        "--fps", type=float, default=None,
        help="TRUE source frame rate; only needed for npz without a "
             "source_fps stamp (sonic SOMA corpus: 50). No default on "
             "purpose: auditing 50fps data at an assumed 30fps reproduces "
             "the very labelling bug this script exists to catch.",
    )
    ap.add_argument("--vel-thr", type=float, default=0.1, help="candidate m/s bound")
    ap.add_argument(
        "--vel-thresholds",
        type=float,
        nargs="+",
        default=(0.05, 0.1, 0.2, 0.3),
    )
    ap.add_argument("--ht-thr", type=float, default=0.08, help="candidate height bound (m)")
    ap.add_argument("--max-clips", type=int, default=None)
    args = ap.parse_args()

    files = sorted(Path(args.data_dir).rglob("*.npz"))
    if args.max_clips:
        files = files[: args.max_clips]
    if not files:
        raise SystemExit(f"No npz files under {args.data_dir}")

    ik = _G1LegIK(args.xml)
    agg: dict[str, list[dict[str, float]]] = {"stored": [], "fixed": []}
    agree_all: list[float] = []
    records: list[tuple[np.ndarray, np.ndarray, float]] = []

    header = (f"{'clip':<32} {'foot':<5} {'src':<7} {'ratio':>6} "
              f"{'flips/s':>8} {'v_mean':>7} {'v_p95':>7}")
    print(header)
    print("-" * len(header))
    for path in files:
        with np.load(path) as z:
            feat = z["feat"].astype(np.float32)
            fps = float(z["source_fps"]) if "source_fps" in z else args.fps
        if fps is None:
            raise SystemExit(
                f"{path} has no source_fps stamp and --fps was not given. "
                "Pass the corpus's true frame rate explicitly "
                "(sonic SOMA: --fps 50)."
            )
        qpos = _delta69_raw_clip_to_qpos(feat)
        for side, col in (("left", 5), ("right", 6)):
            foot = ik.foot_positions(qpos, side)          # (T, 3) world
            speed = np.linalg.norm(np.diff(foot, axis=0), axis=1) * fps
            stored = (feat[:-1, col] > 0.5).astype(np.float32)
            fixed = _foot_contacts(
                foot, fps=fps, vel_thr=args.vel_thr, ht_thr=args.ht_thr
            )
            records.append((stored, foot, fps))
            agree = float((stored == fixed).mean())
            agree_all.append(agree)
            for src, c in (("stored", stored), ("fixed", fixed)):
                s = _stats(c, speed, fps)
                agg[src].append(s)
                print(f"{path.stem[:32]:<32} {side:<5} {src:<7} "
                      f"{s['ratio']:>6.2f} {s['flips_per_s']:>8.2f} "
                      f"{s['v_mean']:>7.3f} {s['v_p95']:>7.3f}")

    print("\n=== aggregate (mean over clips x feet) ===")
    for src in ("stored", "fixed"):
        rows = agg[src]
        mean = {k: float(np.mean([r[k] for r in rows])) for k in rows[0]}
        print(f"{src:<7} ratio={mean['ratio']:.2f} flips/s={mean['flips_per_s']:.2f} "
              f"v_mean={mean['v_mean']:.3f} m/s  v_p95={mean['v_p95']:.3f} m/s")
    print(f"stored-vs-fixed agreement: {float(np.mean(agree_all)):.1%}")
    print("\n=== velocity-threshold calibration ===")
    print(
        f"{'threshold':>9} {'ratio':>7} {'v_p95':>8} "
        f"{'precision':>10} {'recall':>8} {'agreement':>10} {'pos_weight':>11}"
    )
    for threshold in args.vel_thresholds:
        counts = {"tp": 0, "fp": 0, "fn": 0, "tn": 0}
        contacts: list[np.ndarray] = []
        contact_speeds: list[np.ndarray] = []
        for stored, foot, fps in records:
            speed = np.linalg.norm(np.diff(foot, axis=0), axis=1) * fps
            candidate = _foot_contacts(
                foot, fps=fps, vel_thr=threshold, ht_thr=args.ht_thr
            )
            contacts.append(candidate)
            contact_speeds.append(speed[candidate > 0.5])
            row_counts = _confusion(candidate, stored)
            for key in counts:
                counts[key] += row_counts[key]
        labels = np.concatenate(contacts)
        nonempty_speeds = [values for values in contact_speeds if values.size]
        speeds = (
            np.concatenate(nonempty_speeds)
            if nonempty_speeds
            else np.empty((0,), dtype=np.float32)
        )
        metrics = _classification_metrics(counts)
        positives = int(np.sum(labels > 0.5))
        negatives = labels.size - positives
        pos_weight = negatives / max(positives, 1)
        print(
            f"{threshold:>9.3f} {labels.mean():>7.3f} "
            f"{(np.percentile(speeds, 95) if speeds.size else 0.0):>8.3f} "
            f"{metrics['precision']:>10.3f} {metrics['recall']:>8.3f} "
            f"{metrics['agreement']:>10.3f} {pos_weight:>11.3f}"
        )
    print("\nReading: clean contact labels should give v_mean ~0.02-0.05 m/s and "
          "v_p95 well under the vel threshold. If 'stored' shows large v_p95 "
          "(swing frames labelled as contact), regenerate the dataset with the "
          "fixed _foot_contacts before retraining delta73_skate_small.")


if __name__ == "__main__":
    main()
