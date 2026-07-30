"""Step-0 drift diagnosis: quantify per-frame root-delta bias vs GT and plot
cumulative world error. Decides which fix path to take:

  * Large NET SIGNED BIAS in delta_yaw / body-frame step  -> linear drift.
    Fixable cheaply by inference-time bias subtraction, durably by the
    absolute-channel representation (cfg.abs_root_channels).
  * Bias ~0 but cumulative error still grows ~sqrt(T)     -> random walk.
    Representation change helps less; invest in closed-loop re-anchoring.

Pure numpy on purpose (no torch/scipy): runs anywhere the training box has
python + numpy.

Usage (on the training box):

  # 1) produce a predicted rollout for a long clip, e.g. via the recon script
  #    or the bridge with --mock --save-qpos:
  python -m pipeline.Experiment.bridge.sonic.run_bridge \
      --ckpt .../emft_best.pt \
      --data .../data_test/wbt-plates-long/ep_00000.npz \
      --device cuda:0 --mock --save-qpos /tmp/wbt_plates_long_qpos.npy

  # 2) compare against the GT features in the same npz:
  python -m Prior_Recon.Masked_Flow.scripts.measure_root_drift \
      --pred-qpos /tmp/wbt_plates_long_qpos.npy \
      --gt .../data_test/wbt-plates-long/ep_00000.npz \
      --plot /tmp/wbt_plates_long_drift.png
"""

from __future__ import annotations

import argparse

import numpy as np

# delta69/70 channel conventions (see dataset/process_delta_motion.py and
# visual/recon_delta69.py): dim 4 = delta_yaw (t -> t+1), dims 7:9 = world
# delta_xy (t -> t+1), dim 69 (aux, feat70 only) = absolute yaw.
_DELTA_YAW = 4
_DELTA_XY = slice(7, 9)
_ABS_YAW = 69


def _wrap_to_pi(a: np.ndarray) -> np.ndarray:
    return (a + np.pi) % (2.0 * np.pi) - np.pi


def _yaw_from_quat_wxyz(q: np.ndarray) -> np.ndarray:
    """(T, 4) wxyz -> (T,) yaw. Matches euler 'xyz' z-component used repo-wide."""
    w, x, y, z = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
    return np.arctan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))


def _unwrap(a: np.ndarray) -> np.ndarray:
    return np.unwrap(a)


def _rotate_into_heading(dxy: np.ndarray, yaw: np.ndarray) -> np.ndarray:
    """Rotate world-frame per-frame steps into the robot's current heading frame.

    Returns (T, 2) [forward, lateral] components: bias here is directly
    interpretable ("overshoots each step by X mm", "veers left by Y mm").
    """
    c, s = np.cos(yaw), np.sin(yaw)
    fwd = c * dxy[:, 0] + s * dxy[:, 1]
    lat = -s * dxy[:, 0] + c * dxy[:, 1]
    return np.stack([fwd, lat], axis=-1)


def load_gt(path: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """GT world xy trajectory, absolute yaw, and per-frame deltas from feat70.

    GT world positions are the cumsum of the stored world deltas (dims 7:9) —
    the same reconstruction the training features are defined by, so pred and
    GT are compared under identical conventions.
    """
    with np.load(path) as data:
        feat = data["feat"].astype(np.float64)
    if feat.ndim != 2 or feat.shape[1] < 70:
        raise ValueError(f"Expected GT feat (T, 70), got {feat.shape}")
    dxy = feat[:, _DELTA_XY]  # world frame, frame t stores t -> t+1
    xy = np.zeros((feat.shape[0], 2))
    xy[1:] = np.cumsum(dxy[:-1], axis=0)
    yaw = _unwrap(feat[:, _ABS_YAW])
    dyaw = feat[:, _DELTA_YAW]
    return xy, yaw, np.concatenate([dxy, dyaw[:, None]], axis=-1)


def load_pred(path: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Predicted world xy, yaw, and per-frame deltas from a (T, 36) qpos rollout."""
    qpos = np.load(path).astype(np.float64)
    if qpos.ndim != 2 or qpos.shape[1] != 36:
        raise ValueError(f"Expected pred qpos (T, 36), got {qpos.shape}")
    xy = qpos[:, :2]
    yaw = _unwrap(_yaw_from_quat_wxyz(qpos[:, 3:7]))
    dxy = np.zeros_like(xy)
    dxy[:-1] = xy[1:] - xy[:-1]
    dxy[-1] = dxy[-2]
    dyaw = np.zeros_like(yaw)
    dyaw[:-1] = _wrap_to_pi(yaw[1:] - yaw[:-1])
    dyaw[-1] = dyaw[-2]
    return xy, yaw, np.concatenate([dxy, dyaw[:, None]], axis=-1)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pred-qpos", required=True, help="(T, 36) predicted world qpos .npy")
    parser.add_argument("--gt", required=True, help="GT episode .npz containing feat (T, 70)")
    parser.add_argument("--fps", type=float, default=30.0)
    parser.add_argument("--plot", default=None, help="optional output .png (needs matplotlib)")
    args = parser.parse_args()

    gt_xy, gt_yaw, gt_d = load_gt(args.gt)
    pr_xy, pr_yaw, pr_d = load_pred(args.pred_qpos)

    T = min(len(gt_xy), len(pr_xy))
    gt_xy, gt_yaw, gt_d = gt_xy[:T], gt_yaw[:T], gt_d[:T]
    pr_xy, pr_yaw, pr_d = pr_xy[:T], pr_yaw[:T], pr_d[:T]

    # Align frame 0 (the rollout is seeded from the GT start pose; any residual
    # offset is initialization, not drift).
    yaw0_off = _wrap_to_pi(pr_yaw[0] - gt_yaw[0])
    pr_yaw = pr_yaw - yaw0_off
    c, s = np.cos(-yaw0_off), np.sin(-yaw0_off)
    rot0 = np.array([[c, -s], [s, c]])
    pr_xy = (pr_xy - pr_xy[0]) @ rot0.T + gt_xy[0]
    pr_d[:, :2] = pr_d[:, :2] @ rot0.T

    # ---- per-frame delta errors, in the GT heading frame -------------------
    gt_step = _rotate_into_heading(gt_d[:, :2], gt_yaw)
    pr_step = _rotate_into_heading(pr_d[:, :2], gt_yaw)
    step_err = pr_step - gt_step  # (T, 2) [forward, lateral]
    dyaw_err = pr_d[:, 2] - gt_d[:, 2]

    # last delta of each stream is a padded repeat -> exclude from stats
    step_err, dyaw_err = step_err[:-1], dyaw_err[:-1]
    n = len(dyaw_err)

    def _bias_row(name: str, err: np.ndarray, unit: str, scale: float) -> None:
        bias, mae, std = err.mean() * scale, np.abs(err).mean() * scale, err.std() * scale
        # Significance of the mean under an iid-ish assumption: |bias| vs its
        # standard error. > ~3 means the systematic component is real.
        z = abs(err.mean()) / max(err.std() / np.sqrt(n), 1e-12)
        proj = bias * n  # linear extrapolation of pure-bias accumulation
        print(
            f"  {name:<18s} bias {bias:+9.4f} {unit:<8s} (z={z:5.1f})   "
            f"MAE {mae:8.4f}   std {std:8.4f}   bias x T = {proj:+9.3f}"
        )

    print(f"\n=== per-frame delta error (n={n} frames @ {args.fps:.0f} fps) ===")
    _bias_row("delta_yaw", dyaw_err, "deg/frame", 180.0 / np.pi)
    _bias_row("step forward", step_err[:, 0], "mm/frame", 1000.0)
    _bias_row("step lateral", step_err[:, 1], "mm/frame", 1000.0)

    # ---- cumulative world error --------------------------------------------
    yaw_cum = _wrap_to_pi(pr_yaw - gt_yaw)
    xy_cum = np.linalg.norm(pr_xy - gt_xy, axis=-1)

    # xy error caused by yaw alone: integrate GT step lengths under the
    # predicted heading error (isolates the "yaw amplifier" contribution).
    gt_heading_pred = gt_yaw + yaw_cum
    dxy_yaw_only = np.zeros_like(gt_d[:, :2])
    dxy_yaw_only[:, 0] = np.cos(gt_heading_pred) * gt_step[:, 0] - np.sin(gt_heading_pred) * gt_step[:, 1]
    dxy_yaw_only[:, 1] = np.sin(gt_heading_pred) * gt_step[:, 0] + np.cos(gt_heading_pred) * gt_step[:, 1]
    xy_yaw_only = np.zeros_like(gt_xy)
    xy_yaw_only[1:] = gt_xy[0] + np.cumsum(dxy_yaw_only[:-1], axis=0)
    xy_yaw_only[0] = gt_xy[0]
    xy_cum_yaw_only = np.linalg.norm(xy_yaw_only - gt_xy, axis=-1)

    print("\n=== cumulative world error ===")
    print(f"  {'frame':>8s} {'t(s)':>7s} {'yaw err(deg)':>13s} {'xy err(m)':>10s} {'xy from yaw alone(m)':>21s}")
    for frac in (0.25, 0.5, 0.75, 1.0):
        i = min(int(T * frac), T - 1)
        print(
            f"  {i:8d} {i / args.fps:7.1f} {np.degrees(yaw_cum[i]):13.2f} "
            f"{xy_cum[i]:10.3f} {xy_cum_yaw_only[i]:21.3f}"
        )

    # ---- growth-shape check: linear (bias) vs sqrt (random walk) -----------
    # Fit log|err| ~ p*log(t): p ~ 1 -> bias-dominated, p ~ 0.5 -> random walk.
    t = np.arange(1, T)
    for name, series in (("yaw", np.abs(yaw_cum[1:])), ("xy", xy_cum[1:])):
        valid = series > 1e-6
        if valid.sum() > 100:
            p = np.polyfit(np.log(t[valid]), np.log(series[valid]), 1)[0]
            verdict = "bias-dominated (linear)" if p > 0.75 else (
                "random-walk (sqrt)" if p > 0.35 else "bounded/noisy")
            print(f"  {name} error growth exponent p = {p:.2f}  -> {verdict}")

    # ---- staleness probe: does step error grow with in-window turn rate? ---
    print("\n=== delta_xy error binned by |gt delta_yaw| (heading-staleness probe) ===")
    abs_dyaw = np.abs(gt_d[:-1, 2])
    edges = np.quantile(abs_dyaw, [0.0, 0.5, 0.9, 1.0])
    labels = ["straight (p0-50)", "turning (p50-90)", "fast turn (p90-100)"]
    for k, label in enumerate(labels):
        m = (abs_dyaw >= edges[k]) & (abs_dyaw <= edges[k + 1])
        if m.sum() > 10:
            mae = np.abs(step_err[m]).mean() * 1000.0
            print(f"  {label:<20s} n={int(m.sum()):6d}   step MAE {mae:8.3f} mm/frame")

    if args.plot:
        try:
            import matplotlib

            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
        except ImportError:
            print("\n[plot skipped: matplotlib not installed]")
            return
        fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))
        ts = np.arange(T) / args.fps
        axes[0].plot(ts, np.degrees(yaw_cum))
        axes[0].set(title="cumulative yaw error", xlabel="t (s)", ylabel="deg")
        axes[1].plot(ts, xy_cum, label="total")
        axes[1].plot(ts, xy_cum_yaw_only, label="from yaw alone")
        axes[1].set(title="cumulative xy error", xlabel="t (s)", ylabel="m")
        axes[1].legend()
        axes[2].plot(gt_xy[:, 0], gt_xy[:, 1], label="GT")
        axes[2].plot(pr_xy[:, 0], pr_xy[:, 1], label="pred")
        axes[2].set(title="world xy trajectory", xlabel="x (m)", ylabel="y (m)")
        axes[2].axis("equal")
        axes[2].legend()
        for ax in axes:
            ax.grid(alpha=0.3)
        fig.tight_layout()
        fig.savefig(args.plot, dpi=130)
        print(f"\n[plot written to {args.plot}]")


if __name__ == "__main__":
    main()
