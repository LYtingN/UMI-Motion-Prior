"""
delta_dataset.py
  0:4   root_tilt_sincos  [4]
        Root roll/pitch tilt encoded as [sin(roll), cos(roll)-1, sin(pitch), cos(pitch)-1].
        Smoother near small angles, avoids angle wraparound.
  4     delta_yaw         [1]
        Inter-frame yaw delta (rotation relative to previous frame).
        Relative, not world-frame absolute.
  5:7   contact_mask      [2]
        Binary contact flags for left/right foot: 1 = planted, 0 = swing.
        Derived from dual thresholds on foot height and velocity.
  7:10  delta_trans_world [3]
        Root frame-to-frame displacement in world frame [dx, dy, dz], m/frame.
        Dataset's __getitem__ rotates xy to heading-local frame using the window's
        absolute_yaw, so the model sees orientation-aligned local displacement.
  10    height            [1]
        Root absolute height z in meters (not a delta).
  11:40 dof               [29]
        Absolute joint angles (rad) for all 29 joints: legs, waist, arms, wrists.
        Describes “what the current pose looks like.”
  40:69 delta_dof         [29]
        Inter-frame joint angle deltas (rad) for all 29 joints.
        Describes “how much each joint moved this frame.”
  69    absolute_yaw      [1] auxiliary
        Current frame's world-frame yaw. Auxiliary channel only: used to rotate
        delta_trans_world to heading-local coords, then immediately dropped — it
        never serves as a direct model input.

With cfg.abs_root_channels=True the model-facing feature gains 4 channels
(69 -> 73), appended at window-slice time (see append_abs_root_channels):
  69:71 xy_rel            [2]  position relative to window frame 0, anchor frame
  71:73 yaw_rel           [2]  (cos, sin) of accumulated turn since frame 0
These are the exact within-window cumsum of dims 7:9 / dim 4, so both layouts
describe identical motion; reconstruction reads them per frame instead of
integrating deltas (root-drift fix).
"""
from __future__ import annotations

import os
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable, Optional

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, DistributedSampler

from Prior_Recon.Masked_Flow.config import EEMaskedFlowConfig

# ---------------------------------------------------------------------------
# Keypoint column names — identical to g1_sonic_dataset.py
# ---------------------------------------------------------------------------

_HAND_KP_COLS = [
    "left_palm_x",  "left_palm_y",  "left_palm_z",
    "left_palm_qw", "left_palm_qx", "left_palm_qy", "left_palm_qz",
    "right_palm_x", "right_palm_y", "right_palm_z",
    "right_palm_qw","right_palm_qx","right_palm_qy","right_palm_qz",
]
_N_HAND_KP = 2
_KP_DIM = 7  # xyz_m + quat_wxyz


# ---------------------------------------------------------------------------
# Keypoint loading
# ---------------------------------------------------------------------------

def _load_keypoints_abs_from_csv(csv_path: str) -> np.ndarray | None:
    """Load absolute hand keypoints from CSV for segment-level slicing."""
    import pandas as pd

    try:
        df = pd.read_csv(csv_path, usecols=_HAND_KP_COLS)
    except Exception:
        return None
    if not all(c in df.columns for c in _HAND_KP_COLS):
        return None

    kp = df[_HAND_KP_COLS].values.astype(np.float32).reshape(-1, _N_HAND_KP, _KP_DIM)
    kp[:, :, :3] *= 0.01
    return kp


def _load_keypoints_abs_from_csv_path(csv_path: Path) -> np.ndarray | None:
    return _load_keypoints_abs_from_csv(str(csv_path))


def _coerce_hand_keypoints(keypoints_abs: np.ndarray) -> np.ndarray:
    """Accept legacy 5-keypoint bundles and keep only the two hand entries."""
    if keypoints_abs.ndim != 3 or keypoints_abs.shape[-1] != _KP_DIM:
        raise ValueError(f"Expected keypoints with shape (T, K, 7), got {keypoints_abs.shape}")
    if keypoints_abs.shape[1] == _N_HAND_KP:
        return keypoints_abs.astype(np.float32, copy=False)
    if keypoints_abs.shape[1] == 5:
        return keypoints_abs[:, :_N_HAND_KP].astype(np.float32, copy=False)
    raise ValueError(f"Unsupported keypoint count {keypoints_abs.shape[1]} in {keypoints_abs.shape}")


def _relative_keypoint_segment(
    keypoints_abs: np.ndarray,
    start: int,
    seq_len: int,
    yaw_0: float | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Build one heading-local segment of relative hand keypoints.

    Positions are rebased to frame 0 and rotated into the window heading frame by
    ``yaw_0``. Quaternions are ALSO rotated into the heading frame (left-multiply
    by R_z(-yaw_0)); this is invariant for any downstream relative-to-first
    rotation (the R_z cancels in R_0^{-1} R_t), so it does not change the legacy
    relative rot6d condition -- but it makes frame 0's quaternion equal the
    absolute-in-heading-frame initial orientation, which the Layer C EE anchor
    (use_ee_anchor) reads directly. Shared by dataset, recon and bridge so the
    anchor is computed identically everywhere.
    """
    w = keypoints_abs[start : start + seq_len].copy()
    z0 = w[0, :, 2].copy()
    w[:, :, :3] -= w[0:1, :, :3]
    if yaw_0 is not None:
        cos_y = np.cos(yaw_0)
        sin_y = np.sin(yaw_0)
        x = w[:, :, 0].copy()
        y = w[:, :, 1].copy()
        w[:, :, 0] = cos_y * x + sin_y * y
        w[:, :, 1] = -sin_y * x + cos_y * y
        # Rotate the wxyz quats by R_z(-yaw_0): q_heading = q_z(-yaw_0) (x) q_world.
        # q_z(-yaw_0) = [cos(yaw_0/2), 0, 0, -sin(yaw_0/2)]. Only w and z of the
        # left quat are non-zero, so the Hamilton product simplifies.
        cw = np.cos(yaw_0 * 0.5)
        sw = -np.sin(yaw_0 * 0.5)
        # .copy() so the in-place writes below don't alias the RHS reads.
        qw = w[:, :, 3].copy()
        qx = w[:, :, 4].copy()
        qy = w[:, :, 5].copy()
        qz = w[:, :, 6].copy()
        w[:, :, 3] = cw * qw - sw * qz
        w[:, :, 4] = cw * qx - sw * qy
        w[:, :, 5] = cw * qy + sw * qx
        w[:, :, 6] = cw * qz + sw * qw
    return w, z0


# ---------------------------------------------------------------------------
# Heading alignment
# ---------------------------------------------------------------------------

def append_abs_root_channels(w69: np.ndarray) -> np.ndarray:
    """Append the 4 absolute-root channels to a heading-aligned window.

    Input ``w69`` is (T, >=69) with delta channels already rotated into the
    window's anchor frame (``_heading_align`` output). Appends:

      dims 69:71  xy_rel  -- position relative to frame 0, anchor frame.
      dims 71:73  yaw_rel -- (cos, sin) of the accumulated turn since frame 0.

    Defined as the exact within-window cumsum of the delta channels (frame t
    stores the t -> t+1 increment, so frame 0 is the origin: xy_rel[0]=0,
    yaw_rel[0]=(1,0)). This matches ``_delta69_window_to_qpos``'s integration
    convention bit-for-bit, so the two layouts describe identical motion; the
    abs channels merely let reconstruction READ root state per frame instead
    of integrating per-frame deltas (drift fix, cfg.abs_root_channels).
    """
    T = w69.shape[0]
    out = np.zeros((T, 73), dtype=np.float32)
    out[:, :69] = w69[:, :69]
    if T > 1:
        out[1:, 69:71] = np.cumsum(w69[:-1, 7:9], axis=0)
        yaw_rel = np.concatenate([[0.0], np.cumsum(w69[:-1, 4])]).astype(np.float32)
    else:
        yaw_rel = np.zeros(T, dtype=np.float32)
    out[:, 71] = np.cos(yaw_rel)
    out[:, 72] = np.sin(yaw_rel)
    return out


def _heading_align(
    feat_70: np.ndarray, start: int, T: int, abs_root: bool = False
) -> np.ndarray:
    """Extract a window and rotate delta_trans_world into heading-local frame.

    Uses the absolute_yaw at the window's first frame (dim 69) so that
    delta_trans_local is always expressed with the robot's forward axis = +x.

    Args:
        feat_70: (T_all, 70) raw feature array from process_delta_motion.py
        start:   window start frame index
        T:       window length
        abs_root: also append the 4 absolute-root channels (xy_rel + yaw_rel
            cos/sin, see ``append_abs_root_channels``) -> (T, 73).

    Returns:
        (T, 69) heading-aligned feature (aux yaw channel stripped), or
        (T, 73) with the absolute-root channels appended when ``abs_root``.
    """
    w     = feat_70[start : start + T].copy()   # (T, 70)
    yaw_0 = float(w[0, 69])                      # window's reference heading

    cos_y = np.cos(yaw_0)
    sin_y = np.sin(yaw_0)

    # Rotate delta_trans (dims 7, 8) by -yaw_0:
    #   [dx_local]   [ cos  sin] [dx]
    #   [dy_local] = [-sin  cos] [dy]
    dx = w[:, 7].copy()
    dy = w[:, 8].copy()
    w[:, 7] =  cos_y * dx + sin_y * dy
    w[:, 8] = -sin_y * dx + cos_y * dy
    # dim 9 (delta_z) is invariant to heading rotation

    if abs_root:
        return append_abs_root_channels(w[:, :69])
    return w[:, :69]   # strip aux yaw -> (T, 69)


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

DELTA_FEAT_DIM = 69   # model-facing dimension (aux yaw stripped)


# ═══════════════════════════════════════════════════════════════════════════
# On-the-fly contact relabel (v1/v2 npz fix-up)
#
# Two generations of broken labels:
#   v1: velocity bound 0.5 m/frame = 15 m/s @30fps, never binding — labels
#       degenerated to a height-only test.
#   v2: correct 0.3 m/s rule but with fps hardcoded to 30 while the source
#       corpus (sonic SOMA) is 50 fps — speed understated by 0.6x, so swing
#       frames up to 0.5 m/s still passed the gate.
# Instead of regenerating every npz, ``_load_delta_file`` REBUILDS
# feat[:, 5:7] here at load time: FK the clip from its own channels (dof +
# root state), then apply the fixed rule (speed < 0.1 m/s AND height < 0.08 m,
# majority-filtered) at the TRUE source fps — identical to the repaired
# ``process_delta_motion._foot_contacts``.
#
# The true fps comes from the npz ``source_fps`` stamp (v3+), else the
# DELTA_SOURCE_FPS env var, else 30.0 (legacy assumption; see
# ``_check_contact_label_version`` which refuses label-dependent training
# when that fallback would be silently trusted).
#
# npz stamped with contact_label_version >= 4 (regenerated data) skip this
# and use their stored labels. Set DISABLE_CONTACT_RELABEL=1 to keep stale
# labels as-is (debug / exact reproduction only). The FK pass runs once per
# cache miss per worker; regenerating the npz removes this cost permanently.
# ═══════════════════════════════════════════════════════════════════════════

_RELABEL_FK = None  # lazily-built CPU FK, one instance per worker process

# Stored labels are trusted from this version on (see banner above).
_CONTACT_LABEL_VERSION_OK = 4


def _resolve_source_fps(stamped_fps: float | None) -> float | None:
    """True frame rate of a clip: npz stamp, else DELTA_SOURCE_FPS, else None."""
    if stamped_fps is not None:
        return float(stamped_fps)
    env = os.environ.get("DELTA_SOURCE_FPS")
    if env:
        return float(env)
    return None


def _recompute_contact_labels(feat_70: np.ndarray, fps: float) -> np.ndarray:
    """Rebuild feat[:, 5:7] from FK foot trajectories with the fixed rule."""
    from Prior_Recon.Masked_Flow.dataset.process_delta_motion import (
        _foot_contacts,
    )
    from Prior_Recon.Masked_Flow.loss.g1_kinematics import (
        G129DeltaForwardKinematics,
        euler_xyz_to_matrix,
    )

    T = feat_70.shape[0]
    if T < 2:
        return feat_70

    global _RELABEL_FK
    if _RELABEL_FK is None:
        _RELABEL_FK = G129DeltaForwardKinematics().eval()

    # Clip-local root state from the clip's own channels. GT deltas integrate
    # exactly (no model noise), and the contact rule only needs relative foot
    # motion + absolute height, so the arbitrary world origin is irrelevant.
    roll = np.arctan2(feat_70[:, 0], feat_70[:, 1] + 1.0).astype(np.float32)
    pitch = np.arctan2(feat_70[:, 2], feat_70[:, 3] + 1.0).astype(np.float32)
    yaw = feat_70[:, 69].astype(np.float32)
    root_pos = np.zeros((T, 3), dtype=np.float32)
    root_pos[:, 2] = feat_70[:, 10]
    root_pos[1:, :2] = np.cumsum(feat_70[:-1, 7:9], axis=0)  # world-frame deltas

    with torch.no_grad():
        rot = euler_xyz_to_matrix(
            torch.from_numpy(roll), torch.from_numpy(pitch), torch.from_numpy(yaw)
        ).view(1, T, 3, 3)
        out = _RELABEL_FK(
            torch.from_numpy(root_pos).view(1, T, 3),
            rot,
            torch.from_numpy(feat_70[:, 11:40].copy()).view(1, T, 29),
        )
        foot = out["foot_translation"][0].numpy()  # (T, 2, 3): left, right

    feat = feat_70.copy()
    for k, col in ((0, 5), (1, 6)):
        c = _foot_contacts(foot[:, k], fps=fps)  # fixed rule, (T-1,)
        feat[: T - 1, col] = c
        feat[T - 1, col] = c[-1]  # hold-pad the final frame
    return feat


def _load_delta_file(path: Path) -> tuple[np.ndarray, np.ndarray | None]:
    """Load one delta feature file.

    Returns:
        feat_70: (T_full, 70)
        keypoints_abs: optional (T_full, 2, 7)
    """
    if path.suffix == ".npz":
        with np.load(str(path)) as data:
            if "feat" not in data:
                raise KeyError(f"{path} missing 'feat' array")
            feat = data["feat"].astype(np.float32)
            keypoints = (
                _coerce_hand_keypoints(data["keypoints"].astype(np.float32))
                if "keypoints" in data
                else None
            )
            version = (
                int(data["contact_label_version"])
                if "contact_label_version" in data
                else 1
            )
            stamped_fps = (
                float(data["source_fps"]) if "source_fps" in data else None
            )
        # ── HERE: pre-v4 contact labels are rebuilt on the fly (see banner) ──
        if (
            version < _CONTACT_LABEL_VERSION_OK
            and os.environ.get("DISABLE_CONTACT_RELABEL") != "1"
        ):
            fps = _resolve_source_fps(stamped_fps) or 30.0
            feat = _recompute_contact_labels(feat, fps=fps)
        return feat, keypoints

    feat = np.load(str(path)).astype(np.float32)
    # legacy .npy has no version stamp -> same on-the-fly relabel
    if os.environ.get("DISABLE_CONTACT_RELABEL") != "1":
        fps = _resolve_source_fps(None) or 30.0
        feat = _recompute_contact_labels(feat, fps=fps)
    return feat, None


def _check_contact_label_version(cfg, files: list[Path]) -> None:
    """Report how contact labels will be sourced for label-dependent losses.

    Pre-v4 npz get relabelled on the fly in ``_load_delta_file``. That relabel
    needs the TRUE source fps (velocity gate is in m/s); when neither the npz
    ``source_fps`` stamp nor DELTA_SOURCE_FPS can provide it, the loader would
    silently fall back to 30 fps — the exact v2 bug — so with label-dependent
    losses enabled that combination errors instead. Same for
    DISABLE_CONTACT_RELABEL=1 (stale labels trained on directly).
    """
    lc = getattr(cfg, "loss", None)
    label_dependent_weights = (
        "contact_weight",
        "foot_contact_weight",
        "foot_skate_weight",
        "foot_sole_pos_weight",
        "foot_sole_skate_weight",
        "boundary_foot_skate_weight",
        "foot_rot_weight",
        "contact_pred_weight",
    )
    needs_labels = lc is not None and any(
        getattr(lc, name, 0.0) > 0.0 for name in label_dependent_weights
    )
    if not needs_labels or not files:
        return
    probe = next((p for p in files if p.suffix == ".npz"), None)
    if probe is None:
        return
    with np.load(str(probe)) as data:
        version = int(data["contact_label_version"]) if "contact_label_version" in data else 1
        stamped_fps = float(data["source_fps"]) if "source_fps" in data else None
    if version >= _CONTACT_LABEL_VERSION_OK:
        return
    if os.environ.get("DISABLE_CONTACT_RELABEL") == "1":
        raise RuntimeError(
            f"contact-dependent losses are enabled, {probe} has pre-fix "
            f"v{version} contact labels, and DISABLE_CONTACT_RELABEL=1 turns "
            "the on-the-fly fix off. Unset it or regenerate the dataset."
        )
    fps = _resolve_source_fps(stamped_fps)
    if fps is None:
        raise RuntimeError(
            f"contact-dependent losses are enabled and {probe} has "
            f"v{version} contact labels needing an on-the-fly relabel, but the "
            "true source fps is unknown (no npz source_fps stamp, "
            "DELTA_SOURCE_FPS unset). Set DELTA_SOURCE_FPS to the corpus frame "
            "rate (sonic SOMA: 50) or regenerate with process_delta_motion.py "
            "--fps. Refusing to silently assume 30 fps — that was the v2 bug."
        )
    print(
        f"[delta_dataset] v{version} contact labels detected: rebuilding "
        f"feat[:, 5:7] on the fly at load time (FK + 0.1 m/s rule @ {fps:g} "
        "fps). Regenerate the npz with process_delta_motion.py --fps to "
        "remove the per-load FK cost."
    )


def _feat_cache_capacity() -> int:
    """Number of source files each dataset keeps cached in RAM.

    The dataset draws hundreds of thousands of windows in shuffled order from a
    few tens of thousands of source files, so a tiny LRU cache almost always
    misses and every ``__getitem__`` re-reads + decompresses an npz from disk —
    that CPU I/O, not the GPU, is the training bottleneck. The whole feature set
    is only a few GB while the box has ~1TB RAM, so by default we cache enough
    files to hold everything resident after the first epoch (no repeat disk hits).

    Override with FEAT_CACHE_MAX_FILES (e.g. a smaller number on a RAM-tight box,
    or 0 to disable caching entirely and always stream from disk).
    """
    raw = os.environ.get("FEAT_CACHE_MAX_FILES", "100000")
    try:
        return max(int(raw), 0)
    except ValueError:
        return 100000


class _ArrayCache:
    """Per-process LRU cache for large source files."""

    def __init__(self, loader: Callable[[Path], object], max_files: int = 4) -> None:
        self.loader = loader
        self.max_files = max(int(max_files), 0)
        self._items: OrderedDict[Path, object] = OrderedDict()

    def get(self, path: Path):
        if self.max_files <= 0:
            return self.loader(path)
        key = path.resolve()
        if key in self._items:
            self._items.move_to_end(key)
            return self._items[key]
        value = self.loader(path)
        self._items[key] = value
        if len(self._items) > self.max_files:
            self._items.popitem(last=False)
        return value


@dataclass(frozen=True)
class _WindowRecord:
    feat_path: Path
    csv_path: Path | None
    start: int


def _iter_starts(total_len: int, window_len: int, stride: int) -> Iterable[int]:
    stride = max(int(stride), 1)
    return range(0, total_len - window_len + 1, stride)


def _select_feature_files(
    feat_root: Path,
    include_mirror: bool,
    val_split: float,
    split: str,
) -> list[Path]:
    feat_files = sorted(feat_root.rglob("*.npz")) + sorted(feat_root.rglob("*.npy"))
    if not include_mirror:
        feat_files = [f for f in feat_files if not f.stem.endswith("_M")]
    if not feat_files:
        raise FileNotFoundError(f"No feature files found under {feat_root}")

    n_val = max(1, int(len(feat_files) * val_split))
    return feat_files[:n_val] if split == "val" else feat_files[n_val:]


class G1DeltaFeatDataset:
    """Dataset loading precomputed delta features + embedded or CSV keypoints.
    Returns per item:
        feat      : (T, 69, 1) float32  heading-aligned delta features
        keypoints : (T, 2, 7)  float32  left_palm, right_palm (pos_m + quat_wxyz)
        kp_z0     : (2,)       float32  absolute z of left+right palm at t=0
        fps       : float
    """

    def __init__(
        self,
        cfg: EEMaskedFlowConfig,
        split: str = "train",
        feat_root: Optional[str | Path] = None,
        csv_root: Optional[str | Path] = None,
        include_mirror: bool = True,
        fps: float = 30.0,
    ):
        if feat_root is None:
            raise ValueError("G1DeltaFeatDataset requires feat_root")

        T      = cfg.motion.seq_len
        stride = cfg.motion.window_stride
        feat_root = Path(feat_root)
        csv_root_path = Path(csv_root) if csv_root is not None else None

        chosen = _select_feature_files(
            feat_root,
            include_mirror=include_mirror,
            val_split=cfg.train.val_split,
            split=split,
        )
        _check_contact_label_version(cfg, chosen)

        self._records: list[_WindowRecord] = []
        cache_files = _feat_cache_capacity()
        self._feat_cache = _ArrayCache(_load_delta_file, max_files=cache_files)
        self._csv_cache = _ArrayCache(_load_keypoints_abs_from_csv_path, max_files=cache_files)

        for p in chosen:
            rel_path = p.relative_to(feat_root)
            raw, embedded_keypoints = self._feat_cache.get(p)
            if raw.shape[1] != 70:
                print(f"  [WARN] unexpected shape {raw.shape} in {p}, skipping")
                continue
            T_all = raw.shape[0]
            if T_all < T:
                continue

            csv_path = None
            if embedded_keypoints is not None:
                if embedded_keypoints.shape != (T_all, _N_HAND_KP, _KP_DIM):
                    print(
                        f"  [WARN] embedded keypoints shape {embedded_keypoints.shape} "
                        f"mismatch in {p}, skipping"
                    )
                    continue
            else:
                if csv_root_path is None:
                    print(
                        f"  [WARN] legacy file {p} has no embedded keypoints and csv_root is not set, skipping"
                    )
                    continue
                csv_path = csv_root_path / rel_path.with_suffix(".csv")
                if not csv_path.exists():
                    continue
                keypoints_abs = _load_keypoints_abs_from_csv(str(csv_path))
                if (
                    keypoints_abs is None
                    or keypoints_abs.shape != (T_all, _N_HAND_KP, _KP_DIM)
                ):
                    continue

            for start in _iter_starts(T_all, T, stride):
                self._records.append(_WindowRecord(p, csv_path, int(start)))
            del raw, embedded_keypoints

        if not self._records:
            raise RuntimeError(
                f"G1DeltaFeatDataset: no clips extracted "
                f"(split={split}, feat_root={feat_root}, csv_root={csv_root})"
            )

        self._fps = fps
        self._T   = T
        self._abs_root = bool(getattr(cfg, "abs_root_channels", False))

    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return len(self._records)

    def __getitem__(self, idx: int) -> dict:
        record = self._records[idx]
        raw, embedded_keypoints = self._feat_cache.get(record.feat_path)
        keypoints_abs = embedded_keypoints
        if keypoints_abs is None and record.csv_path is not None:
            keypoints_abs = self._csv_cache.get(record.csv_path)
        if keypoints_abs is None:
            raise RuntimeError(f"No keypoints available for {record.feat_path}")

        feat_raw = _heading_align(raw, record.start, self._T, abs_root=self._abs_root)
        kp, kp_z0 = _relative_keypoint_segment(
            keypoints_abs,
            record.start,
            self._T,
            yaw_0=float(raw[record.start, 69]),
        )
        feat  = torch.from_numpy(feat_raw.astype(np.float32)).unsqueeze(-1)  # (T, 69, 1)
        kp    = torch.from_numpy(kp)                                          # (T, 2, 7)
        kp_z0 = torch.from_numpy(kp_z0)                                       # (2,)
        return {"feat": feat, "keypoints": kp, "kp_z0": kp_z0, "fps": float(self._fps)}

    def summary(self) -> str:
        return (
            f"G1DeltaFeatDataset(clips={len(self)}, lazy=True, feat_dim={DELTA_FEAT_DIM}, "
            f"T={self._T}, fps={self._fps})"
        )


class G1DeltaFeatPrimitiveDataset:
    """Segment-level dataset for primitive rollout training.

    Each item is one heading-aligned segment of length
    ``history_len + num_primitives * future_len``. Primitive slicing happens in
    the wrapper so that both motion features and EE conditioning share the same
    segment-level alignment.
    """

    def __init__(
        self,
        cfg: EEMaskedFlowConfig,
        split: str = "train",
        feat_root: Optional[str | Path] = None,
        csv_root: Optional[str | Path] = None,
        include_mirror: bool = True,
        fps: float = 30.0,
    ):
        if feat_root is None:
            raise ValueError("G1DeltaFeatPrimitiveDataset requires feat_root")
        if not cfg.primitive.enabled:
            raise ValueError("Primitive dataset requires cfg.primitive.enabled=True")

        seg_len = cfg.primitive.segment_len
        segment_unrolls = int(getattr(cfg.primitive, "segment_unrolls", 1))
        if segment_unrolls < 1:
            raise ValueError("primitive.segment_unrolls must be at least 1")
        record_len = cfg.primitive.unroll_len
        stride = cfg.primitive.segment_stride
        feat_root = Path(feat_root)
        csv_root_path = Path(csv_root) if csv_root is not None else None

        chosen = _select_feature_files(
            feat_root,
            include_mirror=include_mirror,
            val_split=cfg.train.val_split,
            split=split,
        )
        _check_contact_label_version(cfg, chosen)

        self._records: list[_WindowRecord] = []
        cache_files = _feat_cache_capacity()
        self._feat_cache = _ArrayCache(_load_delta_file, max_files=cache_files)
        self._csv_cache = _ArrayCache(_load_keypoints_abs_from_csv_path, max_files=cache_files)

        for path in chosen:
            rel_path = path.relative_to(feat_root)
            raw, embedded_keypoints = self._feat_cache.get(path)
            if raw.shape[1] != 70:
                print(f"  [WARN] unexpected shape {raw.shape} in {path}, skipping")
                continue

            total_len = raw.shape[0]
            if total_len < record_len:
                continue

            keypoints_abs = embedded_keypoints
            csv_path = None
            if keypoints_abs is not None:
                if keypoints_abs.shape != (total_len, _N_HAND_KP, _KP_DIM):
                    print(
                        f"  [WARN] embedded keypoints shape {keypoints_abs.shape} "
                        f"mismatch in {path}, skipping"
                    )
                    continue
            else:
                if csv_root_path is None:
                    print(
                        f"  [WARN] legacy file {path} has no embedded keypoints and csv_root is not set, skipping"
                    )
                    continue
                csv_path = csv_root_path / rel_path.with_suffix(".csv")
                if not csv_path.exists():
                    continue
                keypoints_abs = _load_keypoints_abs_from_csv(str(csv_path))
                if (
                    keypoints_abs is None
                    or keypoints_abs.shape != (total_len, _N_HAND_KP, _KP_DIM)
                ):
                    continue

            for start in _iter_starts(total_len, record_len, stride):
                self._records.append(_WindowRecord(path, csv_path, int(start)))
            del raw, embedded_keypoints

        if not self._records:
            raise RuntimeError(
                f"G1DeltaFeatPrimitiveDataset: no segments extracted "
                f"(split={split}, feat_root={feat_root}, csv_root={csv_root})"
            )

        self._fps = fps
        self._segment_len = seg_len
        self._segment_step = cfg.primitive.segment_step
        self._segment_unrolls = segment_unrolls
        self._look_len = int(getattr(cfg, "lookahead_len", 0))
        self._abs_root = bool(getattr(cfg, "abs_root_channels", False))

    def __len__(self) -> int:
        return len(self._records)

    def _segment_item(
        self,
        raw: np.ndarray,
        keypoints_abs: np.ndarray,
        start: int,
    ) -> dict:
        feat_raw = _heading_align(raw, start, self._segment_len, abs_root=self._abs_root)
        seg_len = self._segment_len
        look_len = self._look_len
        # Slice keypoints past the segment end for EE lookahead, so all frames
        # share the segment's frame-0 anchor and heading. Clips that end early
        # are padded by holding the last real frame (the deployment convention
        # for "no more preview available").
        avail = min(keypoints_abs.shape[0] - start, seg_len + look_len)
        kp_ext, kp_z0 = _relative_keypoint_segment(
            keypoints_abs,
            start,
            avail,
            yaw_0=float(raw[start, 69]),
        )
        kp_seg = kp_ext[:seg_len]
        feat = torch.from_numpy(feat_raw.astype(np.float32)).unsqueeze(-1)
        kp = torch.from_numpy(kp_seg)
        kp_z0 = torch.from_numpy(kp_z0)
        item = {
            "feat": feat,
            "keypoints": kp,
            "kp_z0": kp_z0,
            "fps": float(self._fps),
            "anchor_yaw": float(raw[start, 69]),
        }

        if look_len > 0:
            n_real = avail - seg_len
            kp_look = np.repeat(kp_ext[-1:], look_len, axis=0)
            kp_look[:n_real] = kp_ext[seg_len:avail]
            look_valid = np.zeros(look_len, dtype=np.float32)
            look_valid[:n_real] = 1.0
            item["keypoints_look"] = torch.from_numpy(kp_look)
            item["look_frames_valid"] = torch.from_numpy(look_valid)
        return item

    def __getitem__(self, idx: int) -> dict:
        record = self._records[idx]
        raw, embedded_keypoints = self._feat_cache.get(record.feat_path)
        keypoints_abs = embedded_keypoints
        if keypoints_abs is None and record.csv_path is not None:
            keypoints_abs = self._csv_cache.get(record.csv_path)
        if keypoints_abs is None:
            raise RuntimeError(f"No keypoints available for {record.feat_path}")

        segments = [
            self._segment_item(
                raw,
                keypoints_abs,
                record.start + segment_idx * self._segment_step,
            )
            for segment_idx in range(self._segment_unrolls)
        ]
        if self._segment_unrolls == 1:
            return segments[0]
        return {"segment_sequence": segments}

    def summary(self) -> str:
        return (
            f"G1DeltaFeatPrimitiveDataset(segments={len(self)}, lazy=True, feat_dim={DELTA_FEAT_DIM}, "
            f"T={self._segment_len}, unrolls={self._segment_unrolls}, fps={self._fps})"
        )


def _quat_wxyz_to_rot6d(q: torch.Tensor) -> torch.Tensor:
    q = F.normalize(q, dim=-1)
    w, x, y, z = q.unbind(-1)
    mat = torch.stack(
        [
            1 - 2 * (y * y + z * z),
            2 * (x * y - w * z),
            2 * (x * z + w * y),
            2 * (x * y + w * z),
            1 - 2 * (x * x + z * z),
            2 * (y * z - w * x),
            2 * (x * z - w * y),
            2 * (y * z + w * x),
            1 - 2 * (x * x + y * y),
        ],
        dim=-1,
    ).reshape(q.shape[:-1] + (3, 3))
    return mat[..., :2].reshape(q.shape[:-1] + (6,))


def _quat_wxyz_mul(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    w1, x1, y1, z1 = a.unbind(-1)
    w2, x2, y2, z2 = b.unbind(-1)
    return torch.stack(
        [
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
        ],
        dim=-1,
    )


def _quat_wxyz_relative_to_first(q: torch.Tensor) -> torch.Tensor:
    q = F.normalize(q, dim=-1)
    q0_inv = q[:1].clone()
    q0_inv[..., 1:] *= -1.0
    return _quat_wxyz_mul(q0_inv.expand_as(q), q)


class MaskedFlowDataset(Dataset):
    def __init__(
        self,
        base_dataset: Dataset,
        cfg: EEMaskedFlowConfig,
        truncate_lookahead: bool = True,
    ) -> None:
        self.base = base_dataset
        self.cfg = cfg
        self.wrist_idx = cfg.skeleton.wrist_local_indices
        self.ee_state_dim = int(getattr(cfg, "ee_state_dim", 0))
        if self.ee_state_dim not in (0, 18):
            raise ValueError(
                f"ee_state_dim must be 0 or 18 (2 hands x pos3+rot6d), got {self.ee_state_dim}."
            )
        if self.ee_state_dim > 0 and not getattr(cfg, "use_ee_pos", False):
            raise ValueError("ee_state_dim > 0 requires use_ee_pos=True (keypoint data).")
        self.lookahead_len = int(getattr(cfg, "lookahead_len", 0))
        if self.lookahead_len > 0 and not getattr(cfg, "use_ee_pos", False):
            raise ValueError("lookahead_len > 0 requires use_ee_pos=True (keypoint data).")
        # The absolute EE anchor (Layer C) is read from the palm keypoint quats,
        # so it only exists on the keypoint condition path. Without use_ee_pos the
        # wrist-feature fallback in _build_s_ee produces no anchor block, so
        # ee_feat_dim (which counts the anchor) would not match the built s_ee.
        if getattr(cfg, "use_ee_anchor", False) and not getattr(cfg, "use_ee_pos", False):
            raise ValueError("use_ee_anchor=True requires use_ee_pos=True (keypoint data).")
        # Randomly truncate the preview during training so short-horizon
        # deployment states (episode end, stale DP output) stay in-distribution.
        # Validation uses the full preview for a stable metric.
        self.truncate_lookahead = bool(truncate_lookahead)

    def __len__(self) -> int:
        return len(self.base)

    def _output_item(self, item: dict) -> dict:
        if getattr(self.cfg, "use_ee_height_anchor", False):
            return item
        return {key: value for key, value in item.items() if key != "kp_z0"}

    def _build_s_ee_from_keypoints(
        self,
        kp: torch.Tensor,
        kp_z0: torch.Tensor | None,
    ) -> torch.Tensor:
        T = kp.shape[0]
        quat_rel = _quat_wxyz_relative_to_first(kp[..., 3:])
        kp9 = torch.cat([kp[..., :3], _quat_wxyz_to_rot6d(quat_rel)], dim=-1)
        s_ee = kp9.reshape(T, -1)

        if self.cfg.use_ee_height_anchor and kp_z0 is not None:
            s_ee = torch.cat([s_ee, kp_z0.unsqueeze(0).expand(T, -1)], dim=-1)

        if getattr(self.cfg, "use_ee_anchor", False):
            # Layer C: absolute initial hand ORIENTATION in the heading frame.
            # kp is already heading-rotated (see _relative_keypoint_segment), so
            # frame 0's quat IS the heading-frame absolute orientation. Broadcast
            # this static 12D (2 hands x rot6d) anchor across all frames so it can
            # ride in the per-frame condition without a separate token.
            anchor6d = _quat_wxyz_to_rot6d(kp[:1, :, 3:]).reshape(1, -1)  # (1, 12)
            s_ee = torch.cat([s_ee, anchor6d.expand(T, -1)], dim=-1)

        if self.cfg.use_ee_vel:
            palm = kp9.reshape(T, -1)
            vel = torch.zeros_like(palm)
            vel[1:] = palm[1:] - palm[:-1]
            s_ee = torch.cat([s_ee, vel], dim=-1)

        return s_ee

    @staticmethod
    def _rebase_keypoints_to_window(
        kp: torch.Tensor,
        kp_z0: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        kp = kp.clone()
        if kp.shape[0] == 0:
            return kp, kp_z0
        offset = kp[:1, :, :3].clone()
        kp[:, :, :3] -= offset
        if kp_z0 is None:
            return kp, None
        return kp, kp_z0 + offset[0, :, 2]

    def _build_s_ee(self, item: dict) -> torch.Tensor:
        feat = item["feat"]
        T = feat.shape[0]

        if self.cfg.use_ee_pos and "keypoints" in item:
            kp = item["keypoints"]
            kp_z0 = item.get("kp_z0")
            s_ee = self._build_s_ee_from_keypoints(kp, kp_z0)
        else:
            s_ee = feat[:, self.wrist_idx, :].reshape(T, -1)

        return s_ee

    def _build_ee_state_segment(self, item: dict) -> torch.Tensor:
        """Segment-anchored per-frame EE pose columns (T, 18).

        Uses the exact arithmetic of the segment_anchored_v1 condition path
        (`_build_s_ee_primitive_windows`), so the pinned state columns are
        bit-identical to `s_ee_prim[..., :18]` and the existing ee_cond losses
        keep referring to the same values.
        """
        if "keypoints" not in item:
            raise ValueError("ee_state_dim > 0 requires keypoint data in the dataset item.")
        kp_seg, _ = self._rebase_keypoints_to_window(item["keypoints"], None)
        s_ee_seg = self._build_s_ee_from_keypoints(kp_seg, None)
        return s_ee_seg[..., : self.ee_state_dim]

    def _build_lookahead_prim_windows(
        self,
        item: dict,
        prim_cfg,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Per-primitive EE preview windows (N, K, E) plus validity (N, K).

        Preview frames live PAST each primitive's window end and share the
        segment anchor, so preview and window condition are one continuous
        trajectory. Frames past the clip end (base-dataset padding) and past a
        randomly sampled truncation point are replaced by a hold-still repeat
        and flagged invalid — feature content never leaks truth the validity
        mask denies, matching what a deployed DP horizon can actually provide.
        """
        K = self.lookahead_len
        if "keypoints_look" not in item or "look_frames_valid" not in item:
            raise ValueError(
                "lookahead_len > 0 requires the base dataset to emit "
                "'keypoints_look'/'look_frames_valid' (G1DeltaFeatPrimitiveDataset)."
            )
        kp_ext = torch.cat([item["keypoints"], item["keypoints_look"]], dim=0)
        kp_ext_reb, kp_z0_reb = self._rebase_keypoints_to_window(kp_ext, item.get("kp_z0"))
        s_ee_ext = self._build_s_ee_from_keypoints(kp_ext_reb, kp_z0_reb)
        seg_len = item["keypoints"].shape[0]
        valid_ext = torch.cat(
            [
                torch.ones(seg_len, dtype=torch.float32),
                item["look_frames_valid"].to(torch.float32),
            ]
        )

        look_prim: list[torch.Tensor] = []
        valid_prim: list[torch.Tensor] = []
        for prim_idx in range(prim_cfg.num_primitives):
            end = prim_idx * prim_cfg.future_len + prim_cfg.primitive_len
            feat = s_ee_ext[end : end + K].clone()
            valid = valid_ext[end : end + K].clone()
            n_real = int(valid.sum().item())
            if self.truncate_lookahead:
                if torch.rand(()) < 0.5:
                    cutoff = n_real
                else:
                    cutoff = min(int(torch.randint(0, K + 1, ()).item()), n_real)
            else:
                cutoff = n_real
            if cutoff < K:
                pad_src = (feat[cutoff - 1] if cutoff > 0 else s_ee_ext[end - 1]).clone()
                if self.cfg.use_ee_vel:
                    # Hold-still semantics: a held reference has zero velocity.
                    # Copying the last real frame verbatim would claim the hands
                    # keep moving while the pose stays frozen — contradictory
                    # content in tokens the invalid embedding asks the model to
                    # discount. Matches the s_ee a held keypoint would produce.
                    pad_src[-18:] = 0.0
                feat[cutoff:] = pad_src
                valid[cutoff:] = 0.0
            look_prim.append(feat)
            valid_prim.append(valid)
        return torch.stack(look_prim, dim=0), torch.stack(valid_prim, dim=0)

    def _build_s_ee_primitive_windows(self, item: dict, prim_cfg) -> torch.Tensor:
        if self.cfg.use_ee_pos and "keypoints" in item:
            kp = item["keypoints"]
            kp_z0 = item.get("kp_z0")
            # segment_anchored_v1: rebase keypoints ONCE to the segment's first frame
            # and build s_ee on the whole segment, so that EE position, rot6d, and
            # velocity stay continuous across primitive boundaries. Primitive slicing
            # is then a pure view over the segment-wide condition.
            kp_seg, kp_z0_seg = self._rebase_keypoints_to_window(kp, kp_z0)
            s_ee_seg = self._build_s_ee_from_keypoints(kp_seg, kp_z0_seg)
            s_ee_prim = []
            for prim_idx in range(prim_cfg.num_primitives):
                start = prim_idx * prim_cfg.future_len
                end = start + prim_cfg.primitive_len
                s_ee_prim.append(s_ee_seg[start:end])
            return torch.stack(s_ee_prim, dim=0)

        s_ee = self._build_s_ee(item)
        s_ee_prim = []
        for prim_idx in range(prim_cfg.num_primitives):
            start = prim_idx * prim_cfg.future_len
            end = start + prim_cfg.primitive_len
            s_ee_prim.append(s_ee[start:end])
        return torch.stack(s_ee_prim, dim=0)

    def _primitive_item(self, item: dict) -> dict:
        feat = item["feat"]
        T = feat.shape[0]
        dof = feat.reshape(T, -1)
        if self.ee_state_dim > 0:
            dof = torch.cat([dof, self._build_ee_state_segment(item)], dim=-1)

        prim_cfg = self.cfg.primitive
        prim_len = prim_cfg.primitive_len
        seg_len = prim_cfg.segment_len
        if T != seg_len:
            raise ValueError(
                f"Primitive dataset expected segment len {seg_len}, got {T}."
            )

        s_full_prim = []
        for prim_idx in range(prim_cfg.num_primitives):
            start = prim_idx * prim_cfg.future_len
            end = start + prim_len
            s_full_prim.append(dof[start:end])
        s_ee = self._build_s_ee(item)
        s_ee_prim = self._build_s_ee_primitive_windows(item, prim_cfg)
        out_item = self._output_item(item)

        result = {
            **out_item,
            "segment_full": dof,
            "segment_ee": s_ee,
            "s_full_prim": torch.stack(s_full_prim, dim=0),
            "s_ee_prim": s_ee_prim,
        }
        if self.lookahead_len > 0:
            look_prim, look_valid = self._build_lookahead_prim_windows(item, prim_cfg)
            result["s_ee_look_prim"] = look_prim
            result["look_valid_prim"] = look_valid
            # Raw lookahead keypoint tensors are only intermediate inputs;
            # drop them so default_collate sees a stable schema.
            result.pop("keypoints_look", None)
            result.pop("look_frames_valid", None)
        return result

    def __getitem__(self, idx: int) -> dict:
        item = self.base[idx]
        if "segment_sequence" not in item:
            return self._primitive_item(item)

        segments = [self._primitive_item(segment) for segment in item["segment_sequence"]]
        result = {}
        for key in segments[0]:
            values = [segment[key] for segment in segments]
            if isinstance(values[0], torch.Tensor):
                result[key] = torch.stack(values, dim=0)
            else:
                result[key] = torch.tensor(values)
        return result


def build_masked_flow_loaders(
    cfg: EEMaskedFlowConfig,
    dataset_factory: Callable,
) -> tuple[DataLoader, DataLoader]:
    train_base = dataset_factory(cfg, "train")
    val_base = dataset_factory(cfg, "val")

    train_ds = MaskedFlowDataset(train_base, cfg, truncate_lookahead=True)
    val_ds = MaskedFlowDataset(val_base, cfg, truncate_lookahead=False)
    num_workers = getattr(cfg.train, "num_workers", 0)
    # pin_memory only benefits async DMA when num_workers > 0
    pin = torch.cuda.is_available() and num_workers > 0

    # Optional DDP: RANK/WORLD_SIZE are set by torchrun. When absent this block
    # is skipped and the loaders behave exactly as in single-GPU training.
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    rank = int(os.environ.get("RANK", 0))
    train_sampler = val_sampler = None
    batch_size = cfg.train.batch_size
    if world_size > 1:
        # cfg.train.batch_size is the GLOBAL batch; split it across ranks so the
        # effective batch size (and thus LR schedule) matches single-GPU runs.
        if batch_size % world_size != 0:
            raise ValueError(
                f"batch_size={batch_size} must be divisible by world_size={world_size}"
            )
        batch_size = batch_size // world_size
        train_sampler = DistributedSampler(
            train_ds, num_replicas=world_size, rank=rank, shuffle=True, drop_last=True
        )
        val_sampler = DistributedSampler(
            val_ds, num_replicas=world_size, rank=rank, shuffle=False, drop_last=False
        )

    kwargs = dict(batch_size=batch_size, num_workers=num_workers, pin_memory=pin)
    if num_workers > 0:
        # Keep workers alive across epochs (avoid per-epoch respawn) and let each
        # worker stage a few batches ahead so the GPU is not starved by I/O.
        # Both are no-ops when num_workers == 0, preserving legacy behaviour.
        kwargs["persistent_workers"] = True
        kwargs["prefetch_factor"] = 4
    return (
        DataLoader(
            train_ds,
            shuffle=(train_sampler is None),
            sampler=train_sampler,
            drop_last=(train_sampler is not None),
            **kwargs,
        ),
        DataLoader(
            val_ds,
            shuffle=False,
            sampler=val_sampler,
            **kwargs,
        ),
    )
