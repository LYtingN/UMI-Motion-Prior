"""Verify the per-file FK cache reproduces the per-window FK bit-for-bit.

``_foot_local_from_raw`` + ``MaskedFlowDataset._compose_feet_from_local`` hoist the
30-body FK loop out of ``__getitem__`` by dropping the window-dependent yaw and
re-applying it per window. That is an exact algebraic identity, so this script
asserts it on REAL data at two levels:

  1. ``_oracle_features(feat, feet_local)`` vs ``_oracle_features(feat, None)``
     -- the composition math alone.
  2. Full ``MaskedFlowDataset[idx]`` dicts with the cache vs with the fallback
     forced -- also covers the window/lookahead slicing and hold-still padding.

Usage:
  python Prior_Recon/Masked_Flow/scripts/check_oracle_feet_equivalence.py \
      --config Prior_Recon/Masked_Flow/configs/sim_config/oracle_root_feet.yaml \
      --feat-root /data/nas_ray/home/eason.er/delta_feat_v4
"""

from __future__ import annotations

import argparse
import sys
import tempfile
from pathlib import Path

import torch


def _find_repo_root(path: Path) -> Path:
    for candidate in (path, *path.parents):
        if (candidate / "Prior_Recon").is_dir():
            return candidate
    raise RuntimeError(f"Could not find repo root above {path}")


sys.path.insert(0, str(_find_repo_root(Path(__file__).resolve())))

from Prior_Recon.Masked_Flow.configs.load_config import config_from_yaml
from Prior_Recon.Masked_Flow.dataset.delta_dataset import (
    G1DeltaFeatPrimitiveDataset,
    MaskedFlowDataset,
)

TOL = 1e-5


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--feat-root", required=True)
    parser.add_argument(
        "--num-files", type=int, default=6,
        help="Source files to link into the temp root (keeps index build fast).",
    )
    parser.add_argument("--num-items", type=int, default=24)
    return parser.parse_args()


def _mini_root(feat_root: Path, num_files: int, tmp: Path) -> Path:
    """Symlink a handful of real npz files into a train/+val/ temp root."""
    files = sorted((feat_root / "train").rglob("*.npz"))[:num_files]
    if not files:
        raise SystemExit(f"No npz files under {feat_root}/train")
    for split in ("train", "val"):
        split_dir = tmp / split
        split_dir.mkdir(parents=True)
        for src in files:
            (split_dir / src.name).symlink_to(src)
    return tmp


def main() -> None:
    args = _parse_args()
    cfg = config_from_yaml(args.config)
    if str(getattr(cfg, "oracle_condition", "none")) != "root_feet":
        raise SystemExit(
            f"{args.config} has oracle_condition="
            f"{getattr(cfg, 'oracle_condition', 'none')!r}; this check is for 'root_feet'."
        )

    with tempfile.TemporaryDirectory() as tmp:
        root = _mini_root(Path(args.feat_root), args.num_files, Path(tmp))
        base = G1DeltaFeatPrimitiveDataset(cfg, split="train", feat_root=str(root))
        wrapped = MaskedFlowDataset(base, cfg, truncate_lookahead=True)
        if base._foot_local_cache is None:
            raise SystemExit("per-file foot cache was not built -- nothing to check")

        n = min(args.num_items, len(base))
        step = max(len(base) // n, 1)
        indices = list(range(0, len(base), step))[:n]

        # --- Level 1: composition math on every segment of every sampled item ---
        worst_math = 0.0
        segments_checked = 0
        for idx in indices:
            item = base[idx]
            segments = item.get("segment_sequence", [item])
            for seg in segments:
                T = seg["feat"].shape[0]
                feat_ext = seg["feat"].reshape(T, -1)
                feet_ext = seg["oracle_feet_local"]
                if "oracle_feat_look" in seg:
                    feat_ext = torch.cat([feat_ext, seg["oracle_feat_look"]], dim=0)
                    feet_ext = torch.cat([feet_ext, seg["oracle_feet_local_look"]], dim=0)
                cached = wrapped._oracle_features(feat_ext, feet_ext)
                fk_only = wrapped._oracle_features(feat_ext, None)
                worst_math = max(worst_math, (cached - fk_only).abs().max().item())
                segments_checked += 1
        print(f"[1] composition vs FK: {segments_checked} segments, "
              f"max |diff| = {worst_math:.3e}")

        # --- Level 2: full item dicts, cache vs forced fallback ---
        original = MaskedFlowDataset._oracle_features

        def _fk_only(self, feat, feet_local=None):
            return original(self, feat, None)

        worst_item = 0.0
        keys_checked = set()
        for idx in indices:
            torch.manual_seed(1234 + idx)  # lookahead truncation is random
            new_item = wrapped[idx]
            MaskedFlowDataset._oracle_features = _fk_only
            try:
                torch.manual_seed(1234 + idx)
                old_item = wrapped[idx]
            finally:
                MaskedFlowDataset._oracle_features = original

            if set(new_item) != set(old_item):
                raise SystemExit(
                    f"key mismatch at idx={idx}: "
                    f"{sorted(set(new_item) ^ set(old_item))}"
                )
            for key, new_val in new_item.items():
                old_val = old_item[key]
                if isinstance(new_val, torch.Tensor):
                    if new_val.shape != old_val.shape:
                        raise SystemExit(
                            f"shape mismatch at idx={idx} key={key}: "
                            f"{tuple(new_val.shape)} vs {tuple(old_val.shape)}"
                        )
                    worst_item = max(worst_item, (new_val - old_val).abs().max().item())
                elif new_val != old_val:
                    raise SystemExit(f"value mismatch at idx={idx} key={key}")
                keys_checked.add(key)
        print(f"[2] full items: {len(indices)} items, {len(keys_checked)} keys, "
              f"max |diff| = {worst_item:.3e}")

        worst = max(worst_math, worst_item)
        if worst > TOL:
            raise SystemExit(f"FAIL: max |diff| {worst:.3e} exceeds tol {TOL:.1e}")
        print(f"PASS: max |diff| {worst:.3e} <= tol {TOL:.1e}")


if __name__ == "__main__":
    main()
