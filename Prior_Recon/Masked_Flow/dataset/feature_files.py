from __future__ import annotations

from pathlib import Path


def select_feature_files(
    feat_root: Path,
    include_mirror: bool,
    val_split: float,
    split: str,
) -> list[Path]:
    pre_split_root = feat_root / split
    search_root = pre_split_root if pre_split_root.is_dir() else feat_root
    feat_files = sorted(search_root.rglob("*.npz")) + sorted(search_root.rglob("*.npy"))
    if not include_mirror:
        feat_files = [path for path in feat_files if not path.stem.endswith("_M")]
    if not feat_files:
        raise FileNotFoundError(f"No feature files found under {search_root}")
    if pre_split_root.is_dir():
        return feat_files

    n_val = max(1, int(len(feat_files) * val_split))
    return feat_files[:n_val] if split == "val" else feat_files[n_val:]
