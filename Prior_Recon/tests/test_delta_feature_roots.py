from __future__ import annotations

from pathlib import Path

from Prior_Recon.Masked_Flow.configs.load_config import config_from_yaml
from Prior_Recon.Masked_Flow.dataset.delta_dataset import _select_feature_files


def _create_feature(root: Path, relative: str) -> Path:
    path = root / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    path.touch()
    return path


def test_select_feature_files_when_train_directory_is_pre_split(
    tmp_path: Path,
) -> None:
    # Given
    train_file = _create_feature(tmp_path, "train/walk/train.npz")
    _create_feature(tmp_path, "val/wave/val.npz")
    _create_feature(tmp_path, "test/jump/test.npz")

    # When
    selected = _select_feature_files(tmp_path, True, 0.1, "train")

    # Then
    assert selected == [train_file]


def test_select_feature_files_when_val_directory_is_pre_split(tmp_path: Path) -> None:
    # Given
    _create_feature(tmp_path, "train/walk/train.npz")
    val_file = _create_feature(tmp_path, "val/wave/val.npz")
    _create_feature(tmp_path, "test/jump/test.npz")

    # When
    selected = _select_feature_files(tmp_path, True, 0.1, "val")

    # Then
    assert selected == [val_file]


def test_config_from_yaml_when_feat_root_is_declared(tmp_path: Path) -> None:
    # Given
    config_path = tmp_path / "config.yaml"
    config_path.write_text("feat_root: /datasets/delta_feat_v4\n", encoding="utf-8")

    # When
    config = config_from_yaml(config_path)

    # Then
    assert config.feat_root == "/datasets/delta_feat_v4"
