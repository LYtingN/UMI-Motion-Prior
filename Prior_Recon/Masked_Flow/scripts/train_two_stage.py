"""Train the two-stage (root -> body) cascade masked-flow prior.

Mirrors train_delta_flow.py but loads a TwoStageMaskedFlowConfig and uses the
cascade trainer. --config is required (there is no built-in two-stage default).

    python Prior_Recon/Masked_Flow/scripts/train_two_stage.py \
        --config Prior_Recon/Masked_Flow/configs/twostage_delta73_small.yaml
"""
from __future__ import annotations

import argparse
import os
import re
import sys
from datetime import datetime
from pathlib import Path

import torch

# See train_delta_flow.py: lift the 64MB /dev/shm cap for num_workers>0.
torch.multiprocessing.set_sharing_strategy("file_system")


def _find_repo_root(path: Path) -> Path:
    for candidate in (path, *path.parents):
        if (candidate / "Prior_Recon").is_dir():
            return candidate
    raise RuntimeError(f"Could not find repo root (looked for 'Prior_Recon/') above {path}")


sys.path.insert(0, str(_find_repo_root(Path(__file__).resolve())))

from Prior_Recon.Masked_Flow.configs.load_config_two_stage import (
    two_stage_config_from_yaml,
)
from Prior_Recon.Masked_Flow.dataset.delta_dataset import (
    G1DeltaFeatPrimitiveDataset,
)
from Prior_Recon.Masked_Flow.trainer_two_stage import TwoStageMaskedFlowTrainer

_DEFAULT_FEAT_ROOT = "data/delta_feat"
_DATE_SUFFIX_RE = re.compile(r"_\d{4}_\d{2}_\d{2}$")


def _append_today_to_ckpt_dir(ckpt_dir: str) -> str:
    path = Path(ckpt_dir)
    if _DATE_SUFFIX_RE.search(path.name):
        return ckpt_dir
    dated_name = f"{path.name}_{datetime.now().strftime('%Y_%m_%d')}"
    return str(path.with_name(dated_name))


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train the two-stage (root -> body) cascade masked-flow model",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--config", default=None, metavar="YAML",
        help="Two-stage YAML config (e.g. configs/twostage_delta73_small.yaml). "
             "Required unless --resume is given.",
    )
    parser.add_argument(
        "--feat-root", default=None, metavar="PATH",
        help=f"Root dir of precomputed delta feature files. Default: {_DEFAULT_FEAT_ROOT}",
    )
    parser.add_argument(
        "--csv-root", default=None, metavar="PATH",
        help="Root dir of legacy CSV keypoint files (only needed for .npy datasets).",
    )
    parser.add_argument(
        "--resume", default=None, metavar="CKPT",
        help="Checkpoint (.pt) to resume training from.",
    )
    parser.add_argument(
        "--device", default=None, metavar="DEVICE",
        help="Compute device, e.g. 'cuda', 'cuda:1', 'cpu'. Default: auto.",
    )
    parser.add_argument(
        "--ckpt-dir", default=None, metavar="PATH",
        help="Override cfg.train.ckpt_dir (checkpoint save directory).",
    )
    parser.add_argument(
        "--batch-size", type=int, default=None, metavar="N",
        help="Override cfg.train.batch_size.",
    )
    parser.add_argument(
        "--microbatch-size", type=int, default=None, metavar="N",
        help="Override cfg.train.max_microbatch_size. Use this to cap per-step GPU memory.",
    )
    parser.add_argument(
        "--num-workers", type=int, default=None, metavar="N",
        help="Override DataLoader worker count.",
    )
    parser.add_argument(
        "--gpu-memory-fraction", type=float, default=None, metavar="F",
        help="Limit this process to a fraction of GPU memory, e.g. 0.85.",
    )
    parser.add_argument(
        "--amp",
        choices=("none", "bf16"),
        default=None,
        help="Mixed precision mode. Use bf16 on CUDA to speed up Transformer forward.",
    )
    parser.add_argument(
        "--tb-log-dir", default=None, metavar="PATH",
        help="TensorBoard log directory. Default: <ckpt-dir>/tensorboard.",
    )
    parser.add_argument(
        "--no-tensorboard", action="store_true",
        help="Disable TensorBoard logging.",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()

    if args.resume:
        resume_state = torch.load(args.resume, map_location="cpu", weights_only=False)
        resume_epoch = resume_state.get("epoch")
        if resume_epoch in (None, 0):
            match = re.search(r"ep(\d+)", Path(args.resume).stem)
            if match:
                inferred_epoch = int(match.group(1))
                if inferred_epoch > 0:
                    resume_state["epoch"] = inferred_epoch
                    print(
                        "[WARN] checkpoint saved with epoch=0; "
                        f"inferred resume epoch={inferred_epoch} from filename: {args.resume}"
                    )
        cfg = resume_state.get("cfg")
        if cfg is None or not hasattr(cfg, "root_stage"):
            raise ValueError(
                f"{args.resume} does not carry a two-stage cfg; cannot resume with "
                "the cascade trainer."
            )
        if args.config:
            print(f"[WARN] --config ignored when --resume is set "
                  f"(restoring cfg from checkpoint: {args.resume})")
    else:
        resume_state = None
        if not args.config:
            raise SystemExit("--config is required (no built-in two-stage default).")
        print(f"Loading two-stage config from: {args.config}")
        cfg = two_stage_config_from_yaml(args.config)

    if args.ckpt_dir:
        cfg.train.ckpt_dir = args.ckpt_dir
    elif not args.resume:
        cfg.train.ckpt_dir = _append_today_to_ckpt_dir(cfg.train.ckpt_dir)
    if args.batch_size is not None:
        cfg.train.batch_size = args.batch_size
    if args.microbatch_size is not None:
        cfg.train.max_microbatch_size = args.microbatch_size
    if args.num_workers is not None:
        cfg.train.num_workers = args.num_workers
    if args.gpu_memory_fraction is not None:
        cfg.train.gpu_memory_fraction = args.gpu_memory_fraction
    if args.amp is not None:
        cfg.train.amp = args.amp

    feat_root = os.path.expanduser(args.feat_root or _DEFAULT_FEAT_ROOT)
    csv_root = os.path.expanduser(args.csv_root) if args.csv_root else None

    tensorboard_log_dir = None
    if not args.no_tensorboard:
        tensorboard_log_dir = args.tb_log_dir or str(Path(cfg.train.ckpt_dir) / "tensorboard")

    def dataset_factory(s2cfg, split):
        return G1DeltaFeatPrimitiveDataset(
            s2cfg, split=split, feat_root=feat_root, csv_root=csv_root
        )

    trainer = TwoStageMaskedFlowTrainer(
        cfg,
        device=args.device,
        dataset_factory=dataset_factory,
        tensorboard_log_dir=tensorboard_log_dir,
        tensorboard_log_interval=50,
    )

    if resume_state is not None:
        trainer.load_checkpoint(resume_state)
        del resume_state

    trainer.train()


if __name__ == "__main__":
    main()
