from __future__ import annotations

import copy
import math
import os
import signal
import threading
import time
from contextlib import nullcontext
from pathlib import Path
from typing import Callable, Optional

import torch
import torch.nn as nn

from Prior_Recon.Masked_Flow.config import PrimitiveConfig
from Prior_Recon.Masked_Flow.dataset.delta_dataset import (
    build_masked_flow_loaders,
)
from Prior_Recon.Masked_Flow.loss.masked_flow_loss import (
    MaskedFlowLossOutput,
    MaskedFlowMatchingLoss,
    mean_loss_outputs,
    merge_segment_geometry,
)
from Prior_Recon.Masked_Flow.model.masked_flow_transformer import (
    EEMaskedFlowTransformer,
    apply_ee_state_condition,
    build_prefix_condition_tensors,
    load_masked_flow_state_dict,
    save_masked_flow_checkpoint,
)
from Prior_Recon.Masked_Flow.utils import distributed as dist_utils


class EMA:
    def __init__(self, model: nn.Module, decay: float = 0.999):
        self.model = copy.deepcopy(model)
        self.model.eval()
        self.decay = decay

    @torch.no_grad()
    def update(self, model: nn.Module) -> None:
        for ema_param, param in zip(self.model.parameters(), model.parameters()):
            ema_param.data.lerp_(param.data, 1.0 - self.decay)

def _is_no_decay_param(name: str, param: nn.Parameter) -> bool:
    """True for parameters AdamW must not shrink toward zero.

    Weight decay is a prior that "smaller is better". That is right for the
    weight matrices, and wrong for every parameter whose zero is a degenerate
    operating point rather than a simple model:

    * the adaLN modulation biases (``ada_ln.1.bias``) carry the residual GATES.
      A gate at 0 switches its branch off AND removes the branch weights'
      gradient (d loss / d branch ~ gate), so decay pulling a gate down is
      self-reinforcing -- the dead-branch lock-in measured on
      emft_ep0355_dit0805 (cross-attention gate 0.0064-0.0102 absmean).
    * LayerNorm gains at 0 erase their activations entirely.
    * biases only shift; decaying them adds no capacity control.
    * the learned positional / tag tables (``pos_emb``, ``ee_pos_emb``,
      ``look_pos_emb``, ``look_invalid_emb``) are absolute codes, not
      transformations; decay just makes tokens indistinguishable.

    All of those are exactly the <=1-dim parameters plus the ``*_emb`` tables,
    which is the standard GPT/DiT split.
    """
    if param.ndim <= 1:
        return True
    return name.rsplit(".", 1)[-1].endswith("emb")


def build_adamw_param_groups(
    model: nn.Module, weight_decay: float
) -> list[dict]:
    """AdamW param groups: decay the weight matrices, spare gates/norms/biases."""
    decay: list[nn.Parameter] = []
    no_decay: list[nn.Parameter] = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        (no_decay if _is_no_decay_param(name, param) else decay).append(param)
    groups: list[dict] = [{"params": decay, "weight_decay": float(weight_decay)}]
    if no_decay:
        groups.append({"params": no_decay, "weight_decay": 0.0})
    return groups


def migrate_single_group_optimizer_state(
    model: nn.Module, optimizer: torch.optim.Optimizer, state: dict
) -> dict | None:
    """Re-index a 1-group AdamW state onto the 2-group split.

    Checkpoints saved before the decay/no-decay split hold one group whose
    ``params`` are indices into ``model.parameters()`` order. The split is a
    permutation of the same parameter set, so the Adam moments can be carried
    over exactly -- just renumbered. Returns ``state`` unchanged if it is not a
    single-group state; returns None if the parameter sets do not match (caller
    then starts the moments fresh).
    """
    old_groups = state.get("param_groups") or []
    if len(old_groups) != 1:
        return state
    old_params = list(model.parameters())
    if len(old_groups[0].get("params", [])) != len(old_params):
        return None
    old_index_by_id = {id(p): i for i, p in enumerate(old_params)}
    old_state = state.get("state", {})
    new_state: dict = {}
    new_groups: list[dict] = []
    next_index = 0
    for group in optimizer.param_groups:
        indices: list[int] = []
        for param in group["params"]:
            old_index = old_index_by_id.get(id(param))
            if old_index is None:
                return None
            if old_index in old_state:
                new_state[next_index] = old_state[old_index]
            indices.append(next_index)
            next_index += 1
        merged = {k: v for k, v in old_groups[0].items() if k != "params"}
        merged["weight_decay"] = group["weight_decay"]
        merged["params"] = indices
        new_groups.append(merged)
    return {"state": new_state, "param_groups": new_groups}


def _cosine_warmup(optimizer, warmup_steps: int, total_steps: int):
    def lr_lambda(step: int) -> float:
        if step < warmup_steps:
            return step / max(warmup_steps, 1)
        progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress)))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def reanchor_predicted_history(
    history: torch.Tensor,
    src_anchor_yaw: torch.Tensor,
    dst_anchor_yaw: torch.Tensor,
    abs_root: bool,
) -> torch.Tensor:
    """Express predicted history in the next segment's heading/root frame."""
    out = history.clone()
    delta = src_anchor_yaw - dst_anchor_yaw
    cos_delta = torch.cos(delta).view(-1, 1)
    sin_delta = torch.sin(delta).view(-1, 1)

    delta_x = history[..., 7]
    delta_y = history[..., 8]
    out[..., 7] = cos_delta * delta_x - sin_delta * delta_y
    out[..., 8] = sin_delta * delta_x + cos_delta * delta_y

    if abs_root:
        if history.shape[-1] < 73:
            raise ValueError(
                f"abs-root predicted history requires at least 73 dims, got {history.shape[-1]}."
            )
        xy = history[..., 69:71] - history[:, :1, 69:71]
        out[..., 69] = cos_delta * xy[..., 0] - sin_delta * xy[..., 1]
        out[..., 70] = sin_delta * xy[..., 0] + cos_delta * xy[..., 1]
        yaw = torch.atan2(history[..., 72], history[..., 71])
        yaw_rel = yaw - yaw[:, :1]
        out[..., 71] = torch.cos(yaw_rel)
        out[..., 72] = torch.sin(yaw_rel)

    return out


class MaskedFlowTransformerTrainer:
    def __init__(
        self,
        cfg,
        device: str | None = None,
        dataset_factory: Optional[Callable] = None,
        tensorboard_log_dir: str | Path | None = None,
        tensorboard_log_interval: int | None = None,
    ):
        self.cfg = cfg

        # ── Distributed setup (no-op unless launched via torchrun) ──────────
        (
            self.distributed,
            self.rank,
            self.local_rank,
            self.world_size,
        ) = dist_utils.setup_distributed()
        self.is_main = self.rank == 0

        if self.distributed:
            # Under torchrun each process owns one GPU indexed by local_rank;
            # ignore any explicit --device to avoid every rank landing on cuda:0.
            self.device = torch.device(f"cuda:{self.local_rank}")
        else:
            self.device = torch.device(
                device or ("cuda" if torch.cuda.is_available() else "cpu")
            )
        self._configure_cuda_memory_limit()
        # Enable TF32 on Tensor Cores for fp32 matmul/conv. Range (8 exp bits)
        # matches fp32; only the mantissa is truncated to 10 bits — still finer
        # than the bf16 (7 bits) the model already trains in, so no meaningful
        # accuracy cost, and it roughly 2x's this launch-bound workload.
        if self.device.type == "cuda":
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
            if self.is_main:
                print("  [tf32] enabled (matmul + cudnn)")
        self.primitive_cfg = getattr(cfg, "primitive", PrimitiveConfig())
        if not hasattr(cfg, "primitive"):
            cfg.primitive = self.primitive_cfg
        if not bool(getattr(self.primitive_cfg, "enabled", False)):
            raise ValueError(
                "Training only supports primitive mode; set cfg.primitive.enabled=True."
            )
        self.amp = str(getattr(cfg.train, "amp", "none") or "none").lower()
        if self.amp not in {"none", "bf16"}:
            raise ValueError(f"Unsupported cfg.train.amp={self.amp!r}; expected 'none' or 'bf16'.")
        self.amp_dtype = torch.bfloat16 if self.amp == "bf16" else None
        bf16_supported = (
            self.device.type == "cuda"
            and getattr(torch.cuda, "is_bf16_supported", lambda: False)()
        )
        self.amp_enabled = self.amp == "bf16" and bf16_supported
        if self.amp != "none" and not self.amp_enabled:
            print(
                f"  [amp] requested {self.amp}, disabled "
                f"(device={self.device.type}, bf16_supported={bf16_supported})"
            )
        elif self.amp_enabled:
            print(f"  [amp] enabled autocast dtype={self.amp}")

        self.model = self._build_model(cfg).to(self.device)
        # Ensure every rank starts from bit-identical weights, then keep them in
        # sync by averaging gradients each step (see _step / _step_train_*).
        if self.distributed:
            dist_utils.broadcast_module(self.model, src=0)
        self.ema = EMA(self.model, decay=cfg.train.ema_decay)
        self.loss_fn = MaskedFlowMatchingLoss(cfg)
        if cfg.motion.seq_len != self.primitive_cfg.primitive_len:
            raise ValueError(
                "Primitive mode requires cfg.motion.seq_len to equal "
                f"history_len + future_len = {self.primitive_cfg.primitive_len}, "
                f"got {cfg.motion.seq_len}."
            )

        if dataset_factory is None:
            raise ValueError("dataset_factory is required")
        self.train_loader, self.val_loader = build_masked_flow_loaders(
            cfg, dataset_factory=dataset_factory
        )

        self.optimizer = torch.optim.AdamW(
            build_adamw_param_groups(self.model, cfg.train.weight_decay),
            lr=cfg.train.lr,
            weight_decay=cfg.train.weight_decay,
        )
        self.total_steps = cfg.train.n_epochs * len(self.train_loader)
        warmup_steps = cfg.train.warmup_epochs * len(self.train_loader)
        self.scheduler = _cosine_warmup(self.optimizer, warmup_steps, self.total_steps)

        self.ckpt_dir = Path(cfg.train.ckpt_dir)
        self.ckpt_dir.mkdir(parents=True, exist_ok=True)
        self.best_val = float("inf")
        self.global_step = 0
        self.start_epoch = 0
        self.current_epoch = 0
        self.tensorboard_log_interval = int(tensorboard_log_interval or cfg.train.log_interval)
        self._tb_writer = None

        if tensorboard_log_dir and self.is_main:
            try:
                from torch.utils.tensorboard import SummaryWriter

                self._tb_writer = SummaryWriter(log_dir=str(tensorboard_log_dir))
                print(f"  [tensorboard] logging to {tensorboard_log_dir}")
            except ImportError:
                print("  [tensorboard] not installed - skipping")

    def _build_model(self, cfg) -> nn.Module:
        """Model factory hook; the two-stage cascade trainer overrides this."""
        return EEMaskedFlowTransformer(cfg)

    def _amp_context(self):
        if not self.amp_enabled:
            return nullcontext()
        return torch.autocast(device_type=self.device.type, dtype=self.amp_dtype)

    def _configure_cuda_memory_limit(self) -> None:
        if self.device.type != "cuda" or not torch.cuda.is_available():
            return
        fraction = float(getattr(self.cfg.train, "gpu_memory_fraction", 0.85) or 0.0)
        if fraction <= 0.0:
            return
        fraction = min(fraction, 1.0)
        device_index = self.device.index
        if device_index is None:
            device_index = torch.cuda.current_device()
        else:
            torch.cuda.set_device(device_index)
        try:
            torch.cuda.set_per_process_memory_fraction(fraction, device_index)
        except RuntimeError as exc:
            print(f"  [cuda] failed to set memory fraction={fraction:.2f}: {exc}", flush=True)
            return
        total_gb = torch.cuda.get_device_properties(device_index).total_memory / 1024**3
        print(
            f"  [cuda:{device_index}] memory cap: {fraction:.0%} of {total_gb:.1f}GB "
            f"({fraction * total_gb:.1f}GB)",
            flush=True,
        )

    def train(self) -> None:
        params = self.model.n_params()
        if self.is_main:
            dist_note = (
                f"  [ddp] world_size={self.world_size}  "
                f"per-rank batch={self.train_loader.batch_size}\n"
                if self.distributed
                else ""
            )
            print(
                f"MaskedFlowTransformerTrainer  device={self.device}\n"
                f"{dist_note}"
                f"  params: transformer={params['transformer']:,}  total={params['total']:,}\n"
                f"  train={len(self.train_loader)} batches  val={len(self.val_loader)} batches\n"
                f"  batch_size={self.cfg.train.batch_size}  "
                f"max_microbatch_size={getattr(self.cfg.train, 'max_microbatch_size', self.cfg.train.batch_size)}  "
                f"gpu_memory_fraction={getattr(self.cfg.train, 'gpu_memory_fraction', 0.85):.2f}"
            )

        interrupted = threading.Event()
        # signal.signal() only works in the main thread of the main interpreter.
        # Under torchrun each rank's training code runs in the main thread, but
        # Ray Train (torch_trainer.train) runs the training fn in a worker sub-
        # thread, where registering handlers raises ValueError. Only install the
        # graceful-exit handlers when we're actually on the main thread; otherwise
        # skip them (the launcher handles termination) so training still runs.
        can_handle_signals = threading.current_thread() is threading.main_thread()
        orig_sigint = signal.getsignal(signal.SIGINT) if can_handle_signals else None
        orig_sigterm = signal.getsignal(signal.SIGTERM) if can_handle_signals else None

        def _graceful_exit(signum, frame) -> None:
            if not interrupted.is_set():
                interrupted.set()
                if self.is_main:
                    print(f"\n[Signal {signum}] received, saving checkpoint before exit...", flush=True)
                    try:
                        self._save("interrupted", epoch=self.current_epoch)
                    except Exception as exc:
                        print(f"  [Signal] save failed: {exc}", flush=True)
            signal.signal(signal.SIGINT, orig_sigint)
            signal.signal(signal.SIGTERM, orig_sigterm)
            os.kill(os.getpid(), signum)

        if can_handle_signals:
            signal.signal(signal.SIGINT, _graceful_exit)
            signal.signal(signal.SIGTERM, _graceful_exit)

        try:
            for epoch in range(self.start_epoch, self.cfg.train.n_epochs):
                if interrupted.is_set():
                    break
                self.current_epoch = epoch
                # Re-seed the DistributedSampler so each epoch reshuffles and
                # every rank sees a disjoint, freshly-permuted shard.
                if self.distributed:
                    for loader in (self.train_loader, self.val_loader):
                        sampler = getattr(loader, "sampler", None)
                        if hasattr(sampler, "set_epoch"):
                            sampler.set_epoch(epoch)
                start = time.time()
                train_metrics = self._run_epoch(train=True)
                val_metrics = self._run_epoch(train=False)
                # Each rank only saw its own data shard; average metrics so every
                # rank agrees on the numbers (needed for a consistent best_val
                # decision and correct logging).
                if self.distributed:
                    train_metrics = {
                        k: dist_utils.reduce_scalar(v) for k, v in train_metrics.items()
                    }
                    val_metrics = {
                        k: dist_utils.reduce_scalar(v) for k, v in val_metrics.items()
                    }
                next_epoch = epoch + 1

                if val_metrics["loss/total"] < self.best_val:
                    self.best_val = val_metrics["loss/total"]
                    if self.is_main:
                        self._save("best", epoch=next_epoch)

                if next_epoch % 5 == 0 and self.is_main:
                    tag = (
                        f"ep{next_epoch:04d}_loss{val_metrics['loss/total']:.4f}"
                        f"_flow{val_metrics['loss/flow']:.4f}"
                    )
                    self._save(tag, epoch=next_epoch)

                self._log(epoch, train_metrics, val_metrics, time.time() - start)

            if not interrupted.is_set() and self.is_main:
                self._save("final", epoch=self.cfg.train.n_epochs)
        finally:
            if can_handle_signals:
                signal.signal(signal.SIGINT, orig_sigint)
                signal.signal(signal.SIGTERM, orig_sigterm)
            if self._tb_writer is not None:
                try:
                    self._tb_writer.close()
                finally:
                    self._tb_writer = None
            dist_utils.cleanup_distributed()

    def _run_epoch(self, train: bool) -> dict[str, float]:
        loader = self.train_loader if train else self.val_loader
        self.model.train(train)
        totals: dict[str, float] = {}
        teacher_totals: dict[str, float] = {}
        n_batches = 0
        oom_count = 0
        ctx = torch.enable_grad() if train else torch.no_grad()
        with ctx:
            for batch in loader:
                batch = {
                    key: value.to(self.device) if isinstance(value, torch.Tensor) else value
                    for key, value in batch.items()
                }
                teacher_values = None
                if not train:
                    loss_values = self._step_values(
                        batch,
                        train=False,
                        force_teacher=not getattr(self.primitive_cfg, "val_rollout", True),
                    )
                    teacher_values = self._step_values(batch, train=False, force_teacher=True)
                else:
                    loss_values = self._step_values(batch, train=train)

                # Free GPU batch tensors immediately; no longer needed after step.
                del batch

                if loss_values is None:
                    oom_count += 1
                    teacher_values = None
                    continue

                for key, value in loss_values.items():
                    totals[key] = totals.get(key, 0.0) + float(value)

                if teacher_values is not None:
                    for key, value in teacher_values.items():
                        teacher_totals[f"teacher/{key}"] = (
                            teacher_totals.get(f"teacher/{key}", 0.0) + float(value)
                        )

                n_batches += 1
                if torch.cuda.is_available() and n_batches % 50 == 0:
                    torch.cuda.empty_cache()
                if train and self.is_main and self.global_step % self.cfg.train.log_interval == 0:
                    metrics = "  ".join(
                        f"{key.split('/')[-1]}={value:.4f}"
                        for key, value in loss_values.items()
                    )
                    lr = self.scheduler.get_last_lr()[0]
                    print(
                        f"  step {self.global_step:5d} | {metrics}  "
                        f"rollout_prob={self._current_rollout_prob(train=True):.2f}  lr={lr:.2e}"
                    )
                if train and self.is_main and self.global_step % self.tensorboard_log_interval == 0:
                    self._log_train_step(loss_values)

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        if oom_count > 0:
            print(f"  [OOM] skipped {oom_count} batches in this epoch", flush=True)
        out = {key: value / max(n_batches, 1) for key, value in totals.items()}
        out.update({key: value / max(n_batches, 1) for key, value in teacher_totals.items()})
        return out

    def _current_rollout_prob(self, train: bool) -> float:
        max_prob = float(getattr(self.primitive_cfg, "rollout_max_prob", 1.0))
        if not train:
            return max_prob if getattr(self.primitive_cfg, "val_rollout", True) else 0.0

        start_ratio = float(getattr(self.primitive_cfg, "rollout_start_ratio", 0.0))
        end_ratio = float(getattr(self.primitive_cfg, "rollout_end_ratio", start_ratio))
        progress = min(self.global_step / max(self.total_steps - 1, 1), 1.0)
        if progress <= start_ratio:
            return 0.0
        if progress >= end_ratio:
            return max_prob

        ramp = (progress - start_ratio) / max(end_ratio - start_ratio, 1e-8)
        return max_prob * ramp

    def _select_history(
        self,
        gt_history: torch.Tensor,
        prev_pred: torch.Tensor | None,
        train: bool,
        force_teacher: bool,
    ) -> torch.Tensor:
        if prev_pred is None or force_teacher:
            return gt_history

        history_len = gt_history.shape[1]
        prev_history = prev_pred[:, -history_len:, :]
        rollout_prob = self._current_rollout_prob(train=train)
        if rollout_prob <= 0.0:
            return gt_history
        if rollout_prob >= 1.0:
            return prev_history

        chooser = torch.rand(
            gt_history.shape[0], 1, 1, device=gt_history.device, dtype=gt_history.dtype
        )
        use_rollout = chooser < rollout_prob
        return torch.where(use_rollout, prev_history, gt_history)

    def _perturb_history(self, history_in: torch.Tensor, train: bool) -> torch.Tensor:
        """Layer B: inject a per-sample CONSTANT pose bias into the pinned history.

        Deployment seeds the history from the current standing/executed robot
        state, which generally disagrees with the fixed EE reference's first
        frame; training on GT / self-rollout never shows that mismatch, so the
        model assumes history-first-frame == reference-anchor and carries a
        constant offset it cannot correct. Here we add a small offset that is
        CONSTANT across the ``history_len`` frames (a wrong-but-static start pose,
        not per-frame jitter) to the joint (11:40) and root-tilt (0:4) channels of
        the delta69 body block. The offset is NOT added to the delta channels
        (4, 7:9, 40:69) -- a constant pose bias leaves inter-frame deltas
        unchanged -- nor to any trailing EE-state columns (those are pinned to the
        GT condition by design). Only active in training with prob > 0.
        """
        prob = float(getattr(self.cfg, "history_perturb_prob", 0.0))
        if not train or prob <= 0.0:
            return history_in
        j_std = float(getattr(self.cfg, "history_perturb_joint_std", 0.0))
        t_std = float(getattr(self.cfg, "history_perturb_tilt_std", 0.0))
        if j_std <= 0.0 and t_std <= 0.0:
            return history_in

        batch = history_in.shape[0]
        dev, dt = history_in.device, history_in.dtype
        fire = (torch.rand(batch, 1, 1, device=dev, dtype=dt) < prob).to(dt)
        out = history_in.clone()
        if j_std > 0.0:
            # (B, 1, 29) constant-across-time joint offset, broadcast over frames.
            j_off = torch.randn(batch, 1, 29, device=dev, dtype=dt) * j_std * fire
            out[..., 11:40] = out[..., 11:40] + j_off
        if t_std > 0.0:
            # Perturb roll/pitch as an angle, re-encode to the sin/(cos-1) channels
            # so the tilt representation stays on its manifold. Base angles are
            # recovered from the existing channels (dims 0:4).
            roll = torch.atan2(out[..., 0], out[..., 1] + 1.0)
            pitch = torch.atan2(out[..., 2], out[..., 3] + 1.0)
            r_off = torch.randn(batch, 1, device=dev, dtype=dt) * t_std * fire[..., 0]
            p_off = torch.randn(batch, 1, device=dev, dtype=dt) * t_std * fire[..., 0]
            roll = roll + r_off
            pitch = pitch + p_off
            out[..., 0] = torch.sin(roll)
            out[..., 1] = torch.cos(roll) - 1.0
            out[..., 2] = torch.sin(pitch)
            out[..., 3] = torch.cos(pitch) - 1.0
        return out

    def _step_primitive_segment(
        self,
        batch: dict,
        train: bool,
        force_teacher: bool,
        initial_history: torch.Tensor | None,
    ) -> tuple[MaskedFlowLossOutput, torch.Tensor]:
        s_full_prim = batch["s_full_prim"]
        s_ee_prim = batch["s_ee_prim"]
        s_ee_look_prim = batch.get("s_ee_look_prim")
        look_valid_prim = batch.get("look_valid_prim")
        segment_full = batch.get("segment_full")
        prim_cfg = self.primitive_cfg
        history_len = prim_cfg.history_len
        if s_full_prim.shape[1] != prim_cfg.num_primitives:
            raise ValueError(
                f"Expected {prim_cfg.num_primitives} primitives, got {s_full_prim.shape[1]}."
            )
        if s_full_prim.shape[2] != prim_cfg.primitive_len:
            raise ValueError(
                f"Expected primitive length {prim_cfg.primitive_len}, got {s_full_prim.shape[2]}."
            )

        losses: list[MaskedFlowLossOutput] = []
        prev_pred = initial_history
        use_lnt = getattr(self.cfg, "use_logit_normal_t", False)
        lnt_sigma = getattr(self.cfg, "logit_normal_sigma", 1.0)
        per_frame = getattr(self.cfg, "per_frame_noise", False)
        ee_state_dim = int(getattr(self.cfg, "ee_state_dim", 0))
        seg_anchor = bool(getattr(self.cfg, "ee_cond_segment_anchor", False))
        seg_geometry = bool(getattr(self.cfg, "segment_geometry_loss", False))
        if seg_geometry and segment_full is None:
            raise ValueError(
                "segment_geometry_loss=True requires 'segment_full' in the batch "
                "(G1DeltaFeatPrimitiveDataset emits it)."
            )
        ee_anchor = None
        if seg_anchor:
            # Shared ee_cond anchor: GT hand pose at SEGMENT frame 0 (loss-side
            # only; UNPERTURBED, paralleling the Layer B target convention).
            ee_anchor = self.loss_fn.gt_hand_anchor(s_full_prim[:, 0, :1].float())
        # Non-detached per-primitive predictions for the stitched segment loss;
        # the autoregressive conditioning below still uses the detached copy.
        pred_x1_prims: list[torch.Tensor] = []
        obs_mask_prims: list[torch.Tensor] = []

        for prim_idx in range(prim_cfg.num_primitives):
            prim_start = prim_idx * prim_cfg.future_len
            s_full = s_full_prim[:, prim_idx]
            s_ee = s_ee_prim[:, prim_idx]
            if segment_full is not None and prim_start > 0:
                # Primitive features are sliced but still expressed in the
                # parent segment frame; FK geometry losses need this yaw origin.
                yaw_offset = segment_full[:, :prim_start, 4].sum(dim=1)
            else:
                yaw_offset = torch.zeros(
                    s_full.shape[0],
                    device=s_full.device,
                    dtype=s_full.dtype,
                )
            gt_history = s_full[:, :history_len]
            history_in = self._select_history(
                gt_history,
                prev_pred=prev_pred,
                train=train,
                force_teacher=(
                    force_teacher or (prim_idx == 0 and initial_history is None)
                ),
            )
            # Layer B: inject a constant pose mismatch into the pinned history so
            # the model learns to recover from a standing / real-state start whose
            # first frame disagrees with the (unperturbed) EE reference. The target
            # s_full is NOT perturbed -> the loss demands the correct trajectory
            # despite the offset seed.
            history_in = self._perturb_history(history_in, train=train)
            known_full, obs_mask = build_prefix_condition_tensors(history_in, s_full.shape[1])
            if ee_state_dim > 0:
                # Deployment mask pattern: history rows AND EE columns pinned
                # together. EE columns come from the GT features (identical to
                # s_ee[..., :18] by dataset construction).
                apply_ee_state_condition(known_full, obs_mask, s_full[..., -ee_state_dim:])
            x_t, v_gt, t, _ = self.model.sample_training_tuple(
                s_full,
                use_logit_normal_t=use_lnt,
                logit_normal_sigma=lnt_sigma,
                per_frame=per_frame,
                obs_mask=obs_mask,
            )
            with self._amp_context():
                out = self.model(
                    x_t,
                    known_full,
                    obs_mask,
                    s_ee,
                    t,
                    s_ee_look=(
                        s_ee_look_prim[:, prim_idx]
                        if s_ee_look_prim is not None
                        else None
                    ),
                    look_valid=(
                        look_valid_prim[:, prim_idx]
                        if look_valid_prim is not None
                        else None
                    ),
                )
            losses.append(
                self.loss_fn(
                    out.pred_v.float(),
                    out.pred_x1.float(),
                    s_full,
                    v_gt,
                    obs_mask,
                    s_ee=s_ee,
                    yaw_offset=yaw_offset,
                    # Layer C anchor loss only on the primitive whose FK frame 0
                    # is the segment heading anchor (yaw_offset == 0).
                    anchor_active=(prim_idx == 0),
                    ee_anchor=ee_anchor,
                    # In segment mode the geometry suite runs ONCE on the
                    # stitched segment below (moved, not duplicated).
                    include_geometry=not seg_geometry,
                )
            )
            if seg_geometry:
                pred_x1_prims.append(out.pred_x1.float())
                obs_mask_prims.append(obs_mask)
            prev_pred = out.pred_x1.detach()
            del out  # Free x_in / t fields; computation graph stays alive via losses.

        loss_out = mean_loss_outputs(losses)
        if seg_geometry:
            # Stitch the primitives back into the full segment: keep primitive
            # 0 whole, drop each later primitive's history overlap in favour of
            # the PREVIOUS primitive's generated (in-graph) frames. Geometry on
            # the stitched tensor is the only place a gradient crosses a
            # primitive seam -- the autoregressive history is detached.
            pred_seg = torch.cat(
                [pred_x1_prims[0]] + [p[:, history_len:] for p in pred_x1_prims[1:]],
                dim=1,
            )
            obs_seg = torch.cat(
                [obs_mask_prims[0]] + [m[:, history_len:] for m in obs_mask_prims[1:]],
                dim=1,
            )
            geom = self.loss_fn.segment_geometry(
                pred_seg,
                segment_full,
                obs_seg,
                s_ee=batch.get("segment_ee"),
                ee_anchor=ee_anchor,
            )
            loss_out = merge_segment_geometry(loss_out, geom)
        if prev_pred is None:
            raise RuntimeError("Primitive segment produced no prediction.")
        return loss_out, prev_pred

    def _step_primitives(
        self,
        batch: dict,
        train: bool,
        force_teacher: bool,
    ) -> MaskedFlowLossOutput:
        s_full_prim = batch["s_full_prim"]
        if s_full_prim.ndim == 4:
            loss_out, _ = self._step_primitive_segment(
                batch,
                train=train,
                force_teacher=force_teacher,
                initial_history=None,
            )
            return loss_out
        if s_full_prim.ndim != 5:
            raise ValueError(
                "Expected s_full_prim shape (B,N,T,D) or (B,S,N,T,D), "
                f"got {tuple(s_full_prim.shape)}."
            )

        segment_unrolls = int(getattr(self.primitive_cfg, "segment_unrolls", 1))
        if s_full_prim.shape[1] != segment_unrolls:
            raise ValueError(
                f"Expected {segment_unrolls} unrolled segments, got {s_full_prim.shape[1]}."
            )
        anchor_yaw = batch.get("anchor_yaw")
        if anchor_yaw is None or anchor_yaw.shape[:2] != s_full_prim.shape[:2]:
            raise ValueError(
                "Cross-segment rollout requires anchor_yaw with shape "
                f"{tuple(s_full_prim.shape[:2])}."
            )

        losses: list[MaskedFlowLossOutput] = []
        carried_history: torch.Tensor | None = None
        abs_root = bool(getattr(self.cfg, "abs_root_channels", False))
        # Heading frame the carried history currently lives in. Segment 0 is
        # teacher-forced from GT, so it starts at the GT anchor; each re-anchor
        # below moves it to a *predicted* heading.
        # Exact for segment_unrolls == 2 (the only configured value): the single
        # re-anchor then reads anchor_yaw[:, 0], which really is GT. For >= 3 it
        # is approximate, because _select_history mixes GT and rolled-out history
        # per sample at prim_idx 0, so some samples' true frame is the GT anchor.
        carried_anchor_yaw = anchor_yaw[:, 0]
        segment_keys = (
            "s_full_prim",
            "s_ee_prim",
            "s_ee_look_prim",
            "look_valid_prim",
            "segment_full",
            "segment_ee",
        )
        for segment_idx in range(segment_unrolls):
            segment_batch = {
                key: batch[key][:, segment_idx]
                for key in segment_keys
                if key in batch
            }
            if carried_history is not None and not force_teacher:
                # Mirror inference (visual/recon_delta69.py:909, which passes
                # prev_pred_yaws[carry_idx] = prev_anchor + atan2(p[.,72], p[.,71])):
                # the destination frame is the *predicted* world heading at the
                # carried history's frame 0, NOT the GT anchor_yaw[:, segment_idx].
                # reanchor_predicted_history's yaw branch unconditionally rebases
                # [71:73] to that predicted frame (yaw_rel = yaw - yaw[:, :1]), so
                # a GT dst rotates [7:9]/[69:71] by the GT turn while the heading
                # channels follow the predicted one -- the two then disagree by
                # exactly the yaw error, worst on the hardest samples.
                if abs_root:
                    next_anchor_yaw = carried_anchor_yaw + torch.atan2(
                        carried_history[:, 0, 72].float(),
                        carried_history[:, 0, 71].float(),
                    )
                else:
                    # Legacy relative layout carries no accumulated-yaw channel
                    # to read the predicted heading from; keep the GT anchor.
                    next_anchor_yaw = anchor_yaw[:, segment_idx]
                carried_history = reanchor_predicted_history(
                    carried_history,
                    src_anchor_yaw=carried_anchor_yaw,
                    dst_anchor_yaw=next_anchor_yaw,
                    abs_root=abs_root,
                )
                carried_anchor_yaw = next_anchor_yaw
            loss_out, final_prediction = self._step_primitive_segment(
                segment_batch,
                train=train,
                force_teacher=force_teacher,
                initial_history=carried_history,
            )
            losses.append(loss_out)
            carried_history = final_prediction[:, -self.primitive_cfg.history_len :].detach()

        return mean_loss_outputs(losses)

    def _step(
        self,
        batch: dict,
        train: bool,
        force_teacher: bool = False,
    ) -> MaskedFlowLossOutput | None:
        try:
            progress = self.global_step / max(self.total_steps - 1, 1)
            self.loss_fn.set_training_progress(progress)
            loss_out = self._step_primitives(batch, train=train, force_teacher=force_teacher)

            if train:
                self.optimizer.zero_grad(set_to_none=True)
                loss_out.total.backward()
                if self.distributed:
                    dist_utils.average_gradients(self.model.parameters())
                nn.utils.clip_grad_norm_(self.model.parameters(), self.cfg.train.grad_clip)
                self.optimizer.step()
                self.scheduler.step()
                self.ema.update(self.model)
                self.global_step += 1

            return loss_out
        except RuntimeError as exc:
            if "out of memory" not in str(exc).lower():
                raise
            mem_mb = (
                torch.cuda.memory_reserved(self.device) / 1024**2
                if torch.cuda.is_available()
                else 0.0
            )
            print(
                f"\n[OOM] step={self.global_step} CUDA out of memory "
                f"(reserved={mem_mb:.0f}MB), skipping batch.",
                flush=True,
            )
            if train:
                self.optimizer.zero_grad(set_to_none=True)
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            return None

    def _reduce_microbatch_after_oom(self, attempted_microbatch: int) -> None:
        current = int(
            getattr(
                self.cfg.train,
                "max_microbatch_size",
                getattr(self.cfg.train, "batch_size", attempted_microbatch),
            )
            or attempted_microbatch
        )
        attempted_microbatch = max(int(attempted_microbatch), 1)
        new_size = max(min(current, attempted_microbatch) // 2, 1)
        if new_size >= current:
            return
        self.cfg.train.max_microbatch_size = new_size
        print(
            f"  [OOM guard] reducing max_microbatch_size: {current} -> {new_size}",
            flush=True,
        )

    def _batch_size(self, batch: dict) -> int:
        for value in batch.values():
            if isinstance(value, torch.Tensor) and value.ndim > 0:
                return int(value.shape[0])
        raise ValueError("Batch has no tensor with a batch dimension.")

    def _slice_batch(self, batch: dict, start: int, end: int, batch_size: int) -> dict:
        sliced = {}
        for key, value in batch.items():
            if isinstance(value, torch.Tensor) and value.ndim > 0 and value.shape[0] == batch_size:
                sliced[key] = value[start:end]
            else:
                sliced[key] = value
        return sliced

    @staticmethod
    def _accumulate_values(
        totals: dict[str, float],
        values: dict[str, float],
        weight: int,
    ) -> None:
        for key, value in values.items():
            totals[key] = totals.get(key, 0.0) + float(value) * weight

    def _loss_output_as_dict(self, loss_out: MaskedFlowLossOutput) -> dict[str, float]:
        loss_cfg = self.cfg.loss
        return loss_out.as_dict(
            include_quantize_rot=getattr(loss_cfg, "quantize_rot_weight", 0.0) > 0.0,
            include_quantize_trans=getattr(loss_cfg, "quantize_trans_weight", 0.0) > 0.0,
        )

    def _step_values(
        self,
        batch: dict,
        train: bool,
        force_teacher: bool = False,
    ) -> dict[str, float] | None:
        batch_size = self._batch_size(batch)
        max_microbatch = int(getattr(self.cfg.train, "max_microbatch_size", batch_size) or batch_size)
        if max_microbatch <= 0 or max_microbatch >= batch_size:
            loss_out = self._step(batch, train=train, force_teacher=force_teacher)
            if loss_out is None:
                self._reduce_microbatch_after_oom(batch_size)
                return None
            values = self._loss_output_as_dict(loss_out)
            del loss_out
            return values

        if train:
            return self._step_train_microbatches(batch, batch_size, max_microbatch, force_teacher)
        return self._step_eval_microbatches(batch, batch_size, max_microbatch, force_teacher)

    def _step_eval_microbatches(
        self,
        batch: dict,
        batch_size: int,
        max_microbatch: int,
        force_teacher: bool,
    ) -> dict[str, float] | None:
        totals: dict[str, float] = {}
        n_items = 0
        for start in range(0, batch_size, max_microbatch):
            end = min(start + max_microbatch, batch_size)
            micro = self._slice_batch(batch, start, end, batch_size)
            loss_out = self._step(micro, train=False, force_teacher=force_teacher)
            if loss_out is None:
                self._reduce_microbatch_after_oom(end - start)
                del micro
                continue
            values = self._loss_output_as_dict(loss_out)
            del loss_out, micro
            weight = end - start
            self._accumulate_values(totals, values, weight)
            n_items += weight
        if n_items == 0:
            return None
        return {key: value / n_items for key, value in totals.items()}

    def _step_train_microbatches(
        self,
        batch: dict,
        batch_size: int,
        max_microbatch: int,
        force_teacher: bool,
    ) -> dict[str, float] | None:
        totals: dict[str, float] = {}
        n_items = 0
        self.optimizer.zero_grad(set_to_none=True)

        for start in range(0, batch_size, max_microbatch):
            end = min(start + max_microbatch, batch_size)
            micro = self._slice_batch(batch, start, end, batch_size)
            weight = end - start
            try:
                progress = self.global_step / max(self.total_steps - 1, 1)
                self.loss_fn.set_training_progress(progress)
                loss_out = self._step_primitives(
                    micro,
                    train=True,
                    force_teacher=force_teacher,
                )
                (loss_out.total * (weight / batch_size)).backward()
                values = self._loss_output_as_dict(loss_out)
                del loss_out
            except RuntimeError as exc:
                if "out of memory" not in str(exc).lower():
                    raise
                mem_mb = (
                    torch.cuda.memory_reserved(self.device) / 1024**2
                    if torch.cuda.is_available()
                    else 0.0
                )
                print(
                    f"\n[OOM] step={self.global_step} microbatch={weight} "
                    f"CUDA out of memory (reserved={mem_mb:.0f}MB), skipping microbatch.",
                    flush=True,
                )
                self.optimizer.zero_grad(set_to_none=True)
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                self._reduce_microbatch_after_oom(max_microbatch)
                return None
            finally:
                del micro

            self._accumulate_values(totals, values, weight)
            n_items += weight

        if n_items == 0:
            self.optimizer.zero_grad(set_to_none=True)
            return None

        if self.distributed:
            dist_utils.average_gradients(self.model.parameters())
        nn.utils.clip_grad_norm_(self.model.parameters(), self.cfg.train.grad_clip)
        self.optimizer.step()
        self.scheduler.step()
        self.ema.update(self.model)
        self.global_step += 1
        return {key: value / n_items for key, value in totals.items()}

    def _log(self, epoch: int, train_metrics: dict[str, float], val_metrics: dict[str, float], elapsed: float) -> None:
        if not self.is_main:
            return
        teacher_suffix = ""
        if "teacher/loss/total" in val_metrics:
            teacher_suffix = f"  teacher_val={val_metrics['teacher/loss/total']:.4f}"
        print(
            f"[epoch {epoch + 1:3d}/{self.cfg.train.n_epochs}] "
            f"train={train_metrics['loss/total']:.4f}  val={val_metrics['loss/total']:.4f}  "
            f"flow={val_metrics['loss/flow']:.4f}  recon={val_metrics['loss/recon']:.4f}  "
            f"vel={val_metrics['loss/velocity']:.4f}{teacher_suffix}  ({elapsed:.1f}s)"
        )
        if self._tb_writer is not None:
            self._tb_writer.add_scalar("epoch", epoch + 1, self.global_step)
            for key, value in train_metrics.items():
                self._tb_writer.add_scalar(f"epoch_train/{key}", value, self.global_step)
            for key, value in val_metrics.items():
                self._tb_writer.add_scalar(f"epoch_val/{key}", value, self.global_step)
            self._tb_writer.flush()

    def _log_train_step(self, loss_values: dict[str, float]) -> None:
        if self._tb_writer is None:
            return
        for key, value in loss_values.items():
            self._tb_writer.add_scalar(f"train/{key}", value, self.global_step)
        self._tb_writer.add_scalar("train/lr", self.scheduler.get_last_lr()[0], self.global_step)
        self._tb_writer.add_scalar(
            "train/rollout_prob",
            self._current_rollout_prob(train=True),
            self.global_step,
        )
        self._tb_writer.add_scalar(
            "train/ee_loss_scale",
            self.loss_fn._ee_curriculum_scale(),
            self.global_step,
        )
        self._tb_writer.add_scalar(
            "train/root_loss_mult",
            self.loss_fn._root_curriculum_mult(),
            self.global_step,
        )
        self._tb_writer.flush()

    def _save(self, tag: str, epoch: int | None = None) -> None:
        path = self.ckpt_dir / f"emft_{tag}.pt"
        save_epoch = self.start_epoch if epoch is None else epoch
        save_masked_flow_checkpoint(
            self.model,
            self.ema.model,
            self.optimizer,
            self.scheduler,
            self.global_step,
            save_epoch,
            self.best_val,
            self.cfg,
            path,
        )
        print(f"  -> checkpoint: {path}")

    def load_checkpoint(self, path: str | dict) -> None:
        state = (
            torch.load(path, map_location=self.device, weights_only=False)
            if isinstance(path, (str, os.PathLike))
            else path
        )
        load_masked_flow_state_dict(self.model, state["model"])
        load_masked_flow_state_dict(self.ema.model, state["ema_model"])
        opt_state = migrate_single_group_optimizer_state(
            self.model, self.optimizer, state["optimizer"]
        )
        if opt_state is None:
            print(
                "  [resume] optimizer state does not match the current "
                "parameter set - starting Adam moments fresh"
            )
        else:
            self.optimizer.load_state_dict(opt_state)
        if "scheduler" in state:
            self.scheduler.load_state_dict(state["scheduler"])
        self.global_step = state.get("step", 0)
        self.start_epoch = state.get("epoch", 0)
        self.current_epoch = self.start_epoch
        self.best_val = state.get("best_val", float("inf"))
        print(
            f"  -> resumed epoch={self.start_epoch}  "
            f"step={self.global_step}  best_val={self.best_val:.4f}"
        )
