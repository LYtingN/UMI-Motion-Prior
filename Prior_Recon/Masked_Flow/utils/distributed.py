"""Lightweight helpers for optional multi-GPU (DDP) training.

The training loop performs several forward passes per optimizer step
(primitive rollout), which is incompatible with the autograd hooks that
``torch.nn.parallel.DistributedDataParallel`` installs. Instead we keep
the raw model and synchronise gradients manually with a single ``all_reduce``
after ``backward`` — this is correct regardless of how many forwards happen.

When the process is not launched under ``torchrun`` (no ``RANK`` env var) every
helper degrades to single-process behaviour, so the trainer code path is
identical for 1 GPU and N GPUs.
"""
from __future__ import annotations

import os
from typing import Iterable

import torch
import torch.distributed as dist


def is_dist_launch() -> bool:
    """True when launched under torchrun / torch.distributed (RANK is set)."""
    return "RANK" in os.environ and "WORLD_SIZE" in os.environ


def get_rank() -> int:
    if dist.is_available() and dist.is_initialized():
        return dist.get_rank()
    return int(os.environ.get("RANK", 0))


def get_local_rank() -> int:
    return int(os.environ.get("LOCAL_RANK", 0))


def get_world_size() -> int:
    if dist.is_available() and dist.is_initialized():
        return dist.get_world_size()
    return int(os.environ.get("WORLD_SIZE", 1))


def is_main_process() -> bool:
    return get_rank() == 0


def setup_distributed() -> tuple[bool, int, int, int]:
    """Initialise the process group if launched under torchrun.

    Returns ``(distributed, rank, local_rank, world_size)``.
    """
    if not is_dist_launch():
        return False, 0, 0, 1

    local_rank = get_local_rank()
    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)
    backend = "nccl" if torch.cuda.is_available() else "gloo"
    if not dist.is_initialized():
        dist.init_process_group(backend=backend)
    return True, dist.get_rank(), local_rank, dist.get_world_size()


def cleanup_distributed() -> None:
    if dist.is_available() and dist.is_initialized():
        dist.barrier()
        dist.destroy_process_group()


def barrier() -> None:
    if dist.is_available() and dist.is_initialized():
        dist.barrier()


@torch.no_grad()
def broadcast_module(module: torch.nn.Module, src: int = 0) -> None:
    """Broadcast all parameters and buffers from ``src`` so every rank starts
    from identical weights."""
    if not (dist.is_available() and dist.is_initialized()):
        return
    for tensor in list(module.parameters()) + list(module.buffers()):
        dist.broadcast(tensor.data, src=src)


@torch.no_grad()
def average_gradients(params: Iterable[torch.nn.Parameter]) -> None:
    """All-reduce (mean) the gradients across ranks. Call after ``backward``
    and before ``optimizer.step``."""
    if not (dist.is_available() and dist.is_initialized()):
        return
    world_size = dist.get_world_size()
    if world_size == 1:
        return
    for param in params:
        if param.grad is None:
            continue
        dist.all_reduce(param.grad.data, op=dist.ReduceOp.SUM)
        param.grad.data /= world_size


@torch.no_grad()
def reduce_scalar(value: float) -> float:
    """Average a python scalar across ranks (for logging aggregated metrics)."""
    if not (dist.is_available() and dist.is_initialized()):
        return value
    world_size = dist.get_world_size()
    if world_size == 1:
        return value
    tensor = torch.tensor(
        [value],
        dtype=torch.float64,
        device="cuda" if torch.cuda.is_available() else "cpu",
    )
    dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
    return float(tensor.item() / world_size)
