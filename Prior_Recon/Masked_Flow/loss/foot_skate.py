from __future__ import annotations

import math

import numpy as np
import torch


def project_contact_probability(raw: torch.Tensor) -> torch.Tensor:
    """Project the model's contact channels onto their public probability domain."""
    return raw.clamp(0.0, 1.0)


def project_contact_probability_numpy(raw: np.ndarray) -> np.ndarray:
    """NumPy equivalent used at the planner/deployment boundary."""
    return np.clip(raw, 0.0, 1.0)


def boundary_sole_skate_loss(
    sole_translation: torch.Tensor,
    contact_probability: torch.Tensor,
    transition_mask: torch.Tensor,
    *,
    fps: float,
    history_len: int,
    future_len: int,
    num_primitives: int,
    topk_ratio: float,
) -> torch.Tensor:
    """Top-k stance-foot speed at every history-to-future primitive boundary."""
    velocity_count = max(sole_translation.shape[1] - 1, 0)
    indices = [
        history_len - 1 + primitive_index * future_len
        for primitive_index in range(num_primitives)
        if 0 <= history_len - 1 + primitive_index * future_len < velocity_count
    ]
    if not indices:
        return sole_translation.sum() * 0.0

    velocity = (sole_translation[:, 1:] - sole_translation[:, :-1]) * fps
    boundary_velocity = velocity[:, indices]
    stance = contact_probability[:, 1:] * contact_probability[:, :-1]
    boundary_stance = stance[:, indices]
    active_transition = transition_mask[:, indices]
    active = boundary_stance.unsqueeze(-1) * active_transition.unsqueeze(-1)
    speed_sq = boundary_velocity.square().sum(dim=-1)
    values = speed_sq[active.expand_as(speed_sq) > 0.5]
    if values.numel() == 0:
        return sole_translation.sum() * 0.0

    ratio = min(max(float(topk_ratio), 0.0), 1.0)
    count = max(1, math.ceil(values.numel() * ratio))
    return torch.topk(values, count, sorted=False).values.mean()
