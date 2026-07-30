from __future__ import annotations

import numpy as np


def rot6d_from_quat_wxyz(q: np.ndarray) -> np.ndarray:
    """Convert quaternion(s) in wxyz format to Zhou et al. 6D rotation."""
    q = q / np.linalg.norm(q, axis=-1, keepdims=True).clip(1e-8)
    w, x, y, z = q[..., 0], q[..., 1], q[..., 2], q[..., 3]
    mat = np.stack(
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
        axis=-1,
    ).reshape(q.shape[:-1] + (3, 3))
    return mat[..., :2].reshape(q.shape[:-1] + (6,))


def quat_wxyz_relative_to_first(q: np.ndarray) -> np.ndarray:
    """Express each quaternion as inv(q[0]) * q[t] along the time axis."""
    q = q / np.linalg.norm(q, axis=-1, keepdims=True).clip(1e-8)
    q0_inv = q[:1].copy()
    q0_inv[..., 1:] *= -1.0

    q0_inv = np.broadcast_to(q0_inv, q.shape)
    w1, x1, y1, z1 = [q0_inv[..., i] for i in range(4)]
    w2, x2, y2, z2 = [q[..., i] for i in range(4)]
    rel = np.stack(
        [
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
        ],
        axis=-1,
    )
    return rel.astype(np.float32)
