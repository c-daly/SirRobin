"""Single donor-Unity to canonical FLU frame boundary."""

from __future__ import annotations

import torch

from sirrobin.numerics.quat import normalize

DONOR_TO_FLU_ROWS = ((0.0, 0.0, -1.0), (-1.0, 0.0, 0.0), (0.0, 1.0, 0.0))


def donor_to_flu_matrix(*, dtype: torch.dtype, device: torch.device | str) -> torch.Tensor:
    return torch.tensor(DONOR_TO_FLU_ROWS, dtype=dtype, device=device)


def donor_vector_to_flu(value: torch.Tensor) -> torch.Tensor:
    basis = donor_to_flu_matrix(dtype=value.dtype, device=value.device)
    return torch.einsum("ij,...j->...i", basis, value)


def quat_to_matrix(q: torch.Tensor) -> torch.Tensor:
    x, y, z, w = normalize(q).unbind(-1)
    return torch.stack(
        (
            1 - 2 * (y * y + z * z),
            2 * (x * y - z * w),
            2 * (x * z + y * w),
            2 * (x * y + z * w),
            1 - 2 * (x * x + z * z),
            2 * (y * z - x * w),
            2 * (x * z - y * w),
            2 * (y * z + x * w),
            1 - 2 * (x * x + y * y),
        ),
        dim=-1,
    ).reshape(*q.shape[:-1], 3, 3)


def matrix_to_quat(matrix: torch.Tensor) -> torch.Tensor:
    """Branchless stable matrix-to-xyzw conversion for proper rotations."""
    m = matrix
    q_abs = torch.sqrt(
        torch.clamp_min(
            torch.stack(
                (
                    1 + m[..., 0, 0] - m[..., 1, 1] - m[..., 2, 2],
                    1 - m[..., 0, 0] + m[..., 1, 1] - m[..., 2, 2],
                    1 - m[..., 0, 0] - m[..., 1, 1] + m[..., 2, 2],
                    1 + m[..., 0, 0] + m[..., 1, 1] + m[..., 2, 2],
                ),
                dim=-1,
            ),
            0.0,
        )
    )
    candidates = torch.stack(
        (
            torch.stack(
                (
                    q_abs[..., 0].square(),
                    m[..., 0, 1] + m[..., 1, 0],
                    m[..., 0, 2] + m[..., 2, 0],
                    m[..., 2, 1] - m[..., 1, 2],
                ),
                -1,
            ),
            torch.stack(
                (
                    m[..., 0, 1] + m[..., 1, 0],
                    q_abs[..., 1].square(),
                    m[..., 1, 2] + m[..., 2, 1],
                    m[..., 0, 2] - m[..., 2, 0],
                ),
                -1,
            ),
            torch.stack(
                (
                    m[..., 0, 2] + m[..., 2, 0],
                    m[..., 1, 2] + m[..., 2, 1],
                    q_abs[..., 2].square(),
                    m[..., 1, 0] - m[..., 0, 1],
                ),
                -1,
            ),
            torch.stack(
                (
                    m[..., 2, 1] - m[..., 1, 2],
                    m[..., 0, 2] - m[..., 2, 0],
                    m[..., 1, 0] - m[..., 0, 1],
                    q_abs[..., 3].square(),
                ),
                -1,
            ),
        ),
        dim=-2,
    )
    denom = (2.0 * q_abs).clamp_min(torch.finfo(matrix.dtype).eps)[..., :, None]
    candidates = candidates / denom
    choice = q_abs.argmax(dim=-1)
    gather = choice[..., None, None].expand(*choice.shape, 1, 4)
    q = torch.gather(candidates, -2, gather).squeeze(-2)
    q = normalize(q)
    return torch.where(q[..., 3:4] < 0, -q, q)


def donor_quat_to_flu(q: torch.Tensor) -> torch.Tensor:
    basis = donor_to_flu_matrix(dtype=q.dtype, device=q.device)
    matrix = basis @ quat_to_matrix(q) @ basis.T
    return matrix_to_quat(matrix)


def mirror_flu_quat(q: torch.Tensor) -> torch.Tensor:
    reflection = torch.diag(torch.tensor([1.0, -1.0, 1.0], dtype=q.dtype, device=q.device))
    return matrix_to_quat(reflection @ quat_to_matrix(q) @ reflection)
