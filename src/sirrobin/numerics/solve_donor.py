"""Untouched donor symmetric 3x3 cofactor solve for gain0 comparison only."""

import torch


def solve_sym3_donor(matrix: torch.Tensor, rhs: torch.Tensor) -> torch.Tensor:
    m00, m01, m02 = matrix[:, 0, 0], matrix[:, 0, 1], matrix[:, 0, 2]
    m11, m12, m22 = matrix[:, 1, 1], matrix[:, 1, 2], matrix[:, 2, 2]
    c00 = m11 * m22 - m12 * m12
    c01 = m02 * m12 - m01 * m22
    c02 = m01 * m12 - m02 * m11
    det = m00 * c00 + m01 * c01 + m02 * c02
    c11 = m00 * m22 - m02 * m02
    c12 = m02 * m01 - m00 * m12
    c22 = m00 * m11 - m01 * m01
    safe = torch.where(det.abs() < 1e-12, torch.ones_like(det), det)
    exact = torch.stack(
        (
            (c00 * rhs[:, 0] + c01 * rhs[:, 1] + c02 * rhs[:, 2]) / safe,
            (c01 * rhs[:, 0] + c11 * rhs[:, 1] + c12 * rhs[:, 2]) / safe,
            (c02 * rhs[:, 0] + c12 * rhs[:, 1] + c22 * rhs[:, 2]) / safe,
        ),
        dim=-1,
    )
    fallback = torch.stack(
        (rhs[:, 0] / m00.clamp_min(1e-6), torch.zeros_like(m00), rhs[:, 2] / m22.clamp_min(1e-6)),
        dim=-1,
    )
    return torch.where((det.abs() < 1e-12)[:, None], fallback, exact)
