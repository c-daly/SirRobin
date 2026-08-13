"""Finite branch-explicit constrained ENU x/y effective-mass solve."""

from __future__ import annotations

import torch

from sirrobin.numerics.solve_constrained_xz import SolveResult


def solve_constrained_xy(
    matrix: torch.Tensor,
    impulse: torch.Tensor,
    body_valid: torch.Tensor,
    *,
    kappa_max: float = 1e6,
    lam_floor: float = 1e-9,
    eps_spd: float = 1e-6,
) -> SolveResult:
    if matrix.shape[-2:] != (3, 3) or impulse.shape[-1] != 3:
        raise ValueError("expected [B,3,3] matrix and [B,3] impulse")
    valid_finite = ~torch.any(
        body_valid & (~torch.isfinite(matrix).all(dim=(-2, -1)) | ~torch.isfinite(impulse).all(-1))
    )
    torch._assert_async(valid_finite, "valid bodies must have finite solve inputs")
    a, b, c = matrix[:, 0, 0], matrix[:, 0, 1], matrix[:, 1, 1]
    px, py = impulse[:, 0], impulse[:, 1]
    trace = a + c
    det = a * c - b.square()
    disc = torch.sqrt(torch.clamp_min((a - c).square() + 4 * b.square(), 0.0))
    lam_max = 0.5 * (trace + disc)
    degenerate = (lam_max < lam_floor) | (det < lam_max.square() / kappa_max)
    regularized = body_valid & degenerate
    exact = body_valid & ~degenerate
    one = torch.ones_like(det)
    reg = eps_spd * lam_max
    det_reg = (a + reg) * (c + reg) - b.square()
    exact_den = torch.where(exact, det, one)
    reg_den = torch.where(regularized, det_reg, one)
    exact_x = (c * px - b * py) / exact_den
    exact_y = (-b * px + a * py) / exact_den
    reg_x = ((c + reg) * px - b * py) / reg_den
    reg_y = (-b * px + (a + reg) * py) / reg_den
    invalid = ~body_valid
    dv_x = torch.where(invalid, 0.0, torch.where(regularized, reg_x, exact_x))
    dv_y = torch.where(invalid, 0.0, torch.where(regularized, reg_y, exact_y))
    dv = torch.stack((dv_x, dv_y, torch.zeros_like(dv_x)), -1)
    j_reg = torch.stack(
        (
            torch.where(regularized, -reg * reg_x, 0.0),
            torch.where(regularized, -reg * reg_y, 0.0),
            torch.zeros_like(reg_x),
        ),
        -1,
    )
    condition = lam_max.square() / det.clamp_min(torch.finfo(matrix.dtype).tiny)
    torch._assert_async(torch.isfinite(dv).all(), "constrained solve produced nonfinite dv")
    return SolveResult(dv, j_reg, regularized, condition)
