"""Finite, branch-explicit constrained x/z effective-mass solve."""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(slots=True)
class SolveResult:
    dv: torch.Tensor
    j_reg: torch.Tensor
    regularized: torch.Tensor
    condition_estimate: torch.Tensor


def solve_constrained_xz(
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
    finite_inputs = ~torch.any(
        body_valid & (~torch.isfinite(matrix).all(dim=(-2, -1)) | ~torch.isfinite(impulse).all(dim=-1))
    )
    torch._assert_async(finite_inputs, "valid bodies must have finite solve inputs")
    a, b, c = matrix[:, 0, 0], matrix[:, 0, 2], matrix[:, 2, 2]
    px, pz = impulse[:, 0], impulse[:, 2]
    tr = a + c
    det = a * c - b * b
    disc = torch.sqrt(torch.clamp((a - c).square() + 4.0 * b.square(), min=0.0))
    lam_max = 0.5 * (tr + disc)
    degenerate = (lam_max < lam_floor) | (det < lam_max.square() / kappa_max)
    invalid = ~body_valid
    regularized = body_valid & degenerate
    exact = body_valid & ~degenerate
    one = torch.ones_like(det)

    reg = eps_spd * lam_max
    det_reg = (a + reg) * (c + reg) - b * b
    denom_exact = torch.where(exact, det, one)
    denom_reg = torch.where(regularized, det_reg, one)
    exact_x = (c * px - b * pz) / denom_exact
    exact_z = (-b * px + a * pz) / denom_exact
    reg_x = ((c + reg) * px - b * pz) / denom_reg
    reg_z = (-b * px + (a + reg) * pz) / denom_reg
    dv_x = torch.where(invalid, 0.0, torch.where(regularized, reg_x, exact_x))
    dv_z = torch.where(invalid, 0.0, torch.where(regularized, reg_z, exact_z))
    dv = torch.stack((dv_x, torch.zeros_like(dv_x), dv_z), dim=-1)
    j_reg = torch.stack(
        (
            torch.where(regularized, -reg * reg_x, 0.0),
            torch.zeros_like(reg_x),
            torch.where(regularized, -reg * reg_z, 0.0),
        ),
        dim=-1,
    )
    tiny = torch.finfo(matrix.dtype).tiny
    condition = lam_max.square() / det.clamp_min(tiny)
    torch._assert_async(torch.isfinite(dv).all(), "constrained solve produced non-finite dv")
    torch._assert_async(torch.isfinite(j_reg).all(), "constrained solve produced non-finite impulse")
    return SolveResult(dv=dv, j_reg=j_reg, regularized=regularized, condition_estimate=condition)
