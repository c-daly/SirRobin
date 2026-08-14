"""Ellipsoid Lamb added-mass coefficients using pinned Gauss-Legendre quadrature."""

from __future__ import annotations

from functools import cache

import numpy as np
import torch


@cache
def gl_nodes_weights(order: int = 256) -> tuple[np.ndarray, np.ndarray]:
    nodes, weights = np.polynomial.legendre.leggauss(order)
    return (nodes + 1.0) * 0.5, weights * 0.5


_GL256_NODES_NP, _GL256_WEIGHTS_NP = gl_nodes_weights()
_GL256_NODES = torch.from_numpy(_GL256_NODES_NP.copy())
_GL256_WEIGHTS = torch.from_numpy(_GL256_WEIGHTS_NP.copy())


def lamb_coefficients(abc: torch.Tensor, *, order: int = 256) -> torch.Tensor:
    """Return normalized `(alpha,beta,gamma)` with sum exactly scaled to two."""
    if abc.shape[-1] != 3 or not torch.is_floating_point(abc):
        raise TypeError("abc must be a floating tensor ending in size 3")
    if torch.any(~torch.isfinite(abc)) or torch.any(abc <= 0):
        raise ValueError("ellipsoid semi-axes must be finite and positive")
    return lamb_coefficients_unchecked(abc, order=order)


def lamb_coefficients_unchecked(
    abc: torch.Tensor,
    *,
    order: int = 256,
) -> torch.Tensor:
    """Evaluate coefficients for boundary-validated positive semi-axes."""

    if order == 256:
        t = _GL256_NODES.to(dtype=abc.dtype, device=abc.device)
        weights = _GL256_WEIGHTS.to(dtype=abc.dtype, device=abc.device)
    else:
        nodes_np, weights_np = gl_nodes_weights(order)
        t = torch.as_tensor(nodes_np, dtype=abc.dtype, device=abc.device)
        weights = torch.as_tensor(weights_np, dtype=abc.dtype, device=abc.device)
    axes2 = abc.square()
    scale = axes2.amax(dim=-1, keepdim=True)
    lam = scale.unsqueeze(-1) * ((1.0 - t) / t)
    jac = scale.unsqueeze(-1) / t.square()
    shifted = axes2.unsqueeze(-1) + lam
    delta = torch.sqrt(shifted.prod(dim=-2))
    integrand = jac / (shifted * delta.unsqueeze(-2))
    integrals = (integrand * weights).sum(dim=-1)
    coeff = abc.prod(dim=-1, keepdim=True) * integrals
    return coeff * (2.0 / coeff.sum(dim=-1, keepdim=True))


def lamb_factors(abc: torch.Tensor, *, order: int = 256) -> torch.Tensor:
    coeff = lamb_coefficients(abc, order=order)
    return coeff / (2.0 - coeff).clamp_min(torch.finfo(coeff.dtype).eps)


def lamb_factors_unchecked(abc: torch.Tensor, *, order: int = 256) -> torch.Tensor:
    coeff = lamb_coefficients_unchecked(abc, order=order)
    return coeff / (2.0 - coeff).clamp_min(torch.finfo(coeff.dtype).eps)


def added_mass(abc: torch.Tensor, rho_water: float = 1000.0, *, order: int = 256) -> torch.Tensor:
    volume = (4.0 / 3.0) * torch.pi * abc.prod(dim=-1, keepdim=True)
    return lamb_factors(abc, order=order) * rho_water * volume


def added_mass_unchecked(
    abc: torch.Tensor,
    rho_water: float = 1000.0,
    *,
    order: int = 256,
) -> torch.Tensor:
    """Evaluate added mass for boundary-validated positive semi-axes."""

    volume = (4.0 / 3.0) * torch.pi * abc.prod(dim=-1, keepdim=True)
    return lamb_factors_unchecked(abc, order=order) * rho_water * volume


def donor_lamb_factors(abc: torch.Tensor) -> torch.Tensor:
    """Historical Simpson-2048 donor rule, isolated to gain0 conformance."""
    source_dtype = abc.dtype
    abc64 = abc.to(torch.float32).to(torch.float64)
    axes2 = abc64.square()
    scale = axes2.amax(dim=-1, keepdim=True)
    i = torch.arange(2049, dtype=torch.float64, device=abc.device)
    t = i / 2048.0
    om = 1.0 - t
    safe_om = torch.where(om < 1e-9, torch.ones_like(om), om)
    lam = scale.unsqueeze(-1) * t / safe_om
    jac = scale.unsqueeze(-1) / safe_om.square()
    shifted = axes2.unsqueeze(-1) + lam
    delta = torch.sqrt(shifted.prod(dim=-2))
    integrand = jac / (shifted * delta.unsqueeze(-2))
    integrand = torch.where((om < 1e-9), torch.zeros_like(integrand), integrand)
    weights = torch.where(
        (i == 0) | (i == 2048), 1.0, torch.where((i.to(torch.int64) % 2) == 1, 4.0, 2.0)
    )
    coeff = abc64.prod(-1, keepdim=True) * (integrand * weights).sum(-1) / (2048.0 * 3.0)
    coeff = coeff * (2.0 / coeff.sum(-1, keepdim=True))
    return (coeff / (2.0 - coeff)).to(torch.float32).to(source_dtype)


def donor_added_mass(abc: torch.Tensor, rho_water: float = 1000.0) -> torch.Tensor:
    abc32 = abc.to(torch.float32)
    volume = (4.0 / 3.0) * torch.tensor(torch.pi, dtype=torch.float32, device=abc.device) * abc32.prod(
        -1, keepdim=True
    )
    return (donor_lamb_factors(abc) * rho_water * volume).to(abc.dtype)
