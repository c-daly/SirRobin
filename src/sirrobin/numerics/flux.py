"""Deterministic conversion of physical flux requests into exact integer transfers."""

from __future__ import annotations

from dataclasses import dataclass

import torch

INT64_SAFE_MAX = 2**62


@dataclass(frozen=True, slots=True)
class CommittedFlux:
    committed_q: torch.Tensor
    carry_mol: torch.Tensor
    shortfall_q: torch.Tensor


def commit_flux(
    requested_mol: torch.Tensor,
    carry_mol: torch.Tensor,
    available_q: torch.Tensor,
    *,
    q_mass_mol: float,
) -> CommittedFlux:
    """Quantize a nonnegative request; availability shortfall never becomes future debt."""
    if requested_mol.dtype != torch.float64 or carry_mol.dtype != torch.float64:
        raise TypeError("requests and carries must be float64")
    if available_q.dtype != torch.int64 or requested_mol.shape != carry_mol.shape != available_q.shape:
        if requested_mol.shape != carry_mol.shape or requested_mol.shape != available_q.shape:
            raise ValueError("request, carry, and availability shapes must match")
        raise TypeError("availability must be int64")
    torch._assert_async(
        (torch.isfinite(requested_mol) & (requested_mol >= 0)).all(),
        "physical requests must be finite and nonnegative",
    )
    torch._assert_async(
        ((carry_mol >= 0) & (carry_mol < q_mass_mol)).all(),
        "carry must be in [0,q_mass)",
    )
    total_mol = requested_mol + carry_mol
    desired_q = torch.floor(total_mol / q_mass_mol).to(torch.int64)
    committed_q = torch.minimum(desired_q, available_q)
    new_carry = total_mol - desired_q.to(torch.float64) * q_mass_mol
    new_carry = torch.clamp(
        new_carry,
        min=0.0,
        max=torch.nextafter(
            torch.tensor(q_mass_mol, dtype=torch.float64, device=new_carry.device),
            torch.tensor(0.0, dtype=torch.float64, device=new_carry.device),
        ),
    )
    return CommittedFlux(committed_q, new_carry, desired_q - committed_q)


def deterministic_fraction(
    total_q: torch.Tensor, fraction: float, carry_q: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Split one integer debit into two exact credits with bounded fractional carry."""
    if total_q.dtype != torch.int64 or carry_q.dtype != torch.float64:
        raise TypeError("total must be int64 and carry float64")
    if total_q.shape != carry_q.shape:
        raise ValueError("total and carry shapes must match")
    if not 0.0 <= fraction <= 1.0:
        raise ValueError("fraction must be in [0,1]")
    raw = total_q.to(torch.float64) * fraction + carry_q
    first_q = torch.floor(raw).to(torch.int64).clamp_min(0)
    first_q = torch.minimum(first_q, total_q)
    new_carry = raw - torch.floor(raw)
    if fraction in (0.0, 1.0):
        new_carry = torch.zeros_like(new_carry)
    return first_q, total_q - first_q, new_carry


def apportion_integer(total_q: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
    """Largest-remainder apportionment with stable low-index tie-breaking."""
    if total_q.dtype != torch.int64 or not torch.is_floating_point(weights):
        raise TypeError("total must be int64 and weights floating")
    if weights.shape[:-1] != total_q.shape:
        raise ValueError("weights must add one choice dimension to total")
    torch._assert_async(
        (torch.isfinite(weights) & (weights >= 0)).all(),
        "weights must be finite and nonnegative",
    )
    norm = weights.sum(dim=-1, keepdim=True)
    safe = torch.where(norm > 0, weights / norm, torch.zeros_like(weights))
    raw = safe * total_q.to(weights.dtype).unsqueeze(-1)
    base = torch.floor(raw).to(torch.int64)
    remaining = total_q - base.sum(dim=-1)
    remainders = raw - base.to(raw.dtype)
    order = torch.argsort(remainders, dim=-1, descending=True, stable=True)
    rank = torch.empty_like(order)
    positions = torch.arange(weights.shape[-1], device=weights.device).expand_as(order)
    rank.scatter_(-1, order, positions)
    extras = rank < remaining.unsqueeze(-1)
    return base + extras.to(torch.int64)
