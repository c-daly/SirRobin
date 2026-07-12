"""Dimensioned step and prefix mechanical residual gates."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from sirrobin.physics.config import LocomotionConfig
from sirrobin.physics.contracts import StepLedger


@dataclass(frozen=True, slots=True)
class PrefixBudget:
    cumulative_residual: torch.Tensor
    cumulative_scale: torch.Tensor
    normalized_prefix: torch.Tensor
    max_normalized_after_warmup: float
    monotone_bias: bool


def step_energy_scale(ledger: StepLedger, atol: float) -> torch.Tensor:
    floor = torch.full_like(ledger.delta_ke, atol)
    return torch.stack(
        (ledger.delta_ke.abs(), ledger.work_impulse.abs(), ledger.work_delta_m.abs(), floor), dim=-1
    ).amax(-1)


def step_closes(ledger: StepLedger, config: LocomotionConfig) -> torch.Tensor:
    f64 = ledger.r_step.dtype == torch.float64
    atol = config.e_atol_f64 if f64 else config.e_atol_f32
    rtol = config.rtol_f64 if f64 else config.rtol_f32
    return ledger.r_step.abs() <= atol + rtol * step_energy_scale(ledger, atol)


def prefix_budget(residuals: torch.Tensor, scales: torch.Tensor, *, warmup: int = 100) -> PrefixBudget:
    if residuals.shape != scales.shape or residuals.ndim != 1:
        raise ValueError("residuals and scales must be equal-length 1-D tensors")
    cumulative_residual = torch.cumsum(residuals, dim=0)
    cumulative_scale = torch.cumsum(scales, dim=0).clamp_min(torch.finfo(scales.dtype).tiny)
    normalized = cumulative_residual.abs() / cumulative_scale
    tail = normalized[min(warmup, normalized.numel() - 1) :]
    signs = torch.sign(residuals[min(warmup, residuals.numel() - 1) :])
    nonzero = signs[signs != 0]
    monotone = bool(nonzero.numel() > 0 and (nonzero == nonzero[0]).float().mean() > 0.95)
    return PrefixBudget(cumulative_residual, cumulative_scale, normalized, float(tail.max().item()), monotone)
