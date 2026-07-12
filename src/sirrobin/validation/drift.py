"""Executable prefix-budget validation for discrete mechanical residuals."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from sirrobin.physics.swim_step import SwimKernel


@dataclass(frozen=True, slots=True)
class DriftResult:
    cumulative_residual: torch.Tensor
    cumulative_scale: torch.Tensor
    prefix_ratio: torch.Tensor
    maximum_after_burnin: torch.Tensor
    monotone_bias: torch.Tensor
    regularization_count: int


def run_prefix_budget(
    kernel: SwimKernel,
    *,
    steps: int = 100_000,
    burnin: int = 100,
) -> DriftResult:
    """Run the exact C3 definition and retain every prefix for audit."""
    if steps < 1 or not 0 <= burnin < steps:
        raise ValueError("require steps >= 1 and 0 <= burnin < steps")
    body = kernel.body
    output_device = body.v_com.device
    cumulative_residual = torch.empty(
        (steps, body.batch_size), dtype=torch.float64, device=output_device
    )
    cumulative_scale = torch.empty_like(cumulative_residual)
    running_residual = torch.zeros(body.batch_size, dtype=torch.float64, device=output_device)
    running_scale = torch.zeros_like(running_residual)
    regularization = torch.zeros((), dtype=torch.int64, device=output_device)
    atol = (
        kernel.config.e_atol_f64
        if body.v_com.dtype == torch.float64
        else kernel.config.e_atol_f32
    )
    for index in range(steps):
        ledger = kernel.step()
        scale = torch.stack(
            (ledger.delta_ke.abs(), ledger.work_impulse.abs(), ledger.work_delta_m.abs()),
            dim=-1,
        ).amax(-1)
        running_residual += ledger.r_step.to(torch.float64)
        running_scale += scale.clamp_min(atol).to(torch.float64)
        cumulative_residual[index].copy_(running_residual)
        cumulative_scale[index].copy_(running_scale)
        regularization += ledger.regularized.sum()
    prefix_ratio = cumulative_residual.abs() / cumulative_scale
    zero = torch.zeros((1, body.batch_size), dtype=torch.float64, device=output_device)
    step_residual = torch.diff(cumulative_residual, dim=0, prepend=zero)[burnin:]
    positive = (step_residual > 0).sum(dim=0)
    negative = (step_residual < 0).sum(dim=0)
    nonzero = positive + negative
    monotone_bias = (nonzero > 0) & (torch.maximum(positive, negative) > 0.95 * nonzero)
    return DriftResult(
        cumulative_residual=cumulative_residual,
        cumulative_scale=cumulative_scale,
        prefix_ratio=prefix_ratio,
        maximum_after_burnin=prefix_ratio[burnin:].amax(dim=0),
        monotone_bias=monotone_bias,
        regularization_count=int(regularization.item()),
    )
