"""Checked exact int64 transfers for the S0 fake-reservoir scaffold."""

from __future__ import annotations

import torch

INT64_MAX = torch.iinfo(torch.int64).max


def transfer_quanta(
    src_q: torch.Tensor,
    dst_q: torch.Tensor,
    requested_q: torch.Tensor,
    mask: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Transfer nonnegative int64 quanta without overdraft or wraparound.

    Returns new source, new destination, and nonnegative shortfall. Inputs are
    not mutated. This is exact only over the explicitly checked int64 domain.
    """
    for name, value in (("src_q", src_q), ("dst_q", dst_q), ("requested_q", requested_q)):
        if value.dtype != torch.int64:
            raise TypeError(f"{name} must be int64")
    if mask.dtype != torch.bool:
        raise TypeError("mask must be bool")
    if not (src_q.shape == dst_q.shape == requested_q.shape == mask.shape):
        raise ValueError("all transfer tensors must have the same shape")
    if torch.any(src_q < 0) or torch.any(dst_q < 0) or torch.any(requested_q < 0):
        raise ValueError("reservoirs and requests must be nonnegative")

    req = torch.where(mask, requested_q, torch.zeros_like(requested_q))
    effective = torch.minimum(req, src_q)
    if torch.any(dst_q > INT64_MAX - effective):
        raise OverflowError("destination int64 overflow")
    return src_q - effective, dst_q + effective, req - effective


def close_books(*reservoirs: torch.Tensor, expected_total: int | torch.Tensor) -> bool:
    if not reservoirs:
        raise ValueError("at least one reservoir is required")
    total = 0
    for reservoir in reservoirs:
        if reservoir.dtype != torch.int64:
            raise TypeError("all reservoirs must be int64")
        if torch.any(reservoir < 0):
            return False
        total += int(reservoir.sum(dtype=torch.int64).item())
    expected = int(expected_total.item()) if isinstance(expected_total, torch.Tensor) else int(expected_total)
    return total == expected
