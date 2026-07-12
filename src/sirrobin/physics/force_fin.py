"""Finite-wing Garrick circulatory channel."""

from __future__ import annotations

import math

import torch


def fin_channel(
    lift_slope: torch.Tensor,
    aspect_ratio: torch.Tensor,
    area: torch.Tensor,
    u: torch.Tensor,
    vt: torch.Tensor,
    slope: torch.Tensor,
    active: torch.Tensor,
    *,
    rho_water: float = 1000.0,
    profile_cd: float = 0.02,
    span_eff: float = 0.9,
    stall_aoa: float = 0.35,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    u_cl = torch.clamp_min(u, 0.0)
    q2 = u_cl.square() + vt.square()
    moving = active & (q2 >= 1e-8)
    speed = torch.sqrt(torch.clamp_min(q2, 1e-30))
    pitch = torch.asin(torch.clamp(slope, -1.0, 1.0))
    alpha = torch.clamp(torch.atan2(vt, u_cl) - pitch, -stall_aoa, stall_aoa)
    cl = lift_slope * alpha
    dynamic = 0.5 * rho_water * u_cl.square() * area
    lift = dynamic * cl
    cdi = profile_cd + cl.square() / (math.pi * span_eff * aspect_ratio.clamp_min(1e-4))
    drag = dynamic * cdi
    sin_b = vt / speed
    cos_b = u_cl / speed
    p_wake = drag * speed
    thrust = lift * sin_b - drag * cos_b
    normal_force = lift * cos_b + drag * sin_b
    p_input = normal_force * vt
    zero = torch.zeros_like(thrust)
    return (
        torch.where(moving, thrust, zero),
        torch.where(moving, p_input, zero),
        torch.where(moving, p_wake, zero),
    )
