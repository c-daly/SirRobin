"""Yaw inertia, drag, and exact discrete angular-work accounting."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from sirrobin.numerics.quat import conjugate, rotate


@dataclass(frozen=True, slots=True)
class YawAdvance:
    momentum: torch.Tensor
    omega_before: torch.Tensor
    omega_after: torch.Tensor
    yaw: torch.Tensor
    floor_hit: torch.Tensor
    backstop_hit: torch.Tensor
    delta_ke_j: torch.Tensor
    work_impulse_j: torch.Tensor
    work_delta_inertia_j: torch.Tensor
    residual_j: torch.Tensor


def yaw_inertia(
    relative_position_enu_m: torch.Tensor,
    rotation_enu: torch.Tensor,
    segment_mass_kg: torch.Tensor,
    added_mass_flu_kg: torch.Tensor,
    mask: torch.Tensor,
) -> torch.Tensor:
    """Return pose-dependent yaw inertia from point mass and Lamb added mass."""
    radius2 = relative_position_enu_m[..., :2].square().sum(-1)
    structural = segment_mass_kg * radius2
    unit_yaw_velocity = torch.stack(
        (-relative_position_enu_m[..., 1], relative_position_enu_m[..., 0],
         torch.zeros_like(radius2)),
        dim=-1,
    )
    velocity_flu = rotate(conjugate(rotation_enu), unit_yaw_velocity)
    added = (added_mass_flu_kg * velocity_flu.square()).sum(-1)
    return torch.where(mask, structural + added, 0.0).sum(-1)


def yaw_drag_coefficient(
    relative_position_enu_m: torch.Tensor,
    broadside_area_m2: torch.Tensor,
    mask: torch.Tensor,
    density_kg_m3: torch.Tensor,
    yaw_cd: float,
) -> torch.Tensor:
    radius = torch.linalg.vector_norm(relative_position_enu_m[..., :2], dim=-1)
    moment = torch.where(mask, broadside_area_m2 * radius.pow(3), 0.0).sum(-1)
    return 0.5 * density_kg_m3 * yaw_cd * moment


def wrap_pi(angle: torch.Tensor) -> torch.Tensor:
    return torch.atan2(torch.sin(angle), torch.cos(angle))


def advance_yaw(
    yaw0: torch.Tensor,
    momentum0: torch.Tensor,
    inertia0: torch.Tensor,
    inertia1: torch.Tensor,
    torque_nm: torch.Tensor,
    dt: float,
    valid: torch.Tensor,
    *,
    inertia_floor: float,
    emergency_omega: float,
) -> YawAdvance:
    floor_hit = valid & ((inertia0 < inertia_floor) | (inertia1 < inertia_floor))
    safe_i0 = inertia0.clamp_min(inertia_floor)
    safe_i1 = inertia1.clamp_min(inertia_floor)
    momentum1 = torch.where(valid, momentum0 + torque_nm * dt, torch.zeros_like(momentum0))
    omega0 = torch.where(valid, momentum0 / safe_i0, torch.zeros_like(momentum0))
    omega1 = torch.where(valid, momentum1 / safe_i1, torch.zeros_like(momentum1))
    backstop = valid & (omega1.abs() >= emergency_omega)
    yaw1 = torch.where(valid, wrap_pi(yaw0 + omega1 * dt), yaw0)
    ke0 = momentum0.square() / (2.0 * safe_i0)
    ke1 = momentum1.square() / (2.0 * safe_i1)
    delta_ke = torch.where(valid, ke1 - ke0, 0.0)
    work_impulse = torch.where(valid, 0.5 * (omega0 + omega1) * (momentum1 - momentum0), 0.0)
    work_delta_i = torch.where(
        valid,
        0.5 * momentum0 * momentum1 * (safe_i1.reciprocal() - safe_i0.reciprocal()),
        0.0,
    )
    residual = delta_ke - work_impulse - work_delta_i
    return YawAdvance(
        momentum1,
        omega0,
        omega1,
        yaw1,
        floor_hit,
        backstop,
        delta_ke,
        work_impulse,
        work_delta_i,
        residual,
    )
