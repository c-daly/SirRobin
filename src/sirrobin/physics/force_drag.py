"""Axial-only anisotropic quadratic form drag."""

from __future__ import annotations

import torch

from sirrobin.numerics.quat import conjugate, rotate


def drag_channel(
    segment_velocity: torch.Tensor,
    rotation: torch.Tensor,
    area_z: torch.Tensor,
    mask: torch.Tensor,
    *,
    rho_water: float = 1000.0,
    cd: float = 0.1,
) -> tuple[torch.Tensor, torch.Tensor]:
    local_velocity = rotate(conjugate(rotation), segment_velocity)
    axial = local_velocity[..., 2]
    local_force = torch.zeros_like(segment_velocity)
    local_force[..., 2] = -0.5 * rho_water * cd * area_z * axial.abs() * axial
    world_force = rotate(rotation, local_force)
    world_force = torch.where(mask[..., None], world_force, torch.zeros_like(world_force))
    total = world_force.sum(dim=1)
    dissipated = torch.clamp_min(-(world_force * segment_velocity).sum(dim=-1), 0.0).sum(dim=1)
    return total, dissipated
