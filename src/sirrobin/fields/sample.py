"""Trilinear sampling of an int64 reservoir without storing a concentration mirror."""

from __future__ import annotations

import torch

from sirrobin.fields.contracts import FieldSample
from sirrobin.fields.geometry import GridGeometry


def _axis_coordinates(
    value: torch.Tensor, spacing: float, count: int, *, periodic: bool
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    cell = value / spacing - 0.5
    if periodic:
        cell = torch.remainder(cell, count)
        lower = torch.floor(cell).to(torch.int64)
        upper = torch.remainder(lower + 1, count)
    else:
        if torch.any(value < 0) or torch.any(value > count * spacing):
            raise ValueError("vertical sample lies outside the closed domain")
        cell = cell.clamp(0.0, count - 1.0)
        lower = torch.floor(cell).to(torch.int64)
        upper = (lower + 1).clamp_max(count - 1)
    return lower, upper, cell - lower.to(cell.dtype)


def sample_reservoir(
    reservoir_q: torch.Tensor,
    positions_m: torch.Tensor,
    geometry: GridGeometry,
    *,
    q_mass_mol: float,
) -> FieldSample:
    if reservoir_q.dtype != torch.int64 or reservoir_q.ndim != 4:
        raise TypeError("reservoir must be int64 [W,Gx,Gy,Gz]")
    if positions_m.ndim != 3 or positions_m.shape[0] != reservoir_q.shape[0] or positions_m.shape[-1] != 3:
        raise ValueError("positions must be [W,P,3]")
    pos = positions_m.to(torch.float64)
    x0, x1, tx = _axis_coordinates(pos[..., 0], geometry.dx_m, geometry.gx, periodic=True)
    y0, y1, ty = _axis_coordinates(pos[..., 1], geometry.dy_m, geometry.gy, periodic=True)
    # Public positions use the ENU world frame: surface z=0, water z<0.
    depth_m = -pos[..., 2]
    z0, z1, tz = _axis_coordinates(depth_m, geometry.dz_m, geometry.gz, periodic=False)
    world = torch.arange(reservoir_q.shape[0], device=reservoir_q.device)[:, None]

    def gather(ix: torch.Tensor, iy: torch.Tensor, iz: torch.Tensor) -> torch.Tensor:
        return reservoir_q[world, ix, iy, iz].to(torch.float64) * q_mass_mol / geometry.cell_volume_m3

    c000, c100 = gather(x0, y0, z0), gather(x1, y0, z0)
    c010, c110 = gather(x0, y1, z0), gather(x1, y1, z0)
    c001, c101 = gather(x0, y0, z1), gather(x1, y0, z1)
    c011, c111 = gather(x0, y1, z1), gather(x1, y1, z1)
    one_x, one_y, one_z = 1.0 - tx, 1.0 - ty, 1.0 - tz
    value = (
        c000 * one_x * one_y * one_z
        + c100 * tx * one_y * one_z
        + c010 * one_x * ty * one_z
        + c110 * tx * ty * one_z
        + c001 * one_x * one_y * tz
        + c101 * tx * one_y * tz
        + c011 * one_x * ty * tz
        + c111 * tx * ty * tz
    )
    dx = (
        (c100 - c000) * one_y * one_z
        + (c110 - c010) * ty * one_z
        + (c101 - c001) * one_y * tz
        + (c111 - c011) * ty * tz
    ) / geometry.dx_m
    dy = (
        (c010 - c000) * one_x * one_z
        + (c110 - c100) * tx * one_z
        + (c011 - c001) * one_x * tz
        + (c111 - c101) * tx * tz
    ) / geometry.dy_m
    depth_gradient = (
        (c001 - c000) * one_x * one_y
        + (c101 - c100) * tx * one_y
        + (c011 - c010) * one_x * ty
        + (c111 - c110) * tx * ty
    ) / geometry.dz_m
    return FieldSample(value, torch.stack((dx, dy, -depth_gradient), dim=-1))
