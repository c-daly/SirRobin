"""Fixed-shape trilinear point stencils for device transactions."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from sirrobin.fields.geometry import GridGeometry
from sirrobin.numerics.flux import apportion_integer


@dataclass(frozen=True, slots=True)
class PointStencil:
    cell_index: torch.Tensor
    weight: torch.Tensor
    vertical_out_of_bounds: torch.Tensor


def point_stencil(
    positions_enu_m: torch.Tensor,
    geometry: GridGeometry,
) -> PointStencil:
    """Return merged eight-wide cell indices and weights for [W,N,3] positions."""

    position = positions_enu_m.to(torch.float64)

    def periodic_axis(value: torch.Tensor, spacing: float, count: int):
        cell = torch.remainder(value / spacing - 0.5, count)
        lower = torch.floor(cell).to(torch.int64)
        upper = torch.remainder(lower + 1, count)
        return lower, upper, cell - lower.to(cell.dtype)

    def closed_axis(value: torch.Tensor, spacing: float, count: int):
        invalid = (value < 0.0) | (value > count * spacing)
        cell = (value / spacing - 0.5).clamp(0.0, count - 1.0)
        lower = torch.floor(cell).to(torch.int64)
        upper = (lower + 1).clamp_max(count - 1)
        return lower, upper, cell - lower.to(cell.dtype), invalid

    x0, x1, tx = periodic_axis(position[..., 0], geometry.dx_m, geometry.gx)
    y0, y1, ty = periodic_axis(position[..., 1], geometry.dy_m, geometry.gy)
    z0, z1, tz, invalid_z = closed_axis(
        -position[..., 2], geometry.dz_m, geometry.gz
    )
    corner = torch.arange(8, dtype=torch.int64, device=position.device)
    bx = ((corner >> 2) & 1).to(torch.bool)
    by = ((corner >> 1) & 1).to(torch.bool)
    bz = (corner & 1).to(torch.bool)
    ix = torch.where(bx, x1[..., None], x0[..., None])
    iy = torch.where(by, y1[..., None], y0[..., None])
    iz = torch.where(bz, z1[..., None], z0[..., None])
    wx = torch.where(bx, tx[..., None], 1.0 - tx[..., None])
    wy = torch.where(by, ty[..., None], 1.0 - ty[..., None])
    wz = torch.where(bz, tz[..., None], 1.0 - tz[..., None])
    cell_index = (ix * geometry.gy + iy) * geometry.gz + iz
    raw_weight = wx * wy * wz

    # Closed and one-cell axes create duplicate corners. Merge their weights into
    # the first occurrence so later integer apportionment sees each physical cell
    # once, matching ScalarGrid.point_stencil semantics without dynamic shapes.
    same_cell = cell_index[..., :, None] == cell_index[..., None, :]
    earlier = torch.tril(
        torch.ones((8, 8), dtype=torch.bool, device=position.device),
        diagonal=-1,
    )
    first = ~(same_cell & earlier).any(dim=-1)
    merged_weight = (same_cell * raw_weight[..., None, :]).sum(dim=-1)
    weight = torch.where(first, merged_weight, 0.0)
    return PointStencil(cell_index, weight, invalid_z)


def gather_stencil(
    reservoir_q: torch.Tensor,
    stencil: PointStencil,
) -> torch.Tensor:
    """Gather one int64 reservoir value per fixed stencil entry."""

    worlds = reservoir_q.shape[0]
    flat = reservoir_q.reshape(worlds, -1)
    index = stencil.cell_index.reshape(worlds, -1)
    return torch.gather(flat, 1, index).reshape_as(stencil.cell_index)


def sample_stencil_mol_m3(
    reservoir_q: torch.Tensor,
    stencil: PointStencil,
    geometry: GridGeometry,
    *,
    q_mass_mol: float,
) -> torch.Tensor:
    values = gather_stencil(reservoir_q, stencil).to(torch.float64)
    return (
        (values * stencil.weight).sum(dim=-1)
        * q_mass_mol
        / geometry.cell_volume_m3
    )


def deposit_stencil(
    reservoir_q: torch.Tensor,
    stencil: PointStencil,
    amount_q: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return reservoir plus exact per-point deposits and per-cell credits."""

    credit = apportion_integer(amount_q, stencil.weight)
    worlds = reservoir_q.shape[0]
    flat_credit = torch.zeros_like(reservoir_q.reshape(worlds, -1))
    flat_credit.scatter_add_(
        1,
        stencil.cell_index.reshape(worlds, -1),
        credit.reshape(worlds, -1),
    )
    return (reservoir_q.reshape(worlds, -1) + flat_credit).reshape_as(
        reservoir_q
    ), flat_credit.reshape_as(reservoir_q)
