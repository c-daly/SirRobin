"""Reservoir-backed scalar-grid read surface and transactional point depletion probe."""

from __future__ import annotations

from itertools import product

import torch

from sirrobin.fields.contracts import FieldSample
from sirrobin.fields.geometry import GridGeometry
from sirrobin.fields.sample import _axis_coordinates, sample_reservoir
from sirrobin.numerics.flux import apportion_integer


class ScalarGrid:
    def __init__(self, reservoir_q: torch.Tensor, geometry: GridGeometry, *, q_mass_mol: float) -> None:
        if reservoir_q.dtype != torch.int64 or reservoir_q.ndim != 4:
            raise TypeError("reservoir must be int64 [W,Gx,Gy,Gz]")
        self._reservoir_q = reservoir_q
        self.geometry = geometry
        self.q_mass_mol = q_mass_mol

    @property
    def reservoir_q(self) -> torch.Tensor:
        """Read-only-by-contract view; economy transactions own mutation."""
        return self._reservoir_q

    def sample(self, positions_m: torch.Tensor) -> FieldSample:
        return sample_reservoir(
            self._reservoir_q,
            positions_m,
            self.geometry,
            q_mass_mol=self.q_mass_mol,
        )

    def deplete_at(self, world: int, position_m: torch.Tensor, requested_q: int) -> int:
        """Synthetic single-point transaction; not a biological grazing API."""
        if not 0 <= world < self._reservoir_q.shape[0] or requested_q < 0:
            raise ValueError("invalid world or request")
        position = position_m.to(device=self._reservoir_q.device, dtype=torch.float64).reshape(1, 3)
        x0, x1, tx = _axis_coordinates(position[:, 0], self.geometry.dx_m, self.geometry.gx, periodic=True)
        y0, y1, ty = _axis_coordinates(position[:, 1], self.geometry.dy_m, self.geometry.gy, periodic=True)
        z0, z1, tz = _axis_coordinates(-position[:, 2], self.geometry.dz_m, self.geometry.gz, periodic=False)
        axes = (
            (x0.item(), x1.item(), tx.item()),
            (y0.item(), y1.item(), ty.item()),
            (z0.item(), z1.item(), tz.item()),
        )
        indices: list[tuple[int, int, int]] = []
        weights: list[float] = []
        for bx, by, bz in product((0, 1), repeat=3):
            ix = int(axes[0][bx])
            iy = int(axes[1][by])
            iz = int(axes[2][bz])
            weight = (
                (axes[0][2] if bx else 1.0 - axes[0][2])
                * (axes[1][2] if by else 1.0 - axes[1][2])
                * (axes[2][2] if bz else 1.0 - axes[2][2])
            )
            if (ix, iy, iz) in indices:
                weights[indices.index((ix, iy, iz))] += weight
            else:
                indices.append((ix, iy, iz))
                weights.append(weight)
        available = torch.stack([self._reservoir_q[world, *index] for index in indices])
        target = min(requested_q, int(available.sum().item()))
        remaining = target
        debit = torch.zeros_like(available)
        active = available > 0
        weight_tensor = torch.tensor(weights, dtype=torch.float64, device=available.device)
        for _ in range(len(indices)):
            if remaining == 0 or not bool(active.any()):
                break
            allocation = apportion_integer(
                torch.tensor(remaining, dtype=torch.int64, device=available.device),
                torch.where(active, weight_tensor, 0.0),
            )
            room = available - debit
            realized = torch.minimum(allocation, room)
            debit += realized
            remaining = target - int(debit.sum().item())
            active &= debit < available
        for index, amount in zip(indices, debit, strict=True):
            self._reservoir_q[world, *index] -= amount
        return int(debit.sum().item())
