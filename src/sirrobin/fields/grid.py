"""Reservoir-backed scalar-grid read surface and transactional point depletion probe."""

from __future__ import annotations

from itertools import product

import torch

from sirrobin.fields.contracts import FieldSample
from sirrobin.fields.geometry import GridGeometry
from sirrobin.fields.sample import _axis_coordinates, sample_reservoir
from sirrobin.numerics.flux import INT64_SAFE_MAX, apportion_integer


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

    def point_stencil(
        self, position_m: torch.Tensor
    ) -> tuple[list[tuple[int, int, int]], torch.Tensor]:
        """Return merged positive-weight cells for a continuous position."""
        position = position_m.to(
            device=self._reservoir_q.device, dtype=torch.float64
        ).reshape(1, 3)
        x0, x1, tx = _axis_coordinates(
            position[:, 0], self.geometry.dx_m, self.geometry.gx, periodic=True
        )
        y0, y1, ty = _axis_coordinates(
            position[:, 1], self.geometry.dy_m, self.geometry.gy, periodic=True
        )
        z0, z1, tz = _axis_coordinates(
            -position[:, 2], self.geometry.dz_m, self.geometry.gz, periodic=False
        )
        axes = (
            (x0.item(), x1.item(), tx.item()),
            (y0.item(), y1.item(), ty.item()),
            (z0.item(), z1.item(), tz.item()),
        )
        indices: list[tuple[int, int, int]] = []
        weights: list[float] = []
        for bx, by, bz in product((0, 1), repeat=3):
            index = (int(axes[0][bx]), int(axes[1][by]), int(axes[2][bz]))
            weight = (
                (axes[0][2] if bx else 1.0 - axes[0][2])
                * (axes[1][2] if by else 1.0 - axes[1][2])
                * (axes[2][2] if bz else 1.0 - axes[2][2])
            )
            if weight <= 0.0:
                continue
            if index in indices:
                weights[indices.index(index)] += weight
            else:
                indices.append(index)
                weights.append(weight)
        return indices, torch.tensor(
            weights, dtype=torch.float64, device=self._reservoir_q.device
        )

    def available_at(self, world: int, position_m: torch.Tensor) -> int:
        """Return exact stock on the positive-weight point stencil."""
        if not 0 <= world < self._reservoir_q.shape[0]:
            raise ValueError("invalid world")
        indices, _ = self.point_stencil(position_m)
        values = [int(self._reservoir_q[world, *index].item()) for index in indices]
        if any(value < 0 or value >= INT64_SAFE_MAX for value in values):
            raise ValueError("local reservoir stock is outside the [0,2^62) domain")
        total = sum(values)
        if total >= INT64_SAFE_MAX:
            raise ValueError("local reservoir stock exceeds the safe reduction domain")
        return total

    def value_at(self, world: int, position_m: torch.Tensor) -> float:
        """Sample one selected position without evaluating inactive capacity slots."""
        if not 0 <= world < self._reservoir_q.shape[0]:
            raise ValueError("invalid world")
        indices, weights = self.point_stencil(position_m)
        values = torch.stack(
            [self._reservoir_q[world, *index] for index in indices]
        )
        if bool(torch.any((values < 0) | (values >= INT64_SAFE_MAX))):
            raise ValueError("local reservoir stock is outside the [0,2^62) domain")
        concentration = (
            (values.to(torch.float64) * weights).sum()
            * self.q_mass_mol
            / self.geometry.cell_volume_m3
        )
        return float(concentration.item())

    def deplete_at(self, world: int, position_m: torch.Tensor, requested_q: int) -> int:
        """Synthetic single-point transaction; not a biological grazing API."""
        if not 0 <= world < self._reservoir_q.shape[0] or requested_q < 0:
            raise ValueError("invalid world or request")
        indices, weight_tensor = self.point_stencil(position_m)
        available = torch.stack([self._reservoir_q[world, *index] for index in indices])
        if bool(torch.any((available < 0) | (available >= INT64_SAFE_MAX))):
            raise ValueError("local reservoir stock is outside the [0,2^62) domain")
        available_total = sum(int(value) for value in available.tolist())
        if available_total >= INT64_SAFE_MAX:
            raise ValueError("local reservoir stock exceeds the safe reduction domain")
        target = min(requested_q, available_total)
        remaining = target
        debit = torch.zeros_like(available)
        active = available > 0
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
        if sum(int(value) for value in debit.tolist()) != target:
            raise ValueError("depletion amount cannot be exactly apportioned in float64")
        for index, amount in zip(indices, debit, strict=True):
            self._reservoir_q[world, *index] -= amount
        return int(debit.sum().item())

    def deposit_at(self, world: int, position_m: torch.Tensor, amount_q: int) -> int:
        """Credit an exact integer amount across the local trilinear stencil."""
        indices, credit = self.deposit_plan(world, position_m, amount_q)
        for index, value in zip(indices, credit, strict=True):
            self._reservoir_q[world, *index] += value
        return int(credit.sum().item())

    def require_deposit_capacity(
        self, world: int, position_m: torch.Tensor, amount_q: int
    ) -> None:
        """Validate a future local credit without changing the reservoir."""
        self.deposit_plan(world, position_m, amount_q)

    def deposit_plan(
        self, world: int, position_m: torch.Tensor, amount_q: int
    ) -> tuple[list[tuple[int, int, int]], torch.Tensor]:
        """Return a validated exact local credit without applying it."""
        if not 0 <= world < self._reservoir_q.shape[0] or amount_q < 0:
            raise ValueError("invalid world or amount")
        if amount_q >= INT64_SAFE_MAX:
            raise ValueError("amount must remain below 2^62")
        indices, weights = self.point_stencil(position_m)
        credit = apportion_integer(
            torch.tensor(amount_q, dtype=torch.int64, device=self._reservoir_q.device),
            weights,
        )
        if sum(int(value) for value in credit.tolist()) != amount_q:
            raise ValueError("deposit amount cannot be exactly apportioned in float64")
        current = torch.stack([self._reservoir_q[world, *index] for index in indices])
        if bool(torch.any((current < 0) | (current >= INT64_SAFE_MAX))):
            raise ValueError("local reservoir stock is outside the [0,2^62) domain")
        if bool(torch.any(current > INT64_SAFE_MAX - 1 - credit)):
            raise ValueError("deposit would exceed the reservoir domain")
        return indices, credit
