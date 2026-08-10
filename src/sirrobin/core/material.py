"""Authoritative creature nutrient stores and whole-world exact accounting."""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch

from sirrobin.economy.state import EconomyState
from sirrobin.numerics.flux import INT64_SAFE_MAX


@dataclass(frozen=True, slots=True)
class MaterialEnergyConfig:
    """World-owned chemical-energy densities for persistent material stocks."""

    producer_j_per_q: float
    reserve_j_per_q: float

    def __post_init__(self) -> None:
        values = (self.producer_j_per_q, self.reserve_j_per_q)
        if any(not isinstance(value, (int, float)) or isinstance(value, bool) for value in values):
            raise TypeError("material energy densities must be real numbers")
        if any(not math.isfinite(value) or value <= 0.0 for value in values):
            raise ValueError("material energy densities must be finite and positive")


@dataclass(slots=True)
class CreatureMaterialState:
    """Tracked limiting nutrient held by fixed-capacity creature slots."""

    structure_q: torch.Tensor
    reserve_q: torch.Tensor
    intake_carry_mol: torch.Tensor
    assimilation_carry_q: torch.Tensor
    maintenance_carry_j: torch.Tensor

    @classmethod
    def zeros_like(cls, alive: torch.Tensor) -> CreatureMaterialState:
        return cls(
            torch.zeros_like(alive, dtype=torch.int64),
            torch.zeros_like(alive, dtype=torch.int64),
            torch.zeros_like(alive, dtype=torch.float64),
            torch.zeros_like(alive, dtype=torch.float64),
            torch.zeros_like(alive, dtype=torch.float64),
        )

    @classmethod
    def uniform_live(
        cls,
        alive: torch.Tensor,
        *,
        structure_q_per_creature: int,
        reserve_q_per_creature: int,
    ) -> CreatureMaterialState:
        values = (structure_q_per_creature, reserve_q_per_creature)
        if any(isinstance(value, bool) or not isinstance(value, int) for value in values):
            raise TypeError("creature material starting values must be integers")
        if any(value < 0 or value >= INT64_SAFE_MAX for value in values):
            raise ValueError("creature material starting values must be in [0,2^62)")
        return cls(
            torch.where(
                alive,
                torch.full_like(alive, structure_q_per_creature, dtype=torch.int64),
                torch.zeros_like(alive, dtype=torch.int64),
            ),
            torch.where(
                alive,
                torch.full_like(alive, reserve_q_per_creature, dtype=torch.int64),
                torch.zeros_like(alive, dtype=torch.int64),
            ),
            torch.zeros_like(alive, dtype=torch.float64),
            torch.zeros_like(alive, dtype=torch.float64),
            torch.zeros_like(alive, dtype=torch.float64),
        )

    @property
    def reservoirs(self) -> tuple[torch.Tensor, torch.Tensor]:
        return self.structure_q, self.reserve_q

    @property
    def carries(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return (
            self.intake_carry_mol,
            self.assimilation_carry_q,
            self.maintenance_carry_j,
        )

    def validate(
        self,
        alive: torch.Tensor,
        *,
        q_mass_mol: float,
        reserve_j_per_q: float,
    ) -> None:
        expected = tuple(alive.shape)
        for name, reservoir in zip(
            ("structure_q", "reserve_q"), self.reservoirs, strict=True
        ):
            if reservoir.dtype != torch.int64:
                raise TypeError(f"{name} must be int64")
            if tuple(reservoir.shape) != expected:
                raise ValueError(f"{name} must have shape {expected}")
            if reservoir.device != alive.device:
                raise ValueError(f"{name} must be on the creature-state device")
            if torch.any(reservoir < 0):
                raise ValueError(f"{name} must be nonnegative")
            if torch.any(reservoir >= INT64_SAFE_MAX):
                raise ValueError(f"{name} must remain below 2^62")
            if torch.any((~alive) & (reservoir != 0)):
                raise ValueError(f"{name} cannot assign material to an inactive slot")
        for name, carry, upper in zip(
            ("intake_carry_mol", "assimilation_carry_q", "maintenance_carry_j"),
            self.carries,
            (q_mass_mol, 1.0, reserve_j_per_q),
            strict=True,
        ):
            if carry.dtype != torch.float64:
                raise TypeError(f"{name} must be float64")
            if tuple(carry.shape) != expected:
                raise ValueError(f"{name} must have shape {expected}")
            if carry.device != alive.device:
                raise ValueError(f"{name} must be on the creature-state device")
            if torch.any(~torch.isfinite(carry)) or torch.any((carry < 0) | (carry >= upper)):
                raise ValueError(f"{name} must remain in [0,{upper})")
            if torch.any((~alive) & (carry != 0)):
                raise ValueError(f"{name} cannot assign carry to an inactive slot")

    def total_per_world(self) -> torch.Tensor:
        return self.structure_q.sum(dim=1, dtype=torch.int64) + self.reserve_q.sum(
            dim=1, dtype=torch.int64
        )

@dataclass(frozen=True, slots=True)
class MatterTotals:
    field_q: torch.Tensor
    structure_q: torch.Tensor
    reserve_q: torch.Tensor
    total_q: torch.Tensor
    raw_reservoirs_valid: torch.Tensor


@dataclass(frozen=True, slots=True)
class WholeWorldMatterLedger:
    """Exact before/after census of every currently tracked nutrient reservoir."""

    field_before_q: torch.Tensor
    structure_before_q: torch.Tensor
    reserve_before_q: torch.Tensor
    total_before_q: torch.Tensor
    field_after_q: torch.Tensor
    structure_after_q: torch.Tensor
    reserve_after_q: torch.Tensor
    total_after_q: torch.Tensor
    expected_total_q: torch.Tensor
    books_closed: torch.Tensor


def matter_totals(
    economy: EconomyState,
    creatures: CreatureMaterialState,
    *,
    alive: torch.Tensor,
    field_shape: tuple[int, int, int, int],
    max_inventory_q: int,
    q_mass_mol: float,
    reserve_j_per_q: float,
) -> MatterTotals:
    worlds = int(alive.shape[0])
    device = alive.device
    valid = torch.ones(worlds, dtype=torch.bool, device=device)
    approximate_total = torch.zeros(
        worlds, dtype=torch.float64, device=device
    )
    field = torch.zeros(worlds, dtype=torch.int64, device=device)
    for reservoir in economy.reservoirs:
        schema_valid = (
            isinstance(reservoir, torch.Tensor)
            and reservoir.dtype == torch.int64
            and tuple(reservoir.shape) == field_shape
            and reservoir.device == device
            and field_shape[0] == worlds
        )
        if not schema_valid:
            valid.fill_(False)
            continue
        valid &= (reservoir >= 0).all(dim=(1, 2, 3))
        valid &= (reservoir < INT64_SAFE_MAX).all(dim=(1, 2, 3))
        approximate_total += reservoir.to(torch.float64).sum(dim=(1, 2, 3))
        field += reservoir.sum(dim=(1, 2, 3), dtype=torch.int64)
    creature_totals = []
    for reservoir in creatures.reservoirs:
        schema_valid = (
            isinstance(reservoir, torch.Tensor)
            and reservoir.dtype == torch.int64
            and tuple(reservoir.shape) == tuple(alive.shape)
            and reservoir.device == device
        )
        if not schema_valid:
            valid.fill_(False)
            creature_totals.append(torch.zeros(worlds, dtype=torch.int64, device=device))
            continue
        valid &= (reservoir >= 0).all(dim=1)
        valid &= (reservoir < INT64_SAFE_MAX).all(dim=1)
        valid &= ((reservoir == 0) | alive).all(dim=1)
        approximate_total += reservoir.to(torch.float64).sum(dim=1)
        creature_totals.append(reservoir.sum(dim=1, dtype=torch.int64))
    for carry, upper in zip(
        creatures.carries,
        (q_mass_mol, 1.0, reserve_j_per_q),
        strict=True,
    ):
        schema_valid = (
            isinstance(carry, torch.Tensor)
            and carry.dtype == torch.float64
            and tuple(carry.shape) == tuple(alive.shape)
            and carry.device == device
        )
        if not schema_valid:
            valid.fill_(False)
            continue
        valid &= torch.isfinite(carry).all(dim=1)
        valid &= ((carry >= 0) & (carry < upper)).all(dim=1)
        valid &= ((carry == 0) | alive).all(dim=1)
    valid &= approximate_total < max_inventory_q

    # The exact reductions are trusted only when the raw census above proves their
    # total is below the configured (<2^62) safe bound. Invalid totals may wrap here,
    # but raw_reservoirs_valid remains false and closure cannot accept them.
    structure, reserve = creature_totals
    return MatterTotals(
        field,
        structure,
        reserve,
        field + structure + reserve,
        valid,
    )


def close_world_matter(
    *,
    expected_total_q: torch.Tensor,
    before: MatterTotals,
    after: MatterTotals,
) -> WholeWorldMatterLedger:
    books_closed = before.raw_reservoirs_valid & after.raw_reservoirs_valid
    books_closed &= before.total_q == expected_total_q
    books_closed &= after.total_q == expected_total_q
    return WholeWorldMatterLedger(
        before.field_q,
        before.structure_q,
        before.reserve_q,
        before.total_q,
        after.field_q,
        after.structure_q,
        after.reserve_q,
        after.total_q,
        expected_total_q.clone(),
        books_closed,
    )
