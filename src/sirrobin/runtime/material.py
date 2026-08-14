"""Exact cross-domain material census and spatial organism returns."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from sirrobin.economy.state import EconomyState
from sirrobin.fields.geometry import GridGeometry
from sirrobin.fields.stencil import deposit_stencil, point_stencil
from sirrobin.numerics.flux import INT64_SAFE_MAX
from sirrobin.organisms.state import PopulationState


@dataclass(frozen=True, slots=True)
class ReturnDeposit:
    dissolved_q: torch.Tensor
    credit_by_cell_q: torch.Tensor
    transaction_committed: torch.Tensor
    overflow: torch.Tensor


@dataclass(frozen=True, slots=True)
class RuntimeMatterLedger:
    expected_q: torch.Tensor
    before_q: torch.Tensor
    after_q: torch.Tensor
    books_closed: torch.Tensor


def total_matter_q(
    economy: EconomyState,
    population: PopulationState,
) -> torch.Tensor:
    return economy.total_per_world() + population.structure_q.sum(
        dim=1, dtype=torch.int64
    ) + population.reserve_q.sum(dim=1, dtype=torch.int64)


def deposit_organism_returns(
    dissolved_q: torch.Tensor,
    positions_enu_m: torch.Tensor,
    return_q: torch.Tensor,
    geometry: GridGeometry,
) -> ReturnDeposit:
    """Deposit exact returns, rolling back a world on cell-capacity overflow."""

    stencil = point_stencil(positions_enu_m, geometry)
    candidate, credit = deposit_stencil(dissolved_q, stencil, return_q)
    cell_overflow = credit > ((INT64_SAFE_MAX - 1) - dissolved_q)
    world_overflow = cell_overflow.flatten(1).any(dim=1)
    committed = ~world_overflow
    cell_commit = committed[:, None, None, None]
    return ReturnDeposit(
        torch.where(cell_commit, candidate, dissolved_q),
        torch.where(cell_commit, credit, 0),
        committed,
        world_overflow,
    )


def close_runtime_matter(
    expected_q: torch.Tensor,
    before_q: torch.Tensor,
    economy: EconomyState,
    population: PopulationState,
) -> RuntimeMatterLedger:
    after_q = total_matter_q(economy, population)
    return RuntimeMatterLedger(
        expected_q,
        before_q,
        after_q,
        (before_q == expected_q) & (after_q == expected_q),
    )
