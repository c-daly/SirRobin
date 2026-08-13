"""Batched continuous feeding and exact shared-stock allocation."""

from __future__ import annotations

import math
from dataclasses import dataclass, replace

import torch

from sirrobin.fields.geometry import GridGeometry
from sirrobin.fields.stencil import (
    PointStencil,
    deposit_stencil,
    point_stencil,
    sample_stencil_mol_m3,
)
from sirrobin.numerics.flux import (
    INT64_SAFE_MAX,
    apportion_integer,
    commit_flux,
    deterministic_fraction,
)
from sirrobin.organisms.state import PopulationState


@dataclass(frozen=True, slots=True)
class FeedingConfig:
    interval_s: float
    q_mass_mol: float
    capture_efficiency: float
    assimilation_efficiency: float
    producer_j_per_q: float
    reserve_j_per_q: float
    allocation_rounds: int = 8

    def validate(self) -> None:
        positive = (
            self.interval_s,
            self.q_mass_mol,
            self.producer_j_per_q,
            self.reserve_j_per_q,
        )
        if any(not math.isfinite(value) or value <= 0.0 for value in positive):
            raise ValueError("feeding interval, quantum, and energy densities must be positive")
        fractions = (self.capture_efficiency, self.assimilation_efficiency)
        if any(not math.isfinite(value) or not 0.0 <= value <= 1.0 for value in fractions):
            raise ValueError("feeding efficiencies must be finite and in [0,1]")
        if (
            not isinstance(self.allocation_rounds, int)
            or isinstance(self.allocation_rounds, bool)
            or self.allocation_rounds < 1
        ):
            raise ValueError("allocation rounds must be a positive integer")


@dataclass(frozen=True, slots=True)
class FeedingLedger:
    sampled_producer_mol_m3: torch.Tensor
    clearance_volume_m3: torch.Tensor
    requested_q: torch.Tensor
    actual_debit_q: torch.Tensor
    reserve_credit_q: torch.Tensor
    dissolved_return_q: torch.Tensor
    producer_debit_by_cell_q: torch.Tensor
    dissolved_credit_by_cell_q: torch.Tensor
    producer_chemical_input_j: torch.Tensor
    reserve_chemical_credit_j: torch.Tensor
    assimilation_heat_j: torch.Tensor
    allocation_rounds_exhausted: torch.Tensor
    transaction_committed: torch.Tensor
    invalid: torch.Tensor


@dataclass(frozen=True, slots=True)
class FeedingStep:
    population: PopulationState
    producer_q: torch.Tensor
    dissolved_q: torch.Tensor
    ledger: FeedingLedger


def _identity_gather(value: torch.Tensor, order: torch.Tensor) -> torch.Tensor:
    tail = value.shape[2:]
    index = order[(...,) + (None,) * len(tail)].expand(*order.shape, *tail)
    return torch.gather(value, 1, index)


def _apportion_cells(
    claims: torch.Tensor,
    cell_index: torch.Tensor,
    stock_q: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Realize proportional cell claims with stable input-order remainder ties."""

    worlds = claims.shape[0]
    flat_claim = claims.reshape(worlds, -1)
    flat_cell = cell_index.reshape(worlds, -1)
    cell_total = torch.zeros_like(stock_q)
    cell_total.scatter_add_(1, flat_cell, flat_claim)
    target = torch.minimum(cell_total, stock_q)
    claim_total = torch.gather(cell_total, 1, flat_cell)
    claim_target = torch.gather(target, 1, flat_cell)
    uncongested = claim_total <= torch.gather(stock_q, 1, flat_cell)
    safe_total = claim_total.clamp_min(1).to(torch.float64)
    raw = flat_claim.to(torch.float64) * claim_target.to(torch.float64) / safe_total
    base = torch.floor(raw).to(torch.int64)
    base_by_cell = torch.zeros_like(stock_q)
    base_by_cell.scatter_add_(1, flat_cell, base)
    leftover = target - base_by_cell
    remainder = raw - base.to(torch.float64)

    # Two stable sorts produce cell-major, remainder-descending claims while
    # retaining stable-ID/stencil order for exact ties.
    remainder_order = torch.argsort(
        remainder, dim=1, descending=True, stable=True
    )
    cell_after_remainder = torch.gather(flat_cell, 1, remainder_order)
    cell_order = torch.argsort(cell_after_remainder, dim=1, stable=True)
    order = torch.gather(remainder_order, 1, cell_order)
    sorted_cell = torch.gather(flat_cell, 1, order)
    positions = torch.arange(
        flat_claim.shape[1], dtype=torch.int64, device=claims.device
    )[None, :].expand_as(sorted_cell)
    group_start = torch.cat(
        (
            torch.ones((worlds, 1), dtype=torch.bool, device=claims.device),
            sorted_cell[:, 1:] != sorted_cell[:, :-1],
        ),
        dim=1,
    )
    start_position = torch.where(group_start, positions, 0)
    start_position = torch.cummax(start_position, dim=1).values
    rank_in_cell = positions - start_position
    sorted_claim = torch.gather(flat_claim, 1, order)
    sorted_extra = (sorted_claim > 0) & (
        rank_in_cell < torch.gather(leftover, 1, sorted_cell)
    )
    extra = torch.zeros_like(flat_claim).scatter(
        1, order, sorted_extra.to(torch.int64)
    )
    apportioned = base + extra
    realized = torch.where(uncongested, flat_claim, apportioned)
    debit_by_cell = torch.zeros_like(stock_q)
    debit_by_cell.scatter_add_(1, flat_cell, realized)
    return realized.reshape_as(claims), debit_by_cell


def allocate_shared_stock(
    stock_q: torch.Tensor,
    stencil: PointStencil,
    requested_q: torch.Tensor,
    stable_id: torch.Tensor,
    alive: torch.Tensor,
    *,
    rounds: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Allocate exact shared stock, redistributing against unsaturated cells."""

    worlds, capacity = alive.shape
    flat_stock = stock_q.reshape(worlds, -1)
    remaining_stock = flat_stock
    remaining_request = torch.where(alive, requested_q, 0)
    actual = torch.zeros_like(requested_q)
    identity_key = torch.where(
        alive, stable_id, torch.iinfo(torch.int64).max
    )
    identity_order = torch.argsort(identity_key, dim=1, stable=True)
    identity_cell = _identity_gather(stencil.cell_index, identity_order)
    identity_weight = _identity_gather(stencil.weight, identity_order)

    for _ in range(rounds):
        remaining_identity = torch.gather(remaining_request, 1, identity_order)
        available = torch.gather(
            remaining_stock, 1, identity_cell.reshape(worlds, -1)
        ).reshape_as(identity_cell)
        active_weight = torch.where(
            (available > 0) & (identity_weight > 0.0), identity_weight, 0.0
        )
        has_cell = active_weight.sum(dim=-1) > 0.0
        proposal_total = torch.where(has_cell, remaining_identity, 0)
        claims = apportion_integer(proposal_total, active_weight)
        realized, debit_by_cell = _apportion_cells(
            claims, identity_cell, remaining_stock
        )
        realized_identity = realized.sum(dim=-1, dtype=torch.int64)
        realized_by_slot = torch.zeros_like(actual).scatter(
            1, identity_order, realized_identity
        )
        actual = actual + realized_by_slot
        remaining_request = remaining_request - realized_by_slot
        remaining_stock = remaining_stock - debit_by_cell

    still_available = torch.gather(
        remaining_stock, 1, stencil.cell_index.reshape(worlds, -1)
    ).reshape_as(stencil.cell_index)
    rounds_exhausted = alive & (remaining_request > 0) & (
        ((still_available > 0) & (stencil.weight > 0.0)).any(dim=-1)
    )
    debit_by_cell = flat_stock - remaining_stock
    return (
        actual,
        debit_by_cell.reshape_as(stock_q),
        remaining_stock.reshape_as(stock_q),
        rounds_exhausted,
    )


def feed_population(
    population: PopulationState,
    producer_q: torch.Tensor,
    dissolved_q: torch.Tensor,
    positions_enu_m: torch.Tensor,
    velocity_rel_water_enu_m_s: torch.Tensor,
    intake_area_m2: torch.Tensor,
    geometry: GridGeometry,
    config: FeedingConfig,
) -> FeedingStep:
    """Plan and commit one simultaneous feeding transaction on the device."""

    alive = population.alive
    stencil = point_stencil(positions_enu_m, geometry)
    concentration = sample_stencil_mol_m3(
        producer_q, stencil, geometry, q_mass_mol=config.q_mass_mol
    )
    relative_speed = torch.linalg.vector_norm(
        velocity_rel_water_enu_m_s.to(torch.float64), dim=-1
    )
    clearance = (
        intake_area_m2.to(torch.float64)
        * relative_speed
        * config.interval_s
        * config.capture_efficiency
    )
    requested_mol = torch.where(alive, clearance * concentration, 0.0)
    quantized = commit_flux(
        requested_mol,
        population.intake_carry_mol,
        torch.full_like(population.reserve_q, INT64_SAFE_MAX - 1),
        q_mass_mol=config.q_mass_mol,
    )
    requested_q = torch.where(alive, quantized.committed_q, 0)
    actual_q, producer_debit, producer_after, rounds_exhausted = (
        allocate_shared_stock(
            producer_q,
            stencil,
            requested_q,
            population.stable_id,
            alive,
            rounds=config.allocation_rounds,
        )
    )
    effective_fraction = min(
        config.assimilation_efficiency,
        1.0,
        config.producer_j_per_q / config.reserve_j_per_q,
    )
    reserve_credit_q, dissolved_return_q, assimilation_carry = (
        deterministic_fraction(
            actual_q,
            effective_fraction,
            population.assimilation_carry_q,
        )
    )
    dissolved_candidate, dissolved_credit = deposit_stencil(
        dissolved_q, stencil, dissolved_return_q
    )
    reserve_overflow = reserve_credit_q > (
        (INT64_SAFE_MAX - 1) - population.reserve_q
    )
    reserve_candidate = population.reserve_q + torch.where(
        reserve_overflow, 0, reserve_credit_q
    )
    dissolved_overflow = dissolved_credit > (
        (INT64_SAFE_MAX - 1) - dissolved_q
    )
    producer_energy_j = actual_q.to(torch.float64) * config.producer_j_per_q
    reserve_energy_j = reserve_credit_q.to(torch.float64) * config.reserve_j_per_q
    assimilation_heat_j = (
        producer_energy_j
        + population.assimilation_carry_q * config.reserve_j_per_q
        - reserve_energy_j
        - assimilation_carry * config.reserve_j_per_q
    )
    local_invalid = (
        stencil.vertical_out_of_bounds
        | rounds_exhausted
        | (quantized.shortfall_q != 0)
        | reserve_overflow
        | ~torch.isfinite(assimilation_heat_j)
        | (assimilation_heat_j < 0.0)
    )
    world_invalid = local_invalid.any(dim=1) | dissolved_overflow.flatten(1).any(
        dim=1
    )
    committed = ~world_invalid
    slot_commit = committed[:, None]
    cell_commit = committed[:, None, None, None]
    committed_actual_q = torch.where(slot_commit, actual_q, 0)
    committed_reserve_credit_q = torch.where(slot_commit, reserve_credit_q, 0)
    committed_dissolved_return_q = torch.where(
        slot_commit, dissolved_return_q, 0
    )
    committed_producer_debit = torch.where(cell_commit, producer_debit, 0)
    committed_dissolved_credit = torch.where(cell_commit, dissolved_credit, 0)
    producer_after = torch.where(cell_commit, producer_after, producer_q)
    dissolved_after = torch.where(
        cell_commit, dissolved_candidate, dissolved_q
    )
    next_population = replace(
        population,
        reserve_q=torch.where(slot_commit, reserve_candidate, population.reserve_q),
        intake_carry_mol=torch.where(
            slot_commit & alive,
            quantized.carry_mol,
            population.intake_carry_mol,
        ),
        assimilation_carry_q=torch.where(
            slot_commit & alive,
            assimilation_carry,
            population.assimilation_carry_q,
        ),
    )
    ledger = FeedingLedger(
        sampled_producer_mol_m3=concentration,
        clearance_volume_m3=clearance,
        requested_q=requested_q,
        actual_debit_q=committed_actual_q,
        reserve_credit_q=committed_reserve_credit_q,
        dissolved_return_q=committed_dissolved_return_q,
        producer_debit_by_cell_q=committed_producer_debit,
        dissolved_credit_by_cell_q=committed_dissolved_credit,
        producer_chemical_input_j=torch.where(slot_commit, producer_energy_j, 0.0),
        reserve_chemical_credit_j=torch.where(slot_commit, reserve_energy_j, 0.0),
        assimilation_heat_j=torch.where(slot_commit, assimilation_heat_j, 0.0),
        allocation_rounds_exhausted=rounds_exhausted,
        transaction_committed=committed,
        invalid=local_invalid | world_invalid[:, None],
    )
    return FeedingStep(next_population, producer_after, dissolved_after, ledger)
