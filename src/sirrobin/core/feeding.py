"""One-creature local producer feeding with exact nutrient routing."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch

from sirrobin.fields.grid import ScalarGrid
from sirrobin.numerics.flux import INT64_SAFE_MAX, commit_flux, deterministic_fraction
from sirrobin.physics.morphology import query_morphology

if TYPE_CHECKING:
    from sirrobin.core.world import HeadlessWorld


@dataclass(frozen=True, slots=True)
class FeedingConfig:
    """Declared conversion anchors for the first local feeding transaction."""

    capture_efficiency: float
    assimilation_efficiency: float

    def __post_init__(self) -> None:
        values = (
            self.capture_efficiency,
            self.assimilation_efficiency,
        )
        if any(isinstance(value, bool) or not isinstance(value, (int, float)) for value in values):
            raise TypeError("feeding configuration values must be real numbers")
        if any(not math.isfinite(value) for value in values):
            raise ValueError("feeding configuration values must be finite")
        if not 0.0 <= self.capture_efficiency <= 1.0:
            raise ValueError("capture efficiency must be in [0,1]")
        if not 0.0 <= self.assimilation_efficiency <= 1.0:
            raise ValueError("assimilation efficiency must be in [0,1]")


@dataclass(frozen=True, slots=True)
class FeedingReport:
    """Causal inputs and committed consequences of one feeding opportunity."""

    world_index: int
    creature_slot: int
    sampled_producer_mol_m3: float
    intake_area_m2: float
    relative_speed_m_s: float
    clearance_volume_m3: float
    requested_q: int
    actual_debit_q: int
    reserve_credit_q: int
    dissolved_return_q: int
    intake_carry_before_mol: float
    intake_carry_after_mol: float
    assimilation_carry_before_q: float
    assimilation_carry_after_q: float
    capture_efficiency: float
    assimilation_efficiency: float
    effective_conversion_fraction: float
    producer_j_per_q: float
    reserve_j_per_q: float
    producer_chemical_input_j: float
    reserve_chemical_credit_j: float
    assimilation_heat_j: float


def feed_single_creature(world: HeadlessWorld, config: FeedingConfig) -> FeedingReport:
    """Settle one morphology-derived local feeding request.

    This deliberately rejects population contention. Deterministic shared-stock
    allocation belongs to Slice 3.1; accepting multiple creatures here would make
    iteration order an undeclared ecological advantage.
    """
    live_locations = world.body.alive.nonzero(as_tuple=False)
    if live_locations.shape[0] != 1:
        raise ValueError("one-creature feeding requires exactly one live creature")
    world_index, creature_slot = (int(value) for value in live_locations[0].tolist())
    position = world.live_state.position_enu_m[world_index, creature_slot]
    energy = world.material_energy_config
    if energy is None:
        raise ValueError("feeding requires world-owned material energy densities")

    producer = ScalarGrid(
        world.economy_state.bp_q,
        world.geometry,
        q_mass_mol=world.economy_config.q_mass_mol,
    )
    concentration = producer.value_at(world_index, position)
    morphology = query_morphology(world.body, world.live_config)
    intake_area = float(morphology.intake_area_m2[world_index, creature_slot].item())
    relative_speed = float(
        torch.linalg.vector_norm(
            world.live_state.velocity_rel_water_enu_m_s[world_index, creature_slot]
        ).item()
    )
    causes = (concentration, intake_area, relative_speed)
    if any(not math.isfinite(value) or value < 0.0 for value in causes):
        raise ValueError("feeding causes must be finite and nonnegative")

    clearance = (
        intake_area
        * relative_speed
        * world.economy_config.dt_eco_s
        * config.capture_efficiency
    )
    requested_mol = clearance * concentration
    intake_carry_before_mol = float(
        world.creature_material.intake_carry_mol[world_index, creature_slot].item()
    )
    request_value = (
        requested_mol + intake_carry_before_mol
    ) / world.economy_config.q_mass_mol
    if not math.isfinite(clearance) or not math.isfinite(request_value):
        raise ValueError("feeding request must be finite")
    if request_value >= INT64_SAFE_MAX:
        raise ValueError("feeding request exceeds the supported exact-integer domain")
    available_q = producer.available_at(world_index, position)
    committed = commit_flux(
        torch.tensor(requested_mol, dtype=torch.float64, device=position.device),
        world.creature_material.intake_carry_mol[world_index, creature_slot],
        torch.tensor(available_q, dtype=torch.int64, device=position.device),
        q_mass_mol=world.economy_config.q_mass_mol,
    )
    actual_q = int(committed.committed_q.item())
    requested_q = actual_q + int(committed.shortfall_q.item())
    intake_carry_after_mol = float(committed.carry_mol.item())

    energy_fraction = min(1.0, energy.producer_j_per_q / energy.reserve_j_per_q)
    effective_fraction = min(config.assimilation_efficiency, energy_fraction)
    assimilation_carry_before_q = float(
        world.creature_material.assimilation_carry_q[
            world_index, creature_slot
        ].item()
    )
    reserve_credit, _, assimilation_carry_after = deterministic_fraction(
        torch.tensor(actual_q, dtype=torch.int64, device=position.device),
        effective_fraction,
        torch.tensor(
            assimilation_carry_before_q, dtype=torch.float64, device=position.device
        ),
    )
    reserve_credit_q = int(reserve_credit.item())
    dissolved_return_q = actual_q - reserve_credit_q

    producer_energy_j = actual_q * energy.producer_j_per_q
    reserve_energy_j = reserve_credit_q * energy.reserve_j_per_q
    carry_energy_before_j = assimilation_carry_before_q * energy.reserve_j_per_q
    carry_energy_after_j = float(assimilation_carry_after.item()) * energy.reserve_j_per_q
    assimilation_heat_j = (
        producer_energy_j
        + carry_energy_before_j
        - reserve_energy_j
        - carry_energy_after_j
    )
    energy_terms = (producer_energy_j, reserve_energy_j, assimilation_heat_j)
    if any(not math.isfinite(value) for value in energy_terms) or assimilation_heat_j < 0.0:
        raise ValueError("feeding energy settlement must be finite and nonnegative")

    dissolved = ScalarGrid(
        world.economy_state.nd_q,
        world.geometry,
        q_mass_mol=world.economy_config.q_mass_mol,
    )
    dissolved.require_deposit_capacity(
        world_index, position, dissolved_return_q
    )
    reserve_before_q = int(
        world.creature_material.reserve_q[world_index, creature_slot].item()
    )
    if (
        reserve_before_q < 0
        or reserve_before_q >= INT64_SAFE_MAX
        or reserve_before_q > INT64_SAFE_MAX - 1 - reserve_credit_q
    ):
        raise ValueError("reserve credit would exceed the creature reservoir domain")

    realized_q = producer.deplete_at(world_index, position, requested_q)
    if realized_q != actual_q:
        raise RuntimeError("producer stock changed between feeding plan and commit")
    dissolved.deposit_at(world_index, position, dissolved_return_q)
    world.creature_material.reserve_q[world_index, creature_slot] += reserve_credit_q
    world.creature_material.intake_carry_mol[
        world_index, creature_slot
    ] = committed.carry_mol
    world.creature_material.assimilation_carry_q[
        world_index, creature_slot
    ] = assimilation_carry_after

    return FeedingReport(
        world_index=world_index,
        creature_slot=creature_slot,
        sampled_producer_mol_m3=concentration,
        intake_area_m2=intake_area,
        relative_speed_m_s=relative_speed,
        clearance_volume_m3=clearance,
        requested_q=requested_q,
        actual_debit_q=actual_q,
        reserve_credit_q=reserve_credit_q,
        dissolved_return_q=dissolved_return_q,
        intake_carry_before_mol=intake_carry_before_mol,
        intake_carry_after_mol=intake_carry_after_mol,
        assimilation_carry_before_q=assimilation_carry_before_q,
        assimilation_carry_after_q=float(assimilation_carry_after.item()),
        capture_efficiency=config.capture_efficiency,
        assimilation_efficiency=config.assimilation_efficiency,
        effective_conversion_fraction=effective_fraction,
        producer_j_per_q=energy.producer_j_per_q,
        reserve_j_per_q=energy.reserve_j_per_q,
        producer_chemical_input_j=producer_energy_j,
        reserve_chemical_credit_j=reserve_energy_j,
        assimilation_heat_j=assimilation_heat_j,
    )
