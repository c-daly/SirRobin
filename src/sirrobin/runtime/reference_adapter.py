"""One-way bootstrap adapter from the preserved reference-world state."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from sirrobin.organisms.development import initialize_development_state
from sirrobin.organisms.state import PopulationState
from sirrobin.runtime.material import total_matter_q
from sirrobin.runtime.state import LivingState

if TYPE_CHECKING:
    from sirrobin.core.world import HeadlessWorld


def living_state_from_reference(world: HeadlessWorld) -> LivingState:
    """Copy initial authoritative stores into the cohesive device-state contract.

    This adapter is only a migration/bootstrap boundary. Device domain kernels do
    not import or reach through the reference world after construction.
    """

    alive = world.genotype.alive
    zeros_i64 = torch.zeros_like(alive, dtype=torch.int64)
    zeros_f64 = torch.zeros_like(alive, dtype=torch.float64)
    population = PopulationState(
        alive=alive,
        stable_id=world.genotype.stable_id,
        parent_id=zeros_i64,
        generation=zeros_i64,
        born_at_s=zeros_f64,
        structure_q=world.creature_material.structure_q,
        reserve_q=world.creature_material.reserve_q,
        intake_carry_mol=world.creature_material.intake_carry_mol,
        assimilation_carry_q=world.creature_material.assimilation_carry_q,
        maintenance_carry_j=world.creature_material.maintenance_carry_j,
        next_stable_id=world.next_stable_id,
    )
    return LivingState(
        population,
        world.genotype,
        world.body,
        initialize_development_state(population, world.body),
        world.live_state,
        world.economy_state,
        total_matter_q(world.economy_state, population),
    )
