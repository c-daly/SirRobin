"""Mass-derived reserve maintenance and exact starvation return."""

from __future__ import annotations

import math
from dataclasses import dataclass
from fractions import Fraction
from typing import TYPE_CHECKING

from sirrobin.fields.grid import ScalarGrid
from sirrobin.numerics.flux import INT64_SAFE_MAX
from sirrobin.physics.morphology import query_morphology

if TYPE_CHECKING:
    from sirrobin.core.world import HeadlessWorld
    from sirrobin.physics.contracts import LiveStepLedger


@dataclass(frozen=True, slots=True)
class MaintenanceConfig:
    """Declared baseline chemical power required per developed body mass."""

    maintenance_w_per_kg: float

    def __post_init__(self) -> None:
        value = self.maintenance_w_per_kg
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise TypeError("maintenance_w_per_kg must be a real number")
        if not math.isfinite(value) or value < 0.0:
            raise ValueError("maintenance_w_per_kg must be finite and nonnegative")


@dataclass(frozen=True, slots=True)
class MaintenanceReport:
    """Causal input and committed consequences for one maintenance settlement."""

    world_index: int
    creature_slot: int
    structural_mass_kg: float
    interval_s: float
    maintenance_w_per_kg: float
    demand_j: float
    carry_before_j: float
    carry_after_j: float
    requested_q: int
    debit_q: int
    reserve_before_q: int
    reserve_after_q: int
    maintenance_return_q: int
    death_return_q: int
    maintenance_heat_j: float
    death_dissipation_j: float
    starved: bool


def _quantize_energy_demand(
    terms_j: tuple[float, ...], energy_per_q: float
) -> tuple[int, float]:
    """Floor represented float terms into exact quanta plus a float carry.

    Integer-ratio arithmetic avoids losing a small prior carry when the new demand
    is large and avoids float division misclassifying requests above ``2**53``.
    """
    if any(not math.isfinite(term) or term < 0.0 for term in terms_j):
        raise ValueError("maintenance demand terms must be finite and nonnegative")
    demand = sum((Fraction(term) for term in terms_j), Fraction())
    quantum = Fraction(energy_per_q)
    requested_q = demand // quantum
    if requested_q >= INT64_SAFE_MAX:
        raise ValueError("maintenance demand exceeds the exact-integer domain")
    carry_j = float(demand - requested_q * quantum)
    if carry_j < 0.0 or carry_j >= energy_per_q:
        raise ValueError("maintenance carry cannot be represented inside one quantum")
    return int(requested_q), carry_j


def maintain_single_creature(
    world: HeadlessWorld,
    config: MaintenanceConfig,
    *,
    last_mechanics_substep: LiveStepLedger | None = None,
) -> MaintenanceReport | None:
    """Pay mass-derived maintenance, or die and return all remaining material.

    Population settlement remains deliberately out of scope until shared-stock
    interaction. A dead world is a stable no-op, which makes the death return
    intrinsically one-time.
    """
    live_locations = world.body.alive.nonzero(as_tuple=False)
    if live_locations.shape[0] > 1:
        raise ValueError("maintenance currently supports at most one live creature")
    if live_locations.shape[0] == 0:
        return None
    world_index, creature_slot = (int(value) for value in live_locations[0].tolist())
    energy_per_q = world.material_energy_config.reserve_j_per_q
    morphology = query_morphology(world.body, world.live_config)
    mass_kg = float(
        morphology.structural_mass_kg[world_index, creature_slot].item()
    )
    carry_before_j = float(
        world.creature_material.maintenance_carry_j[
            world_index, creature_slot
        ].item()
    )
    base_demand_j = (
        config.maintenance_w_per_kg
        * mass_kg
        * world.economy_config.dt_eco_s
    )
    requested_q, carry_after_j = _quantize_energy_demand(
        (base_demand_j, carry_before_j), energy_per_q
    )
    demand_j = math.fsum((base_demand_j, carry_before_j))

    reserve_before_q = int(
        world.creature_material.reserve_q[world_index, creature_slot].item()
    )
    debit_q = min(requested_q, reserve_before_q)
    reserve_after_q = reserve_before_q - debit_q
    starved = debit_q < requested_q
    structure_q = int(
        world.creature_material.structure_q[world_index, creature_slot].item()
    )
    death_return_q = structure_q + reserve_after_q if starved else 0
    total_return_q = debit_q + death_return_q
    death_dissipation_j = 0.0
    if starved:
        velocity = world.live_state.velocity_rel_water_enu_m_s[
            world_index, creature_slot
        ]
        vertical_velocity = float(velocity[2].item())
        if not math.isfinite(vertical_velocity) or vertical_velocity != 0.0:
            raise ValueError("death settlement does not support vertical velocity")
        yaw_momentum = float(
            world.live_state.yaw_momentum_kg_m2_s[
                world_index, creature_slot
            ].item()
        )
        if last_mechanics_substep is None:
            if bool((velocity != 0.0).any()) or yaw_momentum != 0.0:
                raise ValueError(
                    "starvation with motion requires the last mechanics ledger"
                )
            kinetic_j = 0.0
        else:
            flat_index = world_index * world.body.capacity + creature_slot
            matrix = last_mechanics_substep.effective_mass_after_kg[flat_index]
            velocity_xy = velocity[:2]
            linear_j = float(
                (0.5 * velocity_xy @ matrix[:2, :2] @ velocity_xy).item()
            )
            yaw_inertia = float(
                last_mechanics_substep.hydrodynamics.yaw_inertia_after_kg_m2[
                    flat_index
                ].item()
            )
            if not math.isfinite(yaw_inertia) or yaw_inertia <= 0.0:
                raise ValueError("death settlement requires positive finite yaw inertia")
            rotational_j = yaw_momentum * yaw_momentum / (2.0 * yaw_inertia)
            kinetic_j = linear_j + rotational_j
        carry_energy_j = float(
            world.creature_material.assimilation_carry_q[
                world_index, creature_slot
            ].item()
        ) * energy_per_q
        death_dissipation_j = kinetic_j + carry_energy_j
        if not math.isfinite(death_dissipation_j) or death_dissipation_j < 0.0:
            raise ValueError("death dissipation must be finite and nonnegative")

    position = world.live_state.position_enu_m[world_index, creature_slot]
    dissolved = ScalarGrid(
        world.economy_state.nd_q,
        world.geometry,
        q_mass_mol=world.economy_config.q_mass_mol,
    )
    dissolved.require_deposit_capacity(world_index, position, total_return_q)
    dissolved.deposit_at(world_index, position, total_return_q)
    world.creature_material.reserve_q[world_index, creature_slot] = reserve_after_q
    world.creature_material.maintenance_carry_j[
        world_index, creature_slot
    ] = 0.0 if starved else carry_after_j

    if starved:
        world.creature_material.structure_q[world_index, creature_slot] = 0
        world.creature_material.reserve_q[world_index, creature_slot] = 0
        world.creature_material.intake_carry_mol[world_index, creature_slot] = 0.0
        world.creature_material.assimilation_carry_q[world_index, creature_slot] = 0.0
        world.genotype.alive[world_index, creature_slot] = False
        world.body.alive[world_index, creature_slot] = False
        world.live_state.velocity_rel_water_enu_m_s[
            world_index, creature_slot
        ] = 0.0
        world.live_state.yaw_momentum_kg_m2_s[world_index, creature_slot] = 0.0

    return MaintenanceReport(
        world_index=world_index,
        creature_slot=creature_slot,
        structural_mass_kg=mass_kg,
        interval_s=world.economy_config.dt_eco_s,
        maintenance_w_per_kg=config.maintenance_w_per_kg,
        demand_j=demand_j,
        carry_before_j=carry_before_j,
        carry_after_j=0.0 if starved else carry_after_j,
        requested_q=requested_q,
        debit_q=debit_q,
        reserve_before_q=reserve_before_q,
        reserve_after_q=0 if starved else reserve_after_q,
        maintenance_return_q=debit_q,
        death_return_q=death_return_q,
        maintenance_heat_j=debit_q * energy_per_q,
        death_dissipation_j=death_dissipation_j,
        starved=starved,
    )
