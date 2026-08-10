"""Local producer feeding with exact shared-stock nutrient routing."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch

from sirrobin.fields.grid import ScalarGrid
from sirrobin.numerics.flux import (
    INT64_SAFE_MAX,
    apportion_integer,
    commit_flux,
    deterministic_fraction,
)
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


@dataclass(frozen=True, slots=True)
class PopulationFeedingReport:
    """One simultaneous transaction over every live feeding request."""

    creatures: tuple[FeedingReport, ...]
    requested_q: int
    actual_debit_q: int
    reserve_credit_q: int
    dissolved_return_q: int
    producer_chemical_input_j: float
    reserve_chemical_credit_j: float
    assimilation_heat_j: float


@dataclass(frozen=True, slots=True)
class _FeedingIntent:
    world_index: int
    creature_slot: int
    position: torch.Tensor
    stencil_indices: tuple[tuple[int, int, int], ...]
    stencil_weights: torch.Tensor
    sampled_producer_mol_m3: float
    intake_area_m2: float
    relative_speed_m_s: float
    clearance_volume_m3: float
    requested_q: int
    intake_carry_before_mol: float
    intake_carry_after_mol: float
    assimilation_carry_before_q: float


def feed_single_creature(world: HeadlessWorld, config: FeedingConfig) -> FeedingReport:
    """Settle through the population transaction while enforcing one live body."""
    if int(world.body.alive.sum().item()) != 1:
        raise ValueError("one-creature feeding requires exactly one live creature")
    return feed_population(world, config).creatures[0]


def _exact_apportion(total_q: int, weights: torch.Tensor, *, context: str) -> list[int]:
    allocation = apportion_integer(
        torch.tensor(total_q, dtype=torch.int64, device=weights.device),
        weights,
    )
    values = [int(value) for value in allocation.tolist()]
    if any(value < 0 for value in values) or sum(values) != total_q:
        raise ValueError(f"{context} cannot be exactly apportioned in float64")
    return values


def _population_intents(
    world: HeadlessWorld,
    config: FeedingConfig,
    producer: ScalarGrid,
) -> list[_FeedingIntent]:
    morphology = query_morphology(world.body, world.live_config)
    locations = [
        (int(world_index), int(creature_slot))
        for world_index, creature_slot in world.body.alive.nonzero(as_tuple=False).tolist()
    ]
    locations.sort(
        key=lambda location: (
            location[0],
            int(world.genotype.stable_id[location].item()),
            location[1],
        )
    )
    intents: list[_FeedingIntent] = []
    for world_index, creature_slot in locations:
        position = world.live_state.position_enu_m[world_index, creature_slot]
        concentration = producer.value_at(world_index, position)
        intake_area = float(
            morphology.intake_area_m2[world_index, creature_slot].item()
        )
        relative_speed = float(
            torch.linalg.vector_norm(
                world.live_state.velocity_rel_water_enu_m_s[
                    world_index, creature_slot
                ]
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
        carry_before = float(
            world.creature_material.intake_carry_mol[
                world_index, creature_slot
            ].item()
        )
        request_value = (
            requested_mol + carry_before
        ) / world.economy_config.q_mass_mol
        if not math.isfinite(clearance) or not math.isfinite(request_value):
            raise ValueError("feeding request must be finite")
        if request_value >= INT64_SAFE_MAX:
            raise ValueError("feeding request exceeds the supported exact-integer domain")
        quantized = commit_flux(
            torch.tensor(
                requested_mol,
                dtype=torch.float64,
                device=position.device,
            ),
            world.creature_material.intake_carry_mol[
                world_index, creature_slot
            ],
            torch.tensor(
                INT64_SAFE_MAX - 1,
                dtype=torch.int64,
                device=position.device,
            ),
            q_mass_mol=world.economy_config.q_mass_mol,
        )
        if int(quantized.shortfall_q.item()) != 0:
            raise RuntimeError("uncapped feeding request unexpectedly reported a shortfall")
        indices, weights = producer.point_stencil(position)
        intents.append(
            _FeedingIntent(
                world_index=world_index,
                creature_slot=creature_slot,
                position=position,
                stencil_indices=tuple(indices),
                stencil_weights=weights,
                sampled_producer_mol_m3=concentration,
                intake_area_m2=intake_area,
                relative_speed_m_s=relative_speed,
                clearance_volume_m3=clearance,
                requested_q=int(quantized.committed_q.item()),
                intake_carry_before_mol=carry_before,
                intake_carry_after_mol=float(quantized.carry_mol.item()),
                assimilation_carry_before_q=float(
                    world.creature_material.assimilation_carry_q[
                        world_index, creature_slot
                    ].item()
                ),
            )
        )
    return intents


def _allocate_shared_stock(
    producer: ScalarGrid,
    intents: list[_FeedingIntent],
) -> tuple[list[int], dict[tuple[int, int, int, int], int]]:
    stock: dict[tuple[int, int, int, int], int] = {}
    for intent in intents:
        for index in intent.stencil_indices:
            key = (intent.world_index, *index)
            if key in stock:
                continue
            value = int(producer.reservoir_q[key].item())
            if value < 0 or value >= INT64_SAFE_MAX:
                raise ValueError(
                    "local reservoir stock is outside the [0,2^62) domain"
                )
            stock[key] = value

    remaining = [intent.requested_q for intent in intents]
    actual = [0 for _ in intents]
    cell_debits: dict[tuple[int, int, int, int], int] = {
        key: 0 for key in stock
    }
    while True:
        proposals: dict[tuple[int, int, int, int], list[tuple[int, int]]] = {}
        for intent_index, intent in enumerate(intents):
            if remaining[intent_index] == 0:
                continue
            active = [
                (local_index, (intent.world_index, *cell_index))
                for local_index, cell_index in enumerate(intent.stencil_indices)
                if stock[(intent.world_index, *cell_index)]
                > cell_debits[(intent.world_index, *cell_index)]
            ]
            if not active:
                continue
            weights = torch.stack(
                [intent.stencil_weights[local_index] for local_index, _ in active]
            )
            allocation = _exact_apportion(
                remaining[intent_index],
                weights,
                context="feeding stencil request",
            )
            for (_, key), amount_q in zip(active, allocation, strict=True):
                if amount_q > 0:
                    proposals.setdefault(key, []).append(
                        (intent_index, amount_q)
                    )
        if not proposals:
            break

        progress_q = 0
        for key in sorted(proposals):
            claims = proposals[key]
            room_q = stock[key] - cell_debits[key]
            proposed_q = sum(amount_q for _, amount_q in claims)
            target_q = min(room_q, proposed_q)
            if target_q == proposed_q:
                realized = [amount_q for _, amount_q in claims]
            else:
                realized = _exact_apportion(
                    target_q,
                    torch.tensor(
                        [amount_q for _, amount_q in claims],
                        dtype=torch.float64,
                        device=producer.reservoir_q.device,
                    ),
                    context="shared producer stock",
                )
            if any(
                amount_q > proposed_q
                for amount_q, (_, proposed_q) in zip(
                    realized, claims, strict=True
                )
            ):
                raise ValueError("shared producer allocation exceeds a creature claim")
            for amount_q, (intent_index, _) in zip(realized, claims, strict=True):
                actual[intent_index] += amount_q
                remaining[intent_index] -= amount_q
                cell_debits[key] += amount_q
                progress_q += amount_q
        if progress_q == 0:
            break

    if any(value < 0 for value in remaining):
        raise RuntimeError("shared feeding allocation exceeded a request")
    if any(cell_debits[key] > stock[key] for key in stock):
        raise RuntimeError("shared feeding allocation exceeded producer stock")
    return actual, {key: value for key, value in cell_debits.items() if value > 0}


def _population_report(creatures: tuple[FeedingReport, ...]) -> PopulationFeedingReport:
    return PopulationFeedingReport(
        creatures=creatures,
        requested_q=sum(creature.requested_q for creature in creatures),
        actual_debit_q=sum(creature.actual_debit_q for creature in creatures),
        reserve_credit_q=sum(creature.reserve_credit_q for creature in creatures),
        dissolved_return_q=sum(
            creature.dissolved_return_q for creature in creatures
        ),
        producer_chemical_input_j=math.fsum(
            creature.producer_chemical_input_j for creature in creatures
        ),
        reserve_chemical_credit_j=math.fsum(
            creature.reserve_chemical_credit_j for creature in creatures
        ),
        assimilation_heat_j=math.fsum(
            creature.assimilation_heat_j for creature in creatures
        ),
    )


def feed_population(
    world: HeadlessWorld,
    config: FeedingConfig,
) -> PopulationFeedingReport:
    """Settle all live local feeding requests against shared stock at once.

    Requests are planned from the same pre-transaction field. Scarce cell stock
    is apportioned by request with stable-ID ordering for exact ties. Unmet demand
    is not carried forward; only the subquantum physical-intake remainder is.
    """
    if not torch.equal(world.body.alive, world.genotype.alive) or not torch.equal(
        world.body.stable_id, world.genotype.stable_id
    ):
        raise ValueError("developed body identity cache differs from genotype authority")
    live_count = int(world.body.alive.sum().item())
    if live_count < 1:
        raise ValueError("population feeding requires at least one live creature")

    energy = world.material_energy_config
    producer = ScalarGrid(
        world.economy_state.bp_q,
        world.geometry,
        q_mass_mol=world.economy_config.q_mass_mol,
    )
    dissolved = ScalarGrid(
        world.economy_state.nd_q,
        world.geometry,
        q_mass_mol=world.economy_config.q_mass_mol,
    )
    intents = _population_intents(world, config, producer)
    actual_by_creature, producer_debits = _allocate_shared_stock(
        producer, intents
    )
    effective_fraction = min(
        config.assimilation_efficiency,
        min(1.0, energy.producer_j_per_q / energy.reserve_j_per_q),
    )

    reports: list[FeedingReport] = []
    assimilation_after: list[float] = []
    dissolved_credits: dict[tuple[int, int, int, int], int] = {}
    for intent, actual_q in zip(intents, actual_by_creature, strict=True):
        reserve_credit, _, carry_after = deterministic_fraction(
            torch.tensor(
                actual_q,
                dtype=torch.int64,
                device=intent.position.device,
            ),
            effective_fraction,
            torch.tensor(
                intent.assimilation_carry_before_q,
                dtype=torch.float64,
                device=intent.position.device,
            ),
        )
        reserve_credit_q = int(reserve_credit.item())
        dissolved_return_q = actual_q - reserve_credit_q
        carry_after_q = float(carry_after.item())
        producer_energy_j = actual_q * energy.producer_j_per_q
        reserve_energy_j = reserve_credit_q * energy.reserve_j_per_q
        assimilation_heat_j = (
            producer_energy_j
            + intent.assimilation_carry_before_q * energy.reserve_j_per_q
            - reserve_energy_j
            - carry_after_q * energy.reserve_j_per_q
        )
        energy_terms = (
            producer_energy_j,
            reserve_energy_j,
            assimilation_heat_j,
        )
        if any(not math.isfinite(value) for value in energy_terms) or (
            assimilation_heat_j < 0.0
        ):
            raise ValueError(
                "feeding energy settlement must be finite and nonnegative"
            )

        reserve_before_q = int(
            world.creature_material.reserve_q[
                intent.world_index, intent.creature_slot
            ].item()
        )
        if (
            reserve_before_q < 0
            or reserve_before_q >= INT64_SAFE_MAX
            or reserve_before_q > INT64_SAFE_MAX - 1 - reserve_credit_q
        ):
            raise ValueError("reserve credit would exceed the creature reservoir domain")
        indices, credits = dissolved.deposit_plan(
            intent.world_index,
            intent.position,
            dissolved_return_q,
        )
        for index, credit in zip(indices, credits.tolist(), strict=True):
            key = (intent.world_index, *index)
            dissolved_credits[key] = dissolved_credits.get(key, 0) + int(credit)

        assimilation_after.append(carry_after_q)
        reports.append(
            FeedingReport(
                world_index=intent.world_index,
                creature_slot=intent.creature_slot,
                sampled_producer_mol_m3=intent.sampled_producer_mol_m3,
                intake_area_m2=intent.intake_area_m2,
                relative_speed_m_s=intent.relative_speed_m_s,
                clearance_volume_m3=intent.clearance_volume_m3,
                requested_q=intent.requested_q,
                actual_debit_q=actual_q,
                reserve_credit_q=reserve_credit_q,
                dissolved_return_q=dissolved_return_q,
                intake_carry_before_mol=intent.intake_carry_before_mol,
                intake_carry_after_mol=intent.intake_carry_after_mol,
                assimilation_carry_before_q=intent.assimilation_carry_before_q,
                assimilation_carry_after_q=carry_after_q,
                capture_efficiency=config.capture_efficiency,
                assimilation_efficiency=config.assimilation_efficiency,
                effective_conversion_fraction=effective_fraction,
                producer_j_per_q=energy.producer_j_per_q,
                reserve_j_per_q=energy.reserve_j_per_q,
                producer_chemical_input_j=producer_energy_j,
                reserve_chemical_credit_j=reserve_energy_j,
                assimilation_heat_j=assimilation_heat_j,
            )
        )

    for key, credit_q in dissolved_credits.items():
        current_q = int(dissolved.reservoir_q[key].item())
        if current_q < 0 or current_q >= INT64_SAFE_MAX:
            raise ValueError(
                "local reservoir stock is outside the [0,2^62) domain"
            )
        if current_q > INT64_SAFE_MAX - 1 - credit_q:
            raise ValueError("deposit would exceed the reservoir domain")

    population = _population_report(tuple(reports))
    if population.actual_debit_q != (
        population.reserve_credit_q + population.dissolved_return_q
    ):
        raise RuntimeError("population feeding credits do not equal the shared debit")

    for key, debit_q in producer_debits.items():
        producer.reservoir_q[key] -= debit_q
    for key, credit_q in dissolved_credits.items():
        dissolved.reservoir_q[key] += credit_q
    for intent, report, carry_after_q in zip(
        intents, reports, assimilation_after, strict=True
    ):
        location = (intent.world_index, intent.creature_slot)
        world.creature_material.reserve_q[location] += report.reserve_credit_q
        world.creature_material.intake_carry_mol[location] = (
            intent.intake_carry_after_mol
        )
        world.creature_material.assimilation_carry_q[location] = carry_after_q

    return population
