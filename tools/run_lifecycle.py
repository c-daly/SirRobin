#!/usr/bin/env python3
"""Run the first paired feed/birth and starve/recycle lifecycle experiment."""

from __future__ import annotations

import argparse
import math
from dataclasses import dataclass, replace

import torch

from sirrobin.core.feeding import FeedingConfig
from sirrobin.core.metabolism import MaintenanceConfig
from sirrobin.core.periodic_motion import DEFAULT_PERIODIC_MOTION_POLICY
from sirrobin.core.reproduction import BirthConfig, BirthReport, attempt_exact_clone_birth
from sirrobin.core.runner import HeadlessRunner
from sirrobin.economy.config import EconomyConfig
from tools.run_world import (
    FIXTURE_BIRTH_CONFIG,
    FIXTURE_FEEDING_CONFIG,
    FIXTURE_MAINTENANCE_CONFIG,
    FIXTURE_RESERVE_Q_PER_BODY,
    _build_fixture_world,
)

STARVATION_MAINTENANCE_CONFIG = MaintenanceConfig(10.0)


@dataclass(frozen=True, slots=True)
class ViableLifecycleReport:
    steps: int
    feeding_events: int
    feeding_producer_debit_q: int
    feeding_reserve_credit_q: int
    feeding_dissolved_return_q: int
    feeding_producer_input_j: float
    feeding_reserve_credit_j: float
    feeding_assimilation_heat_j: float
    final_assimilation_carry_j: float
    maintenance_reserve_debit_q: int
    maintenance_return_q: int
    maintenance_heat_j: float
    birth: BirthReport
    final_population: int
    initial_whole_world_q: int
    final_whole_world_q: int
    books_closed: bool


@dataclass(frozen=True, slots=True)
class StarvedLifecycleReport:
    steps_to_death: int
    starved: bool
    initial_structure_q: int
    initial_reserve_q: int
    maintenance_return_q: int
    death_return_q: int
    maintenance_heat_j: float
    death_dissipation_j: float
    predeath_producer_recycling_q: int
    post_death_recycling_steps: int
    post_death_producer_recycling_q: int
    final_population: int
    initial_whole_world_q: int
    final_whole_world_q: int
    books_closed: bool


@dataclass(frozen=True, slots=True)
class LifecycleScenarioReport:
    """Two controlled arms required before population contention exists."""

    reserve_j_per_q: float
    viable: ViableLifecycleReport
    starved: StarvedLifecycleReport


def _positive_step_bound(name: str, value: int) -> None:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an integer")
    if value < 1:
        raise ValueError(f"{name} must be positive")


def run_first_lifecycle_scenario(
    *,
    device_name: str,
    feeding_config: FeedingConfig = FIXTURE_FEEDING_CONFIG,
    viable_maintenance_config: MaintenanceConfig = FIXTURE_MAINTENANCE_CONFIG,
    starvation_maintenance_config: MaintenanceConfig = STARVATION_MAINTENANCE_CONFIG,
    birth_config: BirthConfig = FIXTURE_BIRTH_CONFIG,
    max_viable_steps: int = 100,
    max_recycling_steps: int = 20,
) -> LifecycleScenarioReport:
    """Run controlled viable and starved arms without shared-stock settlement.

    This is not an unscripted population claim. The arms isolate the immediate
    lifecycle consumers until Slice 3.1 defines within-world stock contention.
    """
    _positive_step_bound("max_viable_steps", max_viable_steps)
    _positive_step_bound("max_recycling_steps", max_recycling_steps)
    try:
        device = torch.device(device_name)
    except RuntimeError as error:
        raise ValueError(f"invalid device {device_name!r}") from error
    if device.type == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA was requested but is not available")

    viable_world = _build_fixture_world(
        bodies=2,
        live_bodies=1,
        reserve_q_per_creature=FIXTURE_RESERVE_Q_PER_BODY,
        device=device,
        economy_interval_s=0.1,
    )
    viable_initial = viable_world.matter_totals()
    viable_runner = HeadlessRunner(
        viable_world,
        periodic_policy=DEFAULT_PERIODIC_MOTION_POLICY,
        feeding_config=feeding_config,
        maintenance_config=viable_maintenance_config,
    )
    feeding_events = 0
    producer_debit_q = 0
    reserve_credit_q = 0
    dissolved_return_q = 0
    producer_input_j = 0.0
    reserve_credit_j = 0.0
    assimilation_heat_j = 0.0
    maintenance_debit_q = 0
    maintenance_return_q = 0
    maintenance_heat_j = 0.0
    viable_books_closed = True
    birth_report = None
    viable_steps = 0
    for step in range(1, max_viable_steps + 1):
        viable_steps = step
        tick = viable_runner.advance()
        viable_books_closed &= bool(tick.matter.books_closed.all())
        if tick.feeding is None or len(tick.maintenance) != 1:
            raise RuntimeError("viable lifecycle arm omitted feeding or maintenance")
        maintenance = tick.maintenance[0]
        feeding_events += 1
        producer_debit_q += tick.feeding.actual_debit_q
        reserve_credit_q += tick.feeding.reserve_credit_q
        dissolved_return_q += tick.feeding.dissolved_return_q
        producer_input_j = math.fsum(
            (producer_input_j, tick.feeding.producer_chemical_input_j)
        )
        reserve_credit_j = math.fsum(
            (reserve_credit_j, tick.feeding.reserve_chemical_credit_j)
        )
        assimilation_heat_j = math.fsum(
            (assimilation_heat_j, tick.feeding.assimilation_heat_j)
        )
        maintenance_debit_q += maintenance.debit_q
        maintenance_return_q += maintenance.maintenance_return_q
        maintenance_heat_j = math.fsum(
            (maintenance_heat_j, maintenance.maintenance_heat_j)
        )
        required_q = int(viable_world.creature_material.structure_q[0, 0])
        required_q += birth_config.initial_reserve_q
        if int(viable_world.creature_material.reserve_q[0, 0]) < required_q:
            continue
        before_birth = viable_world.matter_totals()
        birth_report = attempt_exact_clone_birth(
            viable_world, birth_config, world_index=0, parent_slot=0
        )
        viable_books_closed &= bool(
            viable_world.close_matter_step(before_birth).books_closed.all()
        )
        break
    if birth_report is None or not birth_report.born:
        raise RuntimeError(
            f"viable lifecycle arm did not fund a paid birth within {max_viable_steps} steps"
        )
    viable_final = viable_world.matter_totals()
    viable_report = ViableLifecycleReport(
        steps=viable_steps,
        feeding_events=feeding_events,
        feeding_producer_debit_q=producer_debit_q,
        feeding_reserve_credit_q=reserve_credit_q,
        feeding_dissolved_return_q=dissolved_return_q,
        feeding_producer_input_j=producer_input_j,
        feeding_reserve_credit_j=reserve_credit_j,
        feeding_assimilation_heat_j=assimilation_heat_j,
        final_assimilation_carry_j=(
            float(viable_world.creature_material.assimilation_carry_q.sum().item())
            * viable_world.material_energy_config.reserve_j_per_q
        ),
        maintenance_reserve_debit_q=maintenance_debit_q,
        maintenance_return_q=maintenance_return_q,
        maintenance_heat_j=maintenance_heat_j,
        birth=birth_report,
        final_population=int(viable_world.body.alive.sum().item()),
        initial_whole_world_q=int(viable_initial.total_q.sum().item()),
        final_whole_world_q=int(viable_final.total_q.sum().item()),
        books_closed=viable_books_closed,
    )

    base_economy = EconomyConfig()
    recycling_economy = replace(
        base_economy,
        gx=1,
        gy=1,
        gz=4,
        lx_m=10.0,
        ly_m=10.0,
        lz_m=20.0,
        dt_eco_s=0.1,
        mu_max_s=1.0,
        producer_maintenance_s=0.0,
        producer_mortality_s=0.0,
        density_mortality_m3_mol_s=0.0,
        microbial_turnover_s=0.0,
        sinking_speed_m_s=0.0,
        kz_nd_m2_s=0.0,
        kz_bp_m2_s=0.0,
        kz_bm_m2_s=0.0,
        remin_floor_s=1.0e-4,
    )
    starved_world = _build_fixture_world(
        bodies=1,
        live_bodies=1,
        reserve_q_per_creature=3,
        device=device,
        economy_interval_s=0.1,
        economy_config=recycling_economy,
    )
    # A causal recycling fixture: no dissolved nutrient, detritus, or microbes
    # exist before death. All field nutrient starts in producers, whose configured
    # losses are zero. Producer growth therefore cannot occur until death returns
    # nutrient to the dissolved reservoir.
    field_total_q = sum(
        int(reservoir.sum().item()) for reservoir in starved_world.economy_state.reservoirs
    )
    for reservoir in starved_world.economy_state.reservoirs:
        reservoir.zero_()
    producer = starved_world.economy_state.bp_q.flatten()
    producer.fill_(field_total_q // producer.numel())
    producer[: field_total_q % producer.numel()] += 1
    starved_world.economy_state.validate(starved_world.economy_config)
    starved_initial = starved_world.matter_totals()
    initial_structure_q = int(starved_world.creature_material.structure_q.sum().item())
    initial_reserve_q = int(starved_world.creature_material.reserve_q.sum().item())
    starved_runner = HeadlessRunner(
        starved_world,
        periodic_policy=DEFAULT_PERIODIC_MOTION_POLICY,
        maintenance_config=starvation_maintenance_config,
    )
    death_report = None
    starved_books_closed = True
    steps_to_death = 0
    predeath_production_q = 0
    for step in range(1, max_viable_steps + 1):
        steps_to_death = step
        tick = starved_runner.advance()
        starved_books_closed &= bool(tick.matter.books_closed.all())
        predeath_production_q += int(tick.economy.production_q.sum().item())
        if tick.maintenance and tick.maintenance[0].starved:
            death_report = tick.maintenance[0]
            break
    if death_report is None:
        raise RuntimeError(
            f"starved lifecycle arm did not die within {max_viable_steps} steps"
        )

    recycled_production_q = 0
    recycling_steps = 0
    for step in range(1, max_recycling_steps + 1):
        recycling_steps = step
        tick = starved_runner.advance()
        starved_books_closed &= bool(tick.matter.books_closed.all())
        recycled_production_q += int(tick.economy.production_q.sum().item())
        if recycled_production_q > 0:
            break
    if recycled_production_q <= 0:
        raise RuntimeError(
            "starved lifecycle field did not resume producer recycling within "
            f"{max_recycling_steps} steps"
        )
    starved_final = starved_world.matter_totals()
    starved_report = StarvedLifecycleReport(
        steps_to_death=steps_to_death,
        starved=death_report.starved,
        initial_structure_q=initial_structure_q,
        initial_reserve_q=initial_reserve_q,
        maintenance_return_q=death_report.maintenance_return_q,
        death_return_q=death_report.death_return_q,
        maintenance_heat_j=death_report.maintenance_heat_j,
        death_dissipation_j=death_report.death_dissipation_j,
        predeath_producer_recycling_q=predeath_production_q,
        post_death_recycling_steps=recycling_steps,
        post_death_producer_recycling_q=recycled_production_q,
        final_population=int(starved_world.body.alive.sum().item()),
        initial_whole_world_q=int(starved_initial.total_q.sum().item()),
        final_whole_world_q=int(starved_final.total_q.sum().item()),
        books_closed=starved_books_closed,
    )
    return LifecycleScenarioReport(
        reserve_j_per_q=viable_world.material_energy_config.reserve_j_per_q,
        viable=viable_report,
        starved=starved_report,
    )


def format_report(report: LifecycleScenarioReport) -> str:
    viable = report.viable
    starved = report.starved
    return "\n".join(
        (
            "SirRobin first lifecycle scenario (operational output; not a stable schema)",
            "paired controlled arms: yes",
            f"reserve chemical energy density J/q: {report.reserve_j_per_q:.9g}",
            f"viable steps to paid birth: {viable.steps}",
            f"viable feeding events: {viable.feeding_events}",
            f"viable producer debit q: {viable.feeding_producer_debit_q}",
            f"viable reserve credit q: {viable.feeding_reserve_credit_q}",
            f"viable producer chemical input J: {viable.feeding_producer_input_j:.9g}",
            f"viable reserve chemical credit J: {viable.feeding_reserve_credit_j:.9g}",
            f"viable assimilation heat J: {viable.feeding_assimilation_heat_j:.9g}",
            f"viable assimilation carry J: {viable.final_assimilation_carry_j:.9g}",
            f"viable maintenance debit q: {viable.maintenance_reserve_debit_q}",
            f"viable maintenance heat J: {viable.maintenance_heat_j:.9g}",
            f"viable birth succeeded: {'yes' if viable.birth.born else 'no'}",
            f"viable parent/child IDs: {viable.birth.parent_id} -> {viable.birth.child_id}",
            f"viable population: {viable.final_population}",
            f"viable construction heat J: {viable.birth.construction_heat_j:.9g}",
            f"viable exact books closed: {'yes' if viable.books_closed else 'no'}",
            f"starved steps to death: {starved.steps_to_death}",
            f"starved death occurred: {'yes' if starved.starved else 'no'}",
            f"starved maintenance return q: {starved.maintenance_return_q}",
            f"starved death return q: {starved.death_return_q}",
            f"starved maintenance heat J: {starved.maintenance_heat_j:.9g}",
            f"starved death dissipation J: {starved.death_dissipation_j:.9g}",
            f"starved population: {starved.final_population}",
            "recycling kinetics: accelerated causal fixture",
            "predeath producer recycling q: "
            f"{starved.predeath_producer_recycling_q}",
            f"post-death recycling steps: {starved.post_death_recycling_steps}",
            "post-death producer recycling q: "
            f"{starved.post_death_producer_recycling_q}",
            f"starved exact books closed: {'yes' if starved.books_closed else 'no'}",
        )
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cpu")
    arguments = parser.parse_args(argv)
    try:
        report = run_first_lifecycle_scenario(device_name=arguments.device)
    except (TypeError, ValueError, RuntimeError) as error:
        parser.error(str(error))
    print(format_report(report))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
