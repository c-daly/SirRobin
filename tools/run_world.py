"""Run the existing composed SirRobin world without Unity.

This deliberately emits human-readable operational output, not a persistence or
observability schema.  Its purpose is to make the baseline world runnable and its
current multi-rate cost visible before new biology is added.
"""

from __future__ import annotations

import argparse
import json
import math
import time
from dataclasses import dataclass, replace
from pathlib import Path

import torch

from sirrobin.core.feeding import FeedingConfig
from sirrobin.core.material import CreatureMaterialState, MaterialEnergyConfig
from sirrobin.core.metabolism import MaintenanceConfig
from sirrobin.core.periodic_motion import DEFAULT_PERIODIC_MOTION_POLICY
from sirrobin.core.reproduction import BirthConfig, attempt_exact_clone_birth
from sirrobin.core.runner import HeadlessRunner, WorldSchedule
from sirrobin.core.world import HeadlessWorld
from sirrobin.economy.config import EconomyConfig
from sirrobin.economy.state import EconomyState
from sirrobin.genetics.genotype import GenotypeBatch
from sirrobin.physics.contracts import FluidSample
from sirrobin.physics.live_config import LiveLocomotionConfig

FIXTURE = Path(__file__).resolve().parents[1] / "oracle/fixtures/live/donor_development_live.json"
FIELD_NAMES = ("ND", "BP", "BD", "BM")
# Declared starting-condition values for this operational fixture only. They make the
# new authoritative stores visible without claiming a calibrated nutrient/body-mass
# mapping before feeding or lifecycle consumes one.
FIXTURE_STRUCTURE_Q_PER_BODY = 1_000
FIXTURE_RESERVE_Q_PER_BODY = 500
FIXTURE_FEEDING_CONFIG = FeedingConfig(
    capture_efficiency=0.5,
    assimilation_efficiency=0.5,
)
FIXTURE_MATERIAL_ENERGY_CONFIG = MaterialEnergyConfig(
    producer_j_per_q=0.50,
    reserve_j_per_q=0.45,
)
FIXTURE_MAINTENANCE_CONFIG = MaintenanceConfig(maintenance_w_per_kg=0.01)
FIXTURE_BIRTH_CONFIG = BirthConfig(initial_reserve_q=100)


@dataclass(frozen=True, slots=True)
class WorldRunReport:
    requested_sim_time_s: float
    sim_time_s: float
    economy_steps: int
    mechanics_steps: int
    full_batch_mechanics_steps: int
    representative_mechanics_steps: int
    fast_forwarded_mechanics_steps: int
    mechanics_steps_per_economy_step: int
    shipped_mechanics_steps_per_economy_step: int
    population: int
    initial_fields_q: tuple[int, int, int, int]
    final_fields_q: tuple[int, int, int, int]
    initial_structure_q: int
    initial_reserve_q: int
    final_structure_q: int
    final_reserve_q: int
    initial_whole_world_q: int
    final_whole_world_q: int
    books_closed: bool
    feeding_enabled: bool
    feeding_events: int
    feeding_producer_debit_q: int
    feeding_reserve_credit_q: int
    feeding_dissolved_return_q: int
    feeding_assimilation_heat_j: float
    maintenance_enabled: bool
    maintenance_events: int
    maintenance_reserve_debit_q: int
    maintenance_dissolved_return_q: int
    death_return_q: int
    maintenance_heat_j: float
    death_dissipation_j: float
    starvation_deaths: int
    birth_requested: bool
    birth_succeeded: bool
    birth_reason: str | None
    birth_parent_id: int | None
    birth_child_id: int | None
    birth_structure_q: int
    birth_initial_reserve_q: int
    birth_total_debit_q: int
    birth_construction_heat_j: float
    final_maintenance_carry_j: float
    final_intake_carry_mol: float
    final_assimilation_carry_q: float
    producer_j_per_q: float
    reserve_j_per_q: float
    final_assimilation_carry_energy_j: float
    gait_time_min_s: float
    gait_time_max_s: float
    positions_sample_enu_m: tuple[tuple[float, float, float], ...]
    mechanical_work_j: float
    periodic_projected_translation_drift_m: float
    periodic_projected_yaw_drift_rad: float
    periodic_projected_relative_state_error: float
    periodic_projected_velocity_error_m_s: float
    periodic_projected_yaw_momentum_error_kg_m2_s: float
    periodic_projected_relative_work_error: float
    setup_wall_time_s: float
    advance_wall_time_s: float

    @property
    def total_wall_time_s(self) -> float:
        return self.setup_wall_time_s + self.advance_wall_time_s

    @property
    def sim_seconds_per_wall_second(self) -> float:
        return self.sim_time_s / self.advance_wall_time_s


def _field_totals_q(state: EconomyState) -> tuple[int, int, int, int]:
    return tuple(int(reservoir.sum(dtype=torch.int64).item()) for reservoir in state.reservoirs)


def _build_fixture_world(
    *,
    bodies: int,
    device: torch.device,
    economy_interval_s: float,
    material_energy_config: MaterialEnergyConfig | None = None,
    live_bodies: int | None = None,
    reserve_q_per_creature: int = FIXTURE_RESERVE_Q_PER_BODY,
) -> HeadlessWorld:
    if live_bodies is None:
        live_bodies = bodies
    if not 0 <= live_bodies <= bodies:
        raise ValueError("live_bodies must fit inside body capacity")
    rows = json.loads(FIXTURE.read_text(encoding="utf-8"))["bodies"]
    swimmer = next(row for row in rows if row["id"] == "swimmer")
    genotype = GenotypeBatch.from_donor_rows(
        [swimmer] * bodies,
        dtype=torch.float64,
        device=device,
    )
    genotype.alive[:, live_bodies:] = False
    genotype.stable_id[:, live_bodies:] = 0
    economy_config = replace(
        EconomyConfig(),
        gx=1,
        gy=1,
        gz=4,
        lx_m=10.0,
        ly_m=10.0,
        lz_m=20.0,
        dt_eco_s=economy_interval_s,
        remin_floor_s=max(
            EconomyConfig().remin_floor_s,
            1.0 / (100_000.0 * economy_interval_s),
        ),
    )
    economy_state = EconomyState.zeros(economy_config, device=device)
    economy_state.nd_q.fill_(10_000_000)
    economy_state.bp_q.fill_(1_000_000)
    economy_state.bd_q[..., 0] = 500_000
    lead = (1, bodies)
    alive = genotype.alive
    return HeadlessWorld(
        genotype=genotype,
        fluid=FluidSample(
            torch.full(lead, 1000.0, dtype=torch.float64, device=device),
            torch.zeros((*lead, 3), dtype=torch.float64, device=device),
        ),
        live_config=LiveLocomotionConfig(),
        economy_state=economy_state,
        economy_config=economy_config,
        creature_material_state=CreatureMaterialState.uniform_live(
            alive,
            structure_q_per_creature=FIXTURE_STRUCTURE_Q_PER_BODY,
            reserve_q_per_creature=reserve_q_per_creature,
        ),
        material_energy_config=(
            FIXTURE_MATERIAL_ENERGY_CONFIG
            if material_energy_config is None
            else material_energy_config
        ),
    )


def run_world(
    *,
    seconds: float,
    bodies: int,
    device_name: str,
    economy_interval_s: float = 0.1,
    feed_one: bool = False,
    maintain_one: bool = False,
    birth_one: bool = False,
) -> WorldRunReport:
    if not math.isfinite(seconds) or seconds <= 0.0:
        raise ValueError("seconds must be positive and finite")
    if bodies <= 0:
        raise ValueError("bodies must be positive")
    if feed_one and bodies != 1:
        raise ValueError("one-creature feeding requires exactly one body")
    if maintain_one and bodies != 1:
        raise ValueError("one-creature maintenance requires exactly one body")
    if birth_one and bodies != 1:
        raise ValueError("one-creature birth requires exactly one live body")
    if birth_one and (feed_one or maintain_one):
        raise ValueError("one-creature birth is a standalone transaction in this slice")
    if not math.isfinite(economy_interval_s) or economy_interval_s <= 0.0:
        raise ValueError("economy_interval_s must be positive and finite")
    try:
        device = torch.device(device_name)
    except RuntimeError as error:
        raise ValueError(f"invalid device {device_name!r}") from error
    if device.type == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA was requested but is not available")

    setup_started = time.perf_counter()
    world = _build_fixture_world(
        bodies=2 if birth_one else bodies,
        live_bodies=bodies,
        reserve_q_per_creature=(
            FIXTURE_STRUCTURE_Q_PER_BODY
            + FIXTURE_BIRTH_CONFIG.initial_reserve_q
            + FIXTURE_RESERVE_Q_PER_BODY
            if birth_one
            else FIXTURE_RESERVE_Q_PER_BODY
        ),
        device=device,
        economy_interval_s=economy_interval_s,
    )
    # This command builds the specifically measured all-live swimmer-clone fixture.
    # Other callers remain on canonical mechanics unless they make their own
    # reviewed, explicit policy decision.
    runner = HeadlessRunner(
        world,
        periodic_policy=DEFAULT_PERIODIC_MOTION_POLICY,
        feeding_config=FIXTURE_FEEDING_CONFIG if feed_one else None,
        maintenance_config=FIXTURE_MAINTENANCE_CONFIG if maintain_one else None,
    )
    interval_s = world.economy_config.dt_eco_s
    intervals = round(seconds / interval_s)
    if not math.isclose(seconds, intervals * interval_s, rel_tol=1.0e-12, abs_tol=1.0e-12):
        raise ValueError(f"seconds must be an exact multiple of the fixture interval {interval_s:g}")
    initial_fields_q = _field_totals_q(world.economy_state)
    initial_matter = world.matter_totals()
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    setup_wall_time_s = time.perf_counter() - setup_started

    advance_started = time.perf_counter()
    books_closed = True
    mechanics_steps = 0
    full_batch_mechanics_steps = 0
    representative_mechanics_steps = 0
    fast_forwarded_mechanics_steps = 0
    mechanical_work_j = 0.0
    periodic_projected_translation_drift_m = 0.0
    periodic_projected_yaw_drift_rad = 0.0
    periodic_projected_relative_state_error = 0.0
    periodic_projected_velocity_error_m_s = 0.0
    periodic_projected_yaw_momentum_error_kg_m2_s = 0.0
    periodic_projected_relative_work_error = 0.0
    feeding_events = 0
    feeding_producer_debit_q = 0
    feeding_reserve_credit_q = 0
    feeding_dissolved_return_q = 0
    feeding_assimilation_heat_j = 0.0
    maintenance_events = 0
    maintenance_reserve_debit_q = 0
    maintenance_dissolved_return_q = 0
    death_return_q = 0
    maintenance_heat_j = 0.0
    death_dissipation_j = 0.0
    starvation_deaths = 0
    for _ in range(intervals):
        tick = runner.advance()
        books_closed &= bool(tick.matter.books_closed.all())
        mechanics_steps += tick.mechanics_steps
        full_batch_mechanics_steps += tick.full_batch_mechanics_steps
        representative_mechanics_steps += tick.representative_mechanics_steps
        fast_forwarded_mechanics_steps += tick.fast_forwarded_mechanics_steps
        mechanical_work_j += float(tick.mechanical_work_j.sum().item())
        if tick.periodic_error is not None:
            periodic_projected_translation_drift_m += (
                tick.periodic_error.accumulated_translation_error_m
            )
            periodic_projected_yaw_drift_rad += (
                tick.periodic_error.accumulated_yaw_error_rad
            )
            periodic_projected_relative_state_error += (
                tick.periodic_error.projected_relative_state_error
            )
            periodic_projected_velocity_error_m_s += (
                tick.periodic_error.projected_velocity_error_m_s
            )
            periodic_projected_yaw_momentum_error_kg_m2_s += (
                tick.periodic_error.projected_yaw_momentum_error_kg_m2_s
            )
            periodic_projected_relative_work_error += (
                tick.periodic_error.projected_relative_work_error
            )
        if tick.feeding is not None:
            feeding_events += 1
            feeding_producer_debit_q += tick.feeding.actual_debit_q
            feeding_reserve_credit_q += tick.feeding.reserve_credit_q
            feeding_dissolved_return_q += tick.feeding.dissolved_return_q
            feeding_assimilation_heat_j += tick.feeding.assimilation_heat_j
        if tick.maintenance is not None:
            maintenance_events += 1
            maintenance_reserve_debit_q += tick.maintenance.debit_q
            maintenance_dissolved_return_q += tick.maintenance.maintenance_return_q
            death_return_q += tick.maintenance.death_return_q
            maintenance_heat_j += tick.maintenance.maintenance_heat_j
            death_dissipation_j += tick.maintenance.death_dissipation_j
            starvation_deaths += int(tick.maintenance.starved)
    birth_report = None
    if birth_one:
        birth_before = world.matter_totals()
        birth_report = attempt_exact_clone_birth(world, FIXTURE_BIRTH_CONFIG)
        books_closed &= bool(world.close_matter_step(birth_before).books_closed.all())
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    advance_wall_time_s = time.perf_counter() - advance_started

    live = world.body.alive
    live_gait_time = world.live_state.gait_time_s[live]
    gait_time = (
        live_gait_time
        if live_gait_time.numel() > 0
        else torch.tensor([world.sim_time_s], dtype=torch.float64, device=device)
    )
    positions = world.live_state.position_enu_m[live]
    sample = positions[: min(8, positions.shape[0])].detach().cpu().tolist()
    shipped_schedule = WorldSchedule.from_configs(LiveLocomotionConfig(), EconomyConfig())
    final_matter = world.matter_totals()
    return WorldRunReport(
        requested_sim_time_s=seconds,
        sim_time_s=world.sim_time_s,
        economy_steps=int(world.economy_state.step.item()),
        mechanics_steps=mechanics_steps,
        full_batch_mechanics_steps=full_batch_mechanics_steps,
        representative_mechanics_steps=representative_mechanics_steps,
        fast_forwarded_mechanics_steps=fast_forwarded_mechanics_steps,
        mechanics_steps_per_economy_step=runner.schedule.mechanics_steps_per_economy_step,
        shipped_mechanics_steps_per_economy_step=(
            shipped_schedule.mechanics_steps_per_economy_step
        ),
        population=int(world.body.alive.sum().item()),
        initial_fields_q=initial_fields_q,
        final_fields_q=_field_totals_q(world.economy_state),
        initial_structure_q=int(initial_matter.structure_q.sum().item()),
        initial_reserve_q=int(initial_matter.reserve_q.sum().item()),
        final_structure_q=int(final_matter.structure_q.sum().item()),
        final_reserve_q=int(final_matter.reserve_q.sum().item()),
        initial_whole_world_q=int(initial_matter.total_q.sum().item()),
        final_whole_world_q=int(final_matter.total_q.sum().item()),
        books_closed=books_closed,
        feeding_enabled=feed_one,
        feeding_events=feeding_events,
        feeding_producer_debit_q=feeding_producer_debit_q,
        feeding_reserve_credit_q=feeding_reserve_credit_q,
        feeding_dissolved_return_q=feeding_dissolved_return_q,
        feeding_assimilation_heat_j=feeding_assimilation_heat_j,
        maintenance_enabled=maintain_one,
        maintenance_events=maintenance_events,
        maintenance_reserve_debit_q=maintenance_reserve_debit_q,
        maintenance_dissolved_return_q=maintenance_dissolved_return_q,
        death_return_q=death_return_q,
        maintenance_heat_j=maintenance_heat_j,
        death_dissipation_j=death_dissipation_j,
        starvation_deaths=starvation_deaths,
        birth_requested=birth_one,
        birth_succeeded=bool(birth_report is not None and birth_report.born),
        birth_reason=None if birth_report is None else birth_report.reason,
        birth_parent_id=None if birth_report is None else birth_report.parent_id,
        birth_child_id=None if birth_report is None else birth_report.child_id,
        birth_structure_q=0 if birth_report is None else birth_report.structure_q,
        birth_initial_reserve_q=(
            0 if birth_report is None else birth_report.initial_reserve_q
        ),
        birth_total_debit_q=0 if birth_report is None else birth_report.total_debit_q,
        birth_construction_heat_j=(
            0.0 if birth_report is None else birth_report.construction_heat_j
        ),
        final_maintenance_carry_j=float(
            world.creature_material.maintenance_carry_j.sum().item()
        ),
        final_intake_carry_mol=float(
            world.creature_material.intake_carry_mol.sum().item()
        ),
        final_assimilation_carry_q=float(
            world.creature_material.assimilation_carry_q.sum().item()
        ),
        producer_j_per_q=world.material_energy_config.producer_j_per_q,
        reserve_j_per_q=world.material_energy_config.reserve_j_per_q,
        final_assimilation_carry_energy_j=float(
            world.creature_material.assimilation_carry_q.sum().item()
            * world.material_energy_config.reserve_j_per_q
        ),
        gait_time_min_s=float(gait_time.min().item()),
        gait_time_max_s=float(gait_time.max().item()),
        positions_sample_enu_m=tuple(tuple(float(value) for value in row) for row in sample),
        mechanical_work_j=mechanical_work_j,
        periodic_projected_translation_drift_m=periodic_projected_translation_drift_m,
        periodic_projected_yaw_drift_rad=periodic_projected_yaw_drift_rad,
        periodic_projected_relative_state_error=(
            periodic_projected_relative_state_error
        ),
        periodic_projected_velocity_error_m_s=(
            periodic_projected_velocity_error_m_s
        ),
        periodic_projected_yaw_momentum_error_kg_m2_s=(
            periodic_projected_yaw_momentum_error_kg_m2_s
        ),
        periodic_projected_relative_work_error=(
            periodic_projected_relative_work_error
        ),
        setup_wall_time_s=setup_wall_time_s,
        advance_wall_time_s=advance_wall_time_s,
    )


def _fields_line(values: tuple[int, int, int, int]) -> str:
    return " ".join(f"{name}={value}" for name, value in zip(FIELD_NAMES, values, strict=True))


def format_report(report: WorldRunReport) -> str:
    positions = "\n".join(
        f"  {index}: ({east:.9g}, {north:.9g}, {up:.9g})"
        for index, (east, north, up) in enumerate(report.positions_sample_enu_m)
    )
    return "\n".join(
        (
            "SirRobin composed-world run (operational output; not a stable schema)",
            f"requested simulated time s: {report.requested_sim_time_s:g}",
            f"actual simulated time s: {report.sim_time_s:g}",
            f"economy steps: {report.economy_steps}",
            f"mechanics steps: {report.mechanics_steps}",
            f"full-batch mechanics steps: {report.full_batch_mechanics_steps}",
            f"representative mechanics steps: {report.representative_mechanics_steps}",
            f"periodic fast-forward mechanics steps: {report.fast_forwarded_mechanics_steps}",
            f"mechanics steps / economy step: {report.mechanics_steps_per_economy_step}",
            "shipped mechanics steps / economy step: "
            f"{report.shipped_mechanics_steps_per_economy_step}",
            f"population: {report.population}",
            f"initial field totals q: {_fields_line(report.initial_fields_q)}",
            f"final field totals q: {_fields_line(report.final_fields_q)}",
            f"initial field total q: {sum(report.initial_fields_q)}",
            f"final field total q: {sum(report.final_fields_q)}",
            "initial creature totals q: "
            f"structure={report.initial_structure_q} reserve={report.initial_reserve_q}",
            "final creature totals q: "
            f"structure={report.final_structure_q} reserve={report.final_reserve_q}",
            f"initial whole-world total q: {report.initial_whole_world_q}",
            f"final whole-world total q: {report.final_whole_world_q}",
            f"exact whole-world books closed: {'yes' if report.books_closed else 'no'}",
            "one-creature feeding enabled: "
            f"{'yes' if report.feeding_enabled else 'no'}",
            f"feeding events: {report.feeding_events}",
            f"feeding producer debit q: {report.feeding_producer_debit_q}",
            f"feeding reserve credit q: {report.feeding_reserve_credit_q}",
            f"feeding dissolved return q: {report.feeding_dissolved_return_q}",
            f"feeding assimilation heat J: {report.feeding_assimilation_heat_j:.9g}",
            f"final feeding intake carry mol: {report.final_intake_carry_mol:.9g}",
            "final feeding assimilation carry q: "
            f"{report.final_assimilation_carry_q:.9g}",
            f"producer chemical energy density J/q: {report.producer_j_per_q:.9g}",
            f"reserve chemical energy density J/q: {report.reserve_j_per_q:.9g}",
            "final feeding assimilation carry energy J: "
            f"{report.final_assimilation_carry_energy_j:.9g}",
            "one-creature maintenance enabled: "
            f"{'yes' if report.maintenance_enabled else 'no'}",
            f"maintenance events: {report.maintenance_events}",
            f"maintenance reserve debit q: {report.maintenance_reserve_debit_q}",
            "maintenance dissolved return q: "
            f"{report.maintenance_dissolved_return_q}",
            f"death material return q: {report.death_return_q}",
            f"maintenance heat J: {report.maintenance_heat_j:.9g}",
            f"death dissipation J: {report.death_dissipation_j:.9g}",
            f"starvation deaths: {report.starvation_deaths}",
            "one paid exact-clone birth requested: "
            f"{'yes' if report.birth_requested else 'no'}",
            f"birth succeeded: {'yes' if report.birth_succeeded else 'no'}",
            f"birth refusal reason: {report.birth_reason or 'none'}",
            "birth parent/child IDs: "
            f"{report.birth_parent_id or 0} -> {report.birth_child_id or 0}",
            f"birth structure q: {report.birth_structure_q}",
            f"birth initial reserve q: {report.birth_initial_reserve_q}",
            f"birth total parent debit q: {report.birth_total_debit_q}",
            f"birth construction heat J: {report.birth_construction_heat_j:.9g}",
            f"final maintenance carry J: {report.final_maintenance_carry_j:.9g}",
            f"integrated mechanical work J: {report.mechanical_work_j:.9g}",
            "periodic projected drift totals across economy intervals: "
            f"translation={report.periodic_projected_translation_drift_m:.9g} m "
            f"yaw={report.periodic_projected_yaw_drift_rad:.9g} rad "
            "projected-relative-state="
            f"{report.periodic_projected_relative_state_error:.9g} "
            "projected-velocity="
            f"{report.periodic_projected_velocity_error_m_s:.9g} m/s "
            "projected-yaw-momentum="
            f"{report.periodic_projected_yaw_momentum_error_kg_m2_s:.9g} kg m2/s "
            "projected-relative-work="
            f"{report.periodic_projected_relative_work_error:.9g}",
            f"mechanics clock range s: {report.gait_time_min_s:g} .. {report.gait_time_max_s:g}",
            f"positions sample ENU m ({len(report.positions_sample_enu_m)}/{report.population}):",
            positions,
            f"setup wall time s: {report.setup_wall_time_s:.6f}",
            f"advance wall time s: {report.advance_wall_time_s:.6f}",
            f"total wall time s: {report.total_wall_time_s:.6f}",
            "simulated seconds / wall second: "
            f"{report.sim_seconds_per_wall_second:.6f} (advance only)",
        )
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seconds", type=float, default=0.1)
    parser.add_argument("--bodies", type=int, default=2)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--economy-interval", type=float, default=0.1)
    parser.add_argument(
        "--feed-one",
        action="store_true",
        help="enable the scoped one-live-creature feeding transaction",
    )
    parser.add_argument(
        "--maintain-one",
        action="store_true",
        help="enable mass-derived maintenance and starvation death for one creature",
    )
    parser.add_argument(
        "--birth-one",
        action="store_true",
        help="attempt one paid exact-clone birth after the requested run",
    )
    arguments = parser.parse_args(argv)
    try:
        report = run_world(
            seconds=arguments.seconds,
            bodies=arguments.bodies,
            device_name=arguments.device,
            economy_interval_s=arguments.economy_interval,
            feed_one=arguments.feed_one,
            maintain_one=arguments.maintain_one,
            birth_one=arguments.birth_one,
        )
    except ValueError as error:
        parser.error(str(error))
    print(format_report(report))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
