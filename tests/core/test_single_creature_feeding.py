"""One-creature local feeding transaction and energy boundary."""

from __future__ import annotations

import json
import math
from dataclasses import replace
from pathlib import Path

import pytest
import torch

from sirrobin.core.feeding import FeedingConfig, feed_single_creature
from sirrobin.core.material import CreatureMaterialState, MaterialEnergyConfig
from sirrobin.core.runner import HeadlessRunner
from sirrobin.core.world import HeadlessWorld
from sirrobin.economy.config import EconomyConfig
from sirrobin.economy.state import EconomyState
from sirrobin.fields.grid import ScalarGrid
from sirrobin.genetics.genotype import GenotypeBatch
from sirrobin.numerics.flux import INT64_SAFE_MAX
from sirrobin.physics.contracts import FluidSample
from sirrobin.physics.live_config import LiveLocomotionConfig
from sirrobin.physics.morphology import query_morphology

FIXTURE = Path("oracle/fixtures/live/donor_development_live.json")


def _feeding_config() -> FeedingConfig:
    return FeedingConfig(
        capture_efficiency=0.5,
        assimilation_efficiency=0.5,
    )


def _world(
    *,
    bodies: int = 1,
    live_bodies: int | None = None,
    energy: MaterialEnergyConfig | None = None,
) -> HeadlessWorld:
    rows = json.loads(FIXTURE.read_text(encoding="utf-8"))["bodies"]
    swimmer = next(row for row in rows if row["id"] == "swimmer")
    genotype = GenotypeBatch.from_donor_rows([swimmer] * bodies, dtype=torch.float64)
    if live_bodies is not None:
        genotype.alive[0, live_bodies:] = False
    config = replace(
        EconomyConfig(),
        gx=1,
        gy=1,
        gz=4,
        lx_m=10.0,
        ly_m=10.0,
        lz_m=20.0,
        dt_eco_s=0.1,
        remin_floor_s=1.0e-4,
    )
    economy = EconomyState.zeros(config)
    economy.nd_q.fill_(10_000_000)
    economy.bp_q.fill_(1_000_000)
    economy.bd_q[..., 0] = 500_000
    return HeadlessWorld(
        genotype=genotype,
        fluid=FluidSample(
            torch.full((1, bodies), 1000.0, dtype=torch.float64),
            torch.zeros((1, bodies, 3), dtype=torch.float64),
        ),
        live_config=LiveLocomotionConfig(),
        economy_state=economy,
        economy_config=config,
        creature_material_state=CreatureMaterialState.uniform_live(
            genotype.alive,
            structure_q_per_creature=1_000,
            reserve_q_per_creature=100,
        ),
        material_energy_config=energy
        or MaterialEnergyConfig(producer_j_per_q=0.50, reserve_j_per_q=0.45),
    )


def _raw_total(world: HeadlessWorld) -> int:
    return sum(
        int(value)
        for reservoir in (
            *world.economy_state.reservoirs,
            *world.creature_material.reservoirs,
        )
        for value in reservoir.reshape(-1).tolist()
    )


def _authority_tensors(world: HeadlessWorld) -> tuple[torch.Tensor, ...]:
    return (
        *world.economy_state.reservoirs,
        *world.creature_material.reservoirs,
        *world.creature_material.carries,
    )


def _speed_for_raw_request(world: HeadlessWorld, config: FeedingConfig, raw_q: float) -> float:
    morphology = query_morphology(world.body, world.live_config)
    area_m2 = float(morphology.intake_area_m2[0, 0].item())
    concentration = ScalarGrid(
        world.economy_state.bp_q,
        world.geometry,
        q_mass_mol=world.economy_config.q_mass_mol,
    ).value_at(0, world.live_state.position_enu_m[0, 0])
    return (
        raw_q
        * world.economy_config.q_mass_mol
        / (area_m2 * world.economy_config.dt_eco_s * config.capture_efficiency * concentration)
    )


def test_local_feeding_debits_actual_producer_and_routes_every_quantum() -> None:
    world = _world()
    world.live_state.velocity_rel_water_enu_m_s[0, 0, 0] = 2.0
    config = _feeding_config()
    total_before = _raw_total(world)
    producer_before = int(world.economy_state.bp_q.sum().item())
    dissolved_before = int(world.economy_state.nd_q.sum().item())
    reserve_before = int(world.creature_material.reserve_q.sum().item())

    report = feed_single_creature(world, config)

    assert report.world_index == 0
    assert report.creature_slot == 0
    expected_request = math.floor(
        report.clearance_volume_m3
        * report.sampled_producer_mol_m3
        / world.economy_config.q_mass_mol
    )
    assert report.requested_q == expected_request
    assert report.actual_debit_q > 0
    assert producer_before - int(world.economy_state.bp_q.sum().item()) == report.actual_debit_q
    assert int(world.creature_material.reserve_q.sum().item()) - reserve_before == report.reserve_credit_q
    assert int(world.economy_state.nd_q.sum().item()) - dissolved_before == report.dissolved_return_q
    assert report.actual_debit_q == report.reserve_credit_q + report.dissolved_return_q
    assert _raw_total(world) == total_before
    assert report.producer_chemical_input_j == pytest.approx(
        report.actual_debit_q * report.producer_j_per_q
    )
    assert report.reserve_chemical_credit_j == pytest.approx(
        report.reserve_credit_q * report.reserve_j_per_q
    )
    assert report.assimilation_heat_j >= 0.0
    assert (
        report.producer_chemical_input_j
        + report.assimilation_carry_before_q * report.reserve_j_per_q
    ) == pytest.approx(
        report.reserve_chemical_credit_j
        + report.assimilation_heat_j
        + report.assimilation_carry_after_q * report.reserve_j_per_q
    )


def test_scarce_local_stock_caps_the_actual_debit_without_minting() -> None:
    world = _world()
    world.economy_state.bp_q.zero_()
    world.economy_state.bp_q[0, 0, 0, 0] = 5
    world.live_state.velocity_rel_water_enu_m_s[0, 0, 0] = 1.0e9
    total_before = _raw_total(world)

    report = feed_single_creature(world, _feeding_config())

    assert report.requested_q > 5
    assert report.actual_debit_q == 5
    assert int(world.economy_state.bp_q.sum().item()) == 0
    assert report.reserve_credit_q + report.dissolved_return_q == 5
    assert _raw_total(world) == total_before


def test_lower_source_energy_limits_reserve_credit_before_heat_can_go_negative() -> None:
    world = _world(
        energy=MaterialEnergyConfig(producer_j_per_q=0.10, reserve_j_per_q=0.45)
    )
    world.live_state.velocity_rel_water_enu_m_s[0, 0, 0] = 2.0
    config = _feeding_config()

    report = feed_single_creature(world, config)

    energy_limited_credit = math.floor(
        report.actual_debit_q * report.producer_j_per_q / report.reserve_j_per_q
    )
    assert report.reserve_credit_q == energy_limited_credit
    assert report.reserve_credit_q < math.floor(
        report.actual_debit_q * config.assimilation_efficiency
    )
    assert report.assimilation_heat_j >= 0.0


def test_nonfinite_dynamic_cause_rejects_before_any_reservoir_mutation() -> None:
    world = _world()
    world.live_state.velocity_rel_water_enu_m_s[0, 0, 0] = 1.0e308
    before = tuple(reservoir.clone() for reservoir in _authority_tensors(world))

    with pytest.raises(ValueError, match="feeding causes"):
        feed_single_creature(world, _feeding_config())

    after = _authority_tensors(world)
    assert all(torch.equal(left, right) for left, right in zip(before, after, strict=True))


@pytest.mark.parametrize("full_destination", ["dissolved", "reserve"])
def test_destination_capacity_rejects_before_source_debit(full_destination: str) -> None:
    world = _world()
    world.live_state.velocity_rel_water_enu_m_s[0, 0, 0] = 2.0
    if full_destination == "dissolved":
        world.economy_state.nd_q[0, 0, 0, 0] = INT64_SAFE_MAX - 1
    else:
        world.creature_material.reserve_q[0, 0] = INT64_SAFE_MAX - 1
    before = tuple(reservoir.clone() for reservoir in _authority_tensors(world))

    with pytest.raises(ValueError, match="deposit|reserve"):
        feed_single_creature(world, _feeding_config())

    after = _authority_tensors(world)
    assert all(torch.equal(left, right) for left, right in zip(before, after, strict=True))


def test_energy_overflow_rejects_before_source_debit_and_arrests_runner() -> None:
    energy = MaterialEnergyConfig(
        producer_j_per_q=1.0e308, reserve_j_per_q=1.0e-308
    )
    config = _feeding_config()
    direct = _world(energy=energy)
    direct.live_state.velocity_rel_water_enu_m_s[0, 0, 0] = 2.0
    before = tuple(reservoir.clone() for reservoir in _authority_tensors(direct))

    with pytest.raises(ValueError, match="feeding energy"):
        feed_single_creature(direct, config)

    after = _authority_tensors(direct)
    assert all(torch.equal(left, right) for left, right in zip(before, after, strict=True))

    runner = HeadlessRunner(_world(energy=energy), feeding_config=config)
    with pytest.raises(ValueError, match="feeding energy"):
        runner.advance()
    with pytest.raises(RuntimeError, match="not resumable"):
        runner.advance()


def test_subquantum_intake_and_assimilation_carries_prevent_cadence_starvation() -> None:
    config = _feeding_config()
    intake_world = _world()
    intake_world.live_state.velocity_rel_water_enu_m_s[0, 0, 0] = _speed_for_raw_request(
        intake_world, config, 0.6
    )

    first = feed_single_creature(intake_world, config)
    second = feed_single_creature(intake_world, config)

    assert first.actual_debit_q == 0
    assert first.intake_carry_after_mol > 0.0
    assert second.actual_debit_q == 1

    assimilation_world = _world()
    assimilation_world.live_state.velocity_rel_water_enu_m_s[
        0, 0, 0
    ] = _speed_for_raw_request(assimilation_world, config, 1.1)

    first = feed_single_creature(assimilation_world, config)
    second = feed_single_creature(assimilation_world, config)

    assert first.actual_debit_q == 1
    assert first.reserve_credit_q == 0
    assert first.assimilation_carry_after_q == pytest.approx(0.5)
    assert second.actual_debit_q == 1
    assert second.reserve_credit_q == 1
    assert second.assimilation_carry_before_q == pytest.approx(0.5)
    assert second.assimilation_carry_after_q == 0.0


def test_energy_limited_fraction_uses_the_same_carry() -> None:
    config = replace(_feeding_config(), assimilation_efficiency=1.0)
    world = _world(
        energy=MaterialEnergyConfig(producer_j_per_q=0.10, reserve_j_per_q=0.20)
    )
    world.live_state.velocity_rel_water_enu_m_s[0, 0, 0] = _speed_for_raw_request(
        world, config, 1.1
    )

    first = feed_single_creature(world, config)
    second = feed_single_creature(world, config)

    assert first.actual_debit_q == 1
    assert first.reserve_credit_q == 0
    assert first.assimilation_carry_after_q == pytest.approx(0.5)
    assert second.actual_debit_q == 1
    assert second.reserve_credit_q == 1
    assert second.assimilation_carry_before_q == pytest.approx(0.5)
    assert second.assimilation_carry_after_q == 0.0


def test_energy_density_is_immutable_world_authority_and_reported() -> None:
    world = _world()
    world.live_state.velocity_rel_water_enu_m_s[0, 0, 0] = 2.0

    report = feed_single_creature(world, _feeding_config())

    assert report.producer_j_per_q == world.material_energy_config.producer_j_per_q
    assert report.reserve_j_per_q == world.material_energy_config.reserve_j_per_q
    with pytest.raises(AttributeError):
        world.material_energy_config = MaterialEnergyConfig(1.0, 1.0)


def test_inactive_out_of_domain_position_cannot_abort_selected_live_sample() -> None:
    world = _world(bodies=2, live_bodies=1)
    world.live_state.position_enu_m[0, 1, 2] = 1.0e9
    world.live_state.velocity_rel_water_enu_m_s[0, 0, 0] = 2.0

    report = feed_single_creature(world, _feeding_config())

    assert report.creature_slot == 0
    assert report.actual_debit_q > 0


def test_runner_rejects_corrupt_live_cache_before_advancing_any_clock() -> None:
    world = _world(bodies=2, live_bodies=1)
    runner = HeadlessRunner(world, feeding_config=_feeding_config())
    world.body.alive[0, 1] = True
    time_before = world.sim_time_s
    gait_before = world.live_state.gait_time_s.clone()

    with pytest.raises(RuntimeError, match="identity cache"):
        runner.advance()

    assert world.sim_time_s == time_before
    assert torch.equal(world.live_state.gait_time_s, gait_before)
    with pytest.raises(RuntimeError, match="not resumable"):
        runner.advance()


def test_runner_rejects_broken_economy_output_before_feeding_from_it() -> None:
    world = _world()
    runner = HeadlessRunner(world, feeding_config=_feeding_config())
    reserve_before = world.creature_material.reserve_q.clone()
    original_step = world._step_economy

    def broken_step():
        ledger = original_step()
        return replace(ledger, books_closed=torch.zeros_like(ledger.books_closed))

    world._step_economy = broken_step

    with pytest.raises(RuntimeError, match="exact nutrient books"):
        runner.advance()

    assert torch.equal(world.creature_material.reserve_q, reserve_before)
    assert torch.equal(
        world.creature_material.intake_carry_mol,
        torch.zeros_like(world.creature_material.intake_carry_mol),
    )


@pytest.mark.parametrize("missing_cause", ["speed", "intake", "producer"])
def test_missing_physical_or_local_cause_makes_feeding_an_exact_noop(
    missing_cause: str,
) -> None:
    world = _world()
    world.live_state.velocity_rel_water_enu_m_s[0, 0, 0] = 2.0
    if missing_cause == "speed":
        world.live_state.velocity_rel_water_enu_m_s.zero_()
    elif missing_cause == "intake":
        world.body.intake.zero_()
    else:
        world.economy_state.bp_q.zero_()
    before = _raw_total(world)

    report = feed_single_creature(world, _feeding_config())

    assert report.requested_q == 0
    assert report.actual_debit_q == 0
    assert report.reserve_credit_q == 0
    assert report.dissolved_return_q == 0
    assert report.assimilation_heat_j == 0.0
    assert _raw_total(world) == before


def test_runner_composes_one_feeding_act_inside_whole_world_closure() -> None:
    runner = HeadlessRunner(_world(), feeding_config=_feeding_config())

    tick = runner.advance()

    assert tick.feeding is not None
    assert tick.feeding.actual_debit_q > 0
    assert tick.feeding.reserve_credit_q > 0
    assert tick.feeding.dissolved_return_q > 0
    assert tick.matter.books_closed.tolist() == [True]
    assert tick.matter.total_before_q.tolist() == tick.matter.total_after_q.tolist()


def test_one_creature_api_still_rejects_population_shape() -> None:
    world = _world(bodies=2)

    with pytest.raises(ValueError, match="exactly one live creature"):
        feed_single_creature(world, _feeding_config())


@pytest.mark.parametrize(
    "changes",
    [
        {"capture_efficiency": -0.1},
        {"capture_efficiency": 1.1},
        {"assimilation_efficiency": -0.1},
        {"assimilation_efficiency": 1.1},
        {"capture_efficiency": math.inf},
        {"assimilation_efficiency": math.nan},
    ],
)
def test_feeding_config_rejects_malformed_scientific_inputs(
    changes: dict[str, float],
) -> None:
    with pytest.raises(ValueError):
        replace(_feeding_config(), **changes)


@pytest.mark.parametrize(
    ("producer", "reserve"),
    ((0.0, 1.0), (1.0, 0.0), (math.inf, 1.0), (1.0, math.nan)),
)
def test_material_energy_config_rejects_malformed_densities(
    producer: float, reserve: float
) -> None:
    with pytest.raises(ValueError):
        MaterialEnergyConfig(producer_j_per_q=producer, reserve_j_per_q=reserve)
