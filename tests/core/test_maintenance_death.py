"""Focused contract for mass-derived maintenance and starvation death."""

from __future__ import annotations

import math
from fractions import Fraction

import pytest
import torch

from sirrobin.core.feeding import FeedingConfig
from sirrobin.core.material import MaterialEnergyConfig
from sirrobin.core.metabolism import (
    MaintenanceConfig,
    _quantize_energy_demand,
    maintain_single_creature,
)
from sirrobin.core.runner import HeadlessRunner
from sirrobin.numerics.flux import INT64_SAFE_MAX
from tools.run_world import _build_fixture_world


def _world(
    *,
    reserve_q: int,
    interval_s: float = 0.1,
    energy: MaterialEnergyConfig | None = None,
):
    world = _build_fixture_world(
        bodies=1,
        device=torch.device("cpu"),
        economy_interval_s=interval_s,
        material_energy_config=energy,
    )
    # Change the creature reserve by an equal field transfer so the world's
    # construction-time exact inventory baseline remains authoritative.
    world.economy_state.nd_q[0, 0, 0, 0] += 500 - reserve_q
    world.creature_material.reserve_q[0, 0] = reserve_q
    return world


def _mass_kg(world) -> float:
    mask = world.body.seg_mask & world.body.alive[..., None]
    mass_sim = torch.where(mask, world.body.mass_sim, 0.0).sum(-1)
    return float((mass_sim * world.live_config.kg_per_sim_mass)[0, 0].item())


def test_maintenance_debits_reserve_returns_matter_and_carries_fractional_demand() -> None:
    world = _world(reserve_q=500)
    before = world.matter_totals()
    dissolved_before = int(world.economy_state.nd_q.sum().item())
    config = MaintenanceConfig(maintenance_w_per_kg=1.0)
    mass_kg = _mass_kg(world)
    demand_j = mass_kg * world.economy_config.dt_eco_s
    expected_debit = math.floor(demand_j / world.material_energy_config.reserve_j_per_q)
    expected_carry = demand_j - (
        expected_debit * world.material_energy_config.reserve_j_per_q
    )

    report = maintain_single_creature(world, config)

    assert report is not None
    assert report.structural_mass_kg == mass_kg
    assert report.reserve_before_q == 500
    assert report.requested_q == expected_debit
    assert report.debit_q == expected_debit
    assert report.reserve_after_q == 500 - expected_debit
    assert report.maintenance_return_q == expected_debit
    assert report.death_return_q == 0
    assert report.starved is False
    assert report.maintenance_heat_j == (
        expected_debit * world.material_energy_config.reserve_j_per_q
    )
    assert math.isclose(report.carry_after_j, expected_carry, abs_tol=1.0e-12)
    assert math.isclose(
        float(world.creature_material.maintenance_carry_j[0, 0].item()),
        expected_carry,
        abs_tol=1.0e-12,
    )
    assert int(world.economy_state.nd_q.sum().item()) == (
        dissolved_before + expected_debit
    )
    assert world.close_matter_step(before).books_closed.tolist() == [True]


def test_fractional_carry_makes_split_and_unsplit_maintenance_agree() -> None:
    split = _world(reserve_q=500, interval_s=0.1)
    unsplit = _world(reserve_q=500, interval_s=0.2)
    config = MaintenanceConfig(maintenance_w_per_kg=0.01)

    first = maintain_single_creature(split, config)
    second = maintain_single_creature(split, config)
    whole = maintain_single_creature(unsplit, config)

    assert first is not None and second is not None and whole is not None
    assert first.debit_q + second.debit_q == whole.debit_q
    assert split.creature_material.reserve_q.tolist() == (
        unsplit.creature_material.reserve_q.tolist()
    )
    assert math.isclose(
        first.maintenance_heat_j + second.maintenance_heat_j,
        whole.maintenance_heat_j,
        abs_tol=1.0e-12,
    )
    assert math.isclose(second.carry_after_j, whole.carry_after_j, abs_tol=1.0e-12)


def test_quantization_above_float_integer_precision_is_exact_and_split_stable() -> None:
    quantum_j = 4.057636661790241e-12
    whole_demand_j = 320748.1636319218
    exact_demand = Fraction.from_float(whole_demand_j)
    exact_quantum = Fraction.from_float(quantum_j)
    expected_q = exact_demand // exact_quantum
    expected_carry_j = float(exact_demand - expected_q * exact_quantum)

    whole_q, whole_carry_j = _quantize_energy_demand(
        (whole_demand_j,), quantum_j
    )
    first_q, first_carry_j = _quantize_energy_demand(
        (whole_demand_j / 2.0,), quantum_j
    )
    second_q, second_carry_j = _quantize_energy_demand(
        (whole_demand_j / 2.0, first_carry_j), quantum_j
    )

    assert expected_q > 2**53
    assert whole_q == expected_q
    assert whole_carry_j == expected_carry_j
    assert first_q + second_q == whole_q
    assert second_carry_j == whole_carry_j


def test_integer_energy_density_completes_a_full_maintenance_settlement() -> None:
    world = _world(
        reserve_q=500,
        energy=MaterialEnergyConfig(producer_j_per_q=1, reserve_j_per_q=1),
    )

    report = maintain_single_creature(world, MaintenanceConfig(1.0))

    assert report is not None
    assert report.debit_q > 0
    assert 0.0 <= report.carry_after_j < 1.0


def test_starvation_returns_all_material_and_cannot_return_it_twice() -> None:
    world = _world(reserve_q=3)
    world.creature_material.assimilation_carry_q[0, 0] = 0.5
    before = world.matter_totals()
    dissolved_before = int(world.economy_state.nd_q.sum().item())
    config = MaintenanceConfig(maintenance_w_per_kg=10.0)

    report = maintain_single_creature(world, config)

    assert report is not None
    assert report.requested_q > 3
    assert report.debit_q == 3
    assert report.maintenance_return_q == 3
    assert report.death_return_q == 1_000
    assert report.death_dissipation_j == 0.225
    assert report.starved is True
    assert world.creature_material.structure_q.tolist() == [[0]]
    assert world.creature_material.reserve_q.tolist() == [[0]]
    assert world.creature_material.intake_carry_mol.tolist() == [[0.0]]
    assert world.creature_material.assimilation_carry_q.tolist() == [[0.0]]
    assert world.creature_material.maintenance_carry_j.tolist() == [[0.0]]
    assert world.genotype.alive.tolist() == [[False]]
    assert world.body.alive.tolist() == [[False]]
    assert int(world.economy_state.nd_q.sum().item()) == dissolved_before + 1_003
    assert world.close_matter_step(before).books_closed.tolist() == [True]

    dissolved_after_death = int(world.economy_state.nd_q.sum().item())
    assert maintain_single_creature(world, config) is None
    assert int(world.economy_state.nd_q.sum().item()) == dissolved_after_death


@pytest.mark.parametrize(
    ("zero_linear", "zero_rotational", "assimilation_carry_q"),
    [
        (False, False, 0.5),
        (True, False, 0.5),
        (False, True, 0.5),
        (False, False, 0.0),
    ],
)
def test_death_dissipation_matches_independent_three_channel_oracle(
    zero_linear: bool,
    zero_rotational: bool,
    assimilation_carry_q: float,
) -> None:
    world = _world(reserve_q=3)
    last_ledger = None
    for _ in range(12):
        last_ledger = world._step_mechanics()
    assert last_ledger is not None
    if zero_linear:
        world.live_state.velocity_rel_water_enu_m_s.zero_()
    if zero_rotational:
        world.live_state.yaw_momentum_kg_m2_s.zero_()
    world.creature_material.assimilation_carry_q[0, 0] = assimilation_carry_q

    velocity_xy = world.live_state.velocity_rel_water_enu_m_s[0, 0, :2].clone()
    yaw_momentum = float(world.live_state.yaw_momentum_kg_m2_s[0, 0].item())
    effective_mass = last_ledger.effective_mass_after_kg[0, :2, :2]
    yaw_inertia = float(
        last_ledger.hydrodynamics.yaw_inertia_after_kg_m2[0].item()
    )
    linear_j = float(
        (0.5 * velocity_xy @ effective_mass @ velocity_xy).item()
    )
    rotational_j = yaw_momentum * yaw_momentum / (2.0 * yaw_inertia)
    carry_j = assimilation_carry_q * world.material_energy_config.reserve_j_per_q

    report = maintain_single_creature(
        world,
        MaintenanceConfig(10.0),
        last_mechanics_substep=last_ledger,
    )

    assert report is not None and report.starved
    assert math.isclose(
        report.death_dissipation_j,
        linear_j + rotational_j + carry_j,
        rel_tol=1.0e-12,
        abs_tol=1.0e-12,
    )


def test_death_energy_oracle_binds_the_only_live_nonzero_ledger_row() -> None:
    world = _build_fixture_world(
        bodies=2,
        device=torch.device("cpu"),
        economy_interval_s=0.1,
    )
    world.economy_state.nd_q[0, 0, 0, 0] += 1_500
    world.creature_material.structure_q[0, 0] = 0
    world.creature_material.reserve_q[0, 0] = 0
    for carry in world.creature_material.carries:
        carry[0, 0] = 0.0
    world.genotype.alive[0, 0] = False
    world.rebuild_body()
    world.creature_material.reserve_q[0, 1] = 3
    world.economy_state.nd_q[0, 0, 0, 0] += 497
    world.creature_material.assimilation_carry_q[0, 1] = 0.5

    last_ledger = None
    for _ in range(12):
        last_ledger = world._step_mechanics()
    assert last_ledger is not None
    velocity_xy = world.live_state.velocity_rel_water_enu_m_s[0, 1, :2].clone()
    yaw_momentum = float(world.live_state.yaw_momentum_kg_m2_s[0, 1].item())
    effective_mass = last_ledger.effective_mass_after_kg[1, :2, :2]
    yaw_inertia = float(
        last_ledger.hydrodynamics.yaw_inertia_after_kg_m2[1].item()
    )
    assert not torch.equal(
        last_ledger.effective_mass_after_kg[0],
        last_ledger.effective_mass_after_kg[1],
    )
    expected_j = float(
        (0.5 * velocity_xy @ effective_mass @ velocity_xy).item()
    )
    expected_j += yaw_momentum * yaw_momentum / (2.0 * yaw_inertia)
    expected_j += 0.5 * world.material_energy_config.reserve_j_per_q

    report = maintain_single_creature(
        world,
        MaintenanceConfig(10.0),
        last_mechanics_substep=last_ledger,
    )

    assert report is not None and report.creature_slot == 1
    assert math.isclose(
        report.death_dissipation_j,
        expected_j,
        rel_tol=1.0e-12,
        abs_tol=1.0e-12,
    )


def test_failed_death_deposit_is_atomic_across_all_authoritative_state() -> None:
    world = _world(reserve_q=3)
    world.creature_material.assimilation_carry_q[0, 0] = 0.5
    world.economy_state.nd_q[0, 0, 0, 0] = INT64_SAFE_MAX - 1
    before = (
        world.economy_state.nd_q.clone(),
        world.creature_material.structure_q.clone(),
        world.creature_material.reserve_q.clone(),
        tuple(carry.clone() for carry in world.creature_material.carries),
        world.genotype.alive.clone(),
        world.body.alive.clone(),
        world.live_state.velocity_rel_water_enu_m_s.clone(),
        world.live_state.yaw_momentum_kg_m2_s.clone(),
    )

    with pytest.raises(ValueError, match="deposit would exceed"):
        maintain_single_creature(world, MaintenanceConfig(10.0))

    after = (
        world.economy_state.nd_q,
        world.creature_material.structure_q,
        world.creature_material.reserve_q,
        world.creature_material.carries,
        world.genotype.alive,
        world.body.alive,
        world.live_state.velocity_rel_water_enu_m_s,
        world.live_state.yaw_momentum_kg_m2_s,
    )
    assert torch.equal(after[0], before[0])
    assert torch.equal(after[1], before[1])
    assert torch.equal(after[2], before[2])
    assert all(
        torch.equal(actual, expected)
        for actual, expected in zip(after[3], before[3], strict=True)
    )
    assert torch.equal(after[4], before[4])
    assert torch.equal(after[5], before[5])
    assert torch.equal(after[6], before[6])
    assert torch.equal(after[7], before[7])


def test_death_rejects_unsupported_vertical_motion_before_mutation() -> None:
    world = _world(reserve_q=3)
    world.live_state.velocity_rel_water_enu_m_s[0, 0, 2] = 1.0
    dissolved_before = world.economy_state.nd_q.clone()

    with pytest.raises(ValueError, match="vertical"):
        maintain_single_creature(world, MaintenanceConfig(10.0))

    assert torch.equal(world.economy_state.nd_q, dissolved_before)
    assert world.creature_material.structure_q.tolist() == [[1_000]]
    assert world.creature_material.reserve_q.tolist() == [[3]]
    assert world.genotype.alive.tolist() == [[True]]
    assert world.body.alive.tolist() == [[True]]


def test_maintenance_rejects_population_contention_in_this_slice() -> None:
    world = _build_fixture_world(
        bodies=2,
        device=torch.device("cpu"),
        economy_interval_s=0.1,
    )

    try:
        maintain_single_creature(world, MaintenanceConfig(1.0))
    except ValueError as error:
        assert "at most one live creature" in str(error)
    else:
        raise AssertionError("multi-creature maintenance unexpectedly succeeded")


def test_runner_feeds_before_maintenance_and_closes_the_combined_tick() -> None:
    world = _world(reserve_q=500)
    runner = HeadlessRunner(
        world,
        feeding_config=FeedingConfig(0.5, 0.5),
        maintenance_config=MaintenanceConfig(0.01),
    )

    tick = runner.advance()

    assert tick.feeding is not None
    assert len(tick.maintenance) == 1
    assert tick.maintenance[0].reserve_before_q == (
        500 + tick.feeding.reserve_credit_q
    )
    assert tick.matter.books_closed.tolist() == [True]


def test_runner_can_continue_an_empty_world_after_one_time_starvation_death() -> None:
    world = _world(reserve_q=3)
    runner = HeadlessRunner(
        world,
        maintenance_config=MaintenanceConfig(10.0),
    )

    death_tick = runner.advance()
    empty_tick = runner.advance()

    assert len(death_tick.maintenance) == 1
    assert death_tick.maintenance[0].starved is True
    assert death_tick.positive_actuator_work_j is not None
    assert death_tick.positive_actuator_work_j.tolist() == [[0.0]]
    assert death_tick.maintenance[0].death_dissipation_j == 0.0
    assert death_tick.matter.books_closed.tolist() == [True]
    assert empty_tick.maintenance == ()
    assert empty_tick.positive_actuator_work_j is not None
    assert empty_tick.actuator_braking_work_j is not None
    assert not bool(empty_tick.positive_actuator_work_j.any())
    assert not bool(empty_tick.actuator_braking_work_j.any())
    assert not bool(empty_tick.mechanical_work_j.any())
    assert empty_tick.matter.books_closed.tolist() == [True]
    assert world.body.alive.sum().item() == 0
    assert world.sim_time_s == 0.2
