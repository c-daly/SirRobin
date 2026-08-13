"""Chemical settlement of authoritative actuator work."""

from __future__ import annotations

import math

import pytest
import torch

from sirrobin.core.material import MaterialEnergyConfig
from sirrobin.core.metabolism import MaintenanceConfig, maintain_population
from sirrobin.core.periodic_motion import DEFAULT_PERIODIC_MOTION_POLICY
from sirrobin.core.runner import HeadlessRunner
from tools.run_world import _build_fixture_world


def _world(*, reserve_q: int = 500):
    world = _build_fixture_world(
        bodies=1,
        reserve_q_per_creature=reserve_q,
        device=torch.device("cpu"),
        economy_interval_s=0.1,
        material_energy_config=MaterialEnergyConfig(
            producer_j_per_q=0.5,
            reserve_j_per_q=0.45,
        ),
    )
    return world


def test_positive_actuator_work_debits_reserve_through_declared_efficiency() -> None:
    world = _world()
    before = world.matter_totals()

    reports = maintain_population(
        world,
        MaintenanceConfig(
            maintenance_w_per_kg=0.0,
            chemical_to_mechanical_efficiency=0.5,
        ),
        positive_actuator_work_j=torch.tensor([[0.45]], dtype=torch.float64),
        actuator_braking_work_j=torch.tensor([[0.10]], dtype=torch.float64),
    )

    report = reports[0]
    assert report.positive_actuator_work_j == 0.45
    assert report.baseline_maintenance_demand_j == 0.0
    assert report.actuator_braking_heat_j == 0.10
    assert report.locomotion_chemical_demand_j == 0.9
    assert report.muscle_inefficiency_heat_j == 0.45
    assert report.requested_q == 2
    assert report.debit_q == 2
    assert report.reserve_before_q == 500
    assert report.reserve_after_q == 498
    assert report.maintenance_return_q == 2
    assert report.carry_after_j == 0.0
    assert report.locomotion_chemical_demand_j == pytest.approx(
        report.positive_actuator_work_j + report.muscle_inefficiency_heat_j
    )
    assert report.debit_q * world.material_energy_config.reserve_j_per_q == pytest.approx(
        report.demand_j - report.carry_after_j
    )
    assert world.close_matter_step(before).books_closed.tolist() == [True]


def test_energy_settlement_identity_includes_fractional_carry() -> None:
    world = _world()

    report = maintain_population(
        world,
        MaintenanceConfig(0.0, chemical_to_mechanical_efficiency=1.0),
        positive_actuator_work_j=torch.tensor([[0.44]], dtype=torch.float64),
    )[0]

    assert report.debit_q == 0
    assert math.isclose(report.carry_after_j, 0.44, abs_tol=1.0e-12)
    assert math.isclose(
        report.demand_j
        - report.debit_q * world.material_energy_config.reserve_j_per_q
        - report.carry_after_j,
        0.0,
        abs_tol=1.0e-12,
    )


def test_negative_or_misshaped_work_is_rejected_before_mutation() -> None:
    world = _world()
    reserve_before = world.creature_material.reserve_q.clone()
    dissolved_before = world.economy_state.nd_q.clone()

    with pytest.raises(ValueError, match="positive actuator work"):
        maintain_population(
            world,
            MaintenanceConfig(0.0),
            positive_actuator_work_j=torch.tensor([[-1.0]], dtype=torch.float64),
        )
    with pytest.raises(ValueError, match="population shape"):
        maintain_population(
            world,
            MaintenanceConfig(0.0),
            positive_actuator_work_j=torch.zeros(2, dtype=torch.float64),
        )

    assert torch.equal(world.creature_material.reserve_q, reserve_before)
    assert torch.equal(world.economy_state.nd_q, dissolved_before)


def test_unfunded_actuator_work_is_rejected_before_mutation() -> None:
    world = _world(reserve_q=0)
    reserve_before = world.creature_material.reserve_q.clone()
    dissolved_before = world.economy_state.nd_q.clone()

    with pytest.raises(ValueError, match="funded chemical budget"):
        maintain_population(
            world,
            MaintenanceConfig(0.0),
            positive_actuator_work_j=torch.tensor([[0.45]], dtype=torch.float64),
        )

    assert torch.equal(world.creature_material.reserve_q, reserve_before)
    assert torch.equal(world.economy_state.nd_q, dissolved_before)


def test_runner_settles_the_same_positive_work_emitted_by_canonical_mechanics() -> None:
    world = _world()
    oracle_world = _world()
    runner = HeadlessRunner(
        world,
        maintenance_config=MaintenanceConfig(
            0.0,
            chemical_to_mechanical_efficiency=1.0,
        ),
    )
    expected_positive = torch.zeros((1, 1), dtype=torch.float64)
    expected_braking = torch.zeros((1, 1), dtype=torch.float64)
    expected_dissipation = torch.zeros((1, 1), dtype=torch.float64)
    for _ in range(runner.schedule.mechanics_steps_per_economy_step):
        ledger = oracle_world._step_mechanics()
        input_power = ledger.total.input_power_w.reshape(1, 1)
        expected_positive.add_(input_power.clamp_min(0.0) * oracle_world.live_config.dt)
        expected_braking.add_((-input_power).clamp_min(0.0) * oracle_world.live_config.dt)
        expected_dissipation.add_(
            ledger.total.dissipated_power_w.reshape(1, 1)
            * oracle_world.live_config.dt
        )

    tick = runner.advance()

    assert tick.positive_actuator_work_j is not None
    assert tick.actuator_braking_work_j is not None
    assert torch.equal(tick.positive_actuator_work_j, expected_positive)
    assert torch.equal(tick.actuator_braking_work_j, expected_braking)
    assert torch.equal(tick.mechanical_work_j, expected_dissipation)
    # This negative control fails if hydrodynamic loss is substituted for the
    # positive actuator-input channel that the chemical debit actually consumes.
    assert not torch.equal(expected_positive, expected_dissipation)
    assert len(tick.maintenance) == 1
    assert tick.maintenance[0].positive_actuator_work_j == pytest.approx(
        float(tick.positive_actuator_work_j.item())
    )
    assert tick.matter.books_closed.tolist() == [True]


def test_zero_reserve_cannot_request_unfunded_actuation() -> None:
    world = _world(reserve_q=0)

    tick = HeadlessRunner(
        world,
        maintenance_config=MaintenanceConfig(0.0),
    ).advance()

    assert tick.positive_actuator_work_j is not None
    assert tick.positive_actuator_work_j.tolist() == [[0.0]]
    assert world.creature_material.reserve_q.tolist() == [[0]]
    assert tick.maintenance[0].starved is False
    assert tick.matter.books_closed.tolist() == [True]


def test_funded_effort_avoids_an_economy_interval_all_or_zero_cliff() -> None:
    energy = MaterialEnergyConfig(
        producer_j_per_q=0.25,
        reserve_j_per_q=0.20,
    )
    whole = _build_fixture_world(
        bodies=1,
        reserve_q_per_creature=1,
        device=torch.device("cpu"),
        economy_interval_s=0.1,
        material_energy_config=energy,
    )
    split = _build_fixture_world(
        bodies=1,
        reserve_q_per_creature=1,
        device=torch.device("cpu"),
        economy_interval_s=0.05,
        material_energy_config=energy,
    )
    whole_tick = HeadlessRunner(
        whole,
        maintenance_config=MaintenanceConfig(0.0),
    ).advance()
    split_runner = HeadlessRunner(
        split,
        maintenance_config=MaintenanceConfig(0.0),
    )
    split_ticks = (split_runner.advance(), split_runner.advance())

    assert whole_tick.positive_actuator_work_j is not None
    assert all(tick.positive_actuator_work_j is not None for tick in split_ticks)
    whole_work_j = float(whole_tick.positive_actuator_work_j.item())
    split_work_j = math.fsum(
        float(tick.positive_actuator_work_j.item())  # type: ignore[union-attr]
        for tick in split_ticks
    )
    assert 0.0 < whole_work_j <= energy.reserve_j_per_q
    assert 0.0 < split_work_j <= energy.reserve_j_per_q
    assert whole_tick.matter.books_closed.tolist() == [True]
    assert all(tick.matter.books_closed.tolist() == [True] for tick in split_ticks)


def test_energy_settlement_rejects_unaccounted_periodic_fast_forward() -> None:
    with pytest.raises(ValueError, match="canonical mechanics"):
        HeadlessRunner(
            _world(),
            periodic_policy=DEFAULT_PERIODIC_MOTION_POLICY,
            maintenance_config=MaintenanceConfig(0.0),
        )


def test_periodic_fast_forward_cannot_be_enabled_by_late_config_mutation() -> None:
    runner = HeadlessRunner(
        _world(),
        periodic_policy=DEFAULT_PERIODIC_MOTION_POLICY,
    )
    runner.maintenance_config = MaintenanceConfig(0.0)

    with pytest.raises(ValueError, match="canonical mechanics"):
        runner.advance()


@pytest.mark.parametrize("efficiency", [0.0, -0.1, 1.1, math.nan, True])
def test_efficiency_must_be_a_finite_fraction(efficiency) -> None:
    with pytest.raises((TypeError, ValueError)):
        MaintenanceConfig(
            0.0,
            chemical_to_mechanical_efficiency=efficiency,
        )
