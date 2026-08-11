"""Continuous population lifecycle contracts for the authoritative runner."""

from __future__ import annotations

import pytest
import torch

from sirrobin.core.metabolism import MaintenanceConfig, maintain_population
from sirrobin.core.reproduction import BirthConfig
from sirrobin.core.runner import HeadlessRunner
from tools.run_world import _build_fixture_world


def _world(*, capacity: int, live: int, reserves: tuple[int, ...]):
    world = _build_fixture_world(
        bodies=capacity,
        live_bodies=live,
        reserve_q_per_creature=0,
        device=torch.device("cpu"),
        economy_interval_s=0.1,
    )
    for slot, reserve_q in enumerate(reserves):
        world.creature_material.reserve_q[0, slot] = reserve_q
        world.economy_state.nd_q[0, 0, 0, 0] -= reserve_q
    return world


def test_population_maintenance_settles_every_live_creature_once() -> None:
    world = _world(capacity=3, live=2, reserves=(2_000, 3))
    before = world.matter_totals()

    reports = maintain_population(world, MaintenanceConfig(10.0))

    assert [report.creature_slot for report in reports] == [0, 1]
    assert reports[0].starved is False
    assert reports[1].starved is True
    assert world.body.alive.tolist() == [[True, False, False]]
    assert world.close_matter_step(before).books_closed.tolist() == [True]


def test_population_maintenance_uses_stable_identity_not_slot_order() -> None:
    world = _world(capacity=2, live=2, reserves=(2_000, 2_000))
    world.genotype.stable_id[0] = torch.tensor([2, 1], dtype=torch.int64)
    world.body.stable_id[0].copy_(world.genotype.stable_id[0])

    reports = maintain_population(world, MaintenanceConfig(0.0))

    assert [report.creature_id for report in reports] == [1, 2]
    assert [report.creature_slot for report in reports] == [1, 0]


def test_runner_continuously_composes_death_then_paid_birth_and_closure() -> None:
    world = _world(capacity=3, live=2, reserves=(2_000, 3))
    original_ids = world.body.stable_id.clone()
    runner = HeadlessRunner(
        world,
        maintenance_config=MaintenanceConfig(10.0),
        birth_config=BirthConfig(initial_reserve_q=100),
    )

    tick = runner.advance()

    assert [report.starved for report in tick.maintenance] == [False, True]
    assert len(tick.births) == 1
    assert tick.births[0].born is True
    assert tick.births[0].parent_id == int(original_ids[0, 0])
    assert tick.births[0].child_slot == 1
    assert tick.births[0].child_id not in original_ids.tolist()[0]
    assert world.body.alive.tolist() == [[True, True, False]]
    assert tick.matter.books_closed.tolist() == [True]


def test_newborn_cannot_reproduce_in_its_birth_tick() -> None:
    world = _world(capacity=4, live=1, reserves=(7_000,))
    runner = HeadlessRunner(
        world,
        birth_config=BirthConfig(initial_reserve_q=2_000),
    )

    first = runner.advance()
    second = runner.advance()

    assert len(first.births) == 1
    assert first.births[0].born is True
    assert int(world.body.alive.sum()) == 3
    assert len(second.births) == 1
    assert second.births[0].parent_id == first.births[0].parent_id


def test_runner_clock_remains_valid_after_the_last_original_parent_dies() -> None:
    world = _world(capacity=2, live=1, reserves=(2_000,))
    runner = HeadlessRunner(
        world,
        maintenance_config=MaintenanceConfig(0.0),
        birth_config=BirthConfig(initial_reserve_q=100),
    )
    first = runner.advance()
    assert first.births[0].born is True
    child_slot = first.births[0].child_slot
    assert child_slot is not None

    parent_reserve_q = int(world.creature_material.reserve_q[0, 0])
    world.creature_material.reserve_q[0, 0] = 0
    world.economy_state.nd_q[0, 0, 0, 0] += parent_reserve_q
    world.economy_state.nd_q[0, 0, 0, 0] -= 1_000
    world.creature_material.reserve_q[0, child_slot] += 1_000
    runner.maintenance_config = MaintenanceConfig(10.0)
    runner.birth_config = None

    second = runner.advance()

    assert second.maintenance[0].starved is True
    assert world.body.alive.tolist() == [[False, True]]
    assert world.sim_time_s == 0.2
    assert world.live_state.gait_time_s[0, child_slot] == pytest.approx(0.1)
    assert second.matter.books_closed.tolist() == [True]


def test_birth_records_capacity_refusal_without_population_repair() -> None:
    world = _world(capacity=1, live=1, reserves=(2_000,))
    runner = HeadlessRunner(
        world,
        birth_config=BirthConfig(initial_reserve_q=100),
    )
    allocator_before = world.next_stable_id

    tick = runner.advance()

    assert len(tick.births) == 1
    assert tick.births[0].born is False
    assert tick.births[0].reason == "slot_exhausted"
    assert torch.equal(world.next_stable_id, allocator_before)
    assert int(world.body.alive.sum()) == 1
    assert tick.matter.books_closed.tolist() == [True]


def test_insufficient_reserve_is_not_promoted_to_a_birth_attempt() -> None:
    world = _world(capacity=2, live=1, reserves=(1_099,))
    runner = HeadlessRunner(
        world,
        birth_config=BirthConfig(initial_reserve_q=100),
    )

    tick = runner.advance()

    assert tick.births == ()
    assert int(world.body.alive.sum()) == 1
    assert tick.matter.books_closed.tolist() == [True]
