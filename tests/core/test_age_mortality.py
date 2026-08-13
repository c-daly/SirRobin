"""Deterministic age mortality shares the exact lifecycle settlement."""

from __future__ import annotations

import math

import pytest
import torch

from sirrobin.core.metabolism import MaintenanceConfig, maintain_single_creature
from sirrobin.core.mortality import AgeMortalityConfig
from sirrobin.core.runner import HeadlessRunner
from sirrobin.numerics.flux import INT64_SAFE_MAX
from tools.run_world import _build_fixture_world


def _world(*, reserve_q: int):
    world = _build_fixture_world(
        bodies=1,
        live_bodies=1,
        device=torch.device("cpu"),
        economy_interval_s=0.1,
    )
    world.economy_state.nd_q[0, 0, 0, 0] += 500 - reserve_q
    world.creature_material.reserve_q[0, 0] = reserve_q
    return world


def test_identity_derived_lifespan_is_bounded_deterministic_and_order_free() -> None:
    config = AgeMortalityConfig(
        min_lifespan_s=40.0,
        max_lifespan_s=80.0,
        seed=17,
    )

    forward = [config.lifespan_s(0, creature_id) for creature_id in range(1, 9)]
    reverse = {
        creature_id: config.lifespan_s(0, creature_id)
        for creature_id in reversed(range(1, 9))
    }

    assert all(40.0 <= value <= 80.0 for value in forward)
    assert len(set(forward)) > 1
    assert forward == [reverse[creature_id] for creature_id in range(1, 9)]
    assert config.lifespan_s(0, 1) != config.lifespan_s(1, 1)


@pytest.mark.parametrize(
    ("kwargs", "error"),
    [
        ({"min_lifespan_s": 0.0, "max_lifespan_s": 1.0}, ValueError),
        ({"min_lifespan_s": 2.0, "max_lifespan_s": 1.0}, ValueError),
        ({"min_lifespan_s": math.inf, "max_lifespan_s": math.inf}, ValueError),
        ({"min_lifespan_s": True, "max_lifespan_s": 1.0}, TypeError),
        ({"min_lifespan_s": 1.0, "max_lifespan_s": 1.0, "seed": -1}, ValueError),
        ({"min_lifespan_s": 1.0, "max_lifespan_s": 1.0, "seed": True}, TypeError),
    ],
)
def test_age_mortality_rejects_malformed_configuration(
    kwargs: dict[str, object], error: type[Exception]
) -> None:
    with pytest.raises(error):
        AgeMortalityConfig(**kwargs)


def test_runner_rejects_age_mortality_without_death_settlement() -> None:
    world = _world(reserve_q=500)

    with pytest.raises(ValueError, match="maintenance"):
        HeadlessRunner(
            world,
            age_mortality_config=AgeMortalityConfig(0.05, 0.05),
        )


def test_well_funded_old_age_returns_all_remaining_matter_and_keeps_lineage() -> None:
    world = _world(reserve_q=500)
    before = world.matter_totals()
    lineage_before = world.lineage_record(0, 1)
    runner = HeadlessRunner(
        world,
        maintenance_config=MaintenanceConfig(0.0),
        age_mortality_config=AgeMortalityConfig(0.05, 0.05),
    )

    tick = runner.advance()

    report = tick.maintenance[0]
    assert report.starved is False
    assert report.death_cause == "old_age"
    assert report.death_return_q == (
        1_000 + report.reserve_before_q - report.debit_q
    )
    assert report.reserve_after_q == 0
    assert world.genotype.alive.tolist() == [[False]]
    assert world.body.alive.tolist() == [[False]]
    assert tick.matter.books_closed.tolist() == [True]
    assert torch.equal(world.matter_totals().total_q, before.total_q)
    assert world.lineage_record(0, 1) == lineage_before


def test_starvation_takes_precedence_when_age_is_also_due() -> None:
    world = _world(reserve_q=0)
    runner = HeadlessRunner(
        world,
        maintenance_config=MaintenanceConfig(10.0),
        age_mortality_config=AgeMortalityConfig(0.05, 0.05),
    )

    tick = runner.advance()

    report = tick.maintenance[0]
    assert report.starved is True
    assert report.death_cause == "starvation"
    assert report.death_return_q == 1_000
    assert tick.matter.books_closed.tolist() == [True]


def test_failed_old_age_recycling_is_atomic() -> None:
    world = _world(reserve_q=500)
    world.economy_state.nd_q[0, 0, 0, 0] = INT64_SAFE_MAX - 1
    before = (
        world.economy_state.nd_q.clone(),
        world.creature_material.structure_q.clone(),
        world.creature_material.reserve_q.clone(),
        tuple(carry.clone() for carry in world.creature_material.carries),
        world.genotype.alive.clone(),
        world.body.alive.clone(),
    )

    with pytest.raises(ValueError, match="deposit would exceed"):
        maintain_single_creature(
            world,
            MaintenanceConfig(0.0),
            old_age_due=True,
        )

    assert torch.equal(world.economy_state.nd_q, before[0])
    assert torch.equal(world.creature_material.structure_q, before[1])
    assert torch.equal(world.creature_material.reserve_q, before[2])
    assert all(
        torch.equal(actual, expected)
        for actual, expected in zip(
            world.creature_material.carries, before[3], strict=True
        )
    )
    assert torch.equal(world.genotype.alive, before[4])
    assert torch.equal(world.body.alive, before[5])
