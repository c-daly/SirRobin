"""Adversarial contract for one atomic shared-stock feeding transaction."""

from __future__ import annotations

import pytest
import torch

from sirrobin.core.feeding import FeedingConfig, feed_population
from sirrobin.core.runner import HeadlessRunner
from sirrobin.numerics.flux import INT64_SAFE_MAX
from tools.run_world import _build_fixture_world


def _world(*, stock_q: int = 5, speed_m_s: float = 1.0e9):
    world = _build_fixture_world(
        bodies=2,
        device=torch.device("cpu"),
        economy_interval_s=0.1,
    )
    producer_before_q = int(world.economy_state.bp_q.sum().item())
    world.economy_state.bp_q.zero_()
    world.economy_state.bp_q[0, 0, 0, 0] = stock_q
    world.economy_state.nd_q[0, 0, 0, 0] += producer_before_q - stock_q
    world.live_state.position_enu_m.zero_()
    world.live_state.velocity_rel_water_enu_m_s.zero_()
    world.live_state.velocity_rel_water_enu_m_s[..., 0] = speed_m_s
    return world


def _raw_total(world) -> int:
    return sum(
        int(value)
        for reservoir in (
            *world.economy_state.reservoirs,
            *world.creature_material.reservoirs,
        )
        for value in reservoir.reshape(-1).tolist()
    )


def _authority_tensors(world) -> tuple[torch.Tensor, ...]:
    return (
        *world.economy_state.reservoirs,
        *world.creature_material.reservoirs,
        *world.creature_material.carries,
    )


def test_two_equal_claimants_share_one_finite_debit_without_minting() -> None:
    world = _world(stock_q=5)
    before_q = _raw_total(world)
    reserve_before_q = int(world.creature_material.reserve_q.sum().item())

    report = feed_population(world, FeedingConfig(1.0, 0.5))

    assert len(report.creatures) == 2
    assert all(creature.requested_q > 5 for creature in report.creatures)
    assert [creature.actual_debit_q for creature in report.creatures] == [3, 2]
    assert report.requested_q == sum(
        creature.requested_q for creature in report.creatures
    )
    assert report.actual_debit_q == 5
    assert int(world.economy_state.bp_q.sum().item()) == 0
    assert report.actual_debit_q == (
        report.reserve_credit_q + report.dissolved_return_q
    )
    assert int(world.creature_material.reserve_q.sum().item()) - reserve_before_q == (
        report.reserve_credit_q
    )
    assert _raw_total(world) == before_q
    for creature in report.creatures:
        assert (
            creature.producer_chemical_input_j
            + creature.assimilation_carry_before_q * creature.reserve_j_per_q
        ) == pytest.approx(
            creature.reserve_chemical_credit_j
            + creature.assimilation_heat_j
            + creature.assimilation_carry_after_q * creature.reserve_j_per_q
        )

    retry = feed_population(world, FeedingConfig(1.0, 0.5))
    assert retry.requested_q == 0
    assert retry.actual_debit_q == 0


def test_scarcity_tie_break_uses_stable_id_not_capacity_slot() -> None:
    world = _world(stock_q=1)
    world.genotype.stable_id[0] = torch.tensor([2, 1], dtype=torch.int64)
    world.body.stable_id[0] = world.genotype.stable_id[0]

    report = feed_population(world, FeedingConfig(1.0, 1.0))

    assert [creature.creature_slot for creature in report.creatures] == [1, 0]
    assert [creature.actual_debit_q for creature in report.creatures] == [1, 0]
    assert report.actual_debit_q == 1


def test_overlapping_stencil_can_use_uncontested_stock_after_shared_cell() -> None:
    world = _world(stock_q=3)
    world.economy_state.bp_q[0, 0, 0, 0] = 1
    world.economy_state.bp_q[0, 0, 0, 1] = 2
    world.live_state.position_enu_m[0, 1, 2] = -5.0

    report = feed_population(world, FeedingConfig(1.0, 1.0))

    assert [creature.actual_debit_q for creature in report.creatures] == [1, 2]
    assert report.actual_debit_q == 3
    assert int(world.economy_state.bp_q.sum().item()) == 0


def test_overlapping_waste_credits_preflight_as_one_atomic_transaction() -> None:
    world = _world(stock_q=5)
    world.economy_state.nd_q[0, 0, 0, 0] = INT64_SAFE_MAX - 4
    before = tuple(tensor.clone() for tensor in _authority_tensors(world))

    try:
        feed_population(world, FeedingConfig(1.0, 0.0))
    except ValueError as error:
        assert "deposit would exceed" in str(error)
    else:
        raise AssertionError("overlapping population deposits unexpectedly committed")

    assert all(
        torch.equal(actual, expected)
        for actual, expected in zip(_authority_tensors(world), before, strict=True)
    )


def test_runner_composes_shared_feeding_inside_whole_world_closure() -> None:
    world = _world(stock_q=5, speed_m_s=2.0)
    runner = HeadlessRunner(world, feeding_config=FeedingConfig(1.0, 0.5))

    tick = runner.advance()

    assert tick.feeding is not None
    assert len(tick.feeding.creatures) == 2
    assert tick.feeding.actual_debit_q == sum(
        creature.actual_debit_q for creature in tick.feeding.creatures
    )
    assert tick.feeding.actual_debit_q == (
        tick.feeding.reserve_credit_q + tick.feeding.dissolved_return_q
    )
    assert tick.matter.books_closed.tolist() == [True]
    assert tick.matter.total_before_q.tolist() == tick.matter.total_after_q.tolist()
