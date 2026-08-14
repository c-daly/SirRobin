from __future__ import annotations

import torch

from sirrobin.fields.geometry import GridGeometry
from sirrobin.fields.stencil import point_stencil
from sirrobin.numerics.flux import INT64_SAFE_MAX
from sirrobin.organisms.feeding import (
    FeedingConfig,
    allocate_shared_stock,
    feed_population,
)
from sirrobin.organisms.state import PopulationState, validate_population_state


def _population(*, stable_id: list[int]) -> PopulationState:
    capacity = len(stable_id)
    alive = torch.ones((1, capacity), dtype=torch.bool)
    zeros_i64 = torch.zeros((1, capacity), dtype=torch.int64)
    zeros_f64 = torch.zeros((1, capacity), dtype=torch.float64)
    return PopulationState(
        alive=alive,
        stable_id=torch.tensor([stable_id], dtype=torch.int64),
        parent_id=zeros_i64,
        generation=zeros_i64,
        born_at_s=zeros_f64,
        structure_q=torch.ones((1, capacity), dtype=torch.int64),
        reserve_q=zeros_i64,
        intake_carry_mol=zeros_f64,
        assimilation_carry_q=zeros_f64,
        maintenance_carry_j=zeros_f64,
        next_stable_id=torch.tensor([max(stable_id) + 1], dtype=torch.int64),
    )


def test_shared_stock_tie_breaks_by_stable_identity_not_slot() -> None:
    geometry = GridGeometry(1, 1, 1, 1.0, 1.0, 1.0)
    positions = torch.zeros((1, 2, 3))
    stencil = point_stencil(positions, geometry)
    actual, debit, remaining, exhausted = allocate_shared_stock(
        torch.tensor([[[[5]]]], dtype=torch.int64),
        stencil,
        torch.tensor([[4, 4]], dtype=torch.int64),
        torch.tensor([[20, 10]], dtype=torch.int64),
        torch.tensor([[True, True]]),
        rounds=8,
    )

    assert actual.tolist() == [[2, 3]]
    assert debit.item() == 5
    assert remaining.item() == 0
    assert not bool(exhausted.any())


def test_one_round_flags_reachable_stock_and_robust_rounds_redistribute_it() -> None:
    geometry = GridGeometry(1, 1, 2, 1.0, 1.0, 2.0)
    positions = torch.tensor([[[0.0, 0.0, 0.0], [0.0, 0.0, -1.0]]])
    stencil = point_stencil(positions, geometry)
    stock = torch.tensor([[[[1, 2]]]], dtype=torch.int64)
    request = torch.tensor([[1, 2]], dtype=torch.int64)
    stable_id = torch.tensor([[1, 2]], dtype=torch.int64)
    alive = torch.tensor([[True, True]])

    fast = allocate_shared_stock(
        stock,
        stencil,
        request,
        stable_id,
        alive,
        rounds=1,
    )
    robust = allocate_shared_stock(
        stock,
        stencil,
        request,
        stable_id,
        alive,
        rounds=8,
    )

    assert fast[0].tolist() == [[1, 1]]
    assert fast[3].tolist() == [[False, True]]
    assert robust[0].tolist() == [[1, 2]]
    assert robust[1].sum().item() == 3
    assert not bool(robust[3].any())


def test_population_feeding_closes_matter_and_named_conversion_energy() -> None:
    population = _population(stable_id=[2, 1])
    geometry = GridGeometry(1, 1, 1, 1.0, 1.0, 1.0)
    producer = torch.tensor([[[[10]]]], dtype=torch.int64)
    dissolved = torch.zeros_like(producer)
    config = FeedingConfig(
        interval_s=1.0,
        q_mass_mol=1.0,
        capture_efficiency=1.0,
        assimilation_efficiency=0.5,
        producer_j_per_q=10.0,
        reserve_j_per_q=10.0,
    )
    config.validate()
    before_q = (
        producer.sum()
        + dissolved.sum()
        + population.structure_q.sum()
        + population.reserve_q.sum()
    )

    step = feed_population(
        population,
        producer,
        dissolved,
        torch.zeros((1, 2, 3)),
        torch.tensor([[[1.0, 0.0, 0.0], [1.0, 0.0, 0.0]]]),
        torch.ones((1, 2)),
        geometry,
        config,
    )

    after_q = (
        step.producer_q.sum()
        + step.dissolved_q.sum()
        + step.population.structure_q.sum()
        + step.population.reserve_q.sum()
    )
    assert torch.equal(before_q, after_q)
    assert step.ledger.requested_q.tolist() == [[10, 10]]
    assert step.ledger.actual_debit_q.tolist() == [[5, 5]]
    assert step.ledger.reserve_credit_q.sum().item() == 4
    assert step.ledger.dissolved_return_q.sum().item() == 6
    assert step.population.assimilation_carry_q.sum().item() == 1.0
    assert step.ledger.producer_chemical_input_j.sum().item() == 100.0
    assert step.ledger.reserve_chemical_credit_j.sum().item() == 40.0
    assert step.ledger.assimilation_heat_j.sum().item() == 50.0
    assert step.ledger.transaction_committed.tolist() == [True]
    assert not bool(step.ledger.invalid.any())
    validate_population_state(step.population)


def test_population_feeding_rolls_back_an_overflowing_world_atomically() -> None:
    population = _population(stable_id=[1])
    geometry = GridGeometry(1, 1, 1, 1.0, 1.0, 1.0)
    producer = torch.tensor([[[[5]]]], dtype=torch.int64)
    dissolved = torch.tensor([[[[INT64_SAFE_MAX - 2]]]], dtype=torch.int64)
    config = FeedingConfig(1.0, 1.0, 1.0, 0.0, 10.0, 10.0)

    step = feed_population(
        population,
        producer,
        dissolved,
        torch.zeros((1, 1, 3)),
        torch.tensor([[[1.0, 0.0, 0.0]]]),
        torch.ones((1, 1)),
        geometry,
        config,
    )

    assert torch.equal(step.producer_q, producer)
    assert torch.equal(step.dissolved_q, dissolved)
    assert torch.equal(step.population.reserve_q, population.reserve_q)
    assert step.ledger.actual_debit_q.sum().item() == 0
    assert step.ledger.dissolved_credit_by_cell_q.sum().item() == 0
    assert step.ledger.transaction_committed.tolist() == [False]
    assert step.ledger.invalid.tolist() == [[True]]


def test_population_feeding_is_one_full_compiled_graph() -> None:
    population = _population(stable_id=[2, 1])
    geometry = GridGeometry(1, 1, 1, 1.0, 1.0, 1.0)
    producer = torch.tensor([[[[10]]]], dtype=torch.int64)
    dissolved = torch.zeros_like(producer)
    positions = torch.zeros((1, 2, 3))
    velocity = torch.tensor([[[1.0, 0.0, 0.0], [1.0, 0.0, 0.0]]])
    intake = torch.ones((1, 2))
    config = FeedingConfig(1.0, 1.0, 1.0, 0.5, 10.0, 10.0)
    compiled = torch.compile(feed_population, fullgraph=True, dynamic=False)

    eager = feed_population(
        population,
        producer,
        dissolved,
        positions,
        velocity,
        intake,
        geometry,
        config,
    )
    actual = compiled(
        population,
        producer,
        dissolved,
        positions,
        velocity,
        intake,
        geometry,
        config,
    )

    assert torch.equal(actual.producer_q, eager.producer_q)
    assert torch.equal(actual.dissolved_q, eager.dissolved_q)
    assert torch.equal(actual.population.reserve_q, eager.population.reserve_q)
