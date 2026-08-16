from __future__ import annotations

import torch

from sirrobin.organisms.lifecycle import (
    LifecycleRequest,
    settle_lifecycle,
    validate_lifecycle_request,
)
from sirrobin.organisms.state import PopulationState, validate_population_state


def _state(
    *,
    alive: list[bool],
    stable_id: list[int],
    structure_q: list[int],
    reserve_q: list[int],
    next_stable_id: int,
    generation: list[int] | None = None,
) -> PopulationState:
    capacity = len(alive)
    generation = generation or [0] * capacity
    active = torch.tensor([alive], dtype=torch.bool)
    state = PopulationState(
        alive=active,
        stable_id=torch.tensor([stable_id], dtype=torch.int64),
        parent_id=torch.zeros((1, capacity), dtype=torch.int64),
        generation=torch.tensor([generation], dtype=torch.int64),
        born_at_s=torch.zeros((1, capacity), dtype=torch.float64),
        structure_q=torch.tensor([structure_q], dtype=torch.int64),
        reserve_q=torch.tensor([reserve_q], dtype=torch.int64),
        intake_carry_mol=torch.zeros((1, capacity), dtype=torch.float64),
        assimilation_carry_q=torch.zeros((1, capacity), dtype=torch.float64),
        maintenance_carry_j=torch.zeros((1, capacity), dtype=torch.float64),
        next_stable_id=torch.tensor([next_stable_id], dtype=torch.int64),
    )
    validate_population_state(state)
    return state


def _request(
    state: PopulationState,
    *,
    death: list[bool],
    birth: list[bool],
    child_structure_q: list[int],
    child_reserve_q: list[int],
    birth_release_energy_q: list[int] | None = None,
) -> LifecycleRequest:
    if birth_release_energy_q is None:
        birth_release_energy_q = [0] * len(death)
    request = LifecycleRequest(
        death=torch.tensor([death], dtype=torch.bool),
        birth=torch.tensor([birth], dtype=torch.bool),
        child_structure_q=torch.tensor([child_structure_q], dtype=torch.int64),
        child_reserve_q=torch.tensor([child_reserve_q], dtype=torch.int64),
        birth_release_energy_q=torch.tensor(
            [birth_release_energy_q], dtype=torch.int64
        ),
        time_s=torch.tensor([4.5], dtype=torch.float64),
    )
    validate_lifecycle_request(state, request)
    return request


def test_death_slot_is_reused_by_paid_birth_without_changing_total_matter() -> None:
    state = _state(
        alive=[True, True, False, False],
        stable_id=[10, 20, 0, 0],
        structure_q=[5, 6, 0, 0],
        reserve_q=[20, 2, 0, 0],
        next_stable_id=30,
    )
    step = settle_lifecycle(
        state,
        _request(
            state,
            death=[False, True, False, False],
            birth=[True, False, False, False],
            child_structure_q=[5, 0, 0, 0],
            child_reserve_q=[3, 0, 0, 0],
        ),
    )

    assert step.state.alive.tolist() == [[True, True, False, False]]
    assert step.state.stable_id.tolist() == [[10, 30, 0, 0]]
    assert step.state.parent_id.tolist() == [[0, 10, 0, 0]]
    assert step.state.generation.tolist() == [[0, 1, 0, 0]]
    assert step.state.born_at_s.tolist() == [[0.0, 4.5, 0.0, 0.0]]
    assert step.state.structure_q.tolist() == [[5, 5, 0, 0]]
    assert step.state.reserve_q.tolist() == [[12, 3, 0, 0]]
    assert step.state.next_stable_id.tolist() == [31]
    assert step.ledger.parent_slot_for_child.tolist() == [[-1, 0, -1, -1]]

    before_q = state.structure_q.sum() + state.reserve_q.sum()
    after_q = step.state.structure_q.sum() + step.state.reserve_q.sum()
    returned_q = (
        step.ledger.death_structure_return_q.sum()
        + step.ledger.death_reserve_return_q.sum()
    )
    assert torch.equal(before_q, after_q + returned_q)
    validate_population_state(step.state)


def test_birth_release_energy_is_paid_and_returned_as_material() -> None:
    state = _state(
        alive=[True, False],
        stable_id=[1, 0],
        structure_q=[2, 0],
        reserve_q=[10, 0],
        next_stable_id=2,
    )

    step = settle_lifecycle(
        state,
        _request(
            state,
            death=[False, False],
            birth=[True, False],
            child_structure_q=[5, 0],
            child_reserve_q=[2, 0],
            birth_release_energy_q=[1, 0],
        ),
    )

    assert step.ledger.accepted_births.tolist() == [1]
    assert step.state.reserve_q.tolist() == [[2, 2]]
    assert step.ledger.birth_release_energy_return_q.tolist() == [[1, 0]]
    before_q = state.structure_q.sum() + state.reserve_q.sum()
    after_q = step.state.structure_q.sum() + step.state.reserve_q.sum()
    assert torch.equal(
        before_q,
        after_q + step.ledger.birth_release_energy_return_q.sum(),
    )


def test_birth_waits_when_parent_cannot_pay_release_energy() -> None:
    state = _state(
        alive=[True, False],
        stable_id=[1, 0],
        structure_q=[2, 0],
        reserve_q=[8, 0],
        next_stable_id=2,
    )

    step = settle_lifecycle(
        state,
        _request(
            state,
            death=[False, False],
            birth=[True, False],
            child_structure_q=[5, 0],
            child_reserve_q=[2, 0],
            birth_release_energy_q=[2, 0],
        ),
    )

    assert step.ledger.accepted_births.tolist() == [0]
    assert step.ledger.unfunded_rejections.tolist() == [1]
    assert torch.equal(step.state.reserve_q, state.reserve_q)


def test_birth_assignment_orders_parents_by_identity_and_destinations_by_slot() -> None:
    state = _state(
        alive=[True, True, True, False, False],
        stable_id=[30, 10, 20, 0, 0],
        structure_q=[2, 2, 2, 0, 0],
        reserve_q=[10, 10, 10, 0, 0],
        next_stable_id=40,
        generation=[4, 1, 2, 0, 0],
    )
    step = settle_lifecycle(
        state,
        _request(
            state,
            death=[False] * 5,
            birth=[True, True, True, False, False],
            child_structure_q=[2, 2, 2, 0, 0],
            child_reserve_q=[1, 1, 1, 0, 0],
        ),
    )

    assert step.ledger.accepted_parent.tolist() == [[False, True, True, False, False]]
    assert step.ledger.born.tolist() == [[False, False, False, True, True]]
    assert step.ledger.parent_slot_for_child.tolist() == [[-1, -1, -1, 1, 2]]
    assert step.state.stable_id.tolist() == [[30, 10, 20, 40, 41]]
    assert step.state.parent_id.tolist() == [[0, 0, 0, 10, 20]]
    assert step.state.generation.tolist() == [[4, 1, 2, 2, 3]]
    assert step.ledger.capacity_rejections.tolist() == [1]


def test_unfunded_and_slot_rejected_births_do_not_debit_parent() -> None:
    state = _state(
        alive=[True, True],
        stable_id=[1, 2],
        structure_q=[4, 4],
        reserve_q=[2, 20],
        next_stable_id=3,
    )
    step = settle_lifecycle(
        state,
        _request(
            state,
            death=[False, False],
            birth=[True, True],
            child_structure_q=[4, 4],
            child_reserve_q=[1, 1],
        ),
    )

    assert torch.equal(step.state.reserve_q, state.reserve_q)
    assert step.ledger.requested_births.tolist() == [2]
    assert step.ledger.accepted_births.tolist() == [0]
    assert step.ledger.unfunded_rejections.tolist() == [1]
    assert step.ledger.capacity_rejections.tolist() == [1]


def test_exhausted_id_space_refuses_birth_without_debit_or_overflow() -> None:
    max_i64 = torch.iinfo(torch.int64).max
    state = _state(
        alive=[True, False],
        stable_id=[max_i64 - 1, 0],
        structure_q=[4, 0],
        reserve_q=[20, 0],
        next_stable_id=max_i64,
    )
    request = _request(
        state,
        death=[False, False],
        birth=[True, False],
        child_structure_q=[4, 0],
        child_reserve_q=[1, 0],
    )

    step = settle_lifecycle(state, request)

    assert torch.equal(step.state.reserve_q, state.reserve_q)
    assert step.state.next_stable_id.tolist() == [max_i64]
    assert step.ledger.accepted_births.tolist() == [0]
    assert step.ledger.id_rejections.tolist() == [1]


def test_lifecycle_transaction_is_one_full_compiled_graph() -> None:
    state = _state(
        alive=[True, True, False, False],
        stable_id=[10, 20, 0, 0],
        structure_q=[5, 6, 0, 0],
        reserve_q=[20, 2, 0, 0],
        next_stable_id=30,
    )
    request = _request(
        state,
        death=[False, True, False, False],
        birth=[True, False, False, False],
        child_structure_q=[5, 0, 0, 0],
        child_reserve_q=[3, 0, 0, 0],
    )
    compiled = torch.compile(settle_lifecycle, fullgraph=True, dynamic=False)
    eager = settle_lifecycle(state, request)
    actual = compiled(state, request)

    assert torch.equal(actual.state.stable_id, eager.state.stable_id)
    assert torch.equal(actual.state.reserve_q, eager.state.reserve_q)
    assert torch.equal(actual.ledger.parent_slot_for_child, eager.ledger.parent_slot_for_child)


def test_randomized_batched_lifecycle_preserves_exact_creature_census() -> None:
    generator = torch.Generator().manual_seed(20260811)
    worlds, capacity = 4, 64
    alive = torch.rand((worlds, capacity), generator=generator) < 0.7
    slot = torch.arange(capacity, dtype=torch.int64)[None, :]
    world_offset = torch.arange(worlds, dtype=torch.int64)[:, None] * 1000
    stable_id = torch.where(alive, world_offset + slot + 1, 0)
    structure = torch.where(
        alive,
        torch.randint(1, 50, (worlds, capacity), generator=generator),
        0,
    )
    reserve = torch.where(
        alive,
        torch.randint(0, 200, (worlds, capacity), generator=generator),
        0,
    )
    state = PopulationState(
        alive=alive,
        stable_id=stable_id,
        parent_id=torch.zeros_like(stable_id),
        generation=torch.zeros_like(stable_id),
        born_at_s=torch.zeros((worlds, capacity), dtype=torch.float64),
        structure_q=structure,
        reserve_q=reserve,
        intake_carry_mol=torch.zeros((worlds, capacity), dtype=torch.float64),
        assimilation_carry_q=torch.zeros(
            (worlds, capacity), dtype=torch.float64
        ),
        maintenance_carry_j=torch.zeros(
            (worlds, capacity), dtype=torch.float64
        ),
        next_stable_id=world_offset[:, 0] + capacity + 1,
    )
    request = LifecycleRequest(
        death=alive & (torch.rand((worlds, capacity), generator=generator) < 0.15),
        birth=alive & (torch.rand((worlds, capacity), generator=generator) < 0.4),
        child_structure_q=structure,
        child_reserve_q=torch.where(alive, 7, 0),
        birth_release_energy_q=torch.zeros_like(structure),
        time_s=torch.full((worlds,), 12.5, dtype=torch.float64),
    )
    validate_population_state(state)
    validate_lifecycle_request(state, request)

    step = settle_lifecycle(state, request)

    before_q = state.structure_q.sum(dim=1) + state.reserve_q.sum(dim=1)
    after_q = step.state.structure_q.sum(dim=1) + step.state.reserve_q.sum(dim=1)
    returned_q = step.ledger.death_structure_return_q.sum(
        dim=1
    ) + step.ledger.death_reserve_return_q.sum(dim=1)
    assert torch.equal(before_q, after_q + returned_q)
    assert bool((step.state.reserve_q >= 0).all())
    assert torch.equal(step.ledger.born.sum(dim=1), step.ledger.accepted_births)
    validate_population_state(step.state)
