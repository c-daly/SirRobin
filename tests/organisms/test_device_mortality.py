from __future__ import annotations

from dataclasses import replace

import torch

from sirrobin.organisms.mortality import (
    MortalityConfig,
    lifespan_s,
    old_age_due,
)
from sirrobin.organisms.random import identity_word_u31
from sirrobin.organisms.state import PopulationState


def _state() -> PopulationState:
    alive = torch.tensor([[True, True, False], [True, False, False]])
    zeros_i64 = torch.zeros_like(alive, dtype=torch.int64)
    zeros_f64 = torch.zeros_like(alive, dtype=torch.float64)
    return PopulationState(
        alive=alive,
        stable_id=torch.tensor([[9, 3, 0], [9, 0, 0]], dtype=torch.int64),
        parent_id=zeros_i64,
        generation=zeros_i64,
        born_at_s=zeros_f64,
        structure_q=torch.where(alive, 4, 0).to(torch.int64),
        reserve_q=torch.where(alive, 8, 0).to(torch.int64),
        intake_carry_mol=zeros_f64,
        assimilation_carry_q=zeros_f64,
        maintenance_carry_j=zeros_f64,
        next_stable_id=torch.tensor([10, 10], dtype=torch.int64),
    )


def test_identity_words_ignore_slot_order_but_include_world_and_stream() -> None:
    stable_id = torch.tensor([[4, 9, 4]], dtype=torch.int64)
    same_world = torch.zeros_like(stable_id)
    words = identity_word_u31(stable_id, same_world, seed=77, stream=2)

    assert words[0, 0] == words[0, 2]
    assert words[0, 0] != words[0, 1]
    assert not torch.equal(
        words,
        identity_word_u31(stable_id, same_world + 1, seed=77, stream=2),
    )
    assert not torch.equal(
        words,
        identity_word_u31(stable_id, same_world, seed=77, stream=3),
    )


def test_lifespan_is_bounded_identity_state_and_inactive_slots_are_zero() -> None:
    state = _state()
    config = MortalityConfig(60.0, 100.0, seed=20260811)
    config.validate()

    first = lifespan_s(state, config)
    second = lifespan_s(state, config)

    assert torch.equal(first, second)
    assert bool((first[state.alive] >= 60.0).all())
    assert bool((first[state.alive] < 100.0).all())
    assert torch.count_nonzero(first[~state.alive]) == 0
    assert first[0, 0] != first[1, 0]


def test_old_age_due_uses_birth_time_and_never_marks_inactive_slots() -> None:
    state = replace(
        _state(),
        born_at_s=torch.tensor(
            [[10.0, 30.0, 0.0], [0.0, 0.0, 0.0]], dtype=torch.float64
        ),
    )
    config = MortalityConfig(50.0, 50.0, seed=0)

    due = old_age_due(state, torch.tensor([70.0, 49.0]), config)

    assert due.tolist() == [[True, False, False], [False, False, False]]


def test_disabled_age_mortality_never_invents_a_death_clock() -> None:
    state = _state()
    config = MortalityConfig(60.0, 100.0, seed=20260811, enabled=False)
    config.validate()

    due = old_age_due(state, torch.tensor([1.0e12, 1.0e12]), config)

    assert not bool(due.any())


def test_mortality_census_is_one_full_compiled_graph() -> None:
    state = _state()
    config = MortalityConfig(60.0, 100.0, seed=20260811)
    compiled = torch.compile(old_age_due, fullgraph=True, dynamic=False)

    eager = old_age_due(state, torch.tensor([80.0, 80.0]), config)
    actual = compiled(state, torch.tensor([80.0, 80.0]), config)

    assert torch.equal(actual, eager)
