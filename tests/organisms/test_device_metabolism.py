from __future__ import annotations

from dataclasses import replace

import torch

from sirrobin.organisms.lifecycle import LifecycleRequest, settle_lifecycle
from sirrobin.organisms.metabolism import (
    MetabolismConfig,
    MetabolismInputs,
    available_actuator_work_j,
    settle_metabolism,
)
from sirrobin.organisms.state import PopulationState, validate_population_state


def _state(*, reserve_q: list[int], structure_q: list[int]) -> PopulationState:
    capacity = len(reserve_q)
    alive = torch.ones((1, capacity), dtype=torch.bool)
    zeros_i64 = torch.zeros((1, capacity), dtype=torch.int64)
    zeros_f64 = torch.zeros((1, capacity), dtype=torch.float64)
    return PopulationState(
        alive=alive,
        stable_id=torch.arange(1, capacity + 1, dtype=torch.int64)[None, :],
        parent_id=zeros_i64,
        generation=zeros_i64,
        born_at_s=zeros_f64,
        structure_q=torch.tensor([structure_q], dtype=torch.int64),
        reserve_q=torch.tensor([reserve_q], dtype=torch.int64),
        intake_carry_mol=zeros_f64,
        assimilation_carry_q=zeros_f64,
        maintenance_carry_j=zeros_f64,
        next_stable_id=torch.tensor([capacity + 1], dtype=torch.int64),
    )


def _inputs(
    state: PopulationState,
    *,
    mass_kg: float,
    positive_work_j: float = 0.0,
    braking_work_j: float = 0.0,
    old_age: bool = False,
) -> MetabolismInputs:
    shape = state.alive.shape
    return MetabolismInputs(
        structural_mass_kg=torch.full(shape, mass_kg),
        positive_actuator_work_j=torch.full(shape, positive_work_j),
        actuator_braking_work_j=torch.full(shape, braking_work_j),
        old_age_due=torch.full(shape, old_age, dtype=torch.bool),
        velocity_enu_m_s=torch.zeros((*shape, 3)),
        yaw_momentum_kg_m2_s=torch.zeros(shape),
        effective_mass_after_kg=torch.eye(3).expand(*shape, 3, 3).clone(),
        yaw_inertia_after_kg_m2=torch.ones(shape),
    )


def _config(**overrides: float) -> MetabolismConfig:
    values = {
        "interval_s": 1.0,
        "maintenance_w_per_kg": 10.0,
        "chemical_to_mechanical_efficiency": 0.5,
        "reserve_j_per_q": 100.0,
    }
    values.update(overrides)
    config = MetabolismConfig(**values)
    config.validate()
    return config


def test_available_work_reserves_baseline_and_prior_fractional_liability() -> None:
    state = replace(
        _state(reserve_q=[3], structure_q=[5]),
        maintenance_carry_j=torch.tensor([[20.0]], dtype=torch.float64),
    )

    budget = available_actuator_work_j(
        state,
        torch.tensor([[2.0]]),
        _config(),
    )

    # 300 J stored - 20 J baseline - 20 J carry, at 50% efficiency.
    assert budget.item() == 130.0


def test_batched_metabolism_quantizes_demand_and_names_heat_channels() -> None:
    state = replace(
        _state(reserve_q=[10], structure_q=[5]),
        maintenance_carry_j=torch.tensor([[20.0]], dtype=torch.float64),
    )
    step = settle_metabolism(
        state,
        _inputs(
            state,
            mass_kg=1.0,
            positive_work_j=50.0,
            braking_work_j=7.0,
        ),
        _config(),
    )

    # 10 J baseline + 100 J locomotion demand + 20 J carry = 1 q + 30 J.
    assert step.ledger.requested_q.tolist() == [[1]]
    assert step.ledger.reserve_debit_q.tolist() == [[1]]
    assert step.state.reserve_q.tolist() == [[9]]
    assert step.state.maintenance_carry_j.tolist() == [[30.0]]
    assert step.ledger.maintenance_return_q.tolist() == [[1]]
    assert step.ledger.maintenance_heat_j.tolist() == [[100.0]]
    assert step.ledger.muscle_inefficiency_heat_j.tolist() == [[50.0]]
    assert step.ledger.actuator_braking_heat_j.tolist() == [[7.0]]
    assert step.ledger.quantization_residual_j.abs().max().item() == 0.0
    assert not bool(step.ledger.death.any())


def test_starvation_then_lifecycle_return_preserves_exact_matter() -> None:
    state = _state(reserve_q=[1], structure_q=[5])
    metabolism = settle_metabolism(
        state,
        _inputs(state, mass_kg=30.0),
        _config(),
    )
    assert metabolism.ledger.requested_q.tolist() == [[3]]
    assert metabolism.ledger.reserve_debit_q.tolist() == [[1]]
    assert metabolism.ledger.starved.tolist() == [[True]]

    lifecycle = settle_lifecycle(
        metabolism.state,
        LifecycleRequest(
            death=metabolism.ledger.death,
            birth=torch.zeros_like(state.alive),
            child_structure_q=torch.zeros_like(state.structure_q),
            child_reserve_q=torch.zeros_like(state.reserve_q),
            birth_release_energy_q=torch.zeros_like(state.reserve_q),
            time_s=torch.tensor([1.0], dtype=torch.float64),
        ),
    )
    before_q = state.structure_q.sum() + state.reserve_q.sum()
    after_q = lifecycle.state.structure_q.sum() + lifecycle.state.reserve_q.sum()
    returned_q = (
        metabolism.ledger.maintenance_return_q.sum()
        + lifecycle.ledger.death_structure_return_q.sum()
        + lifecycle.ledger.death_reserve_return_q.sum()
    )
    assert torch.equal(before_q, after_q + returned_q)
    assert lifecycle.state.alive.tolist() == [[False]]
    validate_population_state(lifecycle.state)


def test_old_age_dissipates_motion_and_fractional_assimilation_energy() -> None:
    state = replace(
        _state(reserve_q=[4], structure_q=[5]),
        assimilation_carry_q=torch.tensor([[0.25]], dtype=torch.float64),
    )
    inputs = _inputs(state, mass_kg=1.0, old_age=True)
    inputs.velocity_enu_m_s[0, 0, 0] = 2.0
    inputs.yaw_momentum_kg_m2_s[0, 0] = 4.0
    inputs.effective_mass_after_kg[0, 0] = 2.0 * torch.eye(3)
    inputs.yaw_inertia_after_kg_m2[0, 0] = 4.0

    step = settle_metabolism(
        state,
        inputs,
        _config(maintenance_w_per_kg=0.0),
    )

    # Linear KE 4 J + rotational KE 2 J + fractional carry energy 25 J.
    assert step.ledger.death.tolist() == [[True]]
    assert step.ledger.death_dissipation_j.tolist() == [[31.0]]
    assert not bool(step.ledger.invalid_death_kinetics.any())


def test_metabolism_is_one_full_compiled_graph() -> None:
    state = _state(reserve_q=[10, 1], structure_q=[5, 5])
    inputs = _inputs(state, mass_kg=2.0, positive_work_j=20.0)
    config = _config()
    compiled = torch.compile(settle_metabolism, fullgraph=True, dynamic=False)

    eager = settle_metabolism(state, inputs, config)
    actual = compiled(state, inputs, config)

    assert torch.equal(actual.state.reserve_q, eager.state.reserve_q)
    assert torch.equal(actual.ledger.death, eager.ledger.death)
    assert torch.allclose(
        actual.ledger.death_dissipation_j,
        eager.ledger.death_dissipation_j,
    )
