from __future__ import annotations

import torch

from sirrobin.organisms.metabolism import MetabolismConfig, MetabolismInputs
from sirrobin.organisms.mortality import MortalityConfig
from sirrobin.organisms.state import PopulationState, validate_population_state
from sirrobin.runtime.organism_step import (
    OrganismIntervalInputs,
    advance_organism_interval,
)


def _fixture() -> tuple[PopulationState, OrganismIntervalInputs]:
    alive = torch.tensor([[True, True, False]])
    zeros_i64 = torch.zeros((1, 3), dtype=torch.int64)
    zeros_f64 = torch.zeros((1, 3), dtype=torch.float64)
    state = PopulationState(
        alive=alive,
        stable_id=torch.tensor([[1, 2, 0]], dtype=torch.int64),
        parent_id=zeros_i64,
        generation=zeros_i64,
        born_at_s=torch.tensor([[0.0, 4.0, 0.0]], dtype=torch.float64),
        structure_q=torch.tensor([[5, 5, 0]], dtype=torch.int64),
        reserve_q=torch.tensor([[10, 20, 0]], dtype=torch.int64),
        intake_carry_mol=zeros_f64,
        assimilation_carry_q=zeros_f64,
        maintenance_carry_j=zeros_f64,
        next_stable_id=torch.tensor([3], dtype=torch.int64),
    )
    metabolism = MetabolismInputs(
        structural_mass_kg=torch.ones((1, 3)),
        positive_actuator_work_j=torch.zeros((1, 3)),
        actuator_braking_work_j=torch.zeros((1, 3)),
        old_age_due=torch.zeros((1, 3), dtype=torch.bool),
        velocity_enu_m_s=torch.zeros((1, 3, 3)),
        yaw_momentum_kg_m2_s=torch.zeros((1, 3)),
        effective_mass_after_kg=torch.eye(3).expand(1, 3, 3, 3).clone(),
        yaw_inertia_after_kg_m2=torch.ones((1, 3)),
    )
    inputs = OrganismIntervalInputs(
        metabolism=metabolism,
        birth_requested=torch.tensor([[False, True, False]]),
        child_structure_q=torch.tensor([[5, 5, 0]], dtype=torch.int64),
        child_reserve_q=torch.tensor([[2, 2, 0]], dtype=torch.int64),
        birth_release_energy_q=torch.zeros((1, 3), dtype=torch.int64),
        time_s=torch.tensor([5.0], dtype=torch.float64),
    )
    return state, inputs


def _metabolism_config() -> MetabolismConfig:
    return MetabolismConfig(
        interval_s=1.0,
        maintenance_w_per_kg=0.0,
        chemical_to_mechanical_efficiency=1.0,
        reserve_j_per_q=100.0,
    )


def test_old_age_slot_can_be_reused_by_a_paid_surviving_parent() -> None:
    state, inputs = _fixture()
    step = advance_organism_interval(
        state,
        inputs,
        _metabolism_config(),
        MortalityConfig(5.0, 5.0, seed=0),
    )

    assert step.metabolism.ledger.old_age_due.tolist() == [[True, False, False]]
    assert step.lifecycle.ledger.died.tolist() == [[True, False, False]]
    assert step.lifecycle.ledger.born.tolist() == [[True, False, False]]
    assert step.lifecycle.ledger.parent_slot_for_child.tolist() == [[1, -1, -1]]
    assert step.state.stable_id.tolist() == [[3, 2, 0]]
    assert step.state.parent_id.tolist() == [[2, 0, 0]]
    assert step.state.generation.tolist() == [[1, 0, 0]]
    assert step.state.born_at_s.tolist() == [[5.0, 4.0, 0.0]]
    assert step.state.structure_q.tolist() == [[5, 5, 0]]
    assert step.state.reserve_q.tolist() == [[2, 13, 0]]

    before_q = state.structure_q.sum() + state.reserve_q.sum()
    after_q = step.state.structure_q.sum() + step.state.reserve_q.sum()
    returns_q = (
        step.metabolism.ledger.maintenance_return_q.sum()
        + step.lifecycle.ledger.death_structure_return_q.sum()
        + step.lifecycle.ledger.death_reserve_return_q.sum()
    )
    assert torch.equal(before_q, after_q + returns_q)
    validate_population_state(step.state)


def test_organism_interval_is_one_full_compiled_graph() -> None:
    state, inputs = _fixture()
    metabolism = _metabolism_config()
    mortality = MortalityConfig(5.0, 5.0, seed=0)
    compiled = torch.compile(
        advance_organism_interval,
        fullgraph=True,
        dynamic=False,
    )

    eager = advance_organism_interval(state, inputs, metabolism, mortality)
    actual = compiled(state, inputs, metabolism, mortality)

    assert torch.equal(actual.state.stable_id, eager.state.stable_id)
    assert torch.equal(actual.state.reserve_q, eager.state.reserve_q)
    assert torch.equal(actual.lifecycle.ledger.born, eager.lifecycle.ledger.born)
