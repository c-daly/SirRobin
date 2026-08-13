from __future__ import annotations

from dataclasses import fields

import pytest
import torch

from sirrobin.genetics.develop import develop, develop_unchecked
from sirrobin.organisms.body_cache import inherit_developed_births
from sirrobin.organisms.lifecycle import LifecycleRequest, settle_lifecycle
from sirrobin.organisms.mutation import (
    JOINT_AMPLITUDE,
    SWIM_FREQUENCY,
    SWIM_WAVE,
    MutationConfig,
    mutate_committed_births,
)
from sirrobin.organisms.state import PopulationState
from tools.run_world import _build_fixture_world


def _paid_birth():
    world = _build_fixture_world(
        bodies=3,
        live_bodies=1,
        reserve_q_per_creature=4_000,
        device=torch.device("cpu"),
        economy_interval_s=0.1,
    )
    alive = world.genotype.alive
    zeros_i64 = torch.zeros_like(alive, dtype=torch.int64)
    zeros_f64 = torch.zeros_like(alive, dtype=torch.float64)
    state = PopulationState(
        alive=alive,
        stable_id=world.genotype.stable_id,
        parent_id=zeros_i64,
        generation=zeros_i64,
        born_at_s=zeros_f64,
        structure_q=world.creature_material.structure_q,
        reserve_q=world.creature_material.reserve_q,
        intake_carry_mol=zeros_f64,
        assimilation_carry_q=zeros_f64,
        maintenance_carry_j=zeros_f64,
        next_stable_id=torch.tensor([2], dtype=torch.int64),
    )
    lifecycle = settle_lifecycle(
        state,
        LifecycleRequest(
            death=torch.zeros_like(alive),
            birth=torch.tensor([[True, False, False]]),
            child_structure_q=state.structure_q,
            child_reserve_q=torch.tensor([[100, 0, 0]], dtype=torch.int64),
            time_s=torch.tensor([1.0], dtype=torch.float64),
        ),
    )
    return world.genotype, lifecycle


def _differences(genotype, parent_slot: int, child_slot: int):
    differences = []
    for field in fields(genotype):
        if field.name in {"alive", "stable_id"}:
            continue
        parent = getattr(genotype, field.name)[0, parent_slot]
        child = getattr(genotype, field.name)[0, child_slot]
        for index in (parent != child).nonzero(as_tuple=False).tolist():
            differences.append((field.name, tuple(index)))
    return differences


@pytest.mark.parametrize(
    ("config", "code", "field_name"),
    [
        (
            MutationConfig(
                seed=17,
                joint_amplitude=True,
                swim_frequency=False,
                swim_wave=False,
            ),
            JOINT_AMPLITUDE,
            "node_joint_amp_rad",
        ),
        (
            MutationConfig(
                seed=17,
                joint_amplitude=False,
                swim_frequency=True,
                swim_wave=False,
            ),
            SWIM_FREQUENCY,
            "swim_freq_hz",
        ),
        (
            MutationConfig(
                seed=17,
                joint_amplitude=False,
                swim_frequency=False,
                swim_wave=True,
            ),
            SWIM_WAVE,
            "swim_wave_rad_per_depth",
        ),
    ],
)
def test_paid_birth_clones_then_mutates_exactly_one_enabled_locus(
    config: MutationConfig,
    code: int,
    field_name: str,
) -> None:
    genotype, lifecycle = _paid_birth()
    config.validate()

    step = mutate_committed_births(
        genotype,
        lifecycle.state,
        lifecycle.ledger,
        config,
    )

    child_slot = int(lifecycle.ledger.born[0].nonzero()[0])
    assert step.ledger.mutated.tolist() == [[False, True, False]]
    assert step.ledger.trait_code[0, child_slot].item() == code
    assert step.ledger.parent_value[0, child_slot] != step.ledger.child_value[0, child_slot]
    assert _differences(step.genotype, 0, child_slot) == [
        (
            field_name,
            (() if code != JOINT_AMPLITUDE else (int(step.ledger.locus[0, child_slot]),)),
        )
    ]
    inherited = inherit_developed_births(
        develop(genotype),
        step.genotype,
        lifecycle.state,
        lifecycle.ledger,
    )
    rebuilt = develop(step.genotype)
    for body_field in fields(rebuilt):
        assert torch.equal(
            getattr(inherited, body_field.name)[0, child_slot],
            getattr(rebuilt, body_field.name)[0, child_slot],
        )
    step.genotype.validate()


def test_same_identity_and_seed_are_independent_of_inactive_slot_contents() -> None:
    first_genotype, first_lifecycle = _paid_birth()
    second_genotype, second_lifecycle = _paid_birth()
    second_genotype.swim_freq_hz[0, 2] = 9.0
    config = MutationConfig(seed=1234)

    first = mutate_committed_births(
        first_genotype, first_lifecycle.state, first_lifecycle.ledger, config
    )
    second = mutate_committed_births(
        second_genotype, second_lifecycle.state, second_lifecycle.ledger, config
    )

    assert torch.equal(first.ledger.trait_code, second.ledger.trait_code)
    assert torch.equal(first.ledger.locus, second.ledger.locus)
    assert torch.equal(first.ledger.child_value, second.ledger.child_value)


def test_paid_frequency_mutation_changes_the_developed_child_capability() -> None:
    genotype, lifecycle = _paid_birth()
    step = mutate_committed_births(
        genotype,
        lifecycle.state,
        lifecycle.ledger,
        MutationConfig(
            seed=4,
            joint_amplitude=False,
            swim_frequency=True,
            swim_wave=False,
        ),
    )
    body = develop(step.genotype)
    child_slot = int(lifecycle.ledger.born[0].nonzero()[0])
    parent_slot = int(lifecycle.ledger.parent_slot_for_child[0, child_slot])

    assert step.ledger.trait_code[0, child_slot].item() == SWIM_FREQUENCY
    assert body.swim_freq_hz[0, child_slot] != body.swim_freq_hz[0, parent_slot]
    assert body.swim_freq_hz[0, child_slot] == step.genotype.swim_freq_hz[0, child_slot]
    assert torch.equal(body.alive, lifecycle.state.alive)
    assert torch.equal(body.stable_id, lifecycle.state.stable_id)


def test_device_development_has_a_host_read_free_full_graph() -> None:
    genotype, lifecycle = _paid_birth()
    step = mutate_committed_births(
        genotype,
        lifecycle.state,
        lifecycle.ledger,
        MutationConfig(
            seed=4,
            joint_amplitude=False,
            swim_frequency=True,
            swim_wave=False,
        ),
    )
    compiled = torch.compile(
        develop_unchecked,
        backend="eager",
        fullgraph=True,
        dynamic=False,
    )

    expected = develop(step.genotype)
    actual = compiled(step.genotype)

    for field in fields(expected):
        assert torch.equal(getattr(actual, field.name), getattr(expected, field.name))


def test_device_mutation_is_one_full_compiled_graph() -> None:
    genotype, lifecycle = _paid_birth()
    config = MutationConfig(
        seed=4,
        joint_amplitude=False,
        swim_frequency=True,
        swim_wave=False,
    )
    compiled = torch.compile(mutate_committed_births, fullgraph=True, dynamic=False)

    eager = mutate_committed_births(
        genotype, lifecycle.state, lifecycle.ledger, config
    )
    actual = compiled(genotype, lifecycle.state, lifecycle.ledger, config)

    assert torch.equal(actual.genotype.swim_freq_hz, eager.genotype.swim_freq_hz)
    assert torch.equal(actual.ledger.trait_code, eager.ledger.trait_code)
