"""Paid birth creates bounded, heritable locomotion variation and lineage."""

from __future__ import annotations

from dataclasses import fields

import pytest
import torch

from sirrobin.core.reproduction import (
    BirthConfig,
    ParametricMutationConfig,
    attempt_paid_birth,
)
from sirrobin.core.runner import HeadlessRunner
from tools.run_world import _build_fixture_world


def _world(*, capacity: int = 3, reserve_q: int = 4_000):
    return _build_fixture_world(
        bodies=capacity,
        live_bodies=1,
        reserve_q_per_creature=reserve_q,
        device=torch.device("cpu"),
        economy_interval_s=0.1,
    )


def _heritable_differences(world, parent_slot: int, child_slot: int):
    differences = []
    for field in fields(world.genotype):
        if field.name in {"alive", "stable_id"}:
            continue
        parent = getattr(world.genotype, field.name)[0, parent_slot]
        child = getattr(world.genotype, field.name)[0, child_slot]
        for index in (parent != child).nonzero(as_tuple=False).tolist():
            location = tuple(index)
            differences.append(
                (
                    field.name,
                    location,
                    float(parent[location]),
                    float(child[location]),
                )
            )
    return differences


@pytest.mark.parametrize(
    ("trait", "field_name", "developed_field"),
    [
        ("joint_amplitude", "node_joint_amp_rad", "joint_amp_rad"),
        ("swim_frequency", "swim_freq_hz", "swim_freq_hz"),
        ("swim_wave", "swim_wave_rad_per_depth", "phase_rad"),
    ],
)
def test_paid_birth_changes_exactly_one_consumed_locomotion_gene(
    trait: str,
    field_name: str,
    developed_field: str,
) -> None:
    world = _world()
    matter_before = world.matter_totals()
    parent_structure_q = int(world.creature_material.structure_q[0, 0])

    report = attempt_paid_birth(
        world,
        BirthConfig(initial_reserve_q=100),
        mutation_config=ParametricMutationConfig(seed=17, traits=(trait,)),
    )

    assert report.born is True
    assert report.child_slot is not None
    assert report.child_id is not None
    assert report.mutation is not None
    assert report.mutation.trait == trait
    assert report.mutation.field_name == field_name
    differences = _heritable_differences(world, 0, report.child_slot)
    assert differences == [
        (
            report.mutation.field_name,
            report.mutation.index,
            report.mutation.parent_value,
            report.mutation.child_value,
        )
    ]
    assert report.mutation.parent_value != report.mutation.child_value
    assert int(world.creature_material.structure_q[0, report.child_slot]) == (
        parent_structure_q
    )
    assert not torch.equal(
        getattr(world.body, developed_field)[0, 0],
        getattr(world.body, developed_field)[0, report.child_slot],
    )
    assert world.close_matter_step(matter_before).books_closed.tolist() == [True]

    lineage = world.lineage_record(0, report.child_id)
    assert lineage.creature_id == report.child_id
    assert lineage.parent_id == report.parent_id
    assert lineage.generation == 1
    assert lineage.born_at_s == pytest.approx(world.sim_time_s)
    assert lineage.mutation == report.mutation


def test_same_seed_and_identity_reproduce_the_same_mutation() -> None:
    first = _world()
    second = _world()
    config = ParametricMutationConfig(seed=1234)

    first_report = attempt_paid_birth(
        first,
        BirthConfig(initial_reserve_q=100),
        mutation_config=config,
    )
    second_report = attempt_paid_birth(
        second,
        BirthConfig(initial_reserve_q=100),
        mutation_config=config,
    )

    assert first_report.mutation == second_report.mutation
    assert first_report.child_slot is not None
    assert second_report.child_slot is not None
    assert _heritable_differences(first, 0, first_report.child_slot) == (
        _heritable_differences(second, 0, second_report.child_slot)
    )


def test_runner_applies_mutation_only_to_a_fully_paid_committed_birth() -> None:
    world = _world(capacity=2, reserve_q=2_000)
    runner = HeadlessRunner(
        world,
        birth_config=BirthConfig(initial_reserve_q=100),
        mutation_config=ParametricMutationConfig(
            seed=9,
            traits=("swim_frequency",),
        ),
    )

    tick = runner.advance()

    assert len(tick.births) == 1
    assert tick.births[0].born is True
    assert tick.births[0].mutation is not None
    assert tick.births[0].mutation.field_name == "swim_freq_hz"
    assert tick.matter.books_closed.tolist() == [True]


def test_unfunded_or_capacity_refused_birth_commits_no_mutation_or_lineage() -> None:
    unfunded = _world(reserve_q=1_099)
    full = _world(capacity=1)
    config = ParametricMutationConfig(seed=3)
    unfunded_before = unfunded.lineage_records
    full_before = full.lineage_records

    unfunded_report = attempt_paid_birth(
        unfunded,
        BirthConfig(initial_reserve_q=100),
        mutation_config=config,
    )
    full_report = attempt_paid_birth(
        full,
        BirthConfig(initial_reserve_q=100),
        mutation_config=config,
    )

    assert unfunded_report.born is False
    assert unfunded_report.mutation is None
    assert unfunded.lineage_records == unfunded_before
    assert full_report.born is False
    assert full_report.mutation is None
    assert full.lineage_records == full_before


def test_dead_ancestry_survives_slot_reuse_and_generation_advances() -> None:
    world = _world(capacity=2, reserve_q=2_000)
    config = ParametricMutationConfig(
        seed=11,
        traits=("swim_frequency",),
    )
    first = attempt_paid_birth(
        world,
        BirthConfig(initial_reserve_q=100),
        mutation_config=config,
    )
    assert first.child_slot == 1
    assert first.child_id == 2

    returned_q = int(
        world.creature_material.structure_q[0, 0]
        + world.creature_material.reserve_q[0, 0]
    )
    world.economy_state.nd_q[0, 0, 0, 0] += returned_q
    world.creature_material.structure_q[0, 0] = 0
    world.creature_material.reserve_q[0, 0] = 0
    for carry in world.creature_material.carries:
        carry[0, 0] = 0.0
    world.genotype.alive[0, 0] = False
    world.rebuild_body()
    world.economy_state.nd_q[0, 0, 0, 0] -= 2_000
    world.creature_material.reserve_q[0, 1] += 2_000

    second = attempt_paid_birth(
        world,
        BirthConfig(initial_reserve_q=100),
        parent_slot=1,
        mutation_config=config,
    )

    assert second.child_slot == 0
    assert second.child_id == 3
    assert world.lineage_record(0, 1).parent_id is None
    assert world.lineage_record(0, 2).parent_id == 1
    grandchild = world.lineage_record(0, 3)
    assert grandchild.parent_id == 2
    assert grandchild.generation == 2
    assert [record.creature_id for record in world.lineage_records] == [1, 2, 3]


@pytest.mark.parametrize(
    "kwargs",
    [
        {"seed": True},
        {"seed": -1},
        {"seed": 1, "traits": ()},
        {"seed": 1, "traits": ("unknown",)},
        {"seed": 1, "traits": ("swim_wave", "swim_wave")},
    ],
)
def test_mutation_config_rejects_invalid_domains(kwargs) -> None:
    error = TypeError if kwargs.get("seed") is True else ValueError
    with pytest.raises(error):
        ParametricMutationConfig(**kwargs)
