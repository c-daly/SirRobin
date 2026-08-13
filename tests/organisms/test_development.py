from __future__ import annotations

from dataclasses import replace

import pytest
import torch

from sirrobin.organisms.development import (
    allocate_segment_structure_q,
    calibrate_development_config,
    initialize_development_state,
    target_structure_cost_q,
    validate_development_state,
)
from sirrobin.runtime.reference_adapter import living_state_from_reference
from tools.run_world import _build_fixture_world


def test_segment_allocation_is_exact_and_uses_stable_low_slot_ties() -> None:
    alive = torch.tensor([[True, True, False]])
    structure_q = torch.tensor([[7, 10, 0]], dtype=torch.int64)
    segment_mask = torch.tensor(
        [
            [
                [False, True, True, True],
                [False, True, True, False],
                [False, False, False, False],
            ]
        ]
    )
    mass_sim = torch.tensor(
        [
            [
                [0.0, 1.0, 1.0, 1.0],
                [0.0, 1.0, 3.0, 0.0],
                [0.0, 0.0, 0.0, 0.0],
            ]
        ],
        dtype=torch.float32,
    )

    actual = allocate_segment_structure_q(
        structure_q,
        mass_sim,
        segment_mask,
        alive,
    )

    assert actual.tolist() == [[[0, 3, 2, 2], [0, 3, 7, 0], [0, 0, 0, 0]]]
    assert torch.equal(actual.sum(dim=-1), structure_q)


def test_reference_initialization_partitions_existing_structure_without_mutation() -> None:
    world = _build_fixture_world(
        bodies=3,
        live_bodies=2,
        reserve_q_per_creature=500,
        device=torch.device("cpu"),
        economy_interval_s=0.1,
    )
    state = living_state_from_reference(world)
    genotype_axes_before = state.genotype.node_log_axes_flu_m.clone()
    body_axes_before = state.body.semi_axes_flu_m.clone()

    development = initialize_development_state(state.population, state.body)

    assert torch.equal(
        development.segment_structure_q.sum(dim=-1),
        state.population.structure_q,
    )
    assert torch.count_nonzero(
        development.segment_structure_q[~state.body.seg_mask]
    ).item() == 0
    assert torch.count_nonzero(
        development.segment_structure_q[~state.population.alive]
    ).item() == 0
    assert torch.equal(state.genotype.node_log_axes_flu_m, genotype_axes_before)
    assert torch.equal(state.body.semi_axes_flu_m, body_axes_before)
    validate_development_state(development, state.population, state.body)


def test_reference_adapter_installs_the_validated_development_authority() -> None:
    world = _build_fixture_world(
        bodies=3,
        live_bodies=2,
        reserve_q_per_creature=500,
        device=torch.device("cpu"),
        economy_interval_s=0.1,
    )

    state = living_state_from_reference(world)

    validate_development_state(
        state.development,
        state.population,
        state.body,
    )
    assert torch.equal(
        state.development.segment_structure_q.sum(dim=-1),
        state.population.structure_q,
    )


def test_reference_calibration_reprices_the_founder_body_exactly() -> None:
    world = _build_fixture_world(
        bodies=3,
        live_bodies=2,
        reserve_q_per_creature=500,
        device=torch.device("cpu"),
        economy_interval_s=0.1,
    )
    state = living_state_from_reference(world)
    config = calibrate_development_config(state.population, state.body)

    cost_q = target_structure_cost_q(state.body, config)

    assert torch.equal(cost_q, state.population.structure_q)


@pytest.mark.parametrize(
    ("replacement", "error", "message"),
    [
        (
            lambda value: value.to(torch.float64),
            TypeError,
            "segment_structure_q must be int64",
        ),
        (
            lambda value: torch.cat((value, value[..., :1]), dim=-1),
            ValueError,
            "segment_structure_q must match developed segment shape",
        ),
        (
            lambda value: value.index_put(
                (torch.tensor([0]), torch.tensor([2]), torch.tensor([0])),
                torch.tensor([1], dtype=torch.int64),
            ),
            ValueError,
            "inactive or undeveloped segment",
        ),
        (
            lambda value: value.index_put(
                (torch.tensor([0]), torch.tensor([0]), torch.tensor([1])),
                value[0, 0, 1:2] + 1,
            ),
            ValueError,
            "must exactly partition population structure",
        ),
    ],
)
def test_development_validation_rejects_malformed_controls(
    replacement,
    error: type[Exception],
    message: str,
) -> None:
    world = _build_fixture_world(
        bodies=3,
        live_bodies=2,
        reserve_q_per_creature=500,
        device=torch.device("cpu"),
        economy_interval_s=0.1,
    )
    state = living_state_from_reference(world)
    development = initialize_development_state(state.population, state.body)
    malformed = replace(
        development,
        segment_structure_q=replacement(development.segment_structure_q.clone()),
    )

    with pytest.raises(error, match=message):
        validate_development_state(malformed, state.population, state.body)


def test_segment_allocation_is_one_full_compiled_graph() -> None:
    alive = torch.tensor([[True, True]])
    structure_q = torch.tensor([[17, 13]], dtype=torch.int64)
    segment_mask = torch.tensor(
        [[[False, True, True, True], [False, True, True, False]]]
    )
    mass_sim = torch.tensor(
        [[[0.0, 2.0, 3.0, 5.0], [0.0, 7.0, 11.0, 0.0]]],
        dtype=torch.float32,
    )
    compiled = torch.compile(
        allocate_segment_structure_q,
        backend="eager",
        fullgraph=True,
        dynamic=False,
    )

    expected = allocate_segment_structure_q(
        structure_q, mass_sim, segment_mask, alive
    )
    actual = compiled(structure_q, mass_sim, segment_mask, alive)

    assert torch.equal(actual, expected)
    assert torch.equal(actual.sum(dim=-1), structure_q)
