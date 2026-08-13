from __future__ import annotations

from dataclasses import fields, replace

import torch

from sirrobin.core.live_world import initialize_live_state
from sirrobin.physics.ecological_motion import (
    advance_affordable_motion,
    advance_requested_motion_with_stage,
    select_affordable_effort,
)
from sirrobin.physics.phase_response import (
    PhaseWindowConfig,
    advance_phase_stage,
    advance_phase_window,
)
from tools.run_world import _build_fixture_world


def test_affordable_effort_is_selected_without_assuming_monotone_work() -> None:
    options = torch.tensor(
        [[[0.0, 0.25, 0.5, 1.0], [0.0, 0.25, 0.5, 1.0]]]
    )
    work = torch.tensor(
        [[[0.0, 3.0, 2.0, 8.0], [0.0, 1.0, 3.0, 5.0]]]
    )
    budget = torch.tensor([[2.5, 20.0]])
    alive = torch.tensor([[True, False]])

    selected = select_affordable_effort(options, work, budget, alive)

    assert selected.effort_fraction.tolist() == [[0.5, 0.0]]
    assert selected.actuator_work_j.tolist() == [[2.0, 0.0]]
    assert selected.option_index.tolist() == [[2, 0]]
    assert selected.requested_was_funded.tolist() == [[False, False]]


def test_requested_effort_is_retained_when_its_work_is_funded() -> None:
    options = torch.tensor([[[0.0, 0.5, 1.0]]])
    work = torch.tensor([[[0.0, 2.0, 7.0]]])

    selected = select_affordable_effort(
        options,
        work,
        torch.tensor([[7.0]]),
        torch.tensor([[True]]),
    )

    assert selected.effort_fraction.item() == 1.0
    assert selected.actuator_work_j.item() == 7.0
    assert selected.requested_was_funded.item() is True


def test_no_affordable_option_cannot_create_an_unfunded_debit() -> None:
    selected = select_affordable_effort(
        torch.tensor([[[0.25, 1.0]]]),
        torch.tensor([[[2.0, 7.0]]]),
        torch.tensor([[1.0]]),
        torch.tensor([[True]]),
    )

    assert selected.effort_fraction.item() == 0.0
    assert selected.actuator_work_j.item() == 0.0
    assert selected.requested_was_funded.item() is False


def test_malformed_work_option_is_not_selected_as_affordable() -> None:
    selected = select_affordable_effort(
        torch.tensor([[[0.0, 0.5, 1.0]]]),
        torch.tensor([[[0.0, -1.0, float("nan")]]]),
        torch.tensor([[10.0]]),
        torch.tensor([[True]]),
    )

    assert selected.effort_fraction.item() == 0.0
    assert selected.actuator_work_j.item() == 0.0
    assert selected.option_index.item() == 0
    assert selected.requested_was_funded.item() is False


def test_batched_affordability_returns_the_selected_physical_trajectory() -> None:
    world = _build_fixture_world(
        bodies=1,
        live_bodies=1,
        device=torch.device("cpu"),
        economy_interval_s=0.1,
        physics_dtype=torch.float32,
    )
    response_config = PhaseWindowConfig(0.1, stages=4, phase_samples=2)
    requested = torch.ones_like(world.body.alive, dtype=torch.float32)

    for budget, expected_effort in ((0.0, 0.0), (1.0e6, 1.0)):
        state = initialize_live_state(world.body)
        actual = advance_affordable_motion(
            world.body,
            state,
            world.fluid,
            world.live_config,
            world.geometry,
            response_config,
            requested_effort=requested,
            budget_j=torch.full_like(requested, budget),
        )
        reference = advance_phase_window(
            world.body,
            initialize_live_state(world.body),
            world.fluid,
            world.live_config,
            world.geometry,
            response_config,
            effort_fraction=torch.full_like(requested, expected_effort),
        )

        assert actual.ledger.selected.effort_fraction.item() == expected_effort
        assert actual.ledger.option_work_j[..., 0].item() == 0.0
        for field in fields(actual.state):
            assert torch.allclose(
                getattr(actual.state, field.name),
                getattr(reference.state, field.name),
                rtol=1.0e-6,
                atol=1.0e-7,
            )
        assert torch.allclose(
            actual.ledger.response.positive_actuator_work_j,
            reference.ledger.positive_actuator_work_j,
        )


def test_requested_only_motion_is_exact_when_funded_and_flags_exact_fallback() -> None:
    world = _build_fixture_world(
        bodies=1,
        live_bodies=1,
        device=torch.device("cpu"),
        economy_interval_s=0.1,
        physics_dtype=torch.float32,
    )
    response_config = PhaseWindowConfig(0.1, stages=4, phase_samples=2)
    requested = torch.ones_like(world.body.alive, dtype=torch.float32)
    funded = advance_requested_motion_with_stage(
        world.body,
        initialize_live_state(world.body),
        world.fluid,
        world.live_config,
        world.geometry,
        response_config,
        requested_effort=requested,
        budget_j=torch.full_like(requested, 1.0e6),
        stage_kernel=advance_phase_stage,
    )
    refused = advance_requested_motion_with_stage(
        world.body,
        initialize_live_state(world.body),
        world.fluid,
        world.live_config,
        world.geometry,
        response_config,
        requested_effort=requested,
        budget_j=torch.zeros_like(requested),
        stage_kernel=advance_phase_stage,
    )
    direct = advance_phase_window(
        world.body,
        initialize_live_state(world.body),
        world.fluid,
        world.live_config,
        world.geometry,
        response_config,
        effort_fraction=requested,
    )

    assert funded.ledger.funding_resolved.tolist() == [[True]]
    assert torch.equal(funded.state.position_enu_m, direct.state.position_enu_m)
    assert torch.equal(
        funded.ledger.response.positive_actuator_work_j,
        direct.ledger.positive_actuator_work_j,
    )
    assert refused.ledger.funding_resolved.tolist() == [[False]]


def test_zero_effort_passive_power_is_not_charged_as_actuator_work() -> None:
    world = _build_fixture_world(
        bodies=1,
        live_bodies=1,
        device=torch.device("cpu"),
        economy_interval_s=0.1,
        physics_dtype=torch.float32,
    )
    requested = torch.zeros_like(world.body.alive, dtype=torch.float32)

    def stage_with_passive_input(*args, **kwargs):
        stage = advance_phase_stage(*args, **kwargs)
        return replace(
            stage,
            ledger=replace(
                stage.ledger,
                positive_actuator_work_j=torch.full_like(
                    stage.ledger.positive_actuator_work_j, 7.0
                ),
                actuator_braking_work_j=torch.full_like(
                    stage.ledger.actuator_braking_work_j, 3.0
                ),
            ),
        )

    advance = advance_requested_motion_with_stage(
        world.body,
        initialize_live_state(world.body),
        world.fluid,
        world.live_config,
        world.geometry,
        PhaseWindowConfig(0.1, stages=1, phase_samples=2),
        requested_effort=requested,
        budget_j=torch.zeros_like(requested),
        stage_kernel=stage_with_passive_input,
    )

    assert advance.ledger.funding_resolved.tolist() == [[True]]
    assert advance.ledger.option_work_j.tolist() == [[[0.0]]]
    assert advance.ledger.response.positive_actuator_work_j.tolist() == [[0.0]]
    assert advance.ledger.response.actuator_braking_work_j.tolist() == [[0.0]]
