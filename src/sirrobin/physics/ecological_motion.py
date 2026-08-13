"""Device contract shared by ecological motion response implementations."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, fields, replace

import torch

from sirrobin.fields.geometry import GridGeometry
from sirrobin.physics.contracts import DevelopedBody, FluidSample, LiveState
from sirrobin.physics.live_config import LiveLocomotionConfig
from sirrobin.physics.phase_response import (
    PhaseWindowAdvance,
    PhaseWindowConfig,
    PhaseWindowLedger,
    advance_phase_stage,
    advance_phase_window_with_stage,
)


@dataclass(frozen=True, slots=True)
class AffordableEffort:
    """The affordable response option selected independently for every slot."""

    effort_fraction: torch.Tensor
    actuator_work_j: torch.Tensor
    option_index: torch.Tensor
    requested_was_funded: torch.Tensor


@dataclass(frozen=True, slots=True)
class AffordableMotionLedger:
    """Selected response plus diagnostics for the evaluated effort set."""

    selected: AffordableEffort
    response: PhaseWindowLedger
    option_work_j: torch.Tensor
    option_nonfinite: torch.Tensor
    funding_resolved: torch.Tensor


@dataclass(frozen=True, slots=True)
class AffordableMotionAdvance:
    state: LiveState
    ledger: AffordableMotionLedger


def select_affordable_effort(
    effort_options: torch.Tensor,
    actuator_work_options_j: torch.Tensor,
    budget_j: torch.Tensor,
    alive: torch.Tensor,
) -> AffordableEffort:
    """Choose the greatest affordable pre-evaluated effort on the device.

    Inputs have shape ``[world, slot, option]`` except budget/alive, which have
    shape ``[world, slot]``. The response generator supplies finite options in
    ``[0, 1]`` and includes a zero-effort/zero-actuator-work option. Options need
    not be ordered and work need not be assumed monotone.

    The final option is the requested effort. Returning its funding flag makes
    the old simulate/restore/retry control path unnecessary while preserving a
    named distinction between requested and accepted actuation.
    """

    valid_option = (
        torch.isfinite(effort_options)
        & (effort_options >= 0.0)
        & (effort_options <= 1.0)
        & torch.isfinite(actuator_work_options_j)
        & (actuator_work_options_j >= 0.0)
    )
    valid_budget = torch.isfinite(budget_j) & (budget_j >= 0.0)
    selectable = alive[..., None] & valid_budget[..., None] & valid_option & (
        actuator_work_options_j <= budget_j[..., None]
    )
    selectable_effort = torch.where(
        selectable, effort_options, torch.full_like(effort_options, -1.0)
    )
    selected_effort, selected_index = selectable_effort.max(dim=-1)
    selected_work = torch.gather(
        actuator_work_options_j, -1, selected_index[..., None]
    ).squeeze(-1)
    has_affordable_option = selectable.any(dim=-1)
    selected_effort = torch.where(
        alive & has_affordable_option, selected_effort.clamp_min(0.0), 0.0
    )
    selected_work = torch.where(alive & has_affordable_option, selected_work, 0.0)
    requested_was_funded = alive & selectable[..., -1]
    return AffordableEffort(
        selected_effort,
        selected_work,
        selected_index,
        requested_was_funded,
    )


def requested_effort_options(requested_effort: torch.Tensor) -> torch.Tensor:
    """Return the fixed ordinary-lane effort basis, including zero and request."""

    fractions = torch.tensor(
        (0.0, 0.25, 0.5, 0.75, 1.0),
        dtype=requested_effort.dtype,
        device=requested_effort.device,
    )
    return requested_effort[..., None] * fractions


def _expand_options(value: torch.Tensor, option_count: int) -> torch.Tensor:
    worlds, capacity = value.shape[:2]
    tail = value.shape[2:]
    return value[:, :, None].expand(
        worlds, capacity, option_count, *tail
    ).reshape(worlds, capacity * option_count, *tail)


def _select_option(
    value: torch.Tensor,
    option_index: torch.Tensor,
    option_count: int,
) -> torch.Tensor:
    worlds, capacity = option_index.shape
    tail = value.shape[2:]
    choices = value.reshape(worlds, capacity, option_count, *tail)
    index = option_index.reshape(
        worlds, capacity, 1, *((1,) * len(tail))
    ).expand(worlds, capacity, 1, *tail)
    return torch.gather(choices, 2, index).squeeze(2)


def _expanded_body(body: DevelopedBody, option_count: int) -> DevelopedBody:
    return DevelopedBody(
        **{
            field.name: _expand_options(getattr(body, field.name), option_count)
            for field in fields(body)
        }
    )


def _expanded_state(state: LiveState, option_count: int) -> LiveState:
    return LiveState(
        **{
            field.name: _expand_options(getattr(state, field.name), option_count)
            for field in fields(state)
        }
    )


def _expanded_fluid(fluid: FluidSample, option_count: int) -> FluidSample:
    return FluidSample(
        **{
            field.name: _expand_options(getattr(fluid, field.name), option_count)
            for field in fields(fluid)
        }
    )


def _select_advance(
    advance: PhaseWindowAdvance,
    selected: AffordableEffort,
    option_count: int,
) -> PhaseWindowAdvance:
    state = LiveState(
        **{
            field.name: _select_option(
                getattr(advance.state, field.name),
                selected.option_index,
                option_count,
            )
            for field in fields(advance.state)
        }
    )
    ledger = PhaseWindowLedger(
        **{
            field.name: _select_option(
                getattr(advance.ledger, field.name),
                selected.option_index,
                option_count,
            )
            for field in fields(advance.ledger)
        }
    )
    return PhaseWindowAdvance(state, ledger)


def _settle_selected_actuation(
    advance: PhaseWindowAdvance,
    effort_fraction: torch.Tensor,
) -> PhaseWindowAdvance:
    """Name actuator work only where the organism actually actuated.

    Canonical hydrodynamic input power may be positive for a passively moving,
    zero-gait body. It remains part of the physical response, but the organism did
    not supply muscle work. This matches the preserved reference settlement.
    """

    actuating = effort_fraction > 0.0
    return PhaseWindowAdvance(
        advance.state,
        replace(
            advance.ledger,
            positive_actuator_work_j=torch.where(
                actuating,
                advance.ledger.positive_actuator_work_j,
                0.0,
            ),
            actuator_braking_work_j=torch.where(
                actuating,
                advance.ledger.actuator_braking_work_j,
                0.0,
            ),
        ),
    )


def advance_affordable_motion(
    body: DevelopedBody,
    state: LiveState,
    fluid: FluidSample,
    live_config: LiveLocomotionConfig,
    geometry: GridGeometry,
    response_config: PhaseWindowConfig,
    *,
    requested_effort: torch.Tensor,
    budget_j: torch.Tensor,
) -> AffordableMotionAdvance:
    """Evaluate fixed effort candidates together and select on the device."""

    return advance_affordable_motion_with_stage(
        body,
        state,
        fluid,
        live_config,
        geometry,
        response_config,
        requested_effort=requested_effort,
        budget_j=budget_j,
        stage_kernel=advance_phase_stage,
    )


def advance_affordable_motion_with_stage(
    body: DevelopedBody,
    state: LiveState,
    fluid: FluidSample,
    live_config: LiveLocomotionConfig,
    geometry: GridGeometry,
    response_config: PhaseWindowConfig,
    *,
    requested_effort: torch.Tensor,
    budget_j: torch.Tensor,
    stage_kernel: Callable[..., PhaseWindowAdvance],
) -> AffordableMotionAdvance:
    """Evaluate effort candidates through a supplied cohesive stage kernel."""

    options = requested_effort_options(requested_effort)
    option_count = options.shape[-1]
    evaluated = advance_phase_window_with_stage(
        _expanded_body(body, option_count),
        _expanded_state(state, option_count),
        _expanded_fluid(fluid, option_count),
        live_config,
        geometry,
        response_config,
        effort_fraction=options.reshape(options.shape[0], -1),
        stage_kernel=stage_kernel,
    )
    raw_option_work = evaluated.ledger.positive_actuator_work_j.reshape_as(options)
    option_work = torch.where(options > 0.0, raw_option_work, 0.0)
    option_nonfinite = evaluated.ledger.nonfinite.reshape_as(options)
    selected = select_affordable_effort(
        options,
        torch.where(option_nonfinite, torch.nan, option_work),
        budget_j,
        body.alive,
    )
    selected_advance = _settle_selected_actuation(
        _select_advance(evaluated, selected, option_count),
        selected.effort_fraction,
    )
    return AffordableMotionAdvance(
        selected_advance.state,
        AffordableMotionLedger(
            selected,
            selected_advance.ledger,
            option_work,
            option_nonfinite,
            ~body.alive
            | (
                ~option_nonfinite
                & torch.isfinite(option_work)
                & (option_work >= 0.0)
                & (option_work <= budget_j[..., None])
            ).any(dim=-1),
        ),
    )


def advance_requested_motion_with_stage(
    body: DevelopedBody,
    state: LiveState,
    fluid: FluidSample,
    live_config: LiveLocomotionConfig,
    geometry: GridGeometry,
    response_config: PhaseWindowConfig,
    *,
    requested_effort: torch.Tensor,
    budget_j: torch.Tensor,
    stage_kernel: Callable[..., PhaseWindowAdvance],
) -> AffordableMotionAdvance:
    """Evaluate only the requested trajectory for speculative chunk execution."""

    evaluated = advance_phase_window_with_stage(
        body,
        state,
        fluid,
        live_config,
        geometry,
        response_config,
        effort_fraction=requested_effort,
        stage_kernel=stage_kernel,
    )
    option_work = torch.where(
        requested_effort[..., None] > 0.0,
        evaluated.ledger.positive_actuator_work_j[..., None],
        0.0,
    )
    option_nonfinite = evaluated.ledger.nonfinite[..., None]
    selected = select_affordable_effort(
        requested_effort[..., None],
        torch.where(option_nonfinite, torch.nan, option_work),
        budget_j,
        body.alive,
    )
    funding_resolved = ~body.alive | selected.requested_was_funded
    settled = _settle_selected_actuation(evaluated, requested_effort)
    return AffordableMotionAdvance(
        evaluated.state,
        AffordableMotionLedger(
            selected,
            settled.ledger,
            option_work,
            option_nonfinite,
            funding_resolved,
        ),
    )
