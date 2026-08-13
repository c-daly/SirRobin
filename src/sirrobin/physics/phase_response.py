"""State-windowed ecological motion derived from canonical hydrodynamics.

This is the candidate ordinary-motion lane for the device runtime. It samples the
actual gait-time window instead of replacing the organism with a terminal speed or
averaging an entire cycle. The canonical 120 Hz solver remains the comparison and
out-of-domain reference.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass, fields

import torch

from sirrobin.fields.geometry import GridGeometry
from sirrobin.numerics.solve_constrained_xy import solve_constrained_xy
from sirrobin.physics.contracts import DevelopedBody, FluidSample, LiveState
from sirrobin.physics.controller import retune_heading_controller_state
from sirrobin.physics.force_hydrodynamic import hydrodynamic_contribution
from sirrobin.physics.live_config import LiveLocomotionConfig
from sirrobin.physics.pose_live import resolve_live_pose
from sirrobin.physics.yaw import advance_yaw


@dataclass(frozen=True, slots=True)
class PhaseWindowConfig:
    """Fixed quadrature and state-update cadence for one ecological interval."""

    interval_s: float
    stages: int = 4
    phase_samples: int = 2

    def validate(self) -> None:
        if not math.isfinite(self.interval_s) or self.interval_s <= 0.0:
            raise ValueError("phase-window interval must be finite and positive")
        if not isinstance(self.stages, int) or isinstance(self.stages, bool):
            raise TypeError("phase-window stages must be an integer")
        if not isinstance(self.phase_samples, int) or isinstance(
            self.phase_samples, bool
        ):
            raise TypeError("phase-window samples must be an integer")
        if self.stages < 1 or self.phase_samples < 1:
            raise ValueError("phase-window stages and samples must be positive")


@dataclass(frozen=True, slots=True)
class PhaseWindowLedger:
    """Named work and intervention channels accumulated over the interval."""

    positive_actuator_work_j: torch.Tensor
    actuator_braking_work_j: torch.Tensor
    dissipated_work_j: torch.Tensor
    regularization_count: torch.Tensor
    yaw_inertia_floor_count: torch.Tensor
    yaw_backstop_hit: torch.Tensor
    nonfinite: torch.Tensor
    effective_mass_after_kg: torch.Tensor
    yaw_inertia_after_kg_m2: torch.Tensor


@dataclass(frozen=True, slots=True)
class PhaseWindowAdvance:
    state: LiveState
    ledger: PhaseWindowLedger


def _expand_phase(value: torch.Tensor, samples: int) -> torch.Tensor:
    """Expand [W,N,...] to [W,N*P,...] without allocating repeated storage."""

    worlds, capacity = value.shape[:2]
    tail = value.shape[2:]
    return value[:, :, None].expand(worlds, capacity, samples, *tail).reshape(
        worlds, capacity * samples, *tail
    )


def _phase_body(body: DevelopedBody, samples: int) -> DevelopedBody:
    return DevelopedBody(
        **{
            field.name: _expand_phase(getattr(body, field.name), samples)
            for field in fields(body)
        }
    )


def _phase_state(state: LiveState, sample_times_s: torch.Tensor) -> LiveState:
    samples = sample_times_s.shape[-1]
    return LiveState(
        position_enu_m=_expand_phase(state.position_enu_m, samples),
        velocity_rel_water_enu_m_s=_expand_phase(
            state.velocity_rel_water_enu_m_s, samples
        ),
        yaw_rad=_expand_phase(state.yaw_rad, samples),
        yaw_momentum_kg_m2_s=_expand_phase(
            state.yaw_momentum_kg_m2_s, samples
        ),
        gait_time_s=sample_times_s.reshape(
            state.gait_time_s.shape[0], state.gait_time_s.shape[1] * samples
        ),
        desired_heading_enu=_expand_phase(state.desired_heading_enu, samples),
        turn_bias_rad_per_depth=_expand_phase(
            state.turn_bias_rad_per_depth, samples
        ),
        heading_initialized=_expand_phase(state.heading_initialized, samples),
    )


def advance_phase_stage(
    body: DevelopedBody,
    state: LiveState,
    fluid: FluidSample,
    live_config: LiveLocomotionConfig,
    geometry: GridGeometry,
    *,
    effort_fraction: torch.Tensor,
    dt_s: float,
    phase_samples: int,
) -> PhaseWindowAdvance:
    """Advance one fixed response stage; intended compiled-kernel boundary."""

    worlds, capacity = body.alive.shape
    phase_body = _phase_body(body, phase_samples)
    offsets = (
        torch.arange(
            phase_samples,
            dtype=state.gait_time_s.dtype,
            device=state.gait_time_s.device,
        )
        + 0.5
    ) / phase_samples
    sample_times_s = state.gait_time_s[..., None] + offsets * dt_s
    sampled_state = _phase_state(state, sample_times_s)
    sampled_effort = _expand_phase(effort_fraction, phase_samples)
    pose0 = resolve_live_pose(
        phase_body,
        sampled_state.gait_time_s,
        sampled_state.turn_bias_rad_per_depth,
        effort=sampled_effort,
    )
    pose1 = resolve_live_pose(
        phase_body,
        sampled_state.gait_time_s + live_config.dt,
        sampled_state.turn_bias_rad_per_depth,
        effort=sampled_effort,
    )
    hydro = hydrodynamic_contribution(
        phase_body,
        sampled_state,
        pose0,
        pose1,
        _expand_phase(fluid.density_kg_m3, phase_samples).reshape(-1),
        live_config,
    )

    def phase_mean(value: torch.Tensor) -> torch.Tensor:
        return value.reshape(worlds, capacity, phase_samples, *value.shape[1:]).mean(
            dim=2
        )

    force = phase_mean(hydro.contribution.force_enu_n)
    torque = phase_mean(hydro.contribution.torque_yaw_nm)
    matrix1 = phase_mean(hydro.diagnostics.effective_mass_after_kg)
    inertia0 = phase_mean(hydro.diagnostics.yaw_inertia_before_kg_m2)
    inertia1 = phase_mean(hydro.diagnostics.yaw_inertia_after_kg_m2)
    input_power = hydro.contribution.input_power_w.reshape(
        worlds, capacity, phase_samples
    )
    dissipated_power = hydro.contribution.dissipated_power_w.reshape(
        worlds, capacity, phase_samples
    )

    alive = body.alive.reshape(-1)
    velocity0 = state.velocity_rel_water_enu_m_s.reshape(-1, 3)
    solve = solve_constrained_xy(
        matrix1.reshape(-1, 3, 3),
        force.reshape(-1, 3) * dt_s,
        alive,
        kappa_max=live_config.kappa_max,
        lam_floor=live_config.lam_floor_kg,
        eps_spd=live_config.eps_spd,
    )
    velocity1 = velocity0 + solve.dv
    velocity1 = torch.stack(
        (
            velocity1[:, 0],
            velocity1[:, 1],
            torch.zeros_like(velocity1[:, 0]),
        ),
        dim=-1,
    )
    velocity1 = torch.where(alive[:, None], velocity1, 0.0)
    yaw = advance_yaw(
        state.yaw_rad.reshape(-1),
        state.yaw_momentum_kg_m2_s.reshape(-1),
        inertia0.reshape(-1),
        inertia1.reshape(-1),
        torque.reshape(-1),
        dt_s,
        alive,
        inertia_floor=live_config.inertia_floor_kg_m2,
        emergency_omega=live_config.emergency_omega_rad_s,
    )
    velocity1_world = velocity1.reshape(worlds, capacity, 3)
    midpoint_transport = (
        0.5 * (state.velocity_rel_water_enu_m_s + velocity1_world)
        + fluid.velocity_enu_m_s
    )
    next_xy = state.position_enu_m[..., :2] + midpoint_transport[..., :2] * dt_s
    wrapped_position = torch.stack(
        (
            torch.remainder(next_xy[..., 0], geometry.lx_m),
            torch.remainder(next_xy[..., 1], geometry.ly_m),
            state.position_enu_m[..., 2],
        ),
        dim=-1,
    )
    next_position = torch.where(
        body.alive[..., None], wrapped_position, state.position_enu_m
    )
    finite = (
        torch.isfinite(velocity1_world).all(dim=-1)
        & torch.isfinite(yaw.yaw.reshape(worlds, capacity))
        & torch.isfinite(yaw.momentum.reshape(worlds, capacity))
        & torch.isfinite(force).all(dim=-1)
        & torch.isfinite(torque)
        & torch.isfinite(matrix1).all(dim=(-2, -1))
        & torch.isfinite(inertia0)
        & torch.isfinite(inertia1)
        & torch.isfinite(input_power).all(dim=-1)
        & torch.isfinite(dissipated_power).all(dim=-1)
    )
    next_state = LiveState(
        position_enu_m=next_position,
        velocity_rel_water_enu_m_s=velocity1_world,
        yaw_rad=yaw.yaw.reshape(worlds, capacity),
        yaw_momentum_kg_m2_s=yaw.momentum.reshape(worlds, capacity),
        gait_time_s=state.gait_time_s + dt_s,
        desired_heading_enu=state.desired_heading_enu,
        turn_bias_rad_per_depth=state.turn_bias_rad_per_depth,
        heading_initialized=state.heading_initialized,
    )
    ledger = PhaseWindowLedger(
        positive_actuator_work_j=torch.where(
            body.alive,
            input_power.clamp_min(0.0).mean(dim=-1) * dt_s,
            0.0,
        ),
        actuator_braking_work_j=torch.where(
            body.alive,
            (-input_power).clamp_min(0.0).mean(dim=-1) * dt_s,
            0.0,
        ),
        dissipated_work_j=torch.where(
            body.alive,
            dissipated_power.mean(dim=-1) * dt_s,
            0.0,
        ),
        regularization_count=solve.regularized.reshape(worlds, capacity).to(
            torch.int64
        ),
        yaw_inertia_floor_count=yaw.floor_hit.reshape(worlds, capacity).to(
            torch.int64
        ),
        yaw_backstop_hit=yaw.backstop_hit.reshape(worlds, capacity),
        nonfinite=body.alive & ~finite,
        effective_mass_after_kg=matrix1,
        yaw_inertia_after_kg_m2=inertia1,
    )
    return PhaseWindowAdvance(next_state, ledger)


def advance_phase_window(
    body: DevelopedBody,
    state: LiveState,
    fluid: FluidSample,
    live_config: LiveLocomotionConfig,
    geometry: GridGeometry,
    response_config: PhaseWindowConfig,
    *,
    effort_fraction: torch.Tensor,
) -> PhaseWindowAdvance:
    """Advance one interval with fixed phase quadrature and state updates.

    Configuration and effort tensors are validated when a runtime session is
    constructed. This hot function contains no device-to-host decisions; the fixed
    stage loop is unrolled by ``torch.compile``.
    """

    return advance_phase_window_with_stage(
        body,
        state,
        fluid,
        live_config,
        geometry,
        response_config,
        effort_fraction=effort_fraction,
        stage_kernel=advance_phase_stage,
    )


def advance_phase_window_with_stage(
    body: DevelopedBody,
    state: LiveState,
    fluid: FluidSample,
    live_config: LiveLocomotionConfig,
    geometry: GridGeometry,
    response_config: PhaseWindowConfig,
    *,
    effort_fraction: torch.Tensor,
    stage_kernel: Callable[..., PhaseWindowAdvance],
) -> PhaseWindowAdvance:
    """Compose a fixed stage kernel without inspecting device state."""

    dt_s = response_config.interval_s / response_config.stages
    current = state
    positive_work = torch.zeros_like(body.mass_sim.sum(dim=-1))
    braking_work = torch.zeros_like(positive_work)
    dissipated_work = torch.zeros_like(positive_work)
    regularization_count = torch.zeros_like(body.stable_id)
    inertia_floor_count = torch.zeros_like(body.stable_id)
    backstop = torch.zeros_like(body.alive)
    nonfinite = torch.zeros_like(body.alive)
    effective_mass_after = torch.zeros(
        (*body.alive.shape, 3, 3),
        dtype=body.mass_sim.dtype,
        device=body.alive.device,
    )
    yaw_inertia_after = torch.zeros_like(positive_work)
    for _ in range(response_config.stages):
        current = retune_heading_controller_state(body, current, live_config)
        stage = stage_kernel(
            body,
            current,
            fluid,
            live_config,
            geometry,
            effort_fraction=effort_fraction,
            dt_s=dt_s,
            phase_samples=response_config.phase_samples,
        )
        current = stage.state
        positive_work = positive_work + stage.ledger.positive_actuator_work_j
        braking_work = braking_work + stage.ledger.actuator_braking_work_j
        dissipated_work = dissipated_work + stage.ledger.dissipated_work_j
        regularization_count = (
            regularization_count + stage.ledger.regularization_count
        )
        inertia_floor_count = (
            inertia_floor_count + stage.ledger.yaw_inertia_floor_count
        )
        backstop = backstop | stage.ledger.yaw_backstop_hit
        nonfinite = nonfinite | stage.ledger.nonfinite
        effective_mass_after = stage.ledger.effective_mass_after_kg
        yaw_inertia_after = stage.ledger.yaw_inertia_after_kg_m2
    return PhaseWindowAdvance(
        current,
        PhaseWindowLedger(
            positive_work,
            braking_work,
            dissipated_work,
            regularization_count,
            inertia_floor_count,
            backstop,
            nonfinite,
            effective_mass_after,
            yaw_inertia_after,
        ),
    )
