"""Error-controlled periodic acceleration of canonical live mechanics."""

from __future__ import annotations

import math
from dataclasses import dataclass, fields

import torch

from sirrobin.physics.contracts import (
    DevelopedBody,
    FluidSample,
    LiveState,
    LiveStepLedger,
)
from sirrobin.physics.live_step import step_live
from sirrobin.physics.yaw import wrap_pi


@dataclass(frozen=True, slots=True)
class PeriodicMotionPolicy:
    """One recurrence/error contract applied without phenotype-specific branches."""

    max_detection_cycles: int = 64
    required_consecutive_cycles: int = 4
    relative_tolerance: float = 1.0e-9
    max_accumulated_translation_error_m: float = 0.1
    max_accumulated_yaw_error_rad: float = 1.0e-3
    max_projected_relative_state_error: float = 1.0e-4
    max_projected_velocity_error_m_s: float = 1.0e-3
    max_projected_yaw_momentum_error_kg_m2_s: float = 1.0e-3
    max_projected_relative_work_error: float = 1.0e-4

    def __post_init__(self) -> None:
        if self.max_detection_cycles < 2:
            raise ValueError("max_detection_cycles must be at least two")
        if not 1 <= self.required_consecutive_cycles < self.max_detection_cycles:
            raise ValueError("required_consecutive_cycles must fit inside detection")
        scalar_bounds = (
            self.relative_tolerance,
            self.max_accumulated_translation_error_m,
            self.max_accumulated_yaw_error_rad,
            self.max_projected_relative_state_error,
            self.max_projected_velocity_error_m_s,
            self.max_projected_yaw_momentum_error_kg_m2_s,
            self.max_projected_relative_work_error,
        )
        if any(not math.isfinite(value) or value <= 0.0 for value in scalar_bounds):
            raise ValueError("periodic error bounds must be finite and positive")


DEFAULT_PERIODIC_MOTION_POLICY = PeriodicMotionPolicy()


@dataclass(frozen=True, slots=True)
class MechanicsAdvance:
    covered_steps: int
    full_batch_steps: int
    representative_steps: int
    fast_forwarded_steps: int
    mechanical_work_j: torch.Tensor
    last_ledger: LiveStepLedger
    periodic_error: PeriodicErrorEstimate | None


@dataclass(frozen=True, slots=True)
class PeriodicErrorEstimate:
    max_relative_recurrence_error: float
    accumulated_translation_error_m: float
    accumulated_yaw_error_rad: float
    projected_relative_state_error: float
    projected_velocity_error_m_s: float
    projected_yaw_momentum_error_kg_m2_s: float
    projected_relative_work_error: float


@dataclass(frozen=True, slots=True)
class _CycleObservation:
    translation_body_m: torch.Tensor
    yaw_delta_rad: torch.Tensor
    velocity_body_m_s: torch.Tensor
    yaw_momentum_kg_m2_s: torch.Tensor
    work_j: torch.Tensor


def _compose_transform(
    first_translation: torch.Tensor,
    first_yaw: torch.Tensor,
    second_translation: torch.Tensor,
    second_yaw: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    cosine = torch.cos(first_yaw)
    sine = torch.sin(first_yaw)
    rotated_x = cosine * second_translation[..., 0] - sine * second_translation[..., 1]
    rotated_y = sine * second_translation[..., 0] + cosine * second_translation[..., 1]
    translation = first_translation + torch.stack((rotated_x, rotated_y), -1)
    return translation, first_yaw + second_yaw


def repeat_transform(
    translation: torch.Tensor, yaw: torch.Tensor, repetitions: int
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compose one planar body-frame rigid transform `repetitions` times."""
    if repetitions < 0:
        raise ValueError("transform repetitions must be nonnegative")
    result_translation = torch.zeros_like(translation)
    result_yaw = torch.zeros_like(yaw)
    base_translation = translation
    base_yaw = yaw
    count = repetitions
    while count:
        if count & 1:
            result_translation, result_yaw = _compose_transform(
                result_translation, result_yaw, base_translation, base_yaw
            )
        base_translation, base_yaw = _compose_transform(
            base_translation, base_yaw, base_translation, base_yaw
        )
        count >>= 1
    return result_translation, result_yaw


def _body_frame_xy(vector: torch.Tensor, yaw: torch.Tensor) -> torch.Tensor:
    cosine = torch.cos(yaw)
    sine = torch.sin(yaw)
    return torch.stack(
        (
            cosine * vector[..., 0] + sine * vector[..., 1],
            -sine * vector[..., 0] + cosine * vector[..., 1],
        ),
        -1,
    )


def _world_frame_xy(vector: torch.Tensor, yaw: torch.Tensor) -> torch.Tensor:
    cosine = torch.cos(yaw)
    sine = torch.sin(yaw)
    return torch.stack(
        (
            cosine * vector[..., 0] - sine * vector[..., 1],
            sine * vector[..., 0] + cosine * vector[..., 1],
        ),
        -1,
    )


def _same_across_capacity(value: torch.Tensor) -> bool:
    return bool(torch.equal(value, value[:, :1].expand_as(value)))


def _same_pose_normalized_state(value: torch.Tensor) -> bool:
    reference = value[:, :1].expand_as(value)
    return bool(torch.allclose(value, reference, rtol=1.0e-12, atol=1.0e-12))


def _can_share_representative(
    body: DevelopedBody,
    state: LiveState,
    fluid: FluidSample,
    effort_fraction: torch.Tensor | None,
) -> bool:
    if body.worlds != 1 or not bool(body.alive.all()):
        return False
    for field in fields(body):
        if field.name == "stable_id":
            continue
        if not _same_across_capacity(getattr(body, field.name)):
            return False
    if not _same_across_capacity(fluid.density_kg_m3):
        return False
    if not bool((fluid.velocity_enu_m_s == 0.0).all()):
        return False
    if effort_fraction is not None:
        if tuple(effort_fraction.shape) != tuple(body.alive.shape):
            return False
        if not _same_across_capacity(effort_fraction):
            return False
    scalar_state = (
        state.gait_time_s,
        state.yaw_momentum_kg_m2_s,
        state.turn_bias_rad_per_depth,
        state.velocity_rel_water_enu_m_s[..., 2],
    )
    if not all(_same_across_capacity(value) for value in scalar_state):
        return False
    body_velocity = _body_frame_xy(
        state.velocity_rel_water_enu_m_s[..., :2], state.yaw_rad
    )
    return _same_pose_normalized_state(body_velocity)


def _slice_representative(body: DevelopedBody, state: LiveState, fluid: FluidSample):
    representative_body = DevelopedBody(
        **{field.name: getattr(body, field.name)[:, :1] for field in fields(body)}
    )
    representative_state = LiveState(
        **{
            field.name: getattr(state, field.name)[:, :1].clone()
            for field in fields(state)
        }
    )
    representative_fluid = FluidSample(
        fluid.density_kg_m3[:, :1], fluid.velocity_enu_m_s[:, :1]
    )
    return representative_body, representative_state, representative_fluid


def _relative_error(first: torch.Tensor, second: torch.Tensor) -> float:
    difference = (first - second).abs()
    scale = torch.maximum(first.abs(), second.abs())
    relative = torch.where(
        scale == 0.0,
        torch.where(difference == 0.0, 0.0, torch.inf),
        difference / scale,
    )
    return float(relative.max().item())


def _recurrence_error(
    current: _CycleObservation,
    previous: _CycleObservation,
) -> float:
    return max(
        _relative_error(getattr(current, field.name), getattr(previous, field.name))
        for field in fields(current)
        if field.name not in {"yaw_delta_rad", "work_j"}
    )


def _accumulated_error(
    current: _CycleObservation,
    previous: _CycleObservation,
    skipped_cycles: int,
) -> PeriodicErrorEstimate:
    recurrence_error = _recurrence_error(current, previous)
    projected_relative_state_error = skipped_cycles * recurrence_error
    projected_velocity_error = skipped_cycles * float(
        torch.linalg.vector_norm(
            current.velocity_body_m_s - previous.velocity_body_m_s, dim=-1
        ).max().item()
    )
    projected_yaw_momentum_error = skipped_cycles * float(
        (
            current.yaw_momentum_kg_m2_s
            - previous.yaw_momentum_kg_m2_s
        ).abs().max().item()
    )
    yaw_error = skipped_cycles * float(
        (current.yaw_delta_rad - previous.yaw_delta_rad).abs().max().item()
    )
    translation_error = float(
        torch.linalg.vector_norm(
            current.translation_body_m - previous.translation_body_m, dim=-1
        ).max().item()
    )
    path_per_cycle = float(
        torch.linalg.vector_norm(current.translation_body_m, dim=-1).max().item()
    )
    position_error = skipped_cycles * translation_error
    position_error += skipped_cycles * path_per_cycle * yaw_error
    work_scale = torch.maximum(current.work_j.abs(), previous.work_j.abs())
    work_difference = (current.work_j - previous.work_j).abs()
    relative_work_error_per_cycle = torch.where(
        work_scale == 0.0,
        torch.where(work_difference == 0.0, 0.0, torch.inf),
        work_difference / work_scale,
    )
    return PeriodicErrorEstimate(
        max_relative_recurrence_error=recurrence_error,
        accumulated_translation_error_m=position_error,
        accumulated_yaw_error_rad=yaw_error,
        projected_relative_state_error=projected_relative_state_error,
        projected_velocity_error_m_s=projected_velocity_error,
        projected_yaw_momentum_error_kg_m2_s=projected_yaw_momentum_error,
        projected_relative_work_error=(
            skipped_cycles * float(relative_work_error_per_cycle.max().item())
        ),
    )


def _within_policy(error: PeriodicErrorEstimate, policy: PeriodicMotionPolicy) -> bool:
    return (
        error.max_relative_recurrence_error <= policy.relative_tolerance
        and error.accumulated_translation_error_m
        <= policy.max_accumulated_translation_error_m
        and error.accumulated_yaw_error_rad <= policy.max_accumulated_yaw_error_rad
        and error.projected_relative_state_error
        <= policy.max_projected_relative_state_error
        and error.projected_velocity_error_m_s
        <= policy.max_projected_velocity_error_m_s
        and error.projected_yaw_momentum_error_kg_m2_s
        <= policy.max_projected_yaw_momentum_error_kg_m2_s
        and error.projected_relative_work_error
        <= policy.max_projected_relative_work_error
    )


def _worst_error(errors: list[PeriodicErrorEstimate]) -> PeriodicErrorEstimate:
    """Return componentwise worst telemetry across the accepted recurrence window."""
    return PeriodicErrorEstimate(
        **{
            field.name: max(getattr(error, field.name) for error in errors)
            for field in fields(PeriodicErrorEstimate)
        }
    )


def _full_advance(
    world,
    steps: int,
    effort_fraction: torch.Tensor | None,
) -> MechanicsAdvance:
    work = torch.zeros_like(world.body.mass_sim.sum(-1))
    ledger = None
    for _ in range(steps):
        ledger = world._step_mechanics(effort_fraction)
        work += ledger.total.dissipated_power_w.reshape_as(work) * world.live_config.dt
    if ledger is None:
        raise ValueError("mechanics advance requires at least one step")
    return MechanicsAdvance(steps, steps, 0, 0, work, ledger, None)


def advance_mechanics_interval(
    world,
    steps: int,
    policy: PeriodicMotionPolicy | None,
    *,
    effort_fraction: torch.Tensor | None = None,
) -> MechanicsAdvance:
    """Cover one mechanics interval, fast-forwarding only a verified clone orbit."""
    if steps < 1:
        raise ValueError("mechanics interval must contain at least one step")
    if policy is None:
        return _full_advance(world, steps, effort_fraction)
    if not _can_share_representative(
        world.body,
        world.live_state,
        world.fluid,
        effort_fraction,
    ):
        return _full_advance(world, steps, effort_fraction)
    frequency = float(world.body.swim_freq_hz[0, 0].item())
    if not math.isfinite(frequency) or frequency <= 0.0:
        return _full_advance(world, steps, effort_fraction)
    period_steps = round(1.0 / (frequency * world.live_config.dt))
    if period_steps < 1 or not math.isclose(
        period_steps * world.live_config.dt * frequency,
        1.0,
        rel_tol=0.0,
        abs_tol=1.0e-12,
    ):
        return _full_advance(world, steps, effort_fraction)
    whole_cycles, remainder_steps = divmod(steps, period_steps)
    if whole_cycles <= policy.max_detection_cycles + 1:
        return _full_advance(world, steps, effort_fraction)

    body, representative, fluid = _slice_representative(
        world.body, world.live_state, world.fluid
    )
    representative_effort = (
        None if effort_fraction is None else effort_fraction[:, :1]
    )
    initial_position = world.live_state.position_enu_m.clone()
    initial_yaw = world.live_state.yaw_rad.clone()
    total_translation = torch.zeros((1, 2), dtype=initial_position.dtype, device=initial_position.device)
    total_yaw = torch.zeros((1,), dtype=initial_yaw.dtype, device=initial_yaw.device)
    actual_work = torch.zeros((1,), dtype=initial_yaw.dtype, device=initial_yaw.device)
    previous = None
    current = None
    accepted_window: list[PeriodicErrorEstimate] = []
    detected_cycles = 0
    last_ledger = None
    accepted_error = None
    for cycle in range(1, policy.max_detection_cycles + 1):
        position_before = representative.position_enu_m.clone()
        yaw_before = representative.yaw_rad.clone()
        cycle_work = torch.zeros_like(representative.yaw_rad)
        for _ in range(period_steps):
            last_ledger = step_live(
                body,
                representative,
                fluid,
                world.live_config,
                effort_fraction=representative_effort,
            )
            representative.position_enu_m.add_(
                representative.velocity_rel_water_enu_m_s * world.live_config.dt
            )
            cycle_work += last_ledger.total.dissipated_power_w * world.live_config.dt
        displacement = representative.position_enu_m[..., :2] - position_before[..., :2]
        translation = _body_frame_xy(displacement, yaw_before).reshape(1, 2)
        yaw_delta = wrap_pi(representative.yaw_rad - yaw_before).reshape(1)
        velocity_body = _body_frame_xy(
            representative.velocity_rel_water_enu_m_s[..., :2],
            representative.yaw_rad,
        ).reshape(1, 2)
        current = _CycleObservation(
            translation,
            yaw_delta,
            velocity_body,
            representative.yaw_momentum_kg_m2_s.reshape(1).clone(),
            cycle_work.reshape(1),
        )
        total_translation, total_yaw = _compose_transform(
            total_translation, total_yaw, translation, yaw_delta
        )
        actual_work += cycle_work.reshape(1)
        detected_cycles = cycle
        if previous is not None:
            skipped = whole_cycles - cycle - 1
            candidate_error = _accumulated_error(current, previous, skipped)
            if _within_policy(candidate_error, policy):
                accepted_window.append(candidate_error)
            else:
                accepted_window.clear()
            if len(accepted_window) >= policy.required_consecutive_cycles:
                accepted_error = _worst_error(
                    accepted_window[-policy.required_consecutive_cycles :]
                )
                break
        else:
            accepted_window.clear()
        previous = current
    else:
        return _full_advance(world, steps, effort_fraction)

    if current is None or last_ledger is None or accepted_error is None:
        return _full_advance(world, steps, effort_fraction)
    skipped_cycles = whole_cycles - detected_cycles - 1
    if skipped_cycles <= 0:
        return _full_advance(world, steps, effort_fraction)
    skipped_translation, skipped_yaw = repeat_transform(
        current.translation_body_m, current.yaw_delta_rad, skipped_cycles
    )
    total_translation, total_yaw = _compose_transform(
        total_translation, total_yaw, skipped_translation, skipped_yaw
    )
    translated = _world_frame_xy(
        total_translation.expand(world.body.capacity, -1), initial_yaw.reshape(-1)
    ).reshape_as(initial_position[..., :2])
    world.live_state.position_enu_m[..., 0].copy_(
        torch.remainder(initial_position[..., 0] + translated[..., 0], world.geometry.lx_m)
    )
    world.live_state.position_enu_m[..., 1].copy_(
        torch.remainder(initial_position[..., 1] + translated[..., 1], world.geometry.ly_m)
    )
    final_yaw = wrap_pi(initial_yaw + total_yaw[:, None])
    world.live_state.yaw_rad.copy_(final_yaw)
    final_body_velocity = current.velocity_body_m_s[:, None, :].expand(
        1, world.body.capacity, 2
    )
    final_world_velocity = _world_frame_xy(final_body_velocity, final_yaw)
    world.live_state.velocity_rel_water_enu_m_s[..., :2].copy_(final_world_velocity)
    world.live_state.velocity_rel_water_enu_m_s[..., 2].zero_()
    world.live_state.yaw_momentum_kg_m2_s.copy_(
        current.yaw_momentum_kg_m2_s[:, None].expand_as(
            world.live_state.yaw_momentum_kg_m2_s
        )
    )
    covered_before_full = (detected_cycles + skipped_cycles) * period_steps
    world.live_state.gait_time_s.copy_(
        world.live_state.gait_time_s[:, :1].expand_as(world.live_state.gait_time_s)
        + covered_before_full * world.live_config.dt
    )
    work = (actual_work + skipped_cycles * current.work_j)[:, None].expand_as(
        world.body.mass_sim.sum(-1)
    ).clone()
    full_steps = period_steps + remainder_steps
    for _ in range(full_steps):
        last_ledger = world._step_mechanics(effort_fraction)
        work += (
            last_ledger.total.dissipated_power_w.reshape_as(work)
            * world.live_config.dt
        )
    return MechanicsAdvance(
        steps,
        full_steps,
        detected_cycles * period_steps,
        skipped_cycles * period_steps,
        work,
        last_ledger,
        accepted_error,
    )
