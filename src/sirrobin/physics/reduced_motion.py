"""Scalar, form-calibrated surge and yaw motion for ecological time scales."""

from __future__ import annotations

import math
from dataclasses import dataclass, fields


@dataclass(frozen=True, slots=True)
class ReducedMotionResponse:
    """Immutable response knots and physical coefficients for one developed form."""

    stable_id: int
    probe_turn_fraction: float
    surge_half_m_s: float
    surge_full_m_s: float
    yaw_positive_half_rad_s: float
    yaw_positive_full_rad_s: float
    yaw_negative_half_rad_s: float
    yaw_negative_full_rad_s: float
    straight_power_half_w: float
    straight_power_full_w: float
    positive_turn_power_half_w: float
    positive_turn_power_full_w: float
    negative_turn_power_half_w: float
    negative_turn_power_full_w: float
    axial_mass_kg: float
    axial_drag_n_per_m_s2: float
    yaw_inertia_kg_m2: float
    yaw_drag_nm_per_rad_s2: float
    calibration_intervention_count: int
    yaw_inertia_floor_used: bool

    def numeric_values(self) -> tuple[float, ...]:
        return tuple(
            float(getattr(self, field.name))
            for field in fields(self)
            if field.name
            not in {
                "stable_id",
                "calibration_intervention_count",
                "yaw_inertia_floor_used",
            }
        )

    def validate(self) -> None:
        if not isinstance(self.stable_id, int) or isinstance(self.stable_id, bool):
            raise ValueError("stable_id must be an integer")
        if not all(math.isfinite(value) for value in self.numeric_values()):
            raise ValueError("reduced response values must be finite")
        if not 0.0 < self.probe_turn_fraction <= 1.0:
            raise ValueError("probe_turn_fraction must lie in (0, 1]")
        if self.axial_mass_kg <= 0.0:
            raise ValueError("axial_mass_kg must be positive")
        if self.yaw_inertia_kg_m2 <= 0.0:
            raise ValueError("yaw_inertia_kg_m2 must be positive")
        if self.axial_drag_n_per_m_s2 < 0.0:
            raise ValueError("axial drag must be nonnegative")
        if self.yaw_drag_nm_per_rad_s2 < 0.0:
            raise ValueError("yaw drag must be nonnegative")
        powers = (
            self.straight_power_half_w,
            self.straight_power_full_w,
            self.positive_turn_power_half_w,
            self.positive_turn_power_full_w,
            self.negative_turn_power_half_w,
            self.negative_turn_power_full_w,
        )
        if any(power < 0.0 for power in powers):
            raise ValueError("measured power must be nonnegative")
        if self.axial_drag_n_per_m_s2 == 0.0 and (
            self.surge_half_m_s != 0.0 or self.surge_full_m_s != 0.0
        ):
            raise ValueError("nonzero surge response requires axial drag")
        yaw_responses = (
            self.yaw_positive_half_rad_s,
            self.yaw_positive_full_rad_s,
            self.yaw_negative_half_rad_s,
            self.yaw_negative_full_rad_s,
        )
        if self.yaw_drag_nm_per_rad_s2 == 0.0 and any(yaw_responses):
            raise ValueError("nonzero yaw response requires yaw drag")
        if (
            not isinstance(self.calibration_intervention_count, int)
            or isinstance(self.calibration_intervention_count, bool)
            or self.calibration_intervention_count < 0
        ):
            raise ValueError("calibration_intervention_count must be a nonnegative integer")
        if not isinstance(self.yaw_inertia_floor_used, bool):
            raise ValueError("yaw_inertia_floor_used must be boolean")


@dataclass(frozen=True, slots=True)
class ReducedMotionState:
    yaw_rad: float = 0.0
    surge_m_s: float = 0.0
    yaw_rate_rad_s: float = 0.0


@dataclass(frozen=True, slots=True)
class ReducedMotionStep:
    state: ReducedMotionState
    target_surge_m_s: float
    target_yaw_rate_rad_s: float
    surge_distance_m: float
    yaw_delta_rad: float
    measured_cost_power_w: float
    measured_cost_j: float


def _effort_value(effort: float, half: float, full: float) -> float:
    if effort == 0.0:
        return 0.0
    if effort == 0.5:
        return half
    if effort == 1.0:
        return full
    if effort <= 0.5:
        return 2.0 * effort * half
    return half + 2.0 * (effort - 0.5) * (full - half)


def _log_cosh(value: float) -> float:
    magnitude = abs(value)
    return magnitude + math.log1p(math.exp(-2.0 * magnitude)) - math.log(2.0)


def _log_sinh_positive(value: float) -> float:
    if value < 20.0:
        return math.log(math.sinh(value))
    return value + math.log1p(-math.exp(-2.0 * value)) - math.log(2.0)


def _integrate_drag_limited(
    current: float,
    target: float,
    inertia: float,
    quadratic_drag: float,
    dt: float,
) -> tuple[float, float]:
    """Return endpoint and exact integral for the scalar quadratic-drag ODE."""
    if quadratic_drag == 0.0:
        return current, current * dt
    coefficient = quadratic_drag / inertia
    if target == 0.0:
        scale = 1.0 + coefficient * abs(current) * dt
        endpoint = current / scale
        integral = math.copysign(math.log(scale) / coefficient, current)
        return endpoint, integral

    direction = math.copysign(1.0, target)
    terminal = abs(target)
    aligned = direction * current
    rate = coefficient * terminal
    if aligned == terminal:
        return target, target * dt
    if aligned >= 0.0:
        if aligned < terminal:
            phase = math.atanh(aligned / terminal)
            final_phase = phase + rate * dt
            advanced = terminal * math.tanh(final_phase)
            integral = (_log_cosh(final_phase) - _log_cosh(phase)) / coefficient
        else:
            phase = math.atanh(terminal / aligned)
            final_phase = phase + rate * dt
            advanced = terminal / math.tanh(final_phase)
            integral = (
                _log_sinh_positive(final_phase) - _log_sinh_positive(phase)
            ) / coefficient
    else:
        phase = math.atan(aligned / terminal)
        crossing_time = -phase / rate
        if dt <= crossing_time:
            final_phase = phase + rate * dt
            advanced = terminal * math.tan(final_phase)
            integral = (
                math.log(math.cos(phase)) - math.log(math.cos(final_phase))
            ) / coefficient
        else:
            final_phase = rate * (dt - crossing_time)
            advanced = terminal * math.tanh(final_phase)
            integral = (
                math.log(math.cos(phase)) + _log_cosh(final_phase)
            ) / coefficient
    return direction * advanced, direction * integral


def _wrap_pi(angle: float) -> float:
    return math.atan2(math.sin(angle), math.cos(angle))


def step_reduced_motion(
    state: ReducedMotionState,
    response: ReducedMotionResponse,
    *,
    effort: float,
    turn_fraction: float,
    dt: float,
) -> ReducedMotionStep:
    """Advance scalar motion under bounded effort and measured turn fraction.

    The turn domain is deliberately limited to the calibrated probe interval. The
    returned measured cost is an empirical interpolation of standardized probe work,
    not an instantaneous drag/kinetic-energy ledger and not yet a world energy debit.
    World-coordinate trajectory projection is deferred to the runner integration.
    """
    response.validate()
    state_values = tuple(getattr(state, field.name) for field in fields(state))
    if not all(math.isfinite(value) for value in state_values):
        raise ValueError("reduced motion state must be finite")
    if not math.isfinite(effort) or not 0.0 <= effort <= 1.0:
        raise ValueError("effort must be finite and in [0, 1]")
    if (
        not math.isfinite(turn_fraction)
        or not -response.probe_turn_fraction
        <= turn_fraction
        <= response.probe_turn_fraction
    ):
        raise ValueError("turn_fraction must be finite and inside the calibrated interval")
    if not math.isfinite(dt) or dt <= 0.0:
        raise ValueError("dt must be finite and positive")

    target_surge = _effort_value(
        effort, response.surge_half_m_s, response.surge_full_m_s
    )
    if turn_fraction > 0.0:
        turn_yaw = _effort_value(
            effort,
            response.yaw_positive_half_rad_s,
            response.yaw_positive_full_rad_s,
        )
        turn_power = _effort_value(
            effort,
            response.positive_turn_power_half_w,
            response.positive_turn_power_full_w,
        )
    elif turn_fraction < 0.0:
        turn_yaw = _effort_value(
            effort,
            response.yaw_negative_half_rad_s,
            response.yaw_negative_full_rad_s,
        )
        turn_power = _effort_value(
            effort,
            response.negative_turn_power_half_w,
            response.negative_turn_power_full_w,
        )
    else:
        turn_yaw = 0.0
        turn_power = _effort_value(
            effort, response.straight_power_half_w, response.straight_power_full_w
        )
    turn_weight = abs(turn_fraction) / response.probe_turn_fraction
    target_yaw = turn_weight * turn_yaw
    straight_power = _effort_value(
        effort, response.straight_power_half_w, response.straight_power_full_w
    )
    power = straight_power + turn_weight * (turn_power - straight_power)

    surge, surge_distance = _integrate_drag_limited(
        state.surge_m_s,
        target_surge,
        response.axial_mass_kg,
        response.axial_drag_n_per_m_s2,
        dt,
    )
    yaw_rate, yaw_delta = _integrate_drag_limited(
        state.yaw_rate_rad_s,
        target_yaw,
        response.yaw_inertia_kg_m2,
        response.yaw_drag_nm_per_rad_s2,
        dt,
    )
    next_state = ReducedMotionState(
        yaw_rad=_wrap_pi(state.yaw_rad + yaw_delta),
        surge_m_s=surge,
        yaw_rate_rad_s=yaw_rate,
    )
    measured_cost = power * dt
    outputs = (
        surge,
        surge_distance,
        yaw_rate,
        yaw_delta,
        next_state.yaw_rad,
        power,
        measured_cost,
    )
    if not all(math.isfinite(value) for value in outputs):
        raise ValueError("reduced motion step produced nonfinite output")
    return ReducedMotionStep(
        state=next_state,
        target_surge_m_s=target_surge,
        target_yaw_rate_rad_s=target_yaw,
        surge_distance_m=surge_distance,
        yaw_delta_rad=yaw_delta,
        measured_cost_power_w=power,
        measured_cost_j=measured_cost,
    )
