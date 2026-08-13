"""Pure standardized motion measurements from the canonical full physics.

The probe is diagnostic only. It scales the developed gait amplitude by a bounded
effort fraction, runs independent straight and open-loop turning trials in still
water, and returns immutable Python values. It neither mutates the developed body
nor installs a second authoritative capability state in the live world.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, replace

import torch

from sirrobin.core.controller import turn_authority
from sirrobin.core.live_world import initialize_live_state
from sirrobin.physics.contracts import DevelopedBody, FluidSample
from sirrobin.physics.live_config import LiveLocomotionConfig
from sirrobin.physics.live_step import step_live
from sirrobin.physics.pose_live import forward_left
from sirrobin.physics.yaw import wrap_pi


@dataclass(frozen=True, slots=True)
class MotionProbeResult:
    stable_id: int
    effort_fraction: float
    turn_fraction: float
    warmup_steps: int
    measurement_steps: int
    measurement_duration_s: float
    measurement_phase_cycles: float
    cycle_mean_surge_m_s: float
    cycle_mean_yaw_response_rad_s: float
    straight_mechanical_work_j: float
    turning_mechanical_work_j: float
    regularization_count: int
    inertia_floor_count: int
    omega_backstop_count: int
    nonfinite_count: int

    @property
    def intervention_count(self) -> int:
        return (
            self.regularization_count
            + self.inertia_floor_count
            + self.omega_backstop_count
            + self.nonfinite_count
        )


@dataclass(frozen=True, slots=True)
class _TrialResult:
    surge_displacement_m: float
    yaw_delta_rad: float
    mechanical_work_j: float
    regularization_count: int
    inertia_floor_count: int
    omega_backstop_count: int
    nonfinite_count: int


def _steps_for_cycles(cycles: float, frequency_hz: float, dt: float) -> int:
    return max(1, round(cycles / (frequency_hz * dt)))


def _run_trial(
    body: DevelopedBody,
    fluid: FluidSample,
    config: LiveLocomotionConfig,
    *,
    turn_fraction: float,
    warmup_steps: int,
    measurement_steps: int,
) -> _TrialResult:
    state = initialize_live_state(body)
    state.turn_bias_rad_per_depth.copy_(turn_fraction * turn_authority(body, config))
    for _ in range(warmup_steps):
        step_live(body, state, fluid, config)

    surge_displacement_m = 0.0
    yaw_delta_rad = 0.0
    mechanical_work_j = 0.0
    regularization_count = 0
    inertia_floor_count = 0
    omega_backstop_count = 0
    nonfinite_count = 0
    for _ in range(measurement_steps):
        yaw_before = state.yaw_rad.clone()
        ledger = step_live(body, state, fluid, config)
        forward, _ = forward_left(state.yaw_rad)
        surge = (state.velocity_rel_water_enu_m_s * forward).sum(-1)
        surge_displacement_m += float(surge.item()) * config.dt
        yaw_delta_rad += float(wrap_pi(state.yaw_rad - yaw_before).item())
        mechanical_work_j += float(ledger.total.dissipated_power_w.item()) * config.dt
        regularization_count += int(ledger.solve_regularized.sum().item())
        inertia_floor_count += int(ledger.yaw_inertia_floor_hit.sum().item())
        omega_backstop_count += int(ledger.omega_backstop_hit.sum().item())
        nonfinite_count += int(ledger.nonfinite.sum().item())
    return _TrialResult(
        surge_displacement_m,
        yaw_delta_rad,
        mechanical_work_j,
        regularization_count,
        inertia_floor_count,
        omega_backstop_count,
        nonfinite_count,
    )


@torch.inference_mode()
def probe_motion(
    body: DevelopedBody,
    *,
    effort_fraction: float,
    turn_fraction: float,
    warmup_cycles: float = 3.0,
    measurement_cycles: float = 4.0,
    config: LiveLocomotionConfig | None = None,
) -> MotionProbeResult:
    """Measure one body at bounded effort using paired full-physics trials.

    Surge and straight work come from the zero-turn trial. Yaw response is the
    turning trial's mean yaw rate minus the straight trial's mean yaw rate, so a
    body's uncommanded asymmetry is not mislabeled as control authority. Mechanical
    work is the time integral of the full physics' named dissipated power.
    """
    if not math.isfinite(effort_fraction) or not 0.0 <= effort_fraction <= 1.0:
        raise ValueError("effort_fraction must be finite and in [0, 1]")
    if not math.isfinite(turn_fraction) or not -1.0 <= turn_fraction <= 1.0:
        raise ValueError("turn_fraction must be finite and in [-1, 1]")
    if not math.isfinite(warmup_cycles) or warmup_cycles < 0.0:
        raise ValueError("warmup_cycles must be finite and nonnegative")
    if not math.isfinite(measurement_cycles) or measurement_cycles <= 0.0:
        raise ValueError("measurement_cycles must be finite and positive")
    if body.batch_size != 1 or not bool(body.alive.item()):
        raise ValueError("motion probe requires exactly one live developed body")

    resolved_config = config or LiveLocomotionConfig()
    resolved_config.validate()
    frequency_hz = float(body.swim_freq_hz.item())
    if not math.isfinite(frequency_hz) or frequency_hz <= 0.0:
        raise ValueError("motion probe requires positive finite swim frequency")
    warmup_steps = (
        0
        if warmup_cycles == 0.0
        else _steps_for_cycles(warmup_cycles, frequency_hz, resolved_config.dt)
    )
    measurement_steps = _steps_for_cycles(
        measurement_cycles, frequency_hz, resolved_config.dt
    )
    duration_s = measurement_steps * resolved_config.dt

    # Effort is a transient diagnostic input. The original developed form remains
    # untouched and is still the sole authority for maximum gait amplitude.
    probe_body = replace(body, joint_amp_rad=body.joint_amp_rad * effort_fraction)
    fluid = FluidSample(
        torch.full(
            body.alive.shape,
            resolved_config.rho_water,
            dtype=body.mass_sim.dtype,
            device=body.mass_sim.device,
        ),
        torch.zeros(
            (*body.alive.shape, 3),
            dtype=body.mass_sim.dtype,
            device=body.mass_sim.device,
        ),
    )
    straight = _run_trial(
        probe_body,
        fluid,
        resolved_config,
        turn_fraction=0.0,
        warmup_steps=warmup_steps,
        measurement_steps=measurement_steps,
    )
    turning = _run_trial(
        probe_body,
        fluid,
        resolved_config,
        turn_fraction=turn_fraction,
        warmup_steps=warmup_steps,
        measurement_steps=measurement_steps,
    )
    return MotionProbeResult(
        stable_id=int(body.stable_id.item()),
        effort_fraction=effort_fraction,
        turn_fraction=turn_fraction,
        warmup_steps=warmup_steps,
        measurement_steps=measurement_steps,
        measurement_duration_s=duration_s,
        measurement_phase_cycles=duration_s * frequency_hz,
        cycle_mean_surge_m_s=straight.surge_displacement_m / duration_s,
        cycle_mean_yaw_response_rad_s=(
            turning.yaw_delta_rad - straight.yaw_delta_rad
        )
        / duration_s,
        straight_mechanical_work_j=straight.mechanical_work_j,
        turning_mechanical_work_j=turning.mechanical_work_j,
        regularization_count=(
            straight.regularization_count + turning.regularization_count
        ),
        inertia_floor_count=straight.inertia_floor_count + turning.inertia_floor_count,
        omega_backstop_count=straight.omega_backstop_count + turning.omega_backstop_count,
        nonfinite_count=straight.nonfinite_count + turning.nonfinite_count,
    )
