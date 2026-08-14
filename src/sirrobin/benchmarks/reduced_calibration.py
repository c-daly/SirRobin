"""Rebuild reduced ecological motion response from one developed body."""

from __future__ import annotations

import math
from dataclasses import replace

import torch

from sirrobin.benchmarks.motion_probe import MotionProbeResult, probe_motion
from sirrobin.core.live_world import initialize_live_state
from sirrobin.physics.contracts import DevelopedBody, FluidSample
from sirrobin.physics.live_config import LiveLocomotionConfig
from sirrobin.physics.live_step import step_live
from sirrobin.physics.reduced_motion import ReducedMotionResponse


def _require_same_straight(
    positive: MotionProbeResult, negative: MotionProbeResult
) -> None:
    if (
        positive.stable_id != negative.stable_id
        or positive.effort_fraction != negative.effort_fraction
        or positive.measurement_duration_s != negative.measurement_duration_s
        or positive.cycle_mean_surge_m_s != negative.cycle_mean_surge_m_s
        or positive.straight_mechanical_work_j != negative.straight_mechanical_work_j
    ):
        raise RuntimeError("paired reduced-motion probes disagree on the straight trial")


def _static_coefficients(
    body: DevelopedBody, config: LiveLocomotionConfig
) -> tuple[float, float, float, float, bool]:
    static_body = replace(body, joint_amp_rad=torch.zeros_like(body.joint_amp_rad))
    state = initialize_live_state(static_body)
    state.velocity_rel_water_enu_m_s[..., 0].fill_(1.0)
    fluid = FluidSample(
        torch.full(
            body.alive.shape,
            config.rho_water,
            dtype=body.mass_sim.dtype,
            device=body.mass_sim.device,
        ),
        torch.zeros(
            (*body.alive.shape, 3),
            dtype=body.mass_sim.dtype,
            device=body.mass_sim.device,
        ),
    )
    ledger = step_live(static_body, state, fluid, config)
    matrix = ledger.hydrodynamics.effective_mass_before_kg.reshape(-1, 3, 3)[0, :2, :2]
    axial_mobility = torch.linalg.inv(matrix)[0, 0]
    axial_mass = float((1.0 / axial_mobility).item())
    axial_drag = -float(ledger.hydrodynamics.drag_force_enu_n.reshape(-1, 3)[0, 0].item())
    raw_yaw_inertia = float(
        ledger.hydrodynamics.yaw_inertia_before_kg_m2.reshape(-1)[0].item()
    )
    yaw_drag = float(ledger.hydrodynamics.yaw_drag_coefficient.reshape(-1)[0].item())
    floor_used = raw_yaw_inertia < config.inertia_floor_kg_m2
    yaw_inertia = max(raw_yaw_inertia, config.inertia_floor_kg_m2)
    values = (axial_mass, axial_drag, yaw_inertia, yaw_drag)
    if not all(math.isfinite(value) for value in values):
        raise ValueError("developed form produced nonfinite reduced-motion coefficients")
    if axial_mass <= 0.0 or axial_drag < 0.0 or yaw_drag < 0.0:
        raise ValueError("developed form produced invalid reduced-motion coefficients")
    return axial_mass, axial_drag, yaw_inertia, yaw_drag, floor_used


@torch.inference_mode()
def calibrate_reduced_motion(
    body: DevelopedBody,
    *,
    probe_turn_fraction: float = 0.25,
    warmup_cycles: float = 1.0,
    measurement_cycles: float = 2.0,
    config: LiveLocomotionConfig | None = None,
) -> ReducedMotionResponse:
    """Measure the evidenced 1/2-cycle response pending Slice 1.4 comparison."""
    if not math.isfinite(probe_turn_fraction) or not 0.0 < probe_turn_fraction <= 1.0:
        raise ValueError("probe_turn_fraction must be finite and in (0, 1]")
    resolved_config = config or LiveLocomotionConfig()
    resolved_config.validate()

    common = {
        "warmup_cycles": warmup_cycles,
        "measurement_cycles": measurement_cycles,
        "config": resolved_config,
    }
    half_positive = probe_motion(
        body, effort_fraction=0.5, turn_fraction=probe_turn_fraction, **common
    )
    half_negative = probe_motion(
        body, effort_fraction=0.5, turn_fraction=-probe_turn_fraction, **common
    )
    full_positive = probe_motion(
        body, effort_fraction=1.0, turn_fraction=probe_turn_fraction, **common
    )
    full_negative = probe_motion(
        body, effort_fraction=1.0, turn_fraction=-probe_turn_fraction, **common
    )
    _require_same_straight(half_positive, half_negative)
    _require_same_straight(full_positive, full_negative)
    axial_mass, axial_drag, yaw_inertia, yaw_drag, floor_used = _static_coefficients(
        body, resolved_config
    )

    response = ReducedMotionResponse(
        stable_id=half_positive.stable_id,
        probe_turn_fraction=probe_turn_fraction,
        surge_half_m_s=half_positive.cycle_mean_surge_m_s,
        surge_full_m_s=full_positive.cycle_mean_surge_m_s,
        yaw_positive_half_rad_s=half_positive.cycle_mean_yaw_response_rad_s,
        yaw_positive_full_rad_s=full_positive.cycle_mean_yaw_response_rad_s,
        yaw_negative_half_rad_s=half_negative.cycle_mean_yaw_response_rad_s,
        yaw_negative_full_rad_s=full_negative.cycle_mean_yaw_response_rad_s,
        straight_power_half_w=(
            half_positive.straight_mechanical_work_j
            / half_positive.measurement_duration_s
        ),
        straight_power_full_w=(
            full_positive.straight_mechanical_work_j
            / full_positive.measurement_duration_s
        ),
        positive_turn_power_half_w=(
            half_positive.turning_mechanical_work_j
            / half_positive.measurement_duration_s
        ),
        positive_turn_power_full_w=(
            full_positive.turning_mechanical_work_j
            / full_positive.measurement_duration_s
        ),
        negative_turn_power_half_w=(
            half_negative.turning_mechanical_work_j
            / half_negative.measurement_duration_s
        ),
        negative_turn_power_full_w=(
            full_negative.turning_mechanical_work_j
            / full_negative.measurement_duration_s
        ),
        axial_mass_kg=axial_mass,
        axial_drag_n_per_m_s2=axial_drag,
        yaw_inertia_kg_m2=yaw_inertia,
        yaw_drag_nm_per_rad_s2=yaw_drag,
        calibration_intervention_count=sum(
            probe.intervention_count
            for probe in (
                half_positive,
                half_negative,
                full_positive,
                full_negative,
            )
        ),
        yaw_inertia_floor_used=floor_used,
    )
    response.validate()
    return response
