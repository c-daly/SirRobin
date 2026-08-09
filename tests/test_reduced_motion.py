"""Focused contract for the form-derived reduced motion integrator."""

from __future__ import annotations

import json
import math
from dataclasses import FrozenInstanceError, fields, replace
from pathlib import Path

import pytest
import torch

from sirrobin.benchmarks.reduced_calibration import calibrate_reduced_motion
from sirrobin.genetics.develop import develop
from sirrobin.genetics.genotype import GenotypeBatch
from sirrobin.physics.reduced_motion import (
    ReducedMotionResponse,
    ReducedMotionState,
    step_reduced_motion,
)

FIXTURE = Path("oracle/fixtures/live/donor_development_live.json")


def _body(body_id: str = "swimmer"):
    rows = json.loads(FIXTURE.read_text(encoding="utf-8"))["bodies"]
    row = next(item for item in rows if item["id"] == body_id)
    return develop(GenotypeBatch.from_donor_rows([row], dtype=torch.float64))


def _response(**overrides: float | int | bool) -> ReducedMotionResponse:
    values: dict[str, float | int | bool] = {
        "stable_id": 7,
        "probe_turn_fraction": 0.25,
        "surge_half_m_s": 1.0,
        "surge_full_m_s": 3.0,
        "yaw_positive_half_rad_s": 0.2,
        "yaw_positive_full_rad_s": 0.6,
        "yaw_negative_half_rad_s": -0.3,
        "yaw_negative_full_rad_s": -0.8,
        "straight_power_half_w": 4.0,
        "straight_power_full_w": 10.0,
        "positive_turn_power_half_w": 5.0,
        "positive_turn_power_full_w": 13.0,
        "negative_turn_power_half_w": 6.0,
        "negative_turn_power_full_w": 16.0,
        "axial_mass_kg": 2.0,
        "axial_drag_n_per_m_s2": 0.5,
        "yaw_inertia_kg_m2": 3.0,
        "yaw_drag_nm_per_rad_s2": 0.75,
        "calibration_intervention_count": 0,
        "yaw_inertia_floor_used": False,
    }
    values.update(overrides)
    return ReducedMotionResponse(**values)  # type: ignore[arg-type]


def test_calibration_is_repeatable_immutable_and_leaves_form_unchanged() -> None:
    body = _body()
    before = {field.name: getattr(body, field.name).clone() for field in fields(body)}
    arguments = {"warmup_cycles": 0.0, "measurement_cycles": 0.05}

    first = calibrate_reduced_motion(body, **arguments)
    second = calibrate_reduced_motion(body, **arguments)

    assert first == second
    assert first.stable_id == 1
    assert first.probe_turn_fraction == 0.25
    assert first.axial_mass_kg > 0.0
    assert first.axial_drag_n_per_m_s2 > 0.0
    assert first.yaw_inertia_kg_m2 > 0.0
    assert first.yaw_drag_nm_per_rad_s2 >= 0.0
    assert all(math.isfinite(value) for value in first.numeric_values())
    for field in fields(body):
        assert torch.equal(getattr(body, field.name), before[field.name]), field.name
    with pytest.raises(FrozenInstanceError):
        first.surge_full_m_s = 99.0  # type: ignore[misc]


def test_default_calibration_binds_evidenced_signed_form_outcomes() -> None:
    swimmer = calibrate_reduced_motion(_body("swimmer"))
    root_only = calibrate_reduced_motion(_body("root-only"))
    deep_cap = calibrate_reduced_motion(_body("deep-cap"))
    backward = calibrate_reduced_motion(_body("random-14"))

    assert swimmer.surge_full_m_s == pytest.approx(3.843585532456881, abs=1e-12)
    assert swimmer.yaw_positive_full_rad_s == pytest.approx(
        0.22815507731265983, abs=1e-12
    )
    assert swimmer.yaw_negative_full_rad_s == pytest.approx(
        -0.25670417298922177, abs=1e-12
    )
    root_motion = (
        root_only.surge_half_m_s,
        root_only.surge_full_m_s,
        root_only.yaw_positive_half_rad_s,
        root_only.yaw_positive_full_rad_s,
        root_only.yaw_negative_half_rad_s,
        root_only.yaw_negative_full_rad_s,
    )
    assert root_motion == (0.0,) * len(root_motion)
    assert root_only.straight_power_full_w == 0.0
    assert root_only.yaw_inertia_floor_used
    assert root_only.calibration_intervention_count > 0
    assert deep_cap.yaw_positive_full_rad_s > 0.0
    assert deep_cap.yaw_negative_full_rad_s > 0.0
    assert backward.surge_full_m_s < 0.0
    assert backward.straight_power_full_w > 0.0


def test_effort_interpolation_preserves_signed_nonmonotonic_response_knots() -> None:
    response = _response(
        surge_half_m_s=-0.4,
        surge_full_m_s=0.2,
        yaw_positive_half_rad_s=-0.1,
        yaw_positive_full_rad_s=-0.3,
        yaw_negative_half_rad_s=0.05,
        yaw_negative_full_rad_s=0.2,
    )
    rest = ReducedMotionState()

    half_positive = step_reduced_motion(
        rest, response, effort=0.5, turn_fraction=0.25, dt=0.1
    )
    full_negative = step_reduced_motion(
        rest, response, effort=1.0, turn_fraction=-0.25, dt=0.1
    )
    quarter = step_reduced_motion(
        rest, response, effort=0.25, turn_fraction=0.25, dt=0.1
    )
    half_turn = step_reduced_motion(
        rest, response, effort=1.0, turn_fraction=0.125, dt=0.1
    )

    assert half_positive.target_surge_m_s == -0.4
    assert half_positive.target_yaw_rate_rad_s == -0.1
    assert full_negative.target_surge_m_s == 0.2
    assert full_negative.target_yaw_rate_rad_s == 0.2
    assert quarter.target_surge_m_s == -0.2
    assert quarter.target_yaw_rate_rad_s == -0.05
    assert half_turn.target_yaw_rate_rad_s == -0.15


def test_zero_capability_does_not_turn_effort_into_motion_or_work() -> None:
    response = _response(
        surge_half_m_s=0.0,
        surge_full_m_s=0.0,
        yaw_positive_half_rad_s=0.0,
        yaw_positive_full_rad_s=0.0,
        yaw_negative_half_rad_s=0.0,
        yaw_negative_full_rad_s=0.0,
        straight_power_half_w=0.0,
        straight_power_full_w=0.0,
        positive_turn_power_half_w=0.0,
        positive_turn_power_full_w=0.0,
        negative_turn_power_half_w=0.0,
        negative_turn_power_full_w=0.0,
        axial_drag_n_per_m_s2=0.0,
        yaw_drag_nm_per_rad_s2=0.0,
    )

    result = step_reduced_motion(
        ReducedMotionState(), response, effort=1.0, turn_fraction=0.25, dt=10.0
    )

    assert result.state == ReducedMotionState()
    assert result.target_surge_m_s == 0.0
    assert result.target_yaw_rate_rad_s == 0.0
    assert result.surge_distance_m == 0.0
    assert result.yaw_delta_rad == 0.0
    assert result.measured_cost_power_w == 0.0
    assert result.measured_cost_j == 0.0


def test_inertia_and_quadratic_drag_preserve_coasting_without_sign_reversal() -> None:
    response = _response()
    state = ReducedMotionState(surge_m_s=2.0, yaw_rate_rad_s=-1.0)

    result = step_reduced_motion(
        state, response, effort=0.0, turn_fraction=0.0, dt=2.0
    )

    expected_surge = 2.0 / (1.0 + 0.5 * 2.0 * 2.0 / 2.0)
    expected_yaw = -1.0 / (1.0 + 0.75 * 1.0 * 2.0 / 3.0)
    assert result.state.surge_m_s == expected_surge
    assert result.state.yaw_rate_rad_s == expected_yaw
    assert 0.0 < result.state.surge_m_s < state.surge_m_s
    assert state.yaw_rate_rad_s < result.state.yaw_rate_rad_s < 0.0
    assert result.surge_distance_m == pytest.approx(4.0 * math.log(2.0))
    assert result.state.yaw_rad < 0.0


def test_measured_power_is_paid_even_for_backward_motion() -> None:
    response = _response(surge_half_m_s=-1.0, surge_full_m_s=-2.0)

    result = step_reduced_motion(
        ReducedMotionState(), response, effort=1.0, turn_fraction=0.125, dt=0.25
    )

    assert result.target_surge_m_s == -2.0
    assert result.state.surge_m_s < 0.0
    assert result.surge_distance_m < 0.0
    assert result.measured_cost_power_w == 11.5
    assert result.measured_cost_j == 2.875


def test_scalar_distance_and_yaw_are_exact_over_large_dt_and_compose() -> None:
    response = _response(
        yaw_positive_half_rad_s=0.3,
        yaw_positive_full_rad_s=0.6,
    )
    start = ReducedMotionState()

    whole = step_reduced_motion(
        start, response, effort=1.0, turn_fraction=0.25, dt=60.0
    )
    first = step_reduced_motion(
        start, response, effort=1.0, turn_fraction=0.25, dt=20.0
    )
    second = step_reduced_motion(
        first.state, response, effort=1.0, turn_fraction=0.25, dt=40.0
    )

    expected_distance = (45.0 - math.log(2.0)) / 0.25
    assert whole.surge_distance_m == pytest.approx(expected_distance, abs=1e-12)
    assert second.state.surge_m_s == pytest.approx(whole.state.surge_m_s, abs=1e-12)
    assert first.surge_distance_m + second.surge_distance_m == pytest.approx(
        whole.surge_distance_m, abs=1e-12
    )
    assert second.state.yaw_rate_rad_s == pytest.approx(
        whole.state.yaw_rate_rad_s, abs=1e-12
    )
    assert first.yaw_delta_rad + second.yaw_delta_rad == pytest.approx(
        whole.yaw_delta_rad, abs=1e-12
    )
    assert first.measured_cost_j + second.measured_cost_j == whole.measured_cost_j


@pytest.mark.parametrize(
    ("call", "message"),
    [
        (
            lambda: step_reduced_motion(
                ReducedMotionState(),
                _response(),
                effort=-0.01,
                turn_fraction=0,
                dt=1,
            ),
            "effort",
        ),
        (
            lambda: step_reduced_motion(
                ReducedMotionState(),
                _response(),
                effort=1.01,
                turn_fraction=0,
                dt=1,
            ),
            "effort",
        ),
        (
            lambda: step_reduced_motion(
                ReducedMotionState(),
                _response(),
                effort=1,
                turn_fraction=-0.251,
                dt=1,
            ),
            "turn_fraction",
        ),
        (
            lambda: step_reduced_motion(
                ReducedMotionState(),
                _response(),
                effort=1,
                turn_fraction=0.251,
                dt=1,
            ),
            "turn_fraction",
        ),
        (
            lambda: step_reduced_motion(
                ReducedMotionState(), _response(), effort=1, turn_fraction=0, dt=0
            ),
            "dt",
        ),
        (
            lambda: step_reduced_motion(
                ReducedMotionState(),
                _response(),
                effort=1,
                turn_fraction=0,
                dt=1.0e308,
            ),
            "nonfinite output",
        ),
        (
            lambda: step_reduced_motion(
                replace(ReducedMotionState(), yaw_rad=math.nan),
                _response(),
                effort=1,
                turn_fraction=0,
                dt=1,
            ),
            "state",
        ),
        (
            lambda: step_reduced_motion(
                ReducedMotionState(),
                _response(axial_mass_kg=0.0),
                effort=1,
                turn_fraction=0,
                dt=1,
            ),
            "axial_mass_kg",
        ),
        (
            lambda: step_reduced_motion(
                ReducedMotionState(),
                _response(straight_power_full_w=-1.0),
                effort=1,
                turn_fraction=0,
                dt=1,
            ),
            "power",
        ),
    ],
)
def test_reduced_step_rejects_malformed_inputs(call, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        call()
