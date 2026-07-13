import json
from pathlib import Path

import pytest
import torch

from sirrobin.core.live_world import initialize_live_state
from sirrobin.genetics.develop import develop
from sirrobin.genetics.genotype import GenotypeBatch
from sirrobin.physics.contracts import FluidSample
from sirrobin.physics.force_sum import sum_contributions, zero_force
from sirrobin.physics.live_config import LiveLocomotionConfig
from sirrobin.physics.live_step import step_live
from sirrobin.physics.yaw import advance_yaw

FIXTURE = Path("oracle/fixtures/live/donor_development_live.json")
GAIN1 = Path("oracle/fixtures/live/gain1_canonical.json")


def _body(body_id: str, dtype: torch.dtype = torch.float64):
    rows = json.loads(FIXTURE.read_text(encoding="utf-8"))["bodies"]
    row = next(row for row in rows if row["id"] == body_id)
    return develop(GenotypeBatch.from_donor_rows([row], dtype=dtype))


def _bodies(body_id: str, count: int, dtype: torch.dtype = torch.float64):
    rows = json.loads(FIXTURE.read_text(encoding="utf-8"))["bodies"]
    row = next(row for row in rows if row["id"] == body_id)
    return develop(GenotypeBatch.from_donor_rows([row] * count, dtype=dtype))


def test_zero_force_contributor_is_exact_and_additive():
    reference = torch.randn(7, 3, dtype=torch.float64)
    zero = zero_force(reference)
    total = sum_contributions((zero, zero), reference)
    assert torch.equal(total.force_enu_n, torch.zeros_like(reference))
    assert torch.equal(total.torque_yaw_nm, torch.zeros(7, dtype=torch.float64))
    assert torch.equal(total.input_power_w, torch.zeros(7, dtype=torch.float64))
    assert torch.equal(total.dissipated_power_w, torch.zeros(7, dtype=torch.float64))


@pytest.mark.parametrize("dtype,rtol", [(torch.float32, 1e-5), (torch.float64, 1e-12)])
def test_angular_work_identity_matches_frozen_oracle(dtype, rtol):
    rows = json.loads(GAIN1.read_text(encoding="utf-8"))["yaw_cases"]["angular_work"]
    for row in rows:
        l0, l1, i0, i1 = (torch.tensor(value, dtype=dtype) for value in row["inputs"])
        result = advance_yaw(
            torch.zeros((), dtype=dtype),
            l0,
            i0,
            i1,
            (l1 - l0) / 0.125,
            0.125,
            torch.ones((), dtype=torch.bool),
            inertia_floor=1e-9,
            emergency_omega=100.0,
        )
        scale = max(abs(row["delta_ke"]), abs(row["work_impulse"]), 1.0)
        assert abs(float(result.residual_j)) <= rtol * scale
        assert float(result.delta_ke_j) == pytest.approx(row["delta_ke"], rel=rtol, abs=rtol)


def test_quadratic_yaw_decay_one_step_tracks_analytic_solution():
    case = json.loads(GAIN1.read_text(encoding="utf-8"))["yaw_cases"]["quadratic_decay"]
    inertia = torch.tensor(case["inertia"], dtype=torch.float64)
    coefficient = torch.tensor(case["coefficient"], dtype=torch.float64)
    omega = torch.tensor(case["omega0"], dtype=torch.float64)
    momentum = inertia * omega
    dt = 1e-4
    for _ in range(10_000):
        torque = -coefficient * omega * omega.abs()
        result = advance_yaw(
            torch.zeros((), dtype=torch.float64),
            momentum,
            inertia,
            inertia,
            torque,
            dt,
            torch.ones((), dtype=torch.bool),
            inertia_floor=1e-9,
            emergency_omega=100.0,
        )
        momentum, omega = result.momentum, result.omega_after
    expected = case["omega0"] / (
        1.0 + case["coefficient"] * abs(case["omega0"]) * 1.0 / case["inertia"]
    )
    assert float(omega) == pytest.approx(expected, rel=2e-5)


def test_live_step_is_finite_tail_aft_and_has_zero_interventions():
    body = _body("swimmer")
    state = initialize_live_state(body)
    fluid = FluidSample(
        torch.full(body.alive.shape, 1000.0, dtype=torch.float64),
        torch.zeros((*body.alive.shape, 3), dtype=torch.float64),
    )
    ledger = step_live(body, state, fluid, LiveLocomotionConfig())
    assert torch.isfinite(state.velocity_rel_water_enu_m_s).all()
    assert torch.isfinite(state.yaw_rad).all()
    assert not torch.any(ledger.solve_regularized)
    assert not torch.any(ledger.yaw_inertia_floor_hit)
    assert not torch.any(ledger.omega_backstop_hit)
    assert not torch.any(ledger.nonfinite)
    assert torch.allclose(state.velocity_rel_water_enu_m_s[..., 2], torch.zeros_like(state.yaw_rad))
    assert abs(float(ledger.residual_linear_j)) < 1e-10
    assert abs(float(ledger.residual_rot_j)) < 1e-10


def test_open_loop_turn_commands_have_opposite_symmetric_yaw():
    body = _bodies("swimmer", 2)
    state = initialize_live_state(body)
    state.turn_bias_rad_per_depth[0] = torch.tensor([0.025, -0.025], dtype=torch.float64)
    # A half gait-period phase shift reflects the sinusoidal pose. Paired with the
    # opposite DC bias, this is the physically symmetric sign fixture.
    state.gait_time_s[0, 1] = 0.25
    fluid = FluidSample(
        torch.full(body.alive.shape, 1000.0, dtype=torch.float64),
        torch.zeros((*body.alive.shape, 3), dtype=torch.float64),
    )
    config = LiveLocomotionConfig()
    for _ in range(120):
        ledger = step_live(body, state, fluid, config)
        assert not torch.any(ledger.solve_regularized | ledger.yaw_inertia_floor_hit)
    yaw = state.yaw_rad[0]
    assert yaw[0] * yaw[1] < 0
    assert abs(float(yaw[0] + yaw[1])) <= 1e-10
