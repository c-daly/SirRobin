import json
from dataclasses import fields
from pathlib import Path

import torch

from sirrobin.physics.config import LocomotionConfig
from sirrobin.physics.contracts import BodyBatch
from sirrobin.physics.force_fin import fin_channel
from sirrobin.physics.force_reactive import reactive_channel
from sirrobin.physics.mass_matrix import mass_properties, prepare_mass_data
from sirrobin.physics.pose import resolve_pose
from sirrobin.physics.swim_step import SwimKernel


def _data():
    return json.loads(Path("oracle/fixtures/corpus.json").read_text()), json.loads(
        Path("oracle/fixtures/gain1_analytic.json").read_text()
    )


def test_reactive_and_fin_power_identities():
    dtype = torch.float64
    mt, u, vt, slope = (torch.tensor([x], dtype=dtype) for x in (2.0, 0.3, 0.1, 0.2))
    thrust, p_in, p_wake, _ = reactive_channel(mt, u, vt, slope)
    assert torch.allclose(p_in, thrust * u + p_wake, rtol=1e-12, atol=1e-14)
    ar = torch.tensor([4.0], dtype=dtype)
    al = 2 * torch.pi * ar / (ar + 2)
    t_fin, p_fin_in, p_fin = fin_channel(
        al, ar, torch.tensor([0.1], dtype=dtype), u, vt, slope, torch.tensor([True])
    )
    assert torch.allclose(p_fin_in, t_fin * u.clamp_min(0) + p_fin, rtol=1e-12, atol=1e-12)


def test_force_channels_match_independent_gain1_fixtures():
    _, oracle = _data()
    for case in oracle["reactive_cases"]:
        mt, u, vt, slope = (torch.tensor([value], dtype=torch.float64) for value in case["inputs"])
        thrust, input_power, wake_power, _ = reactive_channel(mt, u, vt, slope)
        assert torch.allclose(
            thrust, torch.tensor([case["thrust_n"]], dtype=torch.float64), rtol=1e-12, atol=1e-12
        )
        assert torch.allclose(
            input_power, torch.tensor([case["input_w"]], dtype=torch.float64), rtol=1e-12, atol=1e-12
        )
        assert torch.allclose(
            wake_power, torch.tensor([case["wake_w"]], dtype=torch.float64), rtol=1e-12, atol=1e-12
        )
    for case in oracle["fin_cases"]:
        lift_slope, aspect_ratio, area, u, vt, slope = (
            torch.tensor([value], dtype=torch.float64) for value in case["inputs"]
        )
        thrust, input_power, wake_power = fin_channel(
            lift_slope,
            aspect_ratio,
            area,
            u,
            vt,
            slope,
            torch.tensor([True]),
        )
        assert torch.allclose(
            thrust, torch.tensor([case["thrust_n"]], dtype=torch.float64), rtol=1e-12, atol=1e-12
        )
        assert torch.allclose(
            input_power, torch.tensor([case["input_w"]], dtype=torch.float64), rtol=1e-12, atol=1e-12
        )
        assert torch.allclose(
            wake_power, torch.tensor([case["wake_w"]], dtype=torch.float64), rtol=1e-12, atol=1e-12
        )


def test_signed_wake_flux_and_dissipated_wake_power_are_not_conflated():
    corpus, _ = _data()
    row = next(row for row in corpus["bodies"] if row["id"] == "H0-00")
    config = LocomotionConfig(n_cap=1, n_live=1)
    kernel = SwimKernel(BodyBatch.from_rows([row], config, dtype=torch.float64), config)
    saw_reverse_flow = False
    for _ in range(120):
        ledger = kernel.step()
        assert torch.allclose(
            ledger.p_reactive_in,
            ledger.t_react * ledger.u + ledger.p_wake,
            rtol=1e-12,
            atol=1e-12,
        )
        assert torch.all(ledger.p_wake_dissipated >= 0)
        expected = torch.where(
            ledger.u >= 0, ledger.p_wake, torch.zeros_like(ledger.p_wake)
        )
        assert torch.equal(ledger.p_wake_dissipated, expected)
        saw_reverse_flow |= bool((ledger.u < 0).any())
    assert saw_reverse_flow


def test_gain1_static_body_cases_match_independent_oracle():
    corpus, oracle = _data()
    by_id = {row["id"]: row for row in corpus["bodies"]}
    rows = [by_id[case["id"]] for case in oracle["body_cases"]]
    body = BodyBatch.from_rows(rows, LocomotionConfig(), dtype=torch.float64)
    pose = resolve_pose(body, body.gait_time)
    props = mass_properties(body, pose, LocomotionConfig(), prepare_mass_data(body, LocomotionConfig()))
    for i, case in enumerate(oracle["body_cases"]):
        assert torch.allclose(
            props.mass_sim[i], torch.tensor(case["mass_sim"], dtype=torch.float64), rtol=1e-11, atol=1e-12
        )
        assert torch.allclose(
            props.matrix[i], torch.tensor(case["matrix_kg"], dtype=torch.float64), rtol=1e-9, atol=1e-9
        )


def test_f32_gain1_matrices_and_constrained_solve_match_independent_oracle():
    corpus, oracle = _data()
    by_id = {row["id"]: row for row in corpus["bodies"]}
    rows = [by_id[case["id"]] for case in oracle["body_cases"]]
    config = LocomotionConfig(n_cap=len(rows), n_live=len(rows))
    body = BodyBatch.from_rows(rows, config, dtype=torch.float32)
    pose = resolve_pose(body, body.gait_time)
    props = mass_properties(body, pose, config, prepare_mass_data(body, config))
    for i, case in enumerate(oracle["body_cases"]):
        expected_matrix = torch.tensor(case["matrix_kg"], dtype=torch.float32)
        assert torch.allclose(props.matrix[i], expected_matrix, rtol=1e-4, atol=2e-4)
        impulse = torch.tensor(case["test_impulse_ns"], dtype=torch.float32)
        matrix_xz = props.matrix[i][(0, 2), :][:, (0, 2)]
        actual_dv = torch.linalg.solve(matrix_xz, impulse[[0, 2]])
        expected_dv = torch.tensor(case["constrained_dv_m_s"], dtype=torch.float32)[[0, 2]]
        assert torch.allclose(actual_dv, expected_dv, rtol=1e-4, atol=1e-7)


def test_full_step_is_finite_closes_discrete_identity_and_never_regularizes():
    corpus, _ = _data()
    rows = [row for row in corpus["bodies"] if row["class"] in {"H1", "H2"}][:16]
    body = BodyBatch.from_rows(rows, LocomotionConfig(), dtype=torch.float64)
    ledger = SwimKernel(body, LocomotionConfig()).step()
    scale = torch.stack(
        (ledger.delta_ke.abs(), ledger.work_impulse.abs(), ledger.work_delta_m.abs()), dim=-1
    ).amax(-1)
    assert torch.all(
        ledger.r_step.abs() <= LocomotionConfig().e_atol_f64 + LocomotionConfig().rtol_f64 * scale
    )
    assert not ledger.regularized.any()
    for field in fields(ledger):
        value = getattr(ledger, field.name)
        if isinstance(value, torch.Tensor):
            assert torch.isfinite(value).all()
