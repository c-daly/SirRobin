import json
from pathlib import Path

import torch

from sirrobin.numerics.solve_constrained_xz import solve_constrained_xz
from sirrobin.numerics.solve_donor import solve_sym3_donor


def test_exact_and_invalid_branches_are_finite():
    matrix = torch.tensor(
        [[[2.0, 0.0, 0.2], [0.0, 1.0, 0.0], [0.2, 0.0, 3.0]], torch.zeros((3, 3))],
        dtype=torch.float64,
    )
    impulse = torch.tensor([[0.4, 0.0, -0.2], [0.0, 0.0, 0.0]], dtype=torch.float64)
    result = solve_constrained_xz(matrix, impulse, torch.tensor([True, False]))
    expected = torch.linalg.solve(matrix[0][(0, 2), :][:, (0, 2)], impulse[0, (0, 2)])
    assert torch.allclose(result.dv[0, (0, 2)], expected)
    assert torch.equal(result.dv[1], torch.zeros(3, dtype=torch.float64))
    assert torch.isfinite(result.dv).all()


def test_regularization_sign_closes_momentum():
    matrix = torch.diag_embed(torch.tensor([[1e-8, 1.0, 1.0]], dtype=torch.float64))
    impulse = torch.tensor([[1e-4, 0.0, 2e-4]], dtype=torch.float64)
    result = solve_constrained_xz(matrix, impulse, torch.tensor([True]), kappa_max=1e6)
    assert result.regularized.item()
    lhs = matrix[0][(0, 2), :][:, (0, 2)] @ result.dv[0, (0, 2)]
    rhs = impulse[0, (0, 2)] + result.j_reg[0, (0, 2)]
    assert torch.allclose(lhs, rhs, rtol=1e-10, atol=1e-14)


def test_tilted_fixture_exposes_donor_3d_then_zero_bug():
    oracle = json.loads(Path("oracle/fixtures/gain1_analytic.json").read_text())
    case = next(case for case in oracle["body_cases"] if case["id"] == "H2-58")
    matrix = torch.tensor([case["matrix_kg"]], dtype=torch.float64)
    impulse = torch.tensor([case["test_impulse_ns"]], dtype=torch.float64)
    donor_xz = solve_sym3_donor(matrix, impulse)[0, [0, 2]]
    constrained_xz = torch.tensor(case["constrained_dv_m_s"], dtype=torch.float64)[[0, 2]]
    relative_difference = torch.linalg.vector_norm(donor_xz - constrained_xz) / torch.linalg.vector_norm(
        constrained_xz
    )
    assert relative_difference > 5e-4
