import json
from pathlib import Path

import numpy as np
import torch

from sirrobin.physics.lamb import gl_nodes_weights, lamb_coefficients, lamb_factors


def test_pinned_gl256_and_negative_gl32_evidence():
    pinned = json.loads(Path("oracle/fixtures/quadrature_gl256.json").read_text())
    nodes, weights = gl_nodes_weights(256)
    assert np.array_equal(nodes, np.asarray(pinned["nodes_0_1"]))
    assert np.array_equal(weights, np.asarray(pinned["weights_0_1"]))
    oracle = json.loads(Path("oracle/fixtures/gain1_analytic.json").read_text())
    assert max(case["gl256_vs_quad_max_rel"] for case in oracle["lamb_cases"]) < 1e-8
    assert max(case["gl32_negative_max_rel"] for case in oracle["lamb_cases"]) > 1e-5


def test_lamb_matches_independent_fixtures():
    oracle = json.loads(Path("oracle/fixtures/gain1_analytic.json").read_text())
    for case in oracle["lamb_cases"]:
        abc = torch.tensor(case["abc_m"], dtype=torch.float64)
        coeff = lamb_coefficients(abc)
        factor = lamb_factors(abc)
        assert torch.allclose(coeff, torch.tensor(case["coeff"], dtype=torch.float64), rtol=1e-10, atol=1e-12)
        assert torch.allclose(
            factor, torch.tensor(case["factor"], dtype=torch.float64), rtol=1e-10, atol=1e-12
        )
        assert abs(float(coeff.sum()) - 2.0) < 1e-12
