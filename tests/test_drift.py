import json
from pathlib import Path

import torch

from sirrobin.physics.config import LocomotionConfig
from sirrobin.physics.contracts import BodyBatch
from sirrobin.physics.swim_step import SwimKernel
from sirrobin.validation.drift import run_prefix_budget


def test_prefix_budget_uses_all_prefixes_and_passes_small_corpus():
    corpus = json.loads(Path("oracle/fixtures/corpus.json").read_text())
    by_id = {row["id"]: row for row in corpus["bodies"]}
    rows = [by_id[body_id] for body_id in ("H1-32", "H2-00", "H2-56")]
    config = LocomotionConfig(n_cap=3, n_live=3)
    body = BodyBatch.from_rows(rows, config, dtype=torch.float32)
    result = run_prefix_budget(SwimKernel(body, config), steps=200, burnin=100)
    assert result.cumulative_residual.shape == (200, 3)
    assert result.prefix_ratio.shape == (200, 3)
    assert torch.all(result.maximum_after_burnin < 1e-3)
    assert result.monotone_bias.shape == (3,)
    assert result.regularization_count == 0
