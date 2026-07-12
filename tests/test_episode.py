import json
from pathlib import Path

import torch

from sirrobin.benchmarks.episode import run_episode
from sirrobin.physics.config import LocomotionConfig
from sirrobin.physics.contracts import BodyBatch
from sirrobin.physics.swim_step import SwimKernel


def test_short_episode_is_finite_and_has_no_regularization():
    corpus = json.loads(Path("oracle/fixtures/corpus.json").read_text())
    rows = [row for row in corpus["bodies"] if row["class"] == "H1"][:4]
    body = BodyBatch.from_rows(rows, LocomotionConfig(n_cap=4, n_live=4), dtype=torch.float64)
    result = run_episode(
        SwimKernel(body, LocomotionConfig(n_cap=4, n_live=4)), warmup_steps=2, measure_steps=4
    )
    assert torch.isfinite(result.cruise_speed).all()
    assert torch.isfinite(result.cost_of_transport).all()
    assert torch.isfinite(result.reactive_ratio).all()
    assert result.regularization_count == 0
