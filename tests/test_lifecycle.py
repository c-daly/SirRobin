import json
from pathlib import Path

import torch

from sirrobin.benchmarks.lifecycle import apply_fixed_churn, tensor_addresses
from sirrobin.physics.config import LocomotionConfig
from sirrobin.physics.contracts import BodyBatch
from sirrobin.physics.swim_step import SwimKernel


def test_fixed_churn_preserves_addresses_and_reproduces_discrete_records():
    rows = json.loads(Path("oracle/fixtures/corpus.json").read_text())["bodies"][:16]
    first = BodyBatch.from_rows(rows, LocomotionConfig(n_cap=16, n_live=16), dtype=torch.float32)
    second = BodyBatch.from_rows(rows, LocomotionConfig(n_cap=16, n_live=16), dtype=torch.float32)
    addresses = tensor_addresses(first)
    kernel = SwimKernel(first, LocomotionConfig(n_cap=16, n_live=16))
    kernel.step()
    assert tensor_addresses(first) == addresses
    event1 = apply_fixed_churn(first, 1000, fraction=0.125)
    event2 = apply_fixed_churn(second, 1000, fraction=0.125)
    assert event1 == event2
    assert tensor_addresses(first) == addresses
    assert first.seg_mask.shape == (16, 17)
