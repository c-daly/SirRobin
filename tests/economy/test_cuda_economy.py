from __future__ import annotations

from dataclasses import replace

import pytest
import torch

from sirrobin.economy.config import EconomyConfig
from sirrobin.economy.state import EconomyState
from sirrobin.economy.step import EconomyKernel


@pytest.mark.gpu
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
def test_cuda_economy_step_closes_exactly() -> None:
    config = replace(EconomyConfig(), gx=2, gy=2, gz=4, lx_m=20, ly_m=20, lz_m=20)
    state = EconomyState.zeros(config, device="cuda")
    state.nd_q.fill_(10_000_000)
    state.bp_q.fill_(1_000_000)
    state.bd_q[..., 0] = 500_000
    kernel = EconomyKernel(state, config)
    expected = state.total_per_world().clone()
    for _ in range(20):
        ledger = kernel.step()
        assert ledger.books_closed.all()
    assert torch.equal(state.total_per_world(), expected)
    state.validate(config)
