from __future__ import annotations

from dataclasses import replace

import pytest
import torch

from sirrobin.economy.config import EconomyConfig
from sirrobin.economy.state import EconomyState
from sirrobin.economy.step import EconomyKernel
from sirrobin.validation.economy import compare_half_timestep, pulse_state, run_pulse


@pytest.mark.slow
def test_uncapped_bloom_crashes_and_converges_at_half_timestep() -> None:
    coarse_config = replace(
        EconomyConfig(),
        gx=1,
        gy=1,
        gz=16,
        lx_m=10,
        ly_m=10,
        lz_m=160,
        density_mortality_m3_mol_s=0.0,
    )
    fine_config = coarse_config.with_half_timestep()
    coarse = run_pulse(coarse_config, steps=1_000, force_d_dd_zero=True)
    fine = run_pulse(fine_config, steps=2_000, force_d_dd_zero=True)
    assert coarse.bloom_passes
    assert fine.bloom_passes
    convergence = compare_half_timestep(coarse, fine, coarse_config.dt_eco_s)
    assert convergence.passes, convergence
    assert max(convergence.relative_differences.values()) <= 0.05


@pytest.mark.slow
def test_light_creates_vertical_producer_zonation_and_no_light_removes_it() -> None:
    base = replace(EconomyConfig(), gx=1, gy=1, gz=16, lx_m=10, ly_m=10, lz_m=160)
    lit = pulse_state(base)
    dark_config = replace(base, i0_w_m2=0.0)
    dark = pulse_state(dark_config)
    lit_kernel = EconomyKernel(lit, base)
    dark_kernel = EconomyKernel(dark, dark_config)
    initial_dark_bp = int(dark.bp_q.sum())
    dark_production = 0
    for _ in range(300):
        lit_kernel.step()
        dark_ledger = dark_kernel.step()
        dark_production += int(dark_ledger.production_q.sum())
    assert lit.bp_q[..., :4].to(torch.float64).mean() > lit.bp_q[..., -4:].to(torch.float64).mean()
    assert dark_production == 0
    assert int(dark.bp_q.sum()) < initial_dark_bp
    assert torch.equal(dark.bp_q[..., :4].sum(), dark.bp_q[..., -4:].sum())


def test_no_mixing_control_cannot_return_deep_nutrient() -> None:
    mixed_config = replace(EconomyConfig(), gx=1, gy=1, gz=8, lx_m=10, ly_m=10, lz_m=80)
    still_config = replace(mixed_config, kz_nd_m2_s=0.0, kz_bp_m2_s=0.0, kz_bm_m2_s=0.0)

    def nutrient_column(config: EconomyConfig) -> EconomyState:
        state = EconomyState.zeros(config)
        state.nd_q[..., -2:] = 10_000_000
        return state

    mixed = nutrient_column(mixed_config)
    still = nutrient_column(still_config)
    mixed_kernel = EconomyKernel(mixed, mixed_config)
    still_kernel = EconomyKernel(still, still_config)
    for _ in range(100):
        assert mixed_kernel.step().books_closed.all()
        assert still_kernel.step().books_closed.all()
    assert int(mixed.nd_q[..., :2].sum()) > 0
    assert int(still.nd_q[..., :2].sum()) == 0
