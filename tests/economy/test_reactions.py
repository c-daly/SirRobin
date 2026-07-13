from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
import torch

from sirrobin.economy.config import EconomyConfig
from sirrobin.economy.reactions import light_at_depth, limitation, reaction_step, remineralization_rates
from sirrobin.economy.state import EconomyState

ROOT = Path(__file__).resolve().parents[2]


def tiny_config(**changes: object) -> EconomyConfig:
    return replace(EconomyConfig(), gx=1, gy=1, gz=4, lx_m=10, ly_m=10, lz_m=20, **changes)


def test_rate_laws_match_independent_numpy_fixtures() -> None:
    fixture = json.loads((ROOT / "oracle/fixtures/economy/reaction_cases.json").read_text(encoding="utf-8"))
    config = EconomyConfig()
    for case in fixture["cases"]:
        values = case["input"]
        depth = torch.tensor(values["depth_m"], dtype=torch.float64)
        nd = torch.tensor(values["nd_mol_m3"], dtype=torch.float64)
        expected = case["expected"]
        assert float(light_at_depth(depth, config)) == pytest.approx(
            expected["light_w_m2"], rel=1e-10, abs=1e-12
        )
        assert float(limitation(nd, depth, config)) == pytest.approx(
            expected["combined_limitation"], rel=1e-10, abs=1e-12
        )
        rates = remineralization_rates(config, device="cpu")
        index = min(int(values["depth_m"] // config.dz_m), config.gz - 1)
        assert float(rates[index]) == pytest.approx(expected["remin_rate_s"], rel=0.03)


def test_martin_and_remineralization_column_match_independent_fixture() -> None:
    fixture = np.load(ROOT / "oracle/fixtures/economy/column_cases.npz")
    config = EconomyConfig()
    rates = remineralization_rates(config, device="cpu").numpy()
    assert np.allclose(rates, fixture["remin_rate_s"], rtol=1e-10, atol=1e-12)
    assert np.all(rates >= config.remin_floor_s)
    martin = fixture["martin_relative_flux"]
    assert np.all(np.diff(martin) < 0)


def test_reactions_close_and_zero_producers_do_not_spontaneously_appear() -> None:
    config = tiny_config()
    state = EconomyState.zeros(config)
    state.nd_q.fill_(1_000_000)
    before = state.total_per_world()
    result = reaction_step(state, config)
    assert torch.equal(state.total_per_world(), before)
    assert state.bp_q.count_nonzero() == 0
    assert result.production_q.item() == 0


def test_density_mortality_is_not_needed_for_baseline_loss() -> None:
    config = tiny_config(density_mortality_m3_mol_s=0.0)
    state = EconomyState.zeros(config)
    state.bp_q.fill_(1_000_000)
    result = reaction_step(state, config)
    assert result.producer_mortality_q.item() > 0
    assert result.producer_maintenance_q.item() > 0
    assert state.total_per_world().item() == 4_000_000
