from __future__ import annotations

import json
from pathlib import Path

import pytest

from sirrobin.economy.config import DEFAULT_ECONOMY_CONFIG, EconomyConfig

ROOT = Path(__file__).resolve().parents[2]


def test_frozen_anchor_manifest_matches_config() -> None:
    manifest = json.loads((ROOT / "oracle/fixtures/economy/anchor_manifest.json").read_text(encoding="utf-8"))
    assert manifest["frozen_before_dynamics"] is True
    assert manifest["config_hash"] == DEFAULT_ECONOMY_CONFIG.sha256()
    assert all(
        {"symbol", "value", "unit", "valid_range", "classification", "source"} <= set(item)
        for item in manifest["anchors"]
    )


def test_config_pins_stability_and_bottom_turnover() -> None:
    config = DEFAULT_ECONOMY_CONFIG
    config.validate()
    sub_dt = config.dt_eco_s / config.transport_substeps
    assert 2 * config.max_kz_m2_s * sub_dt / config.dz_m**2 <= 1
    assert config.sinking_speed_m_s * sub_dt / config.dz_m <= 1
    assert config.remin_floor_s > 0
    assert 1 / config.remin_floor_s < 1_000_000 * config.dt_eco_s
    assert config.with_half_timestep().dt_eco_s == config.dt_eco_s / 2


def test_invalid_reaction_step_is_rejected() -> None:
    with pytest.raises(ValueError, match="reaction timestep"):
        EconomyConfig(mu_max_s=10 / 86_400).validate()
