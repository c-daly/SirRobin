from __future__ import annotations

from dataclasses import replace

import pytest
import torch

from sirrobin.economy.config import EconomyConfig
from sirrobin.economy.state import EconomyState
from sirrobin.fields.geometry import GridGeometry
from sirrobin.fields.grid import ScalarGrid
from sirrobin.fields.transport import mix_vertical, sink_vertical
from sirrobin.numerics.flux import INT64_SAFE_MAX


def config_2x2x2(**changes: object) -> EconomyConfig:
    return replace(EconomyConfig(), gx=2, gy=2, gz=2, lx_m=2, ly_m=2, lz_m=2, **changes)


def test_grid_center_and_periodic_sampling_are_continuous() -> None:
    config = config_2x2x2()
    state = EconomyState.zeros(config)
    state.nd_q[0, 0, 0, 0] = 1_000_000_000
    grid = ScalarGrid(state.nd_q, GridGeometry.from_config(config), q_mass_mol=config.q_mass_mol)
    center = torch.tensor([[[0.5, 0.5, -0.5]]], dtype=torch.float64)
    wrapped = torch.tensor([[[2.5, 0.5, -0.5]]], dtype=torch.float64)
    first = grid.sample(center)
    second = grid.sample(wrapped)
    assert torch.equal(first.value_mol_m3, second.value_mol_m3)
    assert torch.isfinite(first.gradient_mol_m4).all()


def test_world_z_gradient_uses_enu_sign_and_rejects_above_surface() -> None:
    config = config_2x2x2()
    state = EconomyState.zeros(config)
    state.nd_q[..., 1] = 1_000_000
    grid = ScalarGrid(state.nd_q, GridGeometry.from_config(config), q_mass_mol=config.q_mass_mol)
    sample = grid.sample(torch.tensor([[[0.5, 0.5, -1.0]]], dtype=torch.float64))
    assert sample.gradient_mol_m4[0, 0, 2] < 0
    with pytest.raises(ValueError, match="outside"):
        grid.sample(torch.tensor([[[0.5, 0.5, 0.1]]], dtype=torch.float64))


def test_point_depletion_is_an_exact_transaction() -> None:
    config = config_2x2x2()
    state = EconomyState.zeros(config)
    state.nd_q.fill_(100)
    grid = ScalarGrid(state.nd_q, GridGeometry.from_config(config), q_mass_mol=config.q_mass_mol)
    before = int(state.nd_q.sum())
    realized = grid.deplete_at(0, torch.tensor([1.0, 1.0, -1.0]), 123)
    assert realized == 123
    assert int(state.nd_q.sum()) == before - realized
    assert torch.all(state.nd_q >= 0)


@pytest.mark.parametrize(
    "position",
    (
        (0.5, 0.5, -0.5),
        (0.5, 0.5, 0.0),
        (2.5, 0.5, -0.5),
    ),
)
def test_point_depletion_never_takes_zero_weight_stock(
    position: tuple[float, float, float],
) -> None:
    config = config_2x2x2()
    state = EconomyState.zeros(config)
    state.nd_q.fill_(1_000)
    state.nd_q[0, 0, 0, 0] = 1
    grid = ScalarGrid(state.nd_q, GridGeometry.from_config(config), q_mass_mol=config.q_mass_mol)

    realized = grid.deplete_at(0, torch.tensor(position), 100)

    assert realized == 1
    assert state.nd_q[0, 0, 0, 0].item() == 0
    assert int(state.nd_q.sum().item()) == 7_000


def test_point_deposit_is_an_exact_local_credit() -> None:
    config = config_2x2x2()
    state = EconomyState.zeros(config)
    grid = ScalarGrid(state.nd_q, GridGeometry.from_config(config), q_mass_mol=config.q_mass_mol)

    credited = grid.deposit_at(0, torch.tensor([1.0, 1.0, -1.0]), 123)

    assert credited == 123
    assert int(state.nd_q.sum()) == credited
    assert state.nd_q.count_nonzero() <= 8


def test_point_deposit_rejects_float_apportionment_mint_before_mutation() -> None:
    config = config_2x2x2()
    state = EconomyState.zeros(config)
    grid = ScalarGrid(state.nd_q, GridGeometry.from_config(config), q_mass_mol=config.q_mass_mol)

    with pytest.raises(ValueError, match="exactly apportion"):
        grid.deposit_at(
            0,
            torch.tensor([1.0, 0.5, -0.5]),
            INT64_SAFE_MAX - 2,
        )

    assert int(state.nd_q.sum().item()) == 0


def test_point_depletion_rejects_float_apportionment_overdebit_before_mutation() -> None:
    config = replace(
        EconomyConfig(),
        gx=1,
        gy=1,
        gz=2,
        lx_m=1.0,
        ly_m=1.0,
        lz_m=2.0,
        max_inventory_q=INT64_SAFE_MAX - 1,
    )
    state = EconomyState.zeros(config)
    stock_q = 2**61 - 1
    state.nd_q[0, 0, 0] = torch.tensor([stock_q, stock_q], dtype=torch.int64)
    grid = ScalarGrid(state.nd_q, GridGeometry.from_config(config), q_mass_mol=config.q_mass_mol)
    before = state.nd_q.clone()

    with pytest.raises(ValueError, match="exactly apportioned"):
        grid.deplete_at(
            0,
            torch.tensor([0.5, 0.5, -1.0]),
            INT64_SAFE_MAX - 3,
        )

    assert torch.equal(state.nd_q, before)


def test_mixing_recolonizes_producers_but_cannot_create_them() -> None:
    config = replace(config_2x2x2(), kz_bp_m2_s=1e-4)
    state = EconomyState.zeros(config)
    result = mix_vertical(
        state.bp_q,
        state.carries.mix_bp_mol,
        config.kz_bp_m2_s,
        config,
        dt_s=config.dt_eco_s / config.transport_substeps,
    )
    assert state.bp_q.count_nonzero() == 0
    assert result.moved_q.sum() == 0
    state.bp_q[..., 0] = 1_000_000
    before = state.bp_q.sum()
    mix_vertical(
        state.bp_q,
        state.carries.mix_bp_mol,
        config.kz_bp_m2_s,
        config,
        dt_s=config.dt_eco_s / config.transport_substeps,
    )
    assert torch.equal(state.bp_q.sum(), before)
    assert torch.all(state.bp_q[..., 1] > 0)


def test_multi_face_overdraft_uses_one_source_budget() -> None:
    config = replace(EconomyConfig(), gx=1, gy=1, gz=3, lx_m=10, ly_m=10, lz_m=15)
    state = EconomyState.zeros(config)
    state.nd_q[0, 0, 0] = torch.tensor([0, 5, 0], dtype=torch.int64)
    before = state.nd_q.sum()
    result = mix_vertical(
        state.nd_q,
        state.carries.mix_nd_mol,
        1e-4,
        config,
        dt_s=1_000_000.0,
    )
    assert torch.equal(state.nd_q.sum(), before)
    assert state.nd_q[0, 0, 0].tolist() == [3, 0, 2]
    assert result.shortfall_q.item() > 0


def test_sinking_closes_at_bottom_and_never_deletes_detritus() -> None:
    config = config_2x2x2()
    state = EconomyState.zeros(config)
    state.bd_q[..., 0] = 1_000_000
    state.bd_q[..., 1] = 500_000
    before = state.bd_q.sum()
    sink_vertical(
        state.bd_q,
        state.carries.sinking_mol,
        config,
        dt_s=config.dt_eco_s / config.transport_substeps,
    )
    assert torch.equal(state.bd_q.sum(), before)
    assert torch.all(state.bd_q[..., 1] >= 500_000)
