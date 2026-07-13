from __future__ import annotations

from dataclasses import replace

from sirrobin.economy.config import EconomyConfig
from sirrobin.economy.ledger import MassLedger
from sirrobin.economy.state import EconomyState


def test_close_books_is_exact_per_world() -> None:
    config = replace(EconomyConfig(), worlds=2, gx=1, gy=1, gz=2, lx_m=10, ly_m=10, lz_m=10)
    state = EconomyState.zeros(config)
    state.nd_q[0].fill_(100)
    state.nd_q[1].fill_(200)
    ledger = MassLedger.from_state(state)
    assert ledger.close_books(state).tolist() == [True, True]
    state.nd_q[1, 0, 0, 0] += 1
    assert ledger.close_books(state).tolist() == [True, False]


def test_state_rejects_the_2_to_62_boundary() -> None:
    config = replace(EconomyConfig(), gx=1, gy=1, gz=2, lx_m=10, ly_m=10, lz_m=10)
    state = EconomyState.zeros(config)
    state.nd_q[0, 0, 0, 0] = 2**62
    try:
        state.validate(config)
    except ValueError as error:
        assert "[0,2^62)" in str(error)
    else:
        raise AssertionError("2^62 must not enter reservoir state")


def test_state_rejects_inventory_above_safe_reduction_bound() -> None:
    config = replace(EconomyConfig(), gx=1, gy=1, gz=2, lx_m=10, ly_m=10, lz_m=10)
    state = EconomyState.zeros(config)
    state.nd_q[0, 0, 0, 0] = config.max_inventory_q
    try:
        state.validate(config)
    except ValueError as error:
        assert "safe reduction" in str(error)
    else:
        raise AssertionError("unsafe aggregate inventory must be rejected")
