from __future__ import annotations

from dataclasses import fields, replace

import torch

from sirrobin.economy.config import EconomyConfig
from sirrobin.economy.snapshot import load_snapshot, save_snapshot
from sirrobin.economy.state import EconomyState
from sirrobin.economy.step import EconomyKernel, advance_economy_unchecked


def tiny_config() -> EconomyConfig:
    return replace(EconomyConfig(), gx=1, gy=1, gz=4, lx_m=10, ly_m=10, lz_m=20)


def seeded_state(config: EconomyConfig) -> EconomyState:
    state = EconomyState.zeros(config)
    state.nd_q.fill_(10_000_000)
    state.bp_q.fill_(1_000_000)
    state.bd_q[..., 0] = 500_000
    return state


def test_full_step_closes_exact_books() -> None:
    config = tiny_config()
    state = seeded_state(config)
    kernel = EconomyKernel(state, config)
    expected = state.total_per_world().clone()
    for _ in range(20):
        ledger = kernel.step()
        assert ledger.books_closed.all()
        assert torch.equal(state.total_per_world(), expected)
        assert torch.all(ledger.intervention_count == 0)
        assert torch.all(ledger.transport_shortfall_q == 0)
        state.validate(config)


def test_functional_device_step_matches_reference_without_mutating_input() -> None:
    config = tiny_config()
    state = seeded_state(config)
    before = state.clone()

    expected_state = state.clone()
    expected = EconomyKernel(expected_state, config).step()
    actual = advance_economy_unchecked(state, config)

    for original, unchanged in zip(
        state.reservoirs, before.reservoirs, strict=True
    ):
        assert torch.equal(original, unchanged)
    for expected_value, actual_value in zip(
        expected_state.reservoirs, actual.state.reservoirs, strict=True
    ):
        assert torch.equal(actual_value, expected_value)
    assert torch.equal(actual.ledger.total_after_q, expected.total_after_q)
    assert torch.equal(actual.ledger.books_closed, expected.books_closed)


def test_functional_device_step_has_a_host_read_free_full_graph() -> None:
    config = tiny_config()
    state = seeded_state(config)
    compiled = torch.compile(
        advance_economy_unchecked,
        backend="eager",
        fullgraph=True,
        dynamic=False,
    )

    expected = advance_economy_unchecked(state, config)
    actual = compiled(state, config)

    assert torch.equal(actual.state.nd_q, expected.state.nd_q)
    assert torch.equal(actual.state.bp_q, expected.state.bp_q)
    assert torch.equal(actual.ledger.books_closed, expected.ledger.books_closed)


def test_snapshot_restores_carries_clock_and_continuation(tmp_path) -> None:
    config = tiny_config()
    state = seeded_state(config)
    kernel = EconomyKernel(state, config)
    for _ in range(7):
        kernel.step()
    path = tmp_path / "economy.safetensors"
    save_snapshot(path, state, config)
    restored, restored_config = load_snapshot(path)
    assert restored_config == config
    for left, right in zip(state.reservoirs, restored.reservoirs, strict=True):
        assert torch.equal(left, right)
    for field in fields(state.carries):
        assert torch.equal(getattr(state.carries, field.name), getattr(restored.carries, field.name))
    assert torch.equal(state.step, restored.step)
    assert torch.equal(state.time_s, restored.time_s)
    assert torch.equal(state.buffer_parity, restored.buffer_parity)
    first = EconomyKernel(state, config).step()
    second = EconomyKernel(restored, config).step()
    assert torch.equal(first.total_after_q, second.total_after_q)
    for left, right in zip(state.reservoirs, restored.reservoirs, strict=True):
        assert torch.equal(left, right)


def test_omitting_snapshot_carries_changes_committed_continuation() -> None:
    config = tiny_config()
    state = seeded_state(config)
    kernel = EconomyKernel(state, config)
    for _ in range(7):
        kernel.step()
    complete = state.clone()
    incomplete = state.clone()
    for field in fields(incomplete.carries):
        getattr(incomplete.carries, field.name).zero_()
    complete_kernel = EconomyKernel(complete, config)
    incomplete_kernel = EconomyKernel(incomplete, config)
    diverged = False
    for _ in range(100):
        complete_kernel.step()
        incomplete_kernel.step()
        diverged |= any(
            not torch.equal(left, right)
            for left, right in zip(complete.reservoirs, incomplete.reservoirs, strict=True)
        )
        if diverged:
            break
    assert diverged


def test_snapshot_rejects_tampered_tensor_bytes(tmp_path) -> None:
    config = tiny_config()
    state = seeded_state(config)
    path = tmp_path / "economy.safetensors"
    save_snapshot(path, state, config)
    payload = bytearray(path.read_bytes())
    payload[-1] ^= 1
    path.write_bytes(payload)
    try:
        load_snapshot(path)
    except ValueError as error:
        assert "tensor hash" in str(error)
    else:
        raise AssertionError("tampered tensor payload must be rejected")


def test_row_slices_reproduce_whole_grid_exactly() -> None:
    full_config = replace(EconomyConfig(), gx=4, gy=1, gz=4, lx_m=40, ly_m=10, lz_m=20)
    slice_config = replace(full_config, gx=2, lx_m=20)
    full = seeded_state(full_config)
    full.nd_q[0, :, 0, :] += torch.arange(4, dtype=torch.int64)[:, None] * 1000
    whole = full.clone()

    def slice_state(start: int, stop: int) -> EconomyState:
        result = EconomyState.zeros(slice_config)
        for target, source in zip(result.reservoirs, full.reservoirs, strict=True):
            target.copy_(source[:, start:stop])
        carry_fields = fields(result.carries)
        for field in carry_fields:
            getattr(result.carries, field.name).copy_(getattr(full.carries, field.name)[:, start:stop])
        return result

    left, right = slice_state(0, 2), slice_state(2, 4)
    EconomyKernel(whole, full_config).step()
    EconomyKernel(left, slice_config).step()
    EconomyKernel(right, slice_config).step()
    for expected, left_value, right_value in zip(
        whole.reservoirs, left.reservoirs, right.reservoirs, strict=True
    ):
        actual = torch.cat((left_value, right_value), dim=1)
        assert torch.equal(expected, actual)
