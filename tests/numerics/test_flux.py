from __future__ import annotations

import torch

from sirrobin.numerics.flux import apportion_integer, commit_flux, deterministic_fraction


def test_commit_flux_bounds_carry_and_drops_availability_debt() -> None:
    request = torch.tensor([2.75e-9, 10e-9], dtype=torch.float64)
    carry = torch.tensor([0.5e-9, 0.0], dtype=torch.float64)
    available = torch.tensor([10, 3], dtype=torch.int64)
    result = commit_flux(request, carry, available, q_mass_mol=1e-9)
    assert result.committed_q.tolist() == [3, 3]
    assert result.shortfall_q.tolist() == [0, 7]
    assert torch.all((result.carry_mol >= 0) & (result.carry_mol < 1e-9))
    assert result.carry_mol[1] == 0


def test_bge_partition_has_one_debit_and_two_exact_credits() -> None:
    total = torch.full((100,), 7, dtype=torch.int64)
    carry = torch.zeros(100, dtype=torch.float64)
    first, second, next_carry = deterministic_fraction(total, 0.2, carry)
    assert torch.equal(first + second, total)
    assert torch.all((next_carry >= 0) & (next_carry < 1))
    assert first.sum() == 100


def test_bge_fractional_carry_matches_long_run_anchor() -> None:
    carry = torch.zeros(1, dtype=torch.float64)
    credited = 0
    for _ in range(100):
        first, second, carry = deterministic_fraction(torch.tensor([7], dtype=torch.int64), 0.2, carry)
        assert int(first + second) == 7
        credited += int(first)
    assert credited == 140
    assert 0 <= float(carry) < 1


def test_bge_endpoint_controls_are_exact() -> None:
    total = torch.tensor([7], dtype=torch.int64)
    carry = torch.tensor([0.4], dtype=torch.float64)
    first_zero, second_zero, carry_zero = deterministic_fraction(total, 0.0, carry)
    first_one, second_one, carry_one = deterministic_fraction(total, 1.0, carry)
    assert (int(first_zero), int(second_zero), float(carry_zero)) == (0, 7, 0.0)
    assert (int(first_one), int(second_one), float(carry_one)) == (7, 0, 0.0)


def test_largest_remainder_is_exact_and_stable_on_ties() -> None:
    total = torch.tensor([5, 2], dtype=torch.int64)
    weights = torch.tensor([[1.0, 1.0, 1.0], [0.5, 0.5, 0.0]], dtype=torch.float64)
    result = apportion_integer(total, weights)
    assert result.tolist() == [[2, 2, 1], [1, 1, 0]]
    assert torch.equal(result.sum(-1), total)
