import pytest
import torch

from sirrobin.numerics.transfer import INT64_MAX, close_books, transfer_quanta


def test_transfer_exact_source_cap_and_shortfall():
    src = torch.tensor([10, 2, 7], dtype=torch.int64)
    dst = torch.tensor([1, 5, 0], dtype=torch.int64)
    req = torch.tensor([3, 9, 4], dtype=torch.int64)
    mask = torch.tensor([True, True, False])
    before = int(src.sum() + dst.sum())
    src2, dst2, shortfall = transfer_quanta(src, dst, req, mask)
    assert src2.tolist() == [7, 0, 7]
    assert dst2.tolist() == [4, 7, 0]
    assert shortfall.tolist() == [0, 7, 0]
    assert close_books(src2, dst2, expected_total=before)


def test_transfer_rejects_negative_and_overflow():
    z = torch.tensor([0], dtype=torch.int64)
    with pytest.raises(ValueError):
        transfer_quanta(-torch.ones_like(z), z, z, torch.ones(1, dtype=torch.bool))
    with pytest.raises(OverflowError):
        transfer_quanta(
            torch.tensor([1], dtype=torch.int64),
            torch.tensor([INT64_MAX], dtype=torch.int64),
            torch.tensor([1], dtype=torch.int64),
            torch.tensor([True]),
        )
