import json
from pathlib import Path

import torch

from sirrobin.genetics.develop import develop
from sirrobin.genetics.genotype import GenotypeBatch

FIXTURES = Path("oracle/fixtures/live")


def _rows():
    donor = json.loads((FIXTURES / "donor_development_live.json").read_text(encoding="utf-8"))
    gain1 = json.loads((FIXTURES / "gain1_canonical.json").read_text(encoding="utf-8"))
    return donor["bodies"], gain1["bodies"]


def test_develop_matches_frozen_independent_canonical_fixture():
    donor_rows, expected_rows = _rows()
    genotype = GenotypeBatch.from_donor_rows(donor_rows, dtype=torch.float64)
    body = develop(genotype)
    for bi, expected in enumerate(expected_rows):
        assert int(body.seg_mask[0, bi].sum()) == len(expected["segments"])
        assert int(body.tail_slot[0, bi]) == expected["tail"] + 1
        for si, segment in enumerate(expected["segments"], start=1):
            assert int(body.parent[0, bi, si]) == segment["parent"] + 1
            assert int(body.depth[0, bi, si]) == segment["depth"]
            assert torch.allclose(
                body.local_pos_flu_m[0, bi, si],
                torch.tensor(segment["local_pos"], dtype=torch.float64),
                rtol=1e-12,
                atol=1e-12,
            )
            actual_q = body.local_rot_flu[0, bi, si]
            expected_q = torch.tensor(segment["local_rot"], dtype=torch.float64)
            assert abs(float(torch.dot(actual_q, expected_q))) > 1 - 1e-12
            assert torch.allclose(
                body.semi_axes_flu_m[0, bi, si],
                torch.tensor(segment["axes"], dtype=torch.float64),
                rtol=1e-12,
                atol=1e-12,
            )
            assert torch.allclose(
                body.mass_sim[0, bi, si],
                torch.tensor(segment["mass_sim"], dtype=torch.float64),
                rtol=1e-12,
                atol=1e-12,
            )
            assert torch.allclose(
                body.drag_area_flu_m2[0, bi, si],
                torch.tensor(segment["drag_area"], dtype=torch.float64),
                rtol=1e-12,
                atol=1e-12,
            )
            assert torch.allclose(
                body.added_mass_flu_kg[0, bi, si],
                torch.tensor(segment["added_mass_kg"], dtype=torch.float64),
                rtol=1e-7,
                atol=1e-9,
            )


def test_develop_is_exact_same_device_and_fixed_shape():
    donor_rows, _ = _rows()
    genotype = GenotypeBatch.from_donor_rows(donor_rows, dtype=torch.float32)
    first = develop(genotype)
    second = develop(genotype)
    assert first.seg_mask.shape == (1, 32, 17)
    for name in first.__dataclass_fields__:
        a, b = getattr(first, name), getattr(second, name)
        if isinstance(a, torch.Tensor):
            assert torch.equal(a, b), name


def test_development_preserves_literal_dfs_and_cap_cases():
    donor_rows, _ = _rows()
    by_id = {row["id"]: row for row in donor_rows}
    rows = [by_id["mirrored"], by_id["deep-cap"], by_id["wide-16"]]
    body = develop(GenotypeBatch.from_donor_rows(rows, dtype=torch.float64))
    assert body.seg_mask.sum(-1).tolist() == [[6, 10, 16]]
    assert body.truncated_candidate_count.tolist() == [[0, 0, 0]]
    # The mirrored edge emits positive then negative before the donor DFS continues.
    assert torch.allclose(body.local_pos_flu_m[0, 0, 2, 1], -body.local_pos_flu_m[0, 0, 3, 1])
