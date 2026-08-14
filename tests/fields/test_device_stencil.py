from __future__ import annotations

import torch

from sirrobin.fields.geometry import GridGeometry
from sirrobin.fields.grid import ScalarGrid
from sirrobin.fields.sample import sample_reservoir_device
from sirrobin.fields.stencil import (
    deposit_stencil,
    point_stencil,
    sample_stencil_mol_m3,
)


def test_fixed_stencil_matches_scalar_grid_with_boundary_duplicates_merged() -> None:
    geometry = GridGeometry(3, 2, 2, 6.0, 4.0, 4.0)
    reservoir = torch.arange(12, dtype=torch.int64).reshape(1, 3, 2, 2)
    positions = torch.tensor([[[0.0, 0.0, 0.0], [2.3, 1.7, -1.2]]])
    stencil = point_stencil(positions, geometry)
    sampled = sample_stencil_mol_m3(
        reservoir, stencil, geometry, q_mass_mol=0.5
    )
    scalar = ScalarGrid(reservoir, geometry, q_mass_mol=0.5)

    assert torch.allclose(
        sampled,
        torch.tensor(
            [[scalar.value_at(0, positions[0, 0]), scalar.value_at(0, positions[0, 1])]],
            dtype=torch.float64,
        ),
    )
    for creature in range(2):
        positive_cells = stencil.cell_index[0, creature][
            stencil.weight[0, creature] > 0
        ]
        assert positive_cells.numel() == torch.unique(positive_cells).numel()
        assert stencil.weight[0, creature].sum().item() == 1.0


def test_fixed_stencil_deposit_is_exact_even_when_all_corners_merge() -> None:
    geometry = GridGeometry(1, 1, 1, 1.0, 1.0, 1.0)
    reservoir = torch.zeros((1, 1, 1, 1), dtype=torch.int64)
    stencil = point_stencil(torch.zeros((1, 2, 3)), geometry)

    after, credits = deposit_stencil(
        reservoir, stencil, torch.tensor([[7, 11]], dtype=torch.int64)
    )

    assert after.item() == 18
    assert credits.item() == 18


def test_device_field_sample_matches_scalar_value_and_gradient() -> None:
    geometry = GridGeometry(3, 2, 2, 6.0, 4.0, 4.0)
    reservoir = torch.arange(12, dtype=torch.int64).reshape(1, 3, 2, 2)
    positions = torch.tensor([[[2.3, 1.7, -1.2]]])

    actual = sample_reservoir_device(
        reservoir, positions, geometry, q_mass_mol=0.5
    )
    expected = ScalarGrid(
        reservoir, geometry, q_mass_mol=0.5
    ).sample(positions)

    assert torch.allclose(actual.value_mol_m3, expected.value_mol_m3)
    assert torch.allclose(actual.gradient_mol_m4, expected.gradient_mol_m4)
    assert actual.vertical_out_of_bounds.tolist() == [[False]]


def test_device_field_sample_reports_closed_vertical_violation_in_graph() -> None:
    geometry = GridGeometry(2, 2, 2, 2.0, 2.0, 2.0)
    reservoir = torch.ones((1, 2, 2, 2), dtype=torch.int64)
    positions = torch.tensor([[[0.5, 0.5, 1.0], [0.5, 0.5, -1.0]]])
    compiled = torch.compile(
        sample_reservoir_device,
        fullgraph=True,
        dynamic=False,
        backend="eager",
    )

    actual = compiled(reservoir, positions, geometry, q_mass_mol=0.5)

    assert actual.vertical_out_of_bounds.tolist() == [[True, False]]
