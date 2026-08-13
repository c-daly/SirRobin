"""Read-only generic scalar-field sampling contracts."""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True, slots=True)
class FieldSample:
    value_mol_m3: torch.Tensor
    gradient_mol_m4: torch.Tensor


@dataclass(frozen=True, slots=True)
class BoundedFieldSample:
    """Device sample plus an explicit closed-boundary validity mask."""

    value_mol_m3: torch.Tensor
    gradient_mol_m4: torch.Tensor
    vertical_out_of_bounds: torch.Tensor
