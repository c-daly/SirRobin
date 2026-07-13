"""Read-only economy telemetry projections."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from sirrobin.economy.config import EconomyConfig
from sirrobin.economy.state import EconomyState


@dataclass(frozen=True, slots=True)
class EconomyTotals:
    nd_q: torch.Tensor
    bp_q: torch.Tensor
    bd_q: torch.Tensor
    bm_q: torch.Tensor
    total_q: torch.Tensor


def reservoir_totals(state: EconomyState) -> EconomyTotals:
    dims = (1, 2, 3)
    values = tuple(reservoir.sum(dim=dims, dtype=torch.int64) for reservoir in state.reservoirs)
    return EconomyTotals(*values, state.total_per_world())


def concentrations_mol_m3(state: EconomyState, config: EconomyConfig) -> dict[str, torch.Tensor]:
    scale = config.q_mass_mol / config.cell_volume_m3
    return {
        name: reservoir.to(torch.float64) * scale
        for name, reservoir in zip(("Nd", "Bp", "Bd", "Bm"), state.reservoirs, strict=True)
    }
