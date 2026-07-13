"""Per-world exact mass ledger and reservoir registry."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from sirrobin.economy.config import INT64_SAFE_MAX
from sirrobin.economy.state import EconomyState


@dataclass(frozen=True, slots=True)
class MassLedger:
    expected_total_q: torch.Tensor

    @classmethod
    def from_state(cls, state: EconomyState) -> MassLedger:
        return cls(state.total_per_world().clone())

    def close_books(self, state: EconomyState) -> torch.Tensor:
        valid = torch.ones_like(self.expected_total_q, dtype=torch.bool)
        for reservoir in state.reservoirs:
            valid &= (reservoir >= 0).all(dim=(1, 2, 3))
            valid &= (reservoir < INT64_SAFE_MAX).all(dim=(1, 2, 3))
        return valid & (state.total_per_world() == self.expected_total_q)

    def require_closed(self, state: EconomyState) -> None:
        if not bool(self.close_books(state).all()):
            raise RuntimeError("exact nutrient books do not close")
