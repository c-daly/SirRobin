"""Public contracts for exact nutrient state and step diagnostics."""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True, slots=True)
class EconomyStepLedger:
    production_q: torch.Tensor
    producer_maintenance_q: torch.Tensor
    producer_mortality_q: torch.Tensor
    decomposition_q: torch.Tensor
    microbial_credit_q: torch.Tensor
    dissolved_credit_q: torch.Tensor
    microbial_turnover_q: torch.Tensor
    sinking_q: torch.Tensor
    mixing_q: torch.Tensor
    reaction_shortfall_q: torch.Tensor
    transport_shortfall_q: torch.Tensor
    total_before_q: torch.Tensor
    total_after_q: torch.Tensor
    books_closed: torch.Tensor
    intervention_count: torch.Tensor
