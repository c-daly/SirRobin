"""Canonical four-reservoir economy state and all restart-relevant carries."""

from __future__ import annotations

from dataclasses import dataclass, fields

import torch

from sirrobin.economy.config import EconomyConfig
from sirrobin.numerics.flux import INT64_SAFE_MAX


@dataclass(slots=True)
class EconomyCarries:
    production_mol: torch.Tensor
    producer_maintenance_mol: torch.Tensor
    producer_mortality_mol: torch.Tensor
    decomposition_mol: torch.Tensor
    microbial_turnover_mol: torch.Tensor
    bge_fraction_q: torch.Tensor
    sinking_mol: torch.Tensor
    mix_nd_mol: torch.Tensor
    mix_bp_mol: torch.Tensor
    mix_bm_mol: torch.Tensor

    @classmethod
    def zeros(cls, config: EconomyConfig, *, device: torch.device | str = "cpu") -> EconomyCarries:
        def cell() -> torch.Tensor:
            return torch.zeros(config.shape, dtype=torch.float64, device=device)

        def face() -> torch.Tensor:
            return torch.zeros(config.face_shape, dtype=torch.float64, device=device)

        return cls(
            cell(),
            cell(),
            cell(),
            cell(),
            cell(),
            cell(),
            face(),
            cell(),
            cell(),
            cell(),
        )

    def clone(self) -> EconomyCarries:
        return EconomyCarries(**{field.name: getattr(self, field.name).clone() for field in fields(self)})


@dataclass(slots=True)
class EconomyState:
    nd_q: torch.Tensor
    bp_q: torch.Tensor
    bd_q: torch.Tensor
    bm_q: torch.Tensor
    carries: EconomyCarries
    step: torch.Tensor
    time_s: torch.Tensor
    buffer_parity: torch.Tensor

    @classmethod
    def zeros(cls, config: EconomyConfig, *, device: torch.device | str = "cpu") -> EconomyState:
        def reservoir() -> torch.Tensor:
            return torch.zeros(config.shape, dtype=torch.int64, device=device)

        return cls(
            reservoir(),
            reservoir(),
            reservoir(),
            reservoir(),
            EconomyCarries.zeros(config, device=device),
            torch.zeros((), dtype=torch.int64, device=device),
            torch.zeros((), dtype=torch.float64, device=device),
            torch.zeros((), dtype=torch.int64, device=device),
        )

    @property
    def reservoirs(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        return self.nd_q, self.bp_q, self.bd_q, self.bm_q

    def total_per_world(self) -> torch.Tensor:
        dims = (1, 2, 3)
        return sum(
            (reservoir.sum(dim=dims, dtype=torch.int64) for reservoir in self.reservoirs),
            start=torch.zeros(self.nd_q.shape[0], dtype=torch.int64, device=self.nd_q.device),
        )

    def validate(self, config: EconomyConfig) -> None:
        for name, reservoir in zip(("nd_q", "bp_q", "bd_q", "bm_q"), self.reservoirs, strict=True):
            if reservoir.dtype != torch.int64 or tuple(reservoir.shape) != config.shape:
                raise TypeError(f"{name} must be int64 with shape {config.shape}")
            if torch.any(reservoir < 0) or torch.any(reservoir >= INT64_SAFE_MAX):
                raise ValueError(f"{name} violates the [0,2^62) domain")
        approximate_total = sum(
            (reservoir.to(torch.float64).sum(dim=(1, 2, 3)) for reservoir in self.reservoirs),
            start=torch.zeros(config.worlds, dtype=torch.float64, device=self.nd_q.device),
        )
        if torch.any(approximate_total >= config.max_inventory_q):
            raise ValueError("per-world inventory exceeds the configured safe reduction bound")
        q = config.q_mass_mol
        for field in fields(self.carries):
            value = getattr(self.carries, field.name)
            if value.dtype != torch.float64 or torch.any(~torch.isfinite(value)):
                raise TypeError(f"{field.name} must be finite float64")
            upper = 1.0 if field.name == "bge_fraction_q" else q
            if torch.any(value < 0) or torch.any(value >= upper):
                raise ValueError(f"{field.name} is outside its carry interval")
        if self.step.dtype != torch.int64 or self.buffer_parity.dtype != torch.int64:
            raise TypeError("step and buffer parity must be int64 scalars")
        if self.time_s.dtype != torch.float64:
            raise TypeError("time must be a float64 scalar")
        if self.step.ndim or self.time_s.ndim or self.buffer_parity.ndim:
            raise ValueError("clock and parity state must be scalar")

    def clone(self) -> EconomyState:
        return EconomyState(
            *(reservoir.clone() for reservoir in self.reservoirs),
            carries=self.carries.clone(),
            step=self.step.clone(),
            time_s=self.time_s.clone(),
            buffer_parity=self.buffer_parity.clone(),
        )
