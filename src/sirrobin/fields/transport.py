"""Exact conservative vertical sinking and finite-volume mixing."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import torch

from sirrobin.numerics.flux import apportion_integer, commit_flux


class VerticalTransportConfig(Protocol):
    worlds: int
    q_mass_mol: float
    sinking_speed_m_s: float
    dz_m: float
    dx_m: float
    dy_m: float
    cell_volume_m3: float


@dataclass(frozen=True, slots=True)
class TransportResult:
    moved_q: torch.Tensor
    shortfall_q: torch.Tensor
    intervention_count: torch.Tensor


def sink_vertical(
    reservoir_q: torch.Tensor,
    carry_mol: torch.Tensor,
    config: VerticalTransportConfig,
    *,
    dt_s: float,
) -> TransportResult:
    source = reservoir_q[..., :-1].clone()
    requested_mol = (
        source.to(torch.float64) * config.q_mass_mol * config.sinking_speed_m_s * dt_s / config.dz_m
    )
    committed = commit_flux(requested_mol, carry_mol, source, q_mass_mol=config.q_mass_mol)
    reservoir_q[..., :-1].sub_(committed.committed_q)
    reservoir_q[..., 1:].add_(committed.committed_q)
    carry_mol.copy_(committed.carry_mol)
    dims = (1, 2, 3)
    return TransportResult(
        committed.committed_q.sum(dim=dims, dtype=torch.int64),
        committed.shortfall_q.sum(dim=dims, dtype=torch.int64),
        torch.zeros(config.worlds, dtype=torch.int64, device=reservoir_q.device),
    )


def mix_vertical(
    reservoir_q: torch.Tensor,
    carry_mol: torch.Tensor,
    diffusivity_m2_s: float,
    config: VerticalTransportConfig,
    *,
    dt_s: float,
) -> TransportResult:
    old = reservoir_q.clone()
    concentration = old.to(torch.float64) * config.q_mass_mol / config.cell_volume_m3
    signed_mol = (
        diffusivity_m2_s
        * (config.dx_m * config.dy_m)
        * (concentration[..., :-1] - concentration[..., 1:])
        / config.dz_m
        * dt_s
    )
    request_up_mol = torch.zeros_like(concentration)
    request_down_mol = torch.zeros_like(concentration)
    request_down_mol[..., :-1] = signed_mol.clamp_min(0.0)
    request_up_mol[..., 1:] = (-signed_mol).clamp_min(0.0)
    total_request_mol = request_up_mol + request_down_mol
    committed = commit_flux(total_request_mol, carry_mol, old, q_mass_mol=config.q_mass_mol)
    carry_mol.copy_(committed.carry_mol)
    allocation = apportion_integer(
        committed.committed_q, torch.stack((request_up_mol, request_down_mol), dim=-1)
    )
    realized_up = allocation[..., 0]
    realized_down = allocation[..., 1]
    face_flow = realized_down[..., :-1] - realized_up[..., 1:]
    reservoir_q[..., :-1].sub_(face_flow)
    reservoir_q[..., 1:].add_(face_flow)
    dims = (1, 2, 3)
    return TransportResult(
        face_flow.abs().sum(dim=dims, dtype=torch.int64),
        committed.shortfall_q.sum(dim=dims, dtype=torch.int64),
        torch.zeros(config.worlds, dtype=torch.int64, device=reservoir_q.device),
    )
