"""Pure rate laws and exact local nutrient transactions."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from sirrobin.economy.config import EconomyConfig
from sirrobin.economy.state import EconomyState
from sirrobin.numerics.flux import commit_flux, deterministic_fraction


@dataclass(frozen=True, slots=True)
class ReactionResult:
    production_q: torch.Tensor
    producer_maintenance_q: torch.Tensor
    producer_mortality_q: torch.Tensor
    decomposition_q: torch.Tensor
    microbial_credit_q: torch.Tensor
    dissolved_credit_q: torch.Tensor
    microbial_turnover_q: torch.Tensor
    shortfall_q: torch.Tensor


def light_at_depth(depth_m: torch.Tensor, config: EconomyConfig) -> torch.Tensor:
    return config.i0_w_m2 * torch.exp(-config.k_light_m_inv * depth_m)


def limitation(
    nd_concentration_mol_m3: torch.Tensor, depth_m: torch.Tensor, config: EconomyConfig
) -> torch.Tensor:
    light = light_at_depth(depth_m, config)
    light_term = light / (light + config.k_i_w_m2)
    nutrient_term = nd_concentration_mol_m3 / (nd_concentration_mol_m3 + config.k_n_mol_m3)
    return torch.minimum(light_term, nutrient_term)


def remineralization_rates(config: EconomyConfig, *, device: torch.device | str) -> torch.Tensor:
    depth = (torch.arange(config.gz, dtype=torch.float64, device=device) + 0.5) * config.dz_m
    mapped = config.martin_b * config.sinking_speed_m_s / (config.martin_reference_depth_m + depth)
    return mapped.clamp_min(config.remin_floor_s)


def reaction_step(state: EconomyState, config: EconomyConfig) -> ReactionResult:
    q = config.q_mass_mol
    volume = config.cell_volume_m3
    nd_old, bp_old, bd_old, bm_old = (reservoir.clone() for reservoir in state.reservoirs)
    nd_mol = nd_old.to(torch.float64) * q
    bp_mol = bp_old.to(torch.float64) * q
    bd_mol = bd_old.to(torch.float64) * q
    bm_mol = bm_old.to(torch.float64) * q
    depth = (torch.arange(config.gz, dtype=torch.float64, device=nd_old.device) + 0.5) * config.dz_m
    depth = depth.view(1, 1, 1, -1)

    production_mol = config.mu_max_s * limitation(nd_mol / volume, depth, config) * bp_mol * config.dt_eco_s
    maintenance_mol = config.producer_maintenance_s * bp_mol * config.dt_eco_s
    bp_concentration = bp_mol / volume
    mortality_rate = config.producer_mortality_s + config.density_mortality_m3_mol_s * bp_concentration
    mortality_mol = mortality_rate * bp_mol * config.dt_eco_s
    remin_rate = remineralization_rates(config, device=nd_old.device).view(1, 1, 1, -1)
    decomposition_mol = remin_rate * bd_mol * config.dt_eco_s
    microbial_turnover_mol = config.microbial_turnover_s * bm_mol * config.dt_eco_s

    production = commit_flux(production_mol, state.carries.production_mol, nd_old, q_mass_mol=q)
    maintenance = commit_flux(maintenance_mol, state.carries.producer_maintenance_mol, bp_old, q_mass_mol=q)
    mortality = commit_flux(
        mortality_mol,
        state.carries.producer_mortality_mol,
        bp_old - maintenance.committed_q,
        q_mass_mol=q,
    )
    decomposition = commit_flux(decomposition_mol, state.carries.decomposition_mol, bd_old, q_mass_mol=q)
    microbial_turnover = commit_flux(
        microbial_turnover_mol,
        state.carries.microbial_turnover_mol,
        bm_old,
        q_mass_mol=q,
    )
    microbial_credit, dissolved_credit, bge_carry = deterministic_fraction(
        decomposition.committed_q, config.bge, state.carries.bge_fraction_q
    )

    state.nd_q.copy_(
        nd_old
        - production.committed_q
        + maintenance.committed_q
        + dissolved_credit
        + microbial_turnover.committed_q
    )
    state.bp_q.copy_(bp_old + production.committed_q - maintenance.committed_q - mortality.committed_q)
    state.bd_q.copy_(bd_old + mortality.committed_q - decomposition.committed_q)
    state.bm_q.copy_(bm_old + microbial_credit - microbial_turnover.committed_q)
    state.carries.production_mol.copy_(production.carry_mol)
    state.carries.producer_maintenance_mol.copy_(maintenance.carry_mol)
    state.carries.producer_mortality_mol.copy_(mortality.carry_mol)
    state.carries.decomposition_mol.copy_(decomposition.carry_mol)
    state.carries.microbial_turnover_mol.copy_(microbial_turnover.carry_mol)
    state.carries.bge_fraction_q.copy_(bge_carry)

    dims = (1, 2, 3)
    total_shortfall = sum(
        (
            result.shortfall_q.sum(dim=dims, dtype=torch.int64)
            for result in (production, maintenance, mortality, decomposition, microbial_turnover)
        ),
        start=torch.zeros(config.worlds, dtype=torch.int64, device=nd_old.device),
    )

    def reduce(value: torch.Tensor) -> torch.Tensor:
        return value.sum(dim=dims, dtype=torch.int64)

    return ReactionResult(
        reduce(production.committed_q),
        reduce(maintenance.committed_q),
        reduce(mortality.committed_q),
        reduce(decomposition.committed_q),
        reduce(microbial_credit),
        reduce(dissolved_credit),
        reduce(microbial_turnover.committed_q),
        total_shortfall,
    )
