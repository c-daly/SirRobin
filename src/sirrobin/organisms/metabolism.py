"""Batched maintenance, locomotion settlement, and mortality decision."""

from __future__ import annotations

import math
from dataclasses import dataclass, replace

import torch

from sirrobin.organisms.state import PopulationState


@dataclass(frozen=True, slots=True)
class MetabolismConfig:
    interval_s: float
    maintenance_w_per_kg: float
    chemical_to_mechanical_efficiency: float
    reserve_j_per_q: float

    def validate(self) -> None:
        values = (
            self.interval_s,
            self.maintenance_w_per_kg,
            self.chemical_to_mechanical_efficiency,
            self.reserve_j_per_q,
        )
        if any(
            isinstance(value, bool) or not isinstance(value, (int, float))
            for value in values
        ):
            raise TypeError("metabolism configuration values must be real numbers")
        if not math.isfinite(self.interval_s) or self.interval_s <= 0.0:
            raise ValueError("metabolism interval must be finite and positive")
        if (
            not math.isfinite(self.maintenance_w_per_kg)
            or self.maintenance_w_per_kg < 0.0
        ):
            raise ValueError("maintenance rate must be finite and nonnegative")
        if (
            not math.isfinite(self.chemical_to_mechanical_efficiency)
            or not 0.0 < self.chemical_to_mechanical_efficiency <= 1.0
        ):
            raise ValueError("mechanical efficiency must be finite and in (0,1]")
        if not math.isfinite(self.reserve_j_per_q) or self.reserve_j_per_q <= 0.0:
            raise ValueError("reserve energy density must be finite and positive")


@dataclass(frozen=True, slots=True)
class MetabolismInputs:
    structural_mass_kg: torch.Tensor
    positive_actuator_work_j: torch.Tensor
    actuator_braking_work_j: torch.Tensor
    old_age_due: torch.Tensor
    velocity_enu_m_s: torch.Tensor
    yaw_momentum_kg_m2_s: torch.Tensor
    effective_mass_after_kg: torch.Tensor
    yaw_inertia_after_kg_m2: torch.Tensor


@dataclass(frozen=True, slots=True)
class MetabolismLedger:
    baseline_demand_j: torch.Tensor
    positive_actuator_work_j: torch.Tensor
    actuator_braking_heat_j: torch.Tensor
    locomotion_chemical_demand_j: torch.Tensor
    muscle_inefficiency_heat_j: torch.Tensor
    total_demand_j: torch.Tensor
    carry_before_j: torch.Tensor
    carry_after_j: torch.Tensor
    quantization_residual_j: torch.Tensor
    requested_q: torch.Tensor
    reserve_debit_q: torch.Tensor
    maintenance_return_q: torch.Tensor
    maintenance_heat_j: torch.Tensor
    starved: torch.Tensor
    old_age_due: torch.Tensor
    death: torch.Tensor
    death_dissipation_j: torch.Tensor
    invalid_death_kinetics: torch.Tensor


@dataclass(frozen=True, slots=True)
class MetabolismStep:
    state: PopulationState
    ledger: MetabolismLedger


def available_actuator_work_j(
    state: PopulationState,
    structural_mass_kg: torch.Tensor,
    config: MetabolismConfig,
) -> torch.Tensor:
    """Return work currently backed after baseline maintenance and prior carry."""

    baseline_j = (
        config.maintenance_w_per_kg * structural_mass_kg.to(torch.float64)
        * config.interval_s
    )
    stored_j = state.reserve_q.to(torch.float64) * config.reserve_j_per_q
    available_chemical_j = (
        stored_j - baseline_j - state.maintenance_carry_j
    ).clamp_min(0.0)
    return torch.where(
        state.alive,
        available_chemical_j * config.chemical_to_mechanical_efficiency,
        0.0,
    )


def settle_metabolism(
    state: PopulationState,
    inputs: MetabolismInputs,
    config: MetabolismConfig,
) -> MetabolismStep:
    """Settle every live slot with fixed-shape tensor operations.

    Configuration and input schema validation belong to the runtime-session
    boundary. This function makes no host decisions and does not deposit returned
    material; its exact integer return ledger is consumed by the spatial field
    transaction before lifecycle clears dead slots.
    """

    alive = state.alive
    baseline_j = torch.where(
        alive,
        config.maintenance_w_per_kg
        * inputs.structural_mass_kg.to(torch.float64)
        * config.interval_s,
        0.0,
    )
    positive_work_j = torch.where(
        alive, inputs.positive_actuator_work_j.to(torch.float64), 0.0
    )
    braking_heat_j = torch.where(
        alive, inputs.actuator_braking_work_j.to(torch.float64), 0.0
    )
    locomotion_chemical_j = (
        positive_work_j / config.chemical_to_mechanical_efficiency
    )
    muscle_inefficiency_j = locomotion_chemical_j - positive_work_j
    carry_before_j = torch.where(alive, state.maintenance_carry_j, 0.0)
    demand_j = baseline_j + locomotion_chemical_j + carry_before_j
    quotient = torch.floor(demand_j / config.reserve_j_per_q)
    requested_q = quotient.to(torch.int64)
    carry_after_j = demand_j - quotient * config.reserve_j_per_q
    quantization_residual_j = (
        demand_j
        - requested_q.to(torch.float64) * config.reserve_j_per_q
        - carry_after_j
    )

    reserve_debit_q = torch.minimum(requested_q, state.reserve_q)
    reserve_after_q = state.reserve_q - reserve_debit_q
    starved = alive & (reserve_debit_q < requested_q)
    old_age_due = alive & inputs.old_age_due
    death = starved | old_age_due

    velocity_xy = inputs.velocity_enu_m_s[..., :2].to(torch.float64)
    matrix_xy = inputs.effective_mass_after_kg[..., :2, :2].to(torch.float64)
    linear_kinetic_j = 0.5 * torch.einsum(
        "...i,...ij,...j->...", velocity_xy, matrix_xy, velocity_xy
    )
    yaw_inertia = inputs.yaw_inertia_after_kg_m2.to(torch.float64)
    safe_yaw_inertia = yaw_inertia.clamp_min(torch.finfo(torch.float64).tiny)
    yaw_momentum = inputs.yaw_momentum_kg_m2_s.to(torch.float64)
    rotational_kinetic_j = yaw_momentum.square() / (2.0 * safe_yaw_inertia)
    carry_energy_j = state.assimilation_carry_q * config.reserve_j_per_q
    death_dissipation_j = torch.where(
        death,
        linear_kinetic_j + rotational_kinetic_j + carry_energy_j,
        0.0,
    )
    invalid_death_kinetics = death & (
        ~torch.isfinite(linear_kinetic_j)
        | ~torch.isfinite(rotational_kinetic_j)
        | ~torch.isfinite(death_dissipation_j)
        | (linear_kinetic_j < 0.0)
        | (rotational_kinetic_j < 0.0)
        | (yaw_inertia <= 0.0)
        | (inputs.velocity_enu_m_s[..., 2] != 0.0)
    )
    next_state = replace(
        state,
        reserve_q=reserve_after_q,
        maintenance_carry_j=torch.where(death, 0.0, carry_after_j),
    )
    ledger = MetabolismLedger(
        baseline_demand_j=baseline_j,
        positive_actuator_work_j=positive_work_j,
        actuator_braking_heat_j=braking_heat_j,
        locomotion_chemical_demand_j=locomotion_chemical_j,
        muscle_inefficiency_heat_j=muscle_inefficiency_j,
        total_demand_j=demand_j,
        carry_before_j=carry_before_j,
        carry_after_j=torch.where(death, 0.0, carry_after_j),
        quantization_residual_j=quantization_residual_j,
        requested_q=requested_q,
        reserve_debit_q=reserve_debit_q,
        maintenance_return_q=reserve_debit_q,
        maintenance_heat_j=reserve_debit_q.to(torch.float64)
        * config.reserve_j_per_q,
        starved=starved,
        old_age_due=old_age_due,
        death=death,
        death_dissipation_j=death_dissipation_j,
        invalid_death_kinetics=invalid_death_kinetics,
    )
    return MetabolismStep(next_state, ledger)
