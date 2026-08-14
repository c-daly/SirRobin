"""Data-only composition of authoritative device-runtime state."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from sirrobin.economy.config import EconomyConfig
from sirrobin.economy.state import EconomyState
from sirrobin.genetics.genotype import GenotypeBatch
from sirrobin.organisms.development import (
    DevelopmentState,
    validate_development_state,
)
from sirrobin.organisms.state import PopulationState, validate_population_state
from sirrobin.physics.contracts import DevelopedBody, LiveState


@dataclass(frozen=True, slots=True)
class LivingState:
    population: PopulationState
    genotype: GenotypeBatch
    body: DevelopedBody
    development: DevelopmentState
    motion: LiveState
    economy: EconomyState
    expected_matter_q: torch.Tensor


def validate_living_state(state: LivingState, economy_config: EconomyConfig) -> None:
    """Validate cross-domain authority and capacity at a session boundary."""

    validate_population_state(state.population)
    state.genotype.validate()
    validate_development_state(state.development, state.population, state.body)
    state.economy.validate(economy_config)
    worlds = state.population.alive.shape[0]
    if state.expected_matter_q.dtype != torch.int64 or tuple(
        state.expected_matter_q.shape
    ) != (worlds,):
        raise TypeError(f"expected matter must be int64 with shape {(worlds,)}")
    if state.expected_matter_q.device != state.population.alive.device:
        raise ValueError("expected matter must share the runtime device")
    authority = (state.population.alive, state.population.stable_id)
    for name, alive, stable_id in (
        ("genotype", state.genotype.alive, state.genotype.stable_id),
        ("developed body", state.body.alive, state.body.stable_id),
    ):
        if not torch.equal(alive, authority[0]) or not torch.equal(
            stable_id, authority[1]
        ):
            raise ValueError(f"{name} identity differs from population authority")
    lead = tuple(state.population.alive.shape)
    motion_shapes = {
        "position_enu_m": (*lead, 3),
        "velocity_rel_water_enu_m_s": (*lead, 3),
        "yaw_rad": lead,
        "yaw_momentum_kg_m2_s": lead,
        "gait_time_s": lead,
        "desired_heading_enu": (*lead, 2),
        "turn_bias_rad_per_depth": lead,
        "heading_initialized": lead,
    }
    for name, expected in motion_shapes.items():
        value = getattr(state.motion, name)
        if tuple(value.shape) != expected:
            raise ValueError(f"motion {name} must have shape {expected}")
        if value.device != state.population.alive.device:
            raise ValueError("all runtime state must share one device")
        if value.dtype != torch.bool and bool((~torch.isfinite(value)).any()):
            raise ValueError(f"motion {name} must be finite")
    combined_q = state.economy.total_per_world() + (
        state.population.structure_q.sum(dim=1, dtype=torch.int64)
        + state.population.reserve_q.sum(dim=1, dtype=torch.int64)
    )
    if bool((combined_q.to(torch.float64) >= economy_config.max_inventory_q).any()):
        raise ValueError("combined field and organism inventory exceeds the safe bound")
    if not torch.equal(combined_q, state.expected_matter_q):
        raise ValueError("runtime state differs from its exact matter baseline")
