"""Boundary-validated configuration composition for the device runtime."""

from __future__ import annotations

import math
from dataclasses import dataclass

from sirrobin.economy.config import EconomyConfig
from sirrobin.fields.geometry import GridGeometry
from sirrobin.numerics.flux import INT64_SAFE_MAX
from sirrobin.organisms.behavior import BehaviorConfig
from sirrobin.organisms.development import DevelopmentConfig
from sirrobin.organisms.feeding import FeedingConfig
from sirrobin.organisms.metabolism import MetabolismConfig
from sirrobin.organisms.mortality import MortalityConfig
from sirrobin.organisms.mutation import MutationConfig
from sirrobin.physics.live_config import LiveLocomotionConfig
from sirrobin.physics.phase_response import PhaseWindowConfig


@dataclass(frozen=True, slots=True)
class LivingRuntimeConfig:
    economy: EconomyConfig
    live: LiveLocomotionConfig
    motion: PhaseWindowConfig
    behavior: BehaviorConfig
    feeding: FeedingConfig
    metabolism: MetabolismConfig
    mortality: MortalityConfig
    mutation: MutationConfig
    development: DevelopmentConfig
    child_initial_reserve_q: int

    @property
    def geometry(self) -> GridGeometry:
        return GridGeometry.from_config(self.economy)


def validate_living_runtime_config(config: LivingRuntimeConfig) -> None:
    """Validate configuration once, before any compiled interval is entered."""

    config.economy.validate()
    config.live.validate()
    config.motion.validate()
    config.behavior.validate()
    config.feeding.validate()
    config.metabolism.validate()
    config.mortality.validate()
    config.mutation.validate()
    config.development.validate()
    intervals = (
        config.motion.interval_s,
        config.feeding.interval_s,
        config.metabolism.interval_s,
    )
    if any(
        not math.isclose(
            value,
            config.economy.dt_eco_s,
            rel_tol=1.0e-12,
            abs_tol=1.0e-12,
        )
        for value in intervals
    ):
        raise ValueError("motion, feeding, metabolism, and field intervals must agree")
    if config.feeding.q_mass_mol != config.economy.q_mass_mol:
        raise ValueError("feeding and field material quanta must agree")
    if config.feeding.reserve_j_per_q != config.metabolism.reserve_j_per_q:
        raise ValueError("feeding and metabolism reserve energy densities must agree")
    if isinstance(config.child_initial_reserve_q, bool) or not isinstance(
        config.child_initial_reserve_q, int
    ):
        raise TypeError("child initial reserve must be an integer")
    if not 0 <= config.child_initial_reserve_q < INT64_SAFE_MAX:
        raise ValueError("child initial reserve exceeds the exact-integer domain")
