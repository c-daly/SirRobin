"""Named operational configuration for the cohesive living runtime."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from sirrobin.organisms.behavior import BehaviorConfig
from sirrobin.organisms.development import calibrate_development_config
from sirrobin.organisms.feeding import FeedingConfig
from sirrobin.organisms.metabolism import MetabolismConfig
from sirrobin.organisms.mortality import MortalityConfig
from sirrobin.organisms.mutation import MutationConfig
from sirrobin.physics.phase_response import PhaseWindowConfig
from sirrobin.runtime.config import LivingRuntimeConfig

if TYPE_CHECKING:
    from sirrobin.core.world import HeadlessWorld
    from sirrobin.runtime.state import LivingState


@dataclass(frozen=True, slots=True)
class RuntimeProfile:
    """Named observation calibration, separate from biological kernels."""

    name: str
    description: str
    mortality: MortalityConfig
    mutation: MutationConfig


BASELINE_RUNTIME_PROFILE = RuntimeProfile(
    name="baseline",
    description="short lifecycle and baseline mutation exposure",
    mortality=MortalityConfig(60.0, 100.0, seed=20260810),
    mutation=MutationConfig(seed=20260810),
)
EVOLUTION_DEMO_RUNTIME_PROFILE = RuntimeProfile(
    name="evolution-demo",
    description="fivefold observation window and mutation exposure for the Unity demo",
    mortality=MortalityConfig(300.0, 500.0, seed=20260810),
    mutation=MutationConfig(seed=20260810, mutation_rate_per_locus=0.01),
)
RUNTIME_PROFILES = {
    profile.name: profile
    for profile in (BASELINE_RUNTIME_PROFILE, EVOLUTION_DEMO_RUNTIME_PROFILE)
}

LIVE_BEHAVIOR_CONFIG = BehaviorConfig(
    food_seeking_effort_fraction=0.5,
    search_effort_fraction=0.25,
    # A 137.5-degree exploratory heading change needs materially more than the
    # old eight-second leg to settle through the physical yaw controller. A
    # thirty-second leg retains deterministic local search while leaving a
    # sustained straight run after each turn.
    search_leg_duration_s=30.0,
    search_duty_fraction=0.65,
    # One structural-equivalent reserve is a morphology-scaled internal target.
    # Local producer must also be present; no world-wide field statistic enters
    # the behavior decision.
    food_sufficient_reserve_ratio=1.0,
    food_cruise_effort_fraction=0.1,
)


def living_runtime_config_from_reference(
    world: HeadlessWorld,
    state: LivingState,
    *,
    profile: RuntimeProfile = BASELINE_RUNTIME_PROFILE,
) -> LivingRuntimeConfig:
    """Build the shared operational runtime configuration at the bootstrap seam."""

    energy = world.material_energy_config
    interval_s = world.economy_config.dt_eco_s
    return LivingRuntimeConfig(
        economy=world.economy_config,
        live=world.live_config,
        motion=PhaseWindowConfig(interval_s, stages=4, phase_samples=3),
        behavior=LIVE_BEHAVIOR_CONFIG,
        feeding=FeedingConfig(
            interval_s=interval_s,
            q_mass_mol=world.economy_config.q_mass_mol,
            capture_efficiency=0.5,
            assimilation_efficiency=0.5,
            producer_j_per_q=energy.producer_j_per_q,
            reserve_j_per_q=energy.reserve_j_per_q,
            allocation_rounds=8,
        ),
        metabolism=MetabolismConfig(
            interval_s=interval_s,
            maintenance_w_per_kg=0.01,
            chemical_to_mechanical_efficiency=1.0,
            reserve_j_per_q=energy.reserve_j_per_q,
        ),
        mortality=profile.mortality,
        mutation=profile.mutation,
        development=calibrate_development_config(
            state.population,
            state.body,
        ),
        child_initial_reserve_q=100,
    )
