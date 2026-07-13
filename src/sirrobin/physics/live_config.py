"""Immutable configuration for canonical live locomotion."""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict, dataclass


@dataclass(frozen=True, slots=True)
class LiveLocomotionConfig:
    dt: float = 1.0 / 120.0
    rho_water: float = 1000.0
    rho_neutral_gene: float = 4.0
    kg_per_sim_mass: float = 250.0
    drag_coeff: float = 0.1
    yaw_drag_coeff: float = 1.0
    fin_profile_cd: float = 0.02
    fin_span_eff: float = 0.9
    fin_stall_aoa: float = 0.35
    amp_max_rad: float = math.radians(58.0)
    full_authority_error_rad: float = math.radians(60.0)
    turn_slew_fraction: float = 0.2
    heading_lowpass_alpha: float = 0.35
    min_heading_speed_m_s: float = 0.2
    kappa_max: float = 1.0e6
    lam_floor_kg: float = 1.0e-9
    eps_spd: float = 1.0e-6
    inertia_floor_kg_m2: float = 1.0e-4
    emergency_omega_rad_s: float = 8.0
    rtol_f32: float = 1.0e-3
    rtol_f64: float = 1.0e-6
    energy_atol_f32_j: float = 1.0e-8
    energy_atol_f64_j: float = 1.0e-12
    frame_version: str = "enu-world_flu-body_v1"

    def validate(self) -> None:
        if self.dt != 1.0 / 120.0:
            raise ValueError("live locomotion dt is frozen at 1/120 s")
        if self.kg_per_sim_mass != self.rho_water / self.rho_neutral_gene:
            raise ValueError("KgPerSimMass must equal rho_water/rho_neutral_gene")
        if self.inertia_floor_kg_m2 <= 0 or self.lam_floor_kg <= 0:
            raise ValueError("numerical assertion floors must be positive")
        if self.emergency_omega_rad_s <= 1.0:
            raise ValueError("emergency omega must remain outside the authorizing operating envelope")
        if not 0 < self.heading_lowpass_alpha <= 1 or not 0 < self.turn_slew_fraction <= 1:
            raise ValueError("controller fractions must lie in (0,1]")
        if self.frame_version != "enu-world_flu-body_v1":
            raise ValueError("public live coordinates must use the frozen ENU/FLU frame")

    def canonical_json(self) -> str:
        return json.dumps(asdict(self), sort_keys=True, separators=(",", ":"))

    def sha256(self) -> str:
        return hashlib.sha256(self.canonical_json().encode()).hexdigest()


DEFAULT_LIVE_CONFIG = LiveLocomotionConfig()
DEFAULT_LIVE_CONFIG.validate()
