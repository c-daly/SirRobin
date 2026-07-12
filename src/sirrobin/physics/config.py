"""Durable locomotion configuration and physical unit anchors."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass


@dataclass(frozen=True, slots=True)
class LocomotionConfig:
    worlds: int = 1
    n_cap: int = 1024
    n_live: int = 1000
    s_max: int = 16
    max_depth: int = 5
    dt: float = 1.0 / 120.0
    rho_water: float = 1000.0
    rho_neutral_gene: float = 4.0
    kg_per_sim_mass: float = 250.0
    drag_coeff: float = 0.1
    fin_profile_cd: float = 0.02
    fin_span_eff: float = 0.9
    fin_stall_aoa: float = 0.35
    ellipsoid_mass_gain: float = 1.0
    fin_plane_gain: float = 1.0
    kappa_max: float = 1.0e6
    lam_floor_kg: float = 1.0e-9
    eps_spd: float = 1.0e-6
    p_atol_f64: float = 1.0e-10
    p_atol_f32: float = 1.0e-6
    e_atol_f64: float = 1.0e-12
    e_atol_f32: float = 1.0e-8
    rtol_f64: float = 1.0e-6
    rtol_f32: float = 1.0e-3
    throughput_floor: float = 9.0e7
    vram_cap_bytes: int = 11 * 1024**3

    @property
    def s_slot(self) -> int:
        return self.s_max + 1

    def validate(self) -> None:
        if self.worlds < 1 or self.n_cap < 1:
            raise ValueError("worlds and n_cap must be positive")
        if not 0 <= self.n_live <= self.n_cap:
            raise ValueError("n_live must be within [0,n_cap]")
        if self.s_max != 16 or self.max_depth != 5:
            raise ValueError("S0 donor caps are frozen at s_max=16, max_depth=5")
        if self.kg_per_sim_mass != self.rho_water / self.rho_neutral_gene:
            raise ValueError("KgPerSimMass must equal rho_water/rho_neutral_gene exactly")
        if self.dt != 1.0 / 120.0:
            raise ValueError("S0 dt is frozen at 1/120 s")
        if self.ellipsoid_mass_gain not in (0.0, 1.0) or not 0.0 <= self.fin_plane_gain <= 1.0:
            raise ValueError("S0 mass gain must be 0 or 1 and fin_plane_gain must be in [0,1]")

    def canonical_json(self) -> str:
        return json.dumps(asdict(self), sort_keys=True, separators=(",", ":"))

    def sha256(self) -> str:
        return hashlib.sha256(self.canonical_json().encode("utf-8")).hexdigest()


DEFAULT_CONFIG = LocomotionConfig()
DEFAULT_CONFIG.validate()
