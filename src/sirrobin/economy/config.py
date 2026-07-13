"""Frozen configuration and scientific anchors for the conserved nutrient economy."""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict, dataclass, replace

from sirrobin.numerics.flux import INT64_SAFE_MAX

DAY_S = 86_400.0


@dataclass(frozen=True, slots=True)
class EconomyConfig:
    worlds: int = 1
    gx: int = 64
    gy: int = 64
    gz: int = 32
    lx_m: float = 640.0
    ly_m: float = 640.0
    lz_m: float = 160.0
    dt_eco_s: float = 8_640.0
    q_mass_mol: float = 1.0e-9
    max_inventory_q: int = 10**15

    i0_w_m2: float = 200.0
    k_light_m_inv: float = 0.04
    k_i_w_m2: float = 20.0
    k_n_mol_m3: float = 2.8e-5
    mu_max_s: float = 0.5 / DAY_S
    producer_maintenance_s: float = 0.10 / DAY_S
    producer_mortality_s: float = 0.05 / DAY_S
    density_mortality_m3_mol_s: float = 500.0 / DAY_S
    max_validation_biomass_mol_m3: float = 2.0e-3

    bge: float = 0.20
    microbial_turnover_s: float = 0.05 / DAY_S
    martin_b: float = 0.858
    martin_reference_depth_m: float = 100.0
    remin_floor_s: float = 0.005 / DAY_S
    sinking_speed_m_s: float = 10.0 / DAY_S
    kz_nd_m2_s: float = 1.0e-4
    kz_bp_m2_s: float = 1.0e-4
    kz_bm_m2_s: float = 1.0e-4

    schema: str = "sirrobin.economy.config.v1"

    @property
    def shape(self) -> tuple[int, int, int, int]:
        return (self.worlds, self.gx, self.gy, self.gz)

    @property
    def face_shape(self) -> tuple[int, int, int, int]:
        return (self.worlds, self.gx, self.gy, self.gz - 1)

    @property
    def dx_m(self) -> float:
        return self.lx_m / self.gx

    @property
    def dy_m(self) -> float:
        return self.ly_m / self.gy

    @property
    def dz_m(self) -> float:
        return self.lz_m / self.gz

    @property
    def cell_volume_m3(self) -> float:
        return self.dx_m * self.dy_m * self.dz_m

    @property
    def max_kz_m2_s(self) -> float:
        return max(self.kz_nd_m2_s, self.kz_bp_m2_s, self.kz_bm_m2_s)

    @property
    def transport_substeps(self) -> int:
        mixing = 2.0 * self.max_kz_m2_s * self.dt_eco_s / self.dz_m**2
        sinking = self.sinking_speed_m_s * self.dt_eco_s / self.dz_m
        return max(1, math.ceil(max(mixing, sinking)))

    def remin_rate_s(self, depth_m: float) -> float:
        mapped = self.martin_b * self.sinking_speed_m_s / (self.martin_reference_depth_m + depth_m)
        return max(self.remin_floor_s, mapped)

    def validate(self) -> None:
        if self.worlds < 1 or min(self.gx, self.gy) < 1 or self.gz < 2:
            raise ValueError("grid must have positive dimensions and at least two vertical cells")
        if min(self.lx_m, self.ly_m, self.lz_m, self.dt_eco_s, self.q_mass_mol) <= 0:
            raise ValueError("dimensions, timestep, and mass quantum must be positive")
        if not 0.0 <= self.bge <= 1.0:
            raise ValueError("BGE must be in [0,1]")
        if self.max_validation_biomass_mol_m3 <= 0 or not math.isfinite(self.max_validation_biomass_mol_m3):
            raise ValueError("validation biomass bound must be finite and positive")
        rates = (
            self.mu_max_s,
            self.producer_maintenance_s,
            self.producer_mortality_s,
            self.density_mortality_m3_mol_s,
            self.microbial_turnover_s,
            self.remin_floor_s,
            self.sinking_speed_m_s,
            self.kz_nd_m2_s,
            self.kz_bp_m2_s,
            self.kz_bm_m2_s,
        )
        if any(rate < 0 or not math.isfinite(rate) for rate in rates):
            raise ValueError("rates and transport anchors must be finite and nonnegative")
        if self.remin_floor_s <= 0:
            raise ValueError("the closed-bottom remineralization floor must be positive")
        if self.max_inventory_q >= INT64_SAFE_MAX:
            raise ValueError("configured inventory must remain below 2^62 quanta")
        sub_dt = self.dt_eco_s / self.transport_substeps
        if 2.0 * self.max_kz_m2_s * sub_dt / self.dz_m**2 > 1.0 + 1e-15:
            raise ValueError("mixing CFL is unstable")
        if self.sinking_speed_m_s * sub_dt / self.dz_m > 1.0 + 1e-15:
            raise ValueError("sinking CFL is unstable")
        if 1.0 / self.remin_floor_s > 1_000_000 * self.dt_eco_s:
            raise ValueError("deep detritus residence time exceeds the validation horizon")
        fastest_reaction = max(
            self.mu_max_s,
            self.producer_maintenance_s + self.producer_mortality_s,
            self.producer_mortality_s + self.density_mortality_m3_mol_s * self.max_validation_biomass_mol_m3,
            self.microbial_turnover_s,
            self.remin_rate_s(0.5 * self.dz_m),
        )
        if fastest_reaction * self.dt_eco_s > 0.25:
            raise ValueError("reaction timestep exceeds the frozen single-step rate bound")

    def with_half_timestep(self) -> EconomyConfig:
        return replace(self, dt_eco_s=self.dt_eco_s / 2.0)

    def canonical_json(self) -> str:
        return json.dumps(asdict(self), sort_keys=True, separators=(",", ":"))

    def sha256(self) -> str:
        return hashlib.sha256(self.canonical_json().encode()).hexdigest()


DEFAULT_ECONOMY_CONFIG = EconomyConfig()
DEFAULT_ECONOMY_CONFIG.validate()
