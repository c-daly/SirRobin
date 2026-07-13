"""Independent NumPy-only fixture generator for the conserved nutrient economy."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

import numpy as np

DAY = 86_400.0
ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "oracle" / "fixtures" / "economy"

CONFIG = {
    "worlds": 1,
    "gx": 64,
    "gy": 64,
    "gz": 32,
    "lx_m": 640.0,
    "ly_m": 640.0,
    "lz_m": 160.0,
    "dt_eco_s": 8_640.0,
    "q_mass_mol": 1.0e-9,
    "max_inventory_q": 10**15,
    "i0_w_m2": 200.0,
    "k_light_m_inv": 0.04,
    "k_i_w_m2": 20.0,
    "k_n_mol_m3": 2.8e-5,
    "mu_max_s": 0.5 / DAY,
    "producer_maintenance_s": 0.10 / DAY,
    "producer_mortality_s": 0.05 / DAY,
    "density_mortality_m3_mol_s": 500.0 / DAY,
    "max_validation_biomass_mol_m3": 2.0e-3,
    "bge": 0.20,
    "microbial_turnover_s": 0.05 / DAY,
    "martin_b": 0.858,
    "martin_reference_depth_m": 100.0,
    "remin_floor_s": 0.005 / DAY,
    "sinking_speed_m_s": 10.0 / DAY,
    "kz_nd_m2_s": 1.0e-4,
    "kz_bp_m2_s": 1.0e-4,
    "kz_bm_m2_s": 1.0e-4,
    "schema": "sirrobin.economy.config.v1",
}


def canonical_hash(value: object) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def file_hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def anchor(symbol: str, value: float, unit: str, valid_range: list[float], kind: str, source: str) -> dict:
    return {
        "symbol": symbol,
        "value": value,
        "unit": unit,
        "valid_range": valid_range,
        "classification": kind,
        "source": source,
    }


def limitation(nd: float, depth: float) -> tuple[float, float, float]:
    light = CONFIG["i0_w_m2"] * math.exp(-CONFIG["k_light_m_inv"] * depth)
    gamma = light / (light + CONFIG["k_i_w_m2"])
    nutrient = nd / (nd + CONFIG["k_n_mol_m3"]) if nd > 0 else 0.0
    return light, gamma, min(gamma, nutrient)


def build_reaction_cases() -> dict:
    cases = []
    volume = 500.0
    for name, nd, bp, bd, bm, depth in (
        ("zero", 0.0, 0.0, 0.0, 0.0, 2.5),
        ("half_nutrient", CONFIG["k_n_mol_m3"], 0.02, 0.01, 0.005, 2.5),
        ("deep", 5.0e-4, 0.02, 0.1, 0.03, 157.5),
        ("saturated", 0.1, 0.5, 0.2, 0.1, 2.5),
    ):
        light, gamma, limit = limitation(nd, depth)
        bp_mol = bp * volume
        bd_mol = bd * volume
        bm_mol = bm * volume
        remin = max(
            CONFIG["remin_floor_s"],
            CONFIG["martin_b"] * CONFIG["sinking_speed_m_s"] / (CONFIG["martin_reference_depth_m"] + depth),
        )
        cases.append(
            {
                "name": name,
                "input": {
                    "nd_mol_m3": nd,
                    "bp_mol_m3": bp,
                    "bd_mol_m3": bd,
                    "bm_mol_m3": bm,
                    "depth_m": depth,
                },
                "expected": {
                    "light_w_m2": light,
                    "light_limitation": gamma,
                    "combined_limitation": limit,
                    "production_mol": CONFIG["mu_max_s"] * limit * bp_mol * CONFIG["dt_eco_s"],
                    "maintenance_mol": CONFIG["producer_maintenance_s"] * bp_mol * CONFIG["dt_eco_s"],
                    "mortality_mol": (
                        CONFIG["producer_mortality_s"] + CONFIG["density_mortality_m3_mol_s"] * bp
                    )
                    * bp_mol
                    * CONFIG["dt_eco_s"],
                    "decomposition_mol": remin * bd_mol * CONFIG["dt_eco_s"],
                    "microbial_turnover_mol": CONFIG["microbial_turnover_s"] * bm_mol * CONFIG["dt_eco_s"],
                    "remin_rate_s": remin,
                },
            }
        )
    return {"schema": "sirrobin.economy.reactions.v1", "config_hash": canonical_hash(CONFIG), "cases": cases}


def build_columns() -> dict[str, np.ndarray]:
    dz = CONFIG["lz_m"] / CONFIG["gz"]
    depth = (np.arange(CONFIG["gz"], dtype=np.float64) + 0.5) * dz
    remin = np.maximum(
        CONFIG["remin_floor_s"],
        CONFIG["martin_b"] * CONFIG["sinking_speed_m_s"] / (CONFIG["martin_reference_depth_m"] + depth),
    )
    martin = ((CONFIG["martin_reference_depth_m"] + depth) / CONFIG["martin_reference_depth_m"]) ** (
        -CONFIG["martin_b"]
    )
    constant = np.full(4, 1000, dtype=np.int64)
    gradient = np.array([4000, 3000, 2000, 1000], dtype=np.int64)
    # Exact expected one-face moves for a deliberately integral fixture.
    mixing_expected = np.array([3990, 3000, 2000, 1010], dtype=np.int64)
    sinking_initial = np.array([100, 0, 0, 0], dtype=np.int64)
    sinking_expected = np.array([80, 20, 0, 0], dtype=np.int64)
    return {
        "depth_m": depth,
        "remin_rate_s": remin,
        "martin_relative_flux": martin,
        "constant_q": constant,
        "gradient_q": gradient,
        "mixing_expected_q": mixing_expected,
        "sinking_initial_q": sinking_initial,
        "sinking_expected_q": sinking_expected,
    }


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    config_hash = canonical_hash(CONFIG)
    anchors = [
        anchor(
            "q_mass", 1e-9, "mol nutrient quantum-1", [1e-12, 1e-6], "numerical schema", "S1 authority §1.1"
        ),
        anchor(
            "mu_max",
            0.5 / DAY,
            "s-1",
            [0.0, 2.0 / DAY],
            "validation anchor",
            "https://fisherybulletin.nmfs.noaa.gov/sites/default/files/pdf-content/1972/704/eppley.pdf",
        ),
        anchor(
            "K_N",
            2.8e-5,
            "mol m-3",
            [1e-7, 1e-2],
            "measured species anchor",
            "Dunaliella salina K_mu=28 nmol P L-1; https://repository.lsu.edu/enviro_sciences_pubs/299/",
        ),
        anchor(
            "BGE",
            0.20,
            "1",
            [0.0, 1.0],
            "validation anchor within observed range",
            "https://doi.org/10.1146/annurev.ecolsys.29.1.503",
        ),
        anchor(
            "martin_b",
            0.858,
            "1",
            [0.0, 2.0],
            "measured composite fit",
            "https://doi.org/10.1016/0198-0149(87)90086-0",
        ),
        anchor(
            "Kz",
            1e-4,
            "m2 s-1",
            [1e-6, 1e-2],
            "validation transport choice",
            "Ledwell et al. measured slow open-ocean mixing; https://doi.org/10.1038/364701a0",
        ),
        anchor(
            "w_sink",
            10.0 / DAY,
            "m s-1",
            [0.0, 200.0 / DAY],
            "validation transport choice",
            "S1 numerical validation condition; not inferred as a universal particle speed",
        ),
        anchor(
            "m_resp",
            0.10 / DAY,
            "s-1",
            [0.0, 1.0 / DAY],
            "validation loss anchor",
            "Master design documented anchor; frozen before dynamics",
        ),
        anchor(
            "d0",
            0.05 / DAY,
            "s-1",
            [0.0, 1.0 / DAY],
            "validation loss anchor",
            "Master design documented anchor; frozen before dynamics",
        ),
        anchor(
            "d_dd",
            500.0 / DAY,
            "m3 mol-1 s-1",
            [0.0, 5000.0 / DAY],
            "non-authorizing mechanism probe",
            "Must be removed in the hard anti-cap authorization ablation",
        ),
        anchor(
            "C_Bp_validation_max",
            2.0e-3,
            "mol m-3",
            [1e-6, 1.0],
            "reaction stability domain",
            "Frozen numerical validity bound, not a carrying capacity or state clamp",
        ),
        anchor(
            "m_microbe",
            0.05 / DAY,
            "s-1",
            [0.0, 1.0 / DAY],
            "provisional closure anchor",
            "Required temporary turnover while S1 has no microbivores",
        ),
        anchor(
            "k_remin_floor",
            0.005 / DAY,
            "s-1",
            [1e-4 / DAY, 1.0 / DAY],
            "closed-bottom trap guard",
            "200-day maximum residence time, frozen validation choice",
        ),
    ]
    manifest = {
        "schema": "sirrobin.economy.anchor-manifest.v1",
        "config": CONFIG,
        "config_hash": config_hash,
        "generator_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "anchors": anchors,
        "frozen_before_dynamics": True,
    }
    (OUT / "anchor_manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    (OUT / "reaction_cases.json").write_text(
        json.dumps(build_reaction_cases(), indent=2, sort_keys=True) + "\n"
    )
    np.savez(OUT / "column_cases.npz", **build_columns())
    bloom = {
        "schema": "sirrobin.economy.bloom-config.v1",
        "config_hash": config_hash,
        "initial_mol_m3": {"Nd_surface": 8e-4, "Nd_deep": 2e-3, "Bp": 2e-5, "Bd": 0.0, "Bm": 0.0},
        "authorizing_requirements": ["d_dd_zero", "half_timestep_convergence", "exact_books_every_step"],
    }
    (OUT / "bloom_config.json").write_text(json.dumps(bloom, indent=2, sort_keys=True) + "\n")
    files = ("anchor_manifest.json", "reaction_cases.json", "column_cases.npz", "bloom_config.json")
    fixture_manifest = {
        "schema": "sirrobin.economy.fixture-manifest.v1",
        "config_hash": config_hash,
        "generator_sha256": file_hash(Path(__file__)),
        "files": {name: file_hash(OUT / name) for name in files},
    }
    (OUT / "manifest.json").write_text(
        json.dumps(fixture_manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
