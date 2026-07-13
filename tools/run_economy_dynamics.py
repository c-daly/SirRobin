"""Emit the frozen uncapped bloom and half-timestep convergence evidence."""

from __future__ import annotations

import json
from dataclasses import asdict, replace
from pathlib import Path

from economy_provenance import economy_source_hash

from sirrobin.economy.config import EconomyConfig
from sirrobin.validation.economy import compare_half_timestep, run_pulse


def main() -> None:
    output = Path("runs/economy-dynamics.json")
    coarse_config = replace(
        EconomyConfig(),
        gx=1,
        gy=1,
        gz=16,
        lx_m=10,
        ly_m=10,
        lz_m=160,
        density_mortality_m3_mol_s=0.0,
    )
    fine_config = coarse_config.with_half_timestep()
    coarse = run_pulse(coarse_config, steps=1_000, force_d_dd_zero=True)
    fine = run_pulse(fine_config, steps=2_000, force_d_dd_zero=True)
    convergence = compare_half_timestep(coarse, fine, coarse_config.dt_eco_s)
    payload = {
        "schema": "sirrobin.economy.dynamics.v1",
        "anti_cap_d_dd": 0.0,
        "source_hash": economy_source_hash(Path(__file__).resolve().parents[1]),
        "coarse_config_hash": coarse_config.sha256(),
        "fine_config_hash": fine_config.sha256(),
        "coarse": asdict(coarse),
        "fine": asdict(fine),
        "convergence": asdict(convergence),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(payload, sort_keys=True))
    if not coarse.bloom_passes or not fine.bloom_passes or not convergence.passes:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
