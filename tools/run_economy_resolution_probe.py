"""Emit the S1 Eulerian interpolation/depletion resolution evidence without claiming grazing fidelity."""

from __future__ import annotations

import json
import math
import time
from pathlib import Path

import torch
from economy_provenance import economy_source_hash

from sirrobin.economy.config import DEFAULT_ECONOMY_CONFIG
from sirrobin.economy.state import EconomyState
from sirrobin.economy.step import EconomyKernel
from sirrobin.fields.geometry import GridGeometry
from sirrobin.fields.grid import ScalarGrid


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    output = root / "runs/economy-resolution.json"
    config = DEFAULT_ECONOMY_CONFIG
    geometry = GridGeometry.from_config(config)
    state = EconomyState.zeros(config)
    state.nd_q.fill_(1_000_000)
    grid = ScalarGrid(state.nd_q, geometry, q_mass_mol=config.q_mass_mol)
    position = torch.tensor([config.dx_m, config.dy_m, -config.dz_m], dtype=torch.float64)
    before = state.nd_q.clone()
    requested = 123_457
    started = time.perf_counter()
    realized = grid.deplete_at(0, position, requested)
    depletion_elapsed = time.perf_counter() - started
    changed = before != state.nd_q
    sample = grid.sample(position.view(1, 1, 3))

    recovery = EconomyState.zeros(config)
    recovery.nd_q.fill_(1_000_000)
    recovery.nd_q[0, config.gx // 2, config.gy // 2, config.gz // 2] = 100_000_000
    variance_before = float(recovery.nd_q.to(torch.float64).var().item())
    kernel = EconomyKernel(recovery, config)
    started = time.perf_counter()
    for _ in range(20):
        ledger = kernel.step()
        if not bool(ledger.books_closed.all()) or int(ledger.transport_shortfall_q.sum()) != 0:
            raise RuntimeError("resolution recovery probe violated transport gates")
    step_elapsed = (time.perf_counter() - started) / 20
    variance_after = float(recovery.nd_q.to(torch.float64).var().item())
    payload = {
        "schema": "sirrobin.economy.resolution.v1",
        "config_hash": config.sha256(),
        "source_hash": economy_source_hash(root),
        "requested_q": requested,
        "realized_q": realized,
        "depletion_footprint_cells": int(changed.sum().item()),
        "depletion_footprint_diagonal_m": math.sqrt(config.dx_m**2 + config.dy_m**2 + config.dz_m**2),
        "smallest_grid_feature_m": min(config.dx_m, config.dy_m, config.dz_m),
        "sample_value_finite": bool(torch.isfinite(sample.value_mol_m3).all()),
        "sample_gradient_finite": bool(torch.isfinite(sample.gradient_mol_m4).all()),
        "hotspot_variance_before": variance_before,
        "hotspot_variance_after_20_steps": variance_after,
        "depletion_elapsed_s": depletion_elapsed,
        "cpu_field_step_elapsed_s": step_elapsed,
        "depletion_to_step_cost_ratio": depletion_elapsed / step_elapsed,
        "interpretation": (
            "synthetic single-point evidence only; dense grazing and the parcel fork remain unresolved"
        ),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(payload, sort_keys=True))
    if realized != requested or payload["depletion_footprint_cells"] != 8:
        raise SystemExit(1)
    if variance_after >= variance_before:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
