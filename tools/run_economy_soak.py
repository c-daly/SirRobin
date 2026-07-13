"""Run the production economy kernel for the exact one-million-step closure gate."""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import replace
from pathlib import Path

import torch
from economy_provenance import economy_source_hash

from sirrobin.economy.config import EconomyConfig
from sirrobin.economy.state import EconomyState
from sirrobin.economy.step import EconomyKernel


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=1_000_000)
    parser.add_argument("--output", type=Path, default=Path("runs/economy-soak.json"))
    args = parser.parse_args()
    config = replace(
        EconomyConfig(),
        gx=1,
        gy=1,
        gz=2,
        lx_m=10,
        ly_m=10,
        lz_m=10,
        sinking_speed_m_s=0.0,
        kz_nd_m2_s=0.0,
        kz_bp_m2_s=0.0,
        kz_bm_m2_s=0.0,
        density_mortality_m3_mol_s=0.0,
    )
    state = EconomyState.zeros(config)
    state.nd_q.fill_(10_000_000)
    state.bp_q.fill_(1_000_000)
    state.bd_q[..., 0] = 500_000
    kernel = EconomyKernel(state, config)
    initial = state.total_per_world().clone()
    failures = torch.zeros(config.worlds, dtype=torch.bool)
    interventions = torch.zeros(config.worlds, dtype=torch.int64)
    shortfalls = torch.zeros(config.worlds, dtype=torch.int64)
    started = time.perf_counter()
    for step in range(args.steps):
        ledger = kernel.step()
        failures |= ~ledger.books_closed
        interventions += ledger.intervention_count
        shortfalls += ledger.transport_shortfall_q
        if (step + 1) % 100_000 == 0:
            print(f"soak {step + 1}/{args.steps}", flush=True)
    elapsed = time.perf_counter() - started
    final = state.total_per_world()
    payload = {
        "schema": "sirrobin.economy.soak.v1",
        "steps": args.steps,
        "elapsed_s": elapsed,
        "steps_per_s": args.steps / elapsed,
        "config_hash": config.sha256(),
        "source_hash": economy_source_hash(Path(__file__).resolve().parents[1]),
        "initial_total_q": initial.tolist(),
        "final_total_q": final.tolist(),
        "book_failure_count": int(failures.sum().item()),
        "intervention_count": int(interventions.sum().item()),
        "transport_shortfall_q": int(shortfalls.sum().item()),
        "reservoir_totals_q": [int(reservoir.sum().item()) for reservoir in state.reservoirs],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(payload, sort_keys=True), flush=True)
    if (
        payload["book_failure_count"]
        or payload["intervention_count"]
        or payload["transport_shortfall_q"]
        or initial.tolist() != final.tolist()
    ):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
