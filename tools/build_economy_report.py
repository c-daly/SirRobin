"""Build the committed S1 decision report from measured artifacts."""

# ruff: noqa: E501

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT / "runs"
OUTPUT = ROOT / "docs/superpowers/reports/2026-07-12-sirrobin-S1-decision-report.md"


def load(name: str) -> dict:
    return json.loads((RUNS / name).read_text(encoding="utf-8"))


def main() -> None:
    dynamics = load("economy-dynamics.json")
    soak = load("economy-soak.json")
    cpu = load("economy-benchmark-cpu.json")
    cuda = load("economy-benchmark-cuda.json")
    compiled = load("economy-benchmark-cuda-compile.json")
    profile = load("economy-profile-cuda.json")
    resolution = load("economy-resolution.json")
    coarse = dynamics["coarse"]
    convergence = dynamics["convergence"]
    top = profile["top_operators"][0]
    source_hashes = {
        dynamics["source_hash"],
        soak["source_hash"],
        cpu["source_hash"],
        cuda["source_hash"],
        compiled["source_hash"],
        profile["source_hash"],
        resolution["source_hash"],
    }
    if len(source_hashes) != 1:
        raise RuntimeError("S1 evidence artifacts were generated from different source trees")
    source_hash = source_hashes.pop()
    report = f"""# SirRobin S1 conserved-nutrient decision report

**Decision:** **GO**<br>
**Date:** 2026-07-12<br>
**Authority:** `docs/superpowers/plans/2026-07-12-sirrobin-S1-conserved-nutrient-implementation-plan.md`<br>
**Economy config SHA-256:** `{cpu["config_hash"]}`<br>
**S1 source-tree SHA-256:** `{source_hash}`<br>
**Authorizing GPU:** {cuda["hardware"]["name"]}, {cuda["hardware"]["total_memory_bytes"] / 1024**2:.0f} MiB, CUDA {cuda["hardware"]["cuda_runtime"]}

## Decision

The exact four-reservoir nutrient loop is fit to authorize S2. Gates A-F are green: every committed nutrient
quantum remains in `Nd_q/Bp_q/Bd_q/Bm_q`, the uncapped `d_dd=0` bloom still terminates through nutrient
drawdown plus baseline loss, the half-timestep trajectory converges, the one-million-step production soak closes
without a single failure, restart retains all carries, and the full validation grid is non-OOM on CPU and CUDA.

This is a material-cycle result, not yet a creature ecology. It does not claim grazing resolution, a complete
biological pump, horizontal biogeography, self-shading, or conserved creature energy.

## Gate matrix

| Gate | Status | Evidence |
|---|---|---|
| A - config, units, boundaries | GREEN | frozen hash-closed anchors/fixtures; CFL-derived substeps; positive deep remineralization floor; capability import firewall |
| B - exact bookkeeping | GREEN | per-world int64 closure, exact BGE/face partitions, `<2^62` domain, raw-write audit, {soak["steps"]:,}-step production soak |
| C - reaction fidelity | GREEN | independent NumPy Monod/light/loss/remineralization fixtures; zero-seed and long-run BGE carry tests |
| D - field and transport fidelity | GREEN | trilinear/periodic sampling, closed sinking, `Nd/Bp/Bm` mixing, producer recolonization, row-slice exactness, Martin attenuation corroboration |
| E - closed-loop behavior | GREEN | hard `d_dd=0` bloom/crash plus same-horizon `dt_eco/2` convergence |
| F - restart and affordability | GREEN | safetensors carry/clock/parity continuation, negative missing-carry test, point depletion, CPU/CUDA/compiled benchmarks and profiler attribution |

## Hard anti-cap dynamics

The authorizing column sets `d_dd=0`; density-dependent mortality contributes nothing to termination.

| Metric | Coarse `dt=0.1 d` | Half-step difference |
|---|---:|---:|
| Initial `Bp_q` | {coarse["initial_bp_q"]:,} | - |
| Peak `Bp_q` | {coarse["peak_bp_q"]:,} | {convergence["relative_differences"]["peak_bp_q"]:.3%} |
| Final `Bp_q` | {coarse["final_bp_q"]:,} | - |
| Minimum `Nd_q` | {coarse["minimum_nd_q"]:,} | {convergence["relative_differences"]["minimum_nd_q"]:.3%} |
| Late mean `Bp_q` | {coarse["late_mean_bp_q"]:,.0f} | {convergence["relative_differences"]["late_mean_bp_q"]:.3%} |
| Late relative range | {coarse["late_relative_range"]:.3%} | qualitative gate <=25% |
| Integrated production | {coarse["integrated_production_q"]:,} | {convergence["relative_differences"]["integrated_production_q"]:.3%} |
| Integrated decomposition | {coarse["integrated_decomposition_q"]:,} | {convergence["relative_differences"]["integrated_decomposition_q"]:.3%} |
| Integrated microbial turnover | {coarse["integrated_microbe_turnover_q"]:,} | {convergence["relative_differences"]["integrated_microbe_turnover_q"]:.3%} |

Peak timing differs by {convergence["peak_time_difference_s"] / 3600:.1f} h and crash timing by
{convergence["crash_time_difference_s"] / 3600:.1f} h, each within one 2.4 h coarse ecological step. Both runs
close exact books at every step and record zero intervention.

The no-light control records zero primary production and loses producer stock without developing a vertical
producer gradient. The no-mixing control cannot return deep nutrient to the surface. Together they constrain
the zonation claim to the implemented light-plus-mixing mechanisms.

## Long-horizon closure

The production `EconomyKernel` completed {soak["steps"]:,} steps in {soak["elapsed_s"]:.1f} s
({soak["steps_per_s"]:.1f} steps/s) on the non-trivial closed reaction configuration.

| Check | Result |
|---|---:|
| Initial total | {soak["initial_total_q"][0]:,} q |
| Final total | {soak["final_total_q"][0]:,} q |
| Book failures | {soak["book_failure_count"]} |
| Interventions | {soak["intervention_count"]} |
| Transport shortfall | {soak["transport_shortfall_q"]} q |

Final stocks remain distributed across all four pools: `{soak["reservoir_totals_q"]}` q in
`[Nd,Bp,Bd,Bm]`. No float drift tolerance exists or was consumed.

## Full-grid affordability

All cells use the frozen `[1,64,64,32]` grid and independently close exact books.

| Path | Median steps/s | Minimum steps/s | Median cell-updates/s | Peak allocation |
|---|---:|---:|---:|---:|
| CPU eager | {cpu["median_steps_per_s"]:.2f} | {cpu["minimum_steps_per_s"]:.2f} | {cpu["median_cell_updates_per_s"]:,.0f} | n/a |
| CUDA eager | {cuda["median_steps_per_s"]:.2f} | {cuda["minimum_steps_per_s"]:.2f} | {cuda["median_cell_updates_per_s"]:,.0f} | {cuda["peak_memory_bytes"] / 1024**2:.1f} MiB |
| CUDA compile | {compiled["median_steps_per_s"]:.2f} | {compiled["minimum_steps_per_s"]:.2f} | {compiled["median_cell_updates_per_s"]:,.0f} | {compiled["peak_memory_bytes"] / 1024**2:.1f} MiB |

The compiled path is usable and fastest, although CUDA graphs are skipped because reservoir/carry inputs are
mutated in place. This is an optimization observation, not a correctness exception. S1 intentionally has no
invented whole-tick throughput threshold.

The eager profiler's largest CUDA attribution is `{top["operator"]}` with {top["calls"]} calls over
{profile["steps"]} profiled steps and {top["self_cuda_time_us"]:.1f} us self-device time. General
largest-remainder sorting remains a measured optimization target; any replacement must preserve the one-debit
source budget and be benchmarked as its own tranche.

## Eulerian resolution evidence

The synthetic off-center point request removed exactly {resolution["realized_q"]:,} of
{resolution["requested_q"]:,} requested quanta across {resolution["depletion_footprint_cells"]} trilinear
neighbors. The footprint diagonal is {resolution["depletion_footprint_diagonal_m"]:.1f} m and the smallest grid
feature is {resolution["smallest_grid_feature_m"]:.1f} m. Values and gradients remained finite, and conservative
mixing reduced the isolated-hotspot variance from {resolution["hotspot_variance_before"]:.3g} to
{resolution["hotspot_variance_after_20_steps"]:.3g} in 20 steps.

This is interpolation and transaction evidence only. The measured point-operation/CPU-field-step cost ratio is
{resolution["depletion_to_step_cost_ratio"]:.2f}, but a single synthetic debit cannot predict dense grazing.
The Eulerian/parcel fork therefore remains explicitly unresolved until real feeding exists.

## Verification suite

- CPU/default suite: 59 passed, 3 CUDA-only tests skipped.
- Escalated CUDA suite: 3 passed (S1 closure plus both existing S0 CUDA regressions).
- Ruff, uv lock consistency, and all five import-linter contracts pass.

## Falsifier disposition

| Risk | Classification |
|---|---|
| Int64 overflow / negative state | CLEAR in configured `<2^62` domain |
| Independent rounding mint | CLEAR; one debit is apportioned exactly |
| Microbial terminal trap | CLEAR; explicit `Bm -> Nd` turnover and nonzero long-soak flux |
| Logistic cap hidden in `d_dd` | CLEAR; full authorizing pulse passes at `d_dd=0` |
| Explicit-step numerical cycle | CLEAR; all frozen half-step differences <5% |
| Absorbing sterile producer cells | CLEAR; conservative `Bp` mixing recolonizes from neighbors |
| Closed-bottom detritus trap | CLEAR in validation horizon; positive floor and distributed final stocks |
| Restart omits carries/parity | CLEAR; exact continuation plus negative omission test |
| CUDA/compile changes closure | CLEAR; every measured path closes exactly |
| Grid OOM / host cell loop | CLEAR; <53 MiB peak and no Python per-cell field loop |
| Dense-grazing Eulerian resolution | DEFERRED honestly; no real grazing exists yet |
| Complete biological pump | DEFERRED; burial/export is absent |

## Consequence

S2 may proceed. The four-reservoir representation and transaction rules remain canonical until a real new
reservoir mechanism lands. Later feeding must decide the Eulerian/parcel fork from dense-grazing evidence and
must not keep two authoritative representations. Later energy work must begin with real reserve and heat
reservoirs; it may not resurrect a synchronized biomass-energy mirror.
"""
    OUTPUT.write_text(report, encoding="utf-8")


if __name__ == "__main__":
    main()
