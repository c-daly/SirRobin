# SirRobin S1 conserved-nutrient decision report

**Decision:** **GO**<br>
**Date:** 2026-07-12<br>
**Authority:** `docs/superpowers/plans/2026-07-12-sirrobin-S1-conserved-nutrient-implementation-plan.md`<br>
**Economy config SHA-256:** `7ba1174c3afb5d15753e6aff45d2ba2f87fad5e740aa4ab540cb72f7d206036a`<br>
**S1 source-tree SHA-256:** `2e1301d88c5747aa4fca121eff64877a3bf341b337453219fd0412b76e7a3cce`<br>
**Authorizing GPU:** NVIDIA GeForce RTX 5070, 12227 MiB, CUDA 13.0

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
| B - exact bookkeeping | GREEN | per-world int64 closure, exact BGE/face partitions, `<2^62` domain, raw-write audit, 1,000,000-step production soak |
| C - reaction fidelity | GREEN | independent NumPy Monod/light/loss/remineralization fixtures; zero-seed and long-run BGE carry tests |
| D - field and transport fidelity | GREEN | trilinear/periodic sampling, closed sinking, `Nd/Bp/Bm` mixing, producer recolonization, row-slice exactness, Martin attenuation corroboration |
| E - closed-loop behavior | GREEN | hard `d_dd=0` bloom/crash plus same-horizon `dt_eco/2` convergence |
| F - restart and affordability | GREEN | safetensors carry/clock/parity continuation, negative missing-carry test, point depletion, CPU/CUDA/compiled benchmarks and profiler attribution |

## Hard anti-cap dynamics

The authorizing column sets `d_dd=0`; density-dependent mortality contributes nothing to termination.

| Metric | Coarse `dt=0.1 d` | Half-step difference |
|---|---:|---:|
| Initial `Bp_q` | 320,000,000 | - |
| Peak `Bp_q` | 4,986,573,600 | 0.139% |
| Final `Bp_q` | 1,806,642,894 | - |
| Minimum `Nd_q` | 18,335,680,981 | 0.041% |
| Late mean `Bp_q` | 1,955,807,510 | 0.190% |
| Late relative range | 15.974% | qualitative gate <=25% |
| Integrated production | 49,302,959,436 | 0.176% |
| Integrated decomposition | 12,309,093,716 | 0.230% |
| Integrated microbial turnover | 1,874,701,001 | 0.286% |

Peak timing differs by 1.2 h and crash timing by
2.4 h, each within one 2.4 h coarse ecological step. Both runs
close exact books at every step and record zero intervention.

The no-light control records zero primary production and loses producer stock without developing a vertical
producer gradient. The no-mixing control cannot return deep nutrient to the surface. Together they constrain
the zonation claim to the implemented light-plus-mixing mechanisms.

## Long-horizon closure

The production `EconomyKernel` completed 1,000,000 steps in 464.3 s
(2153.6 steps/s) on the non-trivial closed reaction configuration.

| Check | Result |
|---|---:|
| Initial total | 22,500,000 q |
| Final total | 22,500,000 q |
| Book failures | 0 |
| Interventions | 0 |
| Transport shortfall | 0 q |

Final stocks remain distributed across all four pools: `[12000001, 937500, 9375000, 187499]` q in
`[Nd,Bp,Bd,Bm]`. No float drift tolerance exists or was consumed.

## Full-grid affordability

All cells use the frozen `[1,64,64,32]` grid and independently close exact books.

| Path | Median steps/s | Minimum steps/s | Median cell-updates/s | Peak allocation |
|---|---:|---:|---:|---:|
| CPU eager | 18.53 | 17.21 | 2,428,672 | n/a |
| CUDA eager | 92.84 | 88.37 | 12,169,118 | 49.0 MiB |
| CUDA compile | 109.10 | 109.01 | 14,299,439 | 38.1 MiB |

The compiled path is usable and fastest, although CUDA graphs are skipped because reservoir/carry inputs are
mutated in place. This is an optimization observation, not a correctness exception. S1 intentionally has no
invented whole-tick throughput threshold.

The eager profiler's largest CUDA attribution is `aten::sort` with 15 calls over
5 profiled steps and 9491.7 us self-device time. General
largest-remainder sorting remains a measured optimization target; any replacement must preserve the one-debit
source budget and be benchmarked as its own tranche.

## Eulerian resolution evidence

The synthetic off-center point request removed exactly 123,457 of
123,457 requested quanta across 8 trilinear
neighbors. The footprint diagonal is 15.0 m and the smallest grid
feature is 5.0 m. Values and gradients remained finite, and conservative
mixing reduced the isolated-hotspot variance from 7.48e+10 to
1.88e+10 in 20 steps.

This is interpolation and transaction evidence only. The measured point-operation/CPU-field-step cost ratio is
0.88, but a single synthetic debit cannot predict dense grazing.
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
