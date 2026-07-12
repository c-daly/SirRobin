# SirRobin S0 locomotion decision report

**Decision:** **NO-GO / REVISE** under the original 1,000-creature/90M Gate E.
**Generated:** 2026-07-12 from the artifacts under `runs/`
**Corpus SHA-256:** `6721211db0aa42c3f63e4364f6f84c948a1f9157af6e02bb35325f09fab5e96e`
**Authorizing hardware:** NVIDIA GeForce RTX 5070, 12,227 MiB physical VRAM, PyTorch 2.13.0+cu130

## Executive position

The physical kernel is scientifically credible enough to retain, but the original Gate E fails decisively:
the best valid CUDA-graph measurements reach less than one percent of the frozen `9.0e7`
creature-steps/s floor. Thresholds were not changed after measurement, H0 was not allowed to authorize, and
the compile timeout is recorded as a failure rather than silently discarded.

This report preserves the original experiment. The separately pre-registered population-grounded Gate E and
its result are recorded in `2026-07-12-sirrobin-S0-population-gate-revision-report.md`.

## Gate decision matrix

| Gate | Status | Evidence |
|---|---|---|
| A — scaffold/representation | GREEN | 17-slot sentinel layout, checked int64 transfers, import firewall, hash-closed manifest, in-place graph state, fixed churn/address tests |
| B — physical/oracle fidelity | GREEN | untouched donor reconstruction/traces and H0/H1/H2 bug-inert episodes; independent gain1 Lamb/reactive/fin/matrix/solve cases; tilted donor-bug regression; zero authorization regularization |
| C — mechanical consistency | GREEN | force-power identities, per-step `R_step`, full f32/f64 100,000-prefix curves, no monotone bias |
| D — reproducibility posture | GREEN | exact discrete tests; 960-step deterministic CPU replay max difference `0.0`; diagnostic tax `1.123x` |
| E — throughput/affordability | **RED** | H1 and H2 are non-OOM but miss the original frozen floor by more than 120x |

## Original authorizing throughput

Five repetitions, 360 warmup steps, 600 timed steps, f32, `N_cap=1024`, `N_live=1000`. The authorization
statistic is the minimum of five.

| Corpus | Minimum c-steps/s | Median c-steps/s | Fraction of floor | Shortfall | Peak allocation |
|---|---:|---:|---:|---:|---:|
| H0 | 701,821 | 723,546 | 0.0078 | 128.2x | 66.0 MiB |
| H1 | 744,955 | 745,917 | 0.0083 | 120.8x | 66.0 MiB |
| H2 | 741,787 | 743,549 | 0.0082 | 121.3x | 66.0 MiB |

H1/H0 minimum ratio is `1.061` and H2/H0 is
`1.057`. The fixed-slot kernel therefore does not show a harmful heterogeneous
masking tax in these cells. The all-16-slot occupancy cell has minimum `728,201` c-steps/s,
`0.982` of H2, despite analytic padding ratios of 2.485x for H1 and 3.821x for
H2. That is evidence against treating an arena rewrite as the first remedy.

The H1 churn cell minimum is `566,979` and median `742,967` c-steps/s. Its
median/no-churn ratio is `0.996`; the lower minimum records the scheduled churn
event rather than hiding it. The CPU eager B=1024 probe reaches minimum `33,677` c-steps/s.
CUDA is clearly faster at B=1024, but the smallest crossover `B*` remains unresolved because the decision was
already terminal and the complete crossover ladder was not used to manufacture a more favorable conclusion.

The r1 `torch.compile` cell status is `compile-timeout` at its frozen
`180`-second timeout. r2 explicit CUDA graph is therefore the best completed
rung, not evidence that compilation succeeded.

## Long-horizon mechanics

| Dtype | Maximum prefix ratio by body | Threshold | Monotone bias | Regularizations | Result |
|---|---|---:|---|---:|---|
| f32 | `[0.0, 0.0, 9.300699657527636e-07, 6.008714979389287e-07]` | 1e-03 | `[False, False, False, False]` | 0 | PASS |
| f64 | `[0.0, 0.0, 2.1163920339892322e-15, 6.570784585719605e-16]` | 1e-06 | `[False, False, False, False]` | 0 | PASS |

The `.npz` artifacts retain every cumulative residual, cumulative scale, and normalized prefix. Signed
reactive wake flux remains separate from nonnegative dissipated wake power; the former closes the force-law
identity and the latter reproduces donor mechanical-work accounting.

## Profiler attribution

The H1 eager profile records severe fine-grained operator fragmentation over 20 steps:

| Operator | Calls | Self CUDA time (us) |
|---|---:|---:|
| `aten::mul` | 7,060 | 7,337.7 |
| `aten::add` | 3,400 | 3,033.5 |
| `aten::gather` | 620 | 2,415.3 |
| `aten::sub` | 1,960 | 1,620.2 |
| `aten::where` | 920 | 1,130.6 |
| `aten::cat` | 620 | 1,039.0 |
| `aten::linalg_cross` | 920 | 1,014.6 |
| `aten::all` | 200 | 624.1 |

This is F3 (dispatch/operator granularity), not F4 (memory): peak allocation is only about 66 MiB against the
11 GiB cap. A fused pose/quaternion/force path is the first measured optimization target. Any fusion must
preserve both oracle arms and the `StepLedger`; a faster narrowed force law does not qualify.

## Falsifier register

| ID | Classification | Consequence |
|---|---|---|
| F1 heterogeneity | CLEAR in measured cells | H1/H2 do not underperform H0 materially |
| F2 pose/segment domination | UNRESOLVED by phase attribution | retain as a profiler question inside fusion work |
| F3 launch/operator overhead | **TRIPPED** | compile timed out; eager profile shows thousands of small ops; open fusion tranche |
| F4 padding/VRAM | CLEAR | 66 MiB peak; full occupancy is close to H2 timing |
| F5 donor portability | CLEAR | untouched C# arm builds/runs and conformance tests pass |
| F6 analytic disagreement | CLEAR | independent gain1 cases pass |
| F7 mechanical drift | CLEAR | both 100,000-step prefix gates pass |
| F8 regularization | CLEAR | zero across correctness, drift, and authorization timing |
| F9 lifecycle graph/cost | CLEAR with recorded minimum tax | graph state is in-place; churn addresses stable |
| F10 deterministic tax | CLEAR/INFORMATIONAL | 1.123x CPU diagnostic tax, exact replay in this run |
| F11 authorization OOM | CLEAR | H1/H2 both complete far below cap |
| F12 whole-tick allocation | UNRESOLVED/LATER | G-E2E remains later authority; S0 already fails independently |
| F13 donor extraction | CLEAR | Unity-light console links untouched donor source and emits hash-bound fixtures |

## Required next decision

The original experiment does not authorize S1. See the separate population-gate revision report for the
owner-approved replacement criterion and its outcome.
