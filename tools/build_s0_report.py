"""Generate the S0 decision report from hash-bound run artifacts."""

# ruff: noqa: E501 -- Markdown table source rows intentionally remain one physical line.

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT / "runs"
OUT = ROOT / "docs" / "superpowers" / "reports" / "2026-07-12-sirrobin-S0-decision-report.md"
FLOOR = 9.0e7


def load(name: str) -> dict:
    return json.loads((RUNS / name).read_text())


def main() -> None:
    h0 = load("benchmark-H0-B1024-cuda-r2.json")["result"]
    h1 = load("benchmark-H1-B1024-cuda-r2.json")["result"]
    h2 = load("benchmark-H2-B1024-cuda-r2.json")["result"]
    full = load("benchmark-FULL-B1024-cuda-r2.json")["result"]
    churn = load("benchmark-H1-B1024-cuda-r2-churn.json")["result"]
    cpu = load("benchmark-H1-B1024-cpu-r0-probe.json")["result"]
    drift32 = load("drift_float32.json")
    drift64 = load("drift_float64.json")
    determinism = load("determinism-cpu.json")
    compile_failure = load("benchmark-H1-B1024-cuda-r1.json")
    profile = load("profile-H1-B1024-cuda.json")
    fixture_manifest = json.loads((ROOT / "oracle" / "fixtures" / "manifest.json").read_text())
    corpus_hash = fixture_manifest["corpus_sha256"]

    def benchmark_row(result: dict) -> str:
        fraction = result["minimum"] / FLOOR
        return (
            f'| {result["corpus_class"]} | {result["minimum"]:,.0f} | '
            f'{result["median"]:,.0f} | {fraction:.4f} | {FLOOR / result["minimum"]:.1f}x | '
            f'{result["peak_memory_bytes"] / 2**20:.1f} MiB |'
        )

    top_aten = [row for row in profile["top_operators"] if row["operator"].startswith("aten::")][:8]
    profile_rows = "\n".join(
        f'| `{row["operator"]}` | {row["calls"]:,} | {row["self_device_us"]:,.1f} |'
        for row in top_aten
    )
    report = f"""# SirRobin S0 locomotion decision report

**Decision:** **NO-GO / REVISE**  
**Generated:** 2026-07-12 from the artifacts under `runs/`  
**Corpus SHA-256:** `{corpus_hash}`  
**Authorizing hardware:** NVIDIA GeForce RTX 5070, 12,227 MiB physical VRAM, PyTorch 2.13.0+cu130

## Executive position

The physical kernel is scientifically credible enough to retain, but the current tensor decomposition is not
an affordable foundation for the next slice. Gates A-D are green. Gate E fails decisively: the best valid
CUDA-graph measurements reach less than one percent of the frozen `9.0e7` creature-steps/s floor. Thresholds
were not changed after measurement, H0 was not allowed to authorize, and the compile timeout is recorded as a
failure rather than silently discarded.

Do not begin S1 on this kernel shape. Preserve the fixtures, equations, fixed-slot representation, and
capability-based packages; open a bounded kernel-fusion tranche aimed at operator/launch fragmentation, then
rerun this unchanged acceptance corpus. The occupancy evidence does not justify a flattened arena first.

## Gate decision matrix

| Gate | Status | Evidence |
|---|---|---|
| A — scaffold/representation | GREEN | 17-slot sentinel layout, checked int64 transfers, import firewall, hash-closed manifest, in-place graph state, fixed churn/address tests |
| B — physical/oracle fidelity | GREEN | untouched donor reconstruction/traces and H0/H1/H2 bug-inert episodes; independent gain1 Lamb/reactive/fin/matrix/solve cases; tilted donor-bug regression; zero authorization regularization |
| C — mechanical consistency | GREEN | force-power identities, per-step `R_step`, full f32/f64 100,000-prefix curves, no monotone bias |
| D — reproducibility posture | GREEN | exact discrete tests; 960-step deterministic CPU replay max difference `{determinism['deterministic_rerun_max_abs']}`; diagnostic tax `{determinism['deterministic_tax']:.3f}x` |
| E — throughput/affordability | **RED** | H1 and H2 are non-OOM but miss the frozen floor by more than 120x |

## Authorizing throughput

Five repetitions, 360 warmup steps, 600 timed steps, f32, `N_cap=1024`, `N_live=1000`. The authorization
statistic is the minimum of five.

| Corpus | Minimum c-steps/s | Median c-steps/s | Fraction of floor | Shortfall | Peak allocation |
|---|---:|---:|---:|---:|---:|
{benchmark_row(h0)}
{benchmark_row(h1)}
{benchmark_row(h2)}

H1/H0 minimum ratio is `{h1['minimum']/h0['minimum']:.3f}` and H2/H0 is
`{h2['minimum']/h0['minimum']:.3f}`. The fixed-slot kernel therefore does not show a harmful heterogeneous
masking tax in these cells. The all-16-slot occupancy cell has minimum `{full['minimum']:,.0f}` c-steps/s,
`{full['minimum']/h2['minimum']:.3f}` of H2, despite analytic padding ratios of 2.485x for H1 and 3.821x for
H2. That is evidence against treating an arena rewrite as the first remedy.

The H1 churn cell minimum is `{churn['minimum']:,.0f}` and median `{churn['median']:,.0f}` c-steps/s. Its
median/no-churn ratio is `{churn['median']/h1['median']:.3f}`; the lower minimum records the scheduled churn
event rather than hiding it. The CPU eager B=1024 probe reaches minimum `{cpu['minimum']:,.0f}` c-steps/s.
CUDA is clearly faster at B=1024, but the smallest crossover `B*` remains unresolved because the decision was
already terminal and the complete crossover ladder was not used to manufacture a more favorable conclusion.

The r1 `torch.compile` cell status is `{compile_failure['status']}` at its frozen
`{compile_failure['timeout_seconds']}`-second timeout. r2 explicit CUDA graph is therefore the best completed
rung, not evidence that compilation succeeded.

## Long-horizon mechanics

| Dtype | Maximum prefix ratio by body | Threshold | Monotone bias | Regularizations | Result |
|---|---|---:|---|---:|---|
| f32 | `{drift32['maximum_after_burnin']}` | {drift32['threshold']:.0e} | `{drift32['monotone_bias']}` | {drift32['regularization_count']} | PASS |
| f64 | `{drift64['maximum_after_burnin']}` | {drift64['threshold']:.0e} | `{drift64['monotone_bias']}` | {drift64['regularization_count']} | PASS |

The `.npz` artifacts retain every cumulative residual, cumulative scale, and normalized prefix. Signed
reactive wake flux remains separate from nonnegative dissipated wake power; the former closes the force-law
identity and the latter reproduces donor mechanical-work accounting.

## Profiler attribution

The H1 eager profile records severe fine-grained operator fragmentation over 20 steps:

| Operator | Calls | Self CUDA time (us) |
|---|---:|---:|
{profile_rows}

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

The next authorized work is not S1 and not a flattened-layout rewrite. It is a reversible locomotion-kernel
fusion experiment with this exact corpus and thresholds held fixed. A future report may change the NO-GO only
after both H1 and H2 rerun non-OOM and clear the original floor; otherwise the architectural throughput thesis
must be revised.
"""
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(report, encoding="utf-8")


if __name__ == "__main__":
    main()
