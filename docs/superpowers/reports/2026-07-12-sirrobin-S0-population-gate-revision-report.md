# SirRobin S0 population-grounded Gate E report

**Decision:** **GO**
**Date:** 2026-07-12
**Authority:** `docs/archive/plans/2026-07-12-sirrobin-locomotion-gate-E-revision.md`
**Corpus SHA-256:** `6721211db0aa42c3f63e4364f6f84c948a1f9157af6e02bb35325f09fab5e96e`
**Hardware:** NVIDIA GeForce RTX 5070, 12,227 MiB physical VRAM, PyTorch 2.13.0+cu130

## Decision

Gates A-D remain green from the original S0 report. All four pre-registered population cells clear their
real-time floors, complete without OOM, and record zero regularization. Under the revised decision classes,
this is an unconditional **GO**: both the 5,000- and 10,000-creature H1/H2 cells pass.

The original 1,000-creature/90M result remains a NO-GO under its original threshold. It is not relabelled or
discarded. This report answers the subsequently owner-pinned scientific objective of 5,000-10,000 creatures.

## Authorizing measurements

All cells use f32, fixed 17-slot bodies, explicit CUDA graph replay, five repetitions, 360 warmup steps, and
600 timed steps. The minimum of five is the authorization statistic.

| Corpus | Capacity/live | Minimum c-steps/s | Median c-steps/s | 1x floor | Headroom | Peak allocation |
|---|---:|---:|---:|---:|---:|---:|
| H1 | 5,120 / 5,000 | 1,678,627 | 1,705,199 | 600,000 | 2.80x | 73.9 MiB |
| H2 | 5,120 / 5,000 | 1,510,988 | 1,520,918 | 600,000 | 2.52x | 73.9 MiB |
| H1 | 10,240 / 10,000 | 2,441,302 | 2,449,740 | 1,200,000 | 2.03x | 83.9 MiB |
| H2 | 10,240 / 10,000 | 2,384,293 | 2,417,413 | 1,200,000 | 1.99x | 83.9 MiB |

Every result is hash-bound to the same corpus and stored under `runs/benchmark-*-cuda-r2.json`.

## Scaling interpretation

The 10,000-creature cells sustain approximately 244-249 simulation ticks/s against the 120 Hz physical
timestep. The architecture therefore supports the stated population at about 2x real time on the authorizing
GPU.

The reported optimization targets remain:

| Target | Required at 5,000 | Required at 10,000 | Current status |
|---|---:|---:|---|
| 1x viability | 600,000 | 1,200,000 | PASS |
| 5x practical | 3,000,000 | 6,000,000 | Not yet met |
| 10x stretch | 6,000,000 | 12,000,000 | Not yet met |

Failure to meet 5x or 10x does not weaken the GO decision because the pre-registered revision makes them
optimization targets, not authorization gates.

## Implementation finding

The first large-batch attempt exposed a setup bottleneck: `BodyBatch.from_rows` performed one host-to-device
write per segment. It now assembles tensors on CPU and transfers each completed tensor once. This changes only
fixture loading/setup, which remains outside timed windows; physics, row order, corpus values, and the hot-step
contract are unchanged. CPU and CUDA correctness tests remain green after the change.

## Consequence

S1 may proceed. Kernel fusion remains worthwhile for faster-than-real-time research throughput, but it is no
longer a prerequisite for the 5,000-10,000-creature architecture. The fixed-slot layout remains canonical;
the occupancy and memory evidence do not justify a flattened arena rewrite.
