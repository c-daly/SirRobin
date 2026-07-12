# SirRobin locomotion Gate E revision — population-grounded authorization

**Status:** pre-registered before the 5,000/10,000-creature measurements
**Date:** 2026-07-12
**Authority:** owner direction after reviewing the original S0 result
**Scope:** replaces only the locomotion throughput threshold and population anchor; Gates A-D, the corpus,
physics, tolerances, oracle arms, and benchmark statistic remain unchanged

## 1. Why Gate E is revised

The original `9.0e7` creature-steps/s floor corresponds to 90,000 simulation ticks/s for a 1,000-creature
world: 750 times the 120 Hz physical timestep. It tested an ungrounded 750x acceleration requirement rather
than the stated near-term scientific objective.

The owner has now pinned the intended population at 5,000-10,000 creatures. This revision derives throughput
directly from that population and the unchanged 120 Hz timestep. The original result remains recorded as a
failure of the original gate; it is not rewritten as a pass.

## 2. Frozen authorizing cells

All cells use f32, the canonical fixed 17-slot representation, the frozen H1/H2 corpus tiled in committed
order, the best completed CUDA rung, five repetitions, 360 warmup steps, and 600 timed steps. The minimum of
five remains the authorization statistic.

| Capacity | Live population | Corpus | Hard real-time floor |
|---:|---:|---|---:|
| 5,120 | 5,000 | H1 | 600,000 creature-steps/s |
| 5,120 | 5,000 | H2 | 600,000 creature-steps/s |
| 10,240 | 10,000 | H1 | 1,200,000 creature-steps/s |
| 10,240 | 10,000 | H2 | 1,200,000 creature-steps/s |

Every cell must be non-OOM and have zero regularization. Dead lanes and unused segment slots remain in the
measured cost and never enter the numerator.

## 3. Performance classes

For live population `N`, acceleration `A`, and timestep rate 120 Hz:

```text
required creature-steps/s = N * 120 * A
```

The report classifies each population using:

| Class | 5,000 creatures | 10,000 creatures | Decision role |
|---|---:|---:|---|
| 1x real-time viability | 600,000 | 1,200,000 | hard authorization |
| 5x practical target | 3,000,000 | 6,000,000 | reported target, not required for viability |
| 10x stretch target | 6,000,000 | 12,000,000 | reported stretch target |

## 4. Revised decision classes

- **GO:** Gates A-D remain green and all four cells clear their 1x floor, non-OOM, with zero regularization.
- **CONDITIONAL GO:** 5,000-creature H1/H2 clear 1x, but either 10,000-creature cell does not. Work may proceed
  only under a 5,000-creature cap while a measured optimization tranche addresses 10,000.
- **NO-GO / REVISE:** either 5,000-creature H1/H2 cell fails 1x, any authorizing cell OOMs, regularization
  activates, or a correctness/hash gate regresses.

The 5x and 10x classes guide optimization priority. They cannot turn a 1x failure into a pass and are not
silently promoted to hard requirements after measurement.

## 5. Evidence handling

The revised report is separate from
`docs/superpowers/reports/2026-07-12-sirrobin-S0-decision-report.md`. It cites the same corpus and fixture
manifest hashes, records every repetition and peak allocation, and states both the original and revised
decisions so provenance remains intact.
