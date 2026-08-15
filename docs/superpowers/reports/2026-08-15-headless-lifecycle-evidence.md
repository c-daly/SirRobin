# Headless birth-capable lifecycle evidence

Date: 2026-08-15

Base: `eea06fd` (`origin/main`)

Branch: `agent/headless-lifecycle-evidence`

## Scope

The standalone `RuntimeSession` command can now distinguish fixed slot capacity
from the initial live population, select either existing operational profile,
and declare the maximum number of authoritative intervals per host chunk. It also
exposes the existing exact dense-candidate control and reports candidate execution
by host chunk.

These are operational controls only. They do not change behavior, mortality,
mutation, feeding, reproduction, candidate pricing, or any conservation rule. The
default remains the baseline profile with every requested slot initially alive and
one host chunk covering the requested duration.

## Focused CPU capability result

The smallest birth-capable run used one founder in three slots, the existing
`evolution-demo` profile, and one interval per host chunk:

```text
PYTHONPATH=src:. /mnt/c/Users/cddal/SirRobin/.venv/bin/python tools/run_world.py \
  --seconds 0.2 --bodies 3 --live-bodies 1 \
  --profile evolution-demo --chunk-intervals 1 --device cpu
```

Two authoritative intervals accepted two paid births and filled all three slots.
The final whole-world total remained exactly `44,506,000` material quanta, matching
the initial total. This is a fixture capability result, not a general viability or
population target.

## Birth-rich CUDA comparison

The owner-relevant exploratory cell used the compiled CUDA runtime, 64 fixed slots,
8 founders, the existing `evolution-demo` profile, 30 simulated seconds, and one
authoritative interval per host chunk. The exact dense control changed only
candidate execution policy.

```text
PYTHONPATH=src:. /mnt/c/Users/cddal/SirRobin/.venv/bin/python tools/run_world.py \
  --seconds 30 --bodies 64 --live-bodies 8 \
  --profile evolution-demo --chunk-intervals 1 --device cuda

PYTHONPATH=src:. /mnt/c/Users/cddal/SirRobin/.venv/bin/python tools/run_world.py \
  --seconds 30 --bodies 64 --live-bodies 8 \
  --profile evolution-demo --chunk-intervals 1 \
  --dense-candidates --device cuda
```

| Candidate policy | Advance wall s | Simulated s/s | Dense chunks | Deferred chunks |
|---|---:|---:|---:|---:|
| optimistic, cold-first sample | 37.464 | 0.801 | 300 | 0 |
| exact dense control | 19.630 | 1.528 | 300 | 0 |
| optimistic, warm-cache repeat | 30.730 | 0.976 | 300 | 0 |

The dense control was 56.6% faster than the warm-cache optimistic repeat by
advance throughput. Setup and prewarm were timed separately and are excluded from
that comparison. The samples were ordered rather than randomized, so the exact
percentage is exploratory; the candidate-path census is not ambiguous. Every
optimistic host chunk detected work requiring the exact dense path, so every chunk
calculated and discarded its fast attempt before publishing the dense result.

All three CUDA runs reported the same bounded biological and accounting result:

- population grew from 8 to the 64-slot capacity through 56 paid births;
- 15,807 birth requests produced 13,760 unfunded and 1,991 capacity rejections;
- 26 births mutated, with 26 mutation events;
- no death occurred in the 30-second observation window;
- feeding debited 181,788 producer quanta and credited 90,883 reserve quanta;
- the final whole-world total exactly matched the initial `44,548,000` quanta;
- no invalid runtime state was published; and
- the printed lifecycle, behavior, field, energy, clock, and position values agreed.

This does not invalidate the previously measured request-free gain. It establishes
that the gain does not generalize to this birth-saturated autonomous workload and
that optimistic candidate replay materially harms its complete-loop throughput.

## Verification

- complete focused CLI file after the dense control was added: 14 passed;
- full non-GPU suite before the dense-control surface: 429 passed, 1 expected CUDA
  skip, and 7 GPU tests deselected;
- whole-tree Ruff: passed;
- `git diff --check`: passed;
- Import Linter kept 6 contracts and reproduced the existing forbidden
  `sirrobin.physics -> sirrobin.fields.geometry` imports in
  `ecological_motion.py` and `phase_response.py`; this slice changes neither
  module nor that boundary.

## Decision boundary

Retain both exact execution policies and the request-free benchmark evidence. Do
not claim a general evolutionary speedup from candidate deferral. For the existing
birth-saturated operational profile, select the dense path explicitly until a
bounded adaptive policy can use observed request history without changing accepted
biology or hiding workload-dependent retries.

The next performance slice should test that narrow adaptive decision against both
request-free and birth-rich complete-loop controls. Save/resume remains the next
functional milestone after the runtime execution policy is made honest for the
operational workload.
