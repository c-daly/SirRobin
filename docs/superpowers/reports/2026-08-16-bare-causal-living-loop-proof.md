# Bare causal living-loop proof

**Date:** 2026-08-16

**Branch:** `recovery/living-loop`

**Checked-out HEAD:** `59161e6ce763f340f129332a86b4a39605624e7b`

**State proved:** the preserved uncommitted living-loop implementation based on
that commit

**Result:** the bounded causal-mechanism acceptance run passed on CUDA; ecological
and biochemical calibration did not

## Claim

The reference world now executes this connected causal trace:

```text
finite local producer stock
    -> local producer value and gradient
    -> bounded inherited intent
    -> developed actuators and hydrodynamics
    -> paid physical movement
    -> exact local capture and depletion
    -> reserve accumulation
    -> fully paid asexual mutated birth
    -> genotype-derived developed child
    -> paid, momentum-conserving nonoverlapping release
    -> child's own sensing, movement, and feeding
```

This is a mechanism result, not a claim of calibrated organism chemistry,
generation time, ecosystem stability, or universal viability.

## Failed default and isolated blocker

The first declared long-horizon CUDA run used the then-current homogeneous
producer concentration of `0.333333e-6 mol m-3`. It was an exploratory diagnostic,
not a retained product command; only the general fail-closed proof harness remains
in source.

It completed all `12,000` authoritative `0.1 s` intervals. All eight founders
survived and every runtime validity check and independent chunk-boundary raw
matter census passed, but there were zero births and zero mutations. The largest
reserve rose from `2 q` to a peak of `440 q` at `502.4 s`, below the `1,100 q`
minimum complete-child funding requirement, then fell to `7 q` at `1,200 s`.
The run therefore failed its generational acceptance gate rather than treating
movement, feeding, or survival as a substitute.

The historical sparse-patch fixture had a local producer concentration of
`4e-6 mol m-3`; spreading its fixed total stock over the new metre grid had
diluted that local concentration twelvefold. A counterfactual transferred
existing dissolved nutrient to producer biomass to restore `4e-6 mol m-3`
homogeneously. It changed no total matter, organism state, controller, effort,
capture or assimilation efficiency, maintenance, birth price, mutation, or
physics. That run immediately crossed the reproduction boundary, isolating
local producer density as the no-birth blocker.

The reference world now uses that homogeneous `4e-6 mol m-3` producer initial
condition: `288,000,000 q` in a `72,000 m3` world. Total limiting nutrient remains
`22.25e-6 mol m-3`; the producer increase is an equal decrease in the dissolved
pool. The value is a deliberately viable, provenance-bound idealized initial
condition, not an empirical site calibration.

## Release defect found by the proof

The first reproduction counterfactual still failed. Float32 world-coordinate
rounding stored the child `1.43 micrometres` inside the declared conservative
support separation:

- required: `3.900099993 m`;
- stored: `3.900098562 m`.

The release placement now adds an explicit finite-precision coordinate bound
before storing the child position. The acceptance comparison and focused test
require the stored separation to meet the declared separation directly; no
assertion tolerance was widened. The final CUDA value was `3.900158405 m`,
`58.41 micrometres` beyond the required bound.

## Final headless CUDA evidence

Command:

```text
TMPDIR=/var/tmp \
TORCHINDUCTOR_CACHE_DIR=/var/tmp/torchinductor_living_loop_proof \
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=. \
.venv/bin/python tools/prove_living_loop.py \
  --device cuda --max-duration-s 120 --seed 20260809
```

The fail-closed `sirrobin.living-loop-proof.v1` verdict passed all twelve claims
and reported no missing claims. It stopped after the first mutated child's first
complete active interval:

- device: CUDA with compiled domains;
- authoritative interval: `0.1 s`;
- completed interval/time: step `51`, `5.1 s`;
- initial/final population: `8 / 15`;
- exact local producer debit: first at step `1`, `8 q` total;
- depletion causality: restoring only the preceding exact cell debit changed
  creature 1's next requested heading;
- paid movement: `83,794.39319229126 J` positive actuator work across the run;
- negative control: the same developed bodies with joint amplitudes zero had
  zero positive actuator work and zero displacement;
- mutated birth: parent ID `3`, child ID `13`, generation `1`, step `50`;
- mutation: `swim_wave_rad_per_depth[3]`, one committed event;
- child body: exactly redeveloped from the committed mutated genotype;
- construction transfer: `1,000 q` structure and `100 q` reserve;
- release: `1 q = 100 J` chemical debit, `2.7838266925564312 J` kinetic
  increase, `97.21617330744357 J` heat;
- release impulse residual: exactly `[0, 0, 0] N s`;
- child first active interval: local food sampled, effort accepted, `0.162352 J`
  positive actuator work, `0.00193448 m` physical displacement, and `1 q`
  local feeding debit;
- independent raw matter census, runtime matter books, and named energy
  boundaries: closed on every interval;
- every interval: finite and valid.

Warmup took `97.7 wall-s`; the measured 51-interval proof body took
`81.1 wall-s`. Throughput is not an acceptance success and remains a performance
limitation.

## Fresh Unity Play Mode replay

The ordinary server was started on CUDA and fast-forwarded through exactly 49
intervals:

```text
TMPDIR=/var/tmp \
TORCHINDUCTOR_CACHE_DIR=/var/tmp/torchinductor_living_loop_proof \
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=. \
.venv/bin/python tools/serve_unity.py \
  --runtime device --device cuda --profile causal \
  --fast-forward-seconds 4.9 --fast-forward-chunk-intervals 32 \
  --stream-every-steps 1
```

Prewarm completed before the listener opened. Fast-forward ended at step `49`,
time `4.9 s`, population `12`, with four paid births and no mutation. A newly
opened `SirRobinLiving` editor loaded `Assets/Scenes/Viewer.unity`, entered Play
Mode, connected from localhost with `after_sequence: 0`, and accepted the next
render records. Both server and Unity logs recorded:

```text
SirRobin [sim 5.0s] creature 3 reproduced: mutant child 13;
mutations=1; swim_wave_rad_per_depth[3] 1->0.965093
```

Unity then accepted the `5.1 s` child-active render and continued receiving live
frames. No Unity script exception was observed in the replay window. The viewer
was downstream only and did not decide or reconstruct the birth. Play Mode
continued through approximately `25 s`; the population reached the fixed
`64`-slot capacity at `14.9 s`, then the client disconnected when Play Mode
stopped. The temporary server was stopped afterward.

## Verification and limitations

Focused post-fix verification:

- four focused inventory, release, runtime-composition, and proof tests passed;
- whole-tree Ruff passed;
- `git diff --check` passed;
- the final compiled CUDA proof passed;
- the fresh Unity Play Mode replay passed.

The complete Python suite, run with CUDA hidden so it could not contend with the
live CUDA viewer, passed `425` tests with `8` expected CUDA-only skips and `14`
PyTorch deprecation warnings in `564.58 s`. The skipped production path was
exercised separately by the passing compiled-CUDA proof and CUDA Unity replay.
The working tree was deliberately not committed or cleaned.

The result does not resolve the pre-existing mismatch among the `187.45 kg`
developed founder, its `1,000 q` tracked structure, the dual-use reserve/material
currency, fixture energy densities, constant capture/assimilation efficiencies,
or maintenance scaling. The `4e-6 mol m-3` producer state also drives rapid
turnover: Unity observed population `29` by `10 s` and capacity `64` by `14.9 s`.
Those are explicit calibration and representation failures, not hidden pass
conditions. The bounded result proves that the retained causal mechanisms
compose; it does not establish that their current scales describe a real
ecosystem.
