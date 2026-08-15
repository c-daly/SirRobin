# Headless RuntimeSession switch

Date: 2026-08-14

Base: `8e3226c` (`origin/main`)

Branch: `agent/headless-runtime-session`

## Decision

`tools/run_world.py` now defaults to the cohesive `RuntimeSession` path. The
preserved `HeadlessRunner` remains available through `--runtime reference` for
direct comparison and for the existing one-creature feeding, maintenance, and
paid-birth probes. Those probe flags are rejected on the device path rather than
silently changing their meaning.

CPU runs use eager domain kernels by default so the command remains a practical
smoke test. CUDA runs compile the configured motion and domain kernels by default.
`--compile` and `--no-compile` provide an explicit device-path control.

The command reports bounded human-readable evidence, not a persistence schema:
authoritative interval/time state, exact field and organism matter totals,
lifecycle and behavior counts, feeding transfers, energy totals, a small position
sample, and separate setup, warmup, and measured-advance timing.

## Configuration ownership

The baseline and evolution-demo profiles, live behavior configuration, and the
reference-to-runtime configuration builder now live in
`sirrobin.runtime.profile`. The headless command and Unity backend therefore use
one declared operational configuration. Simulation authority remains in
`RuntimeSession` and the domain kernels; neither the command nor the viewer owns a
second biological implementation.

## Direct operational evidence

The default CPU command completed two 0.1-second intervals with two live slots:

```text
tools/run_world.py --seconds 0.2 --bodies 2 --device cpu
```

It published 0.2 simulated seconds, retained a population of two, closed the exact
whole-world books at 44,512,000 material quanta, and reported the autonomous
behavior and lifecycle summary.

The production-path CUDA smoke used the default compiled setting:

```text
tools/run_world.py --seconds 0.1 --bodies 2 --device cuda
```

It published one exact interval with closed books and no invalid state. Cold setup
took 17.893 seconds and compilation/prewarm took 245.581 seconds. The accepted
interval then took 0.0482 seconds, or 2.073 simulated seconds per wall second. This
is a startup and wiring check for a two-slot, one-interval fixture, not a complete-
world throughput result or a deep-time performance claim.

Both organisms requested birth in that CUDA interval and both requests were
rejected because every fixed-capacity slot was occupied. That is expected for this
all-live smoke fixture and is not evidence about lifecycle adequacy or birth-rich
performance.

## Verification

- `tests/core/test_run_world_tool.py`: 10 passed.
- Focused runtime and Unity compatibility tests: 38 passed, 2 CUDA tests skipped
  in the restricted sandbox; those CUDA tests were then exercised in the
  escalated GPU gate.
- Full non-GPU suite: 417 passed, 1 CUDA skip, 7 GPU tests deselected.
- GPU-marked suite on the RTX 5070: 7 passed, 418 deselected.
- Import Linter: all seven configured contracts kept.
- Whole-tree Ruff and `git diff --check`: clean.

## Remaining boundary

This closes the documented standalone-command routing gap, subject to review and
merge. It does not retire the reference runner. Longer lifecycle/population
comparisons, broader motion holdouts, and a justified cadence/response model remain
separate scientific work. The command exposes those future runs through the
authoritative runtime without turning any desired population, trajectory, or rate
into a hidden success condition.
