# Motion-probe starting domain

**Date:** 2026-08-09
**Branch:** `restart/original-baseline`
**Measured commit:** `94c1482`
**Status:** exploratory evidence for Milestone 1.2; no universal capability threshold

## Method

The pure full-physics probe was run on CPU in float64 and still water for 12 existing
developed bodies. The set deliberately includes the root-only structural zero, the
named swimmer/mirrored/deep/wide fixtures, low and high gait frequencies, and low and
high wave parameters:

`root-only`, `swimmer`, `mirrored`, `deep-cap`, `wide-16`, `random-00`, `random-04`,
`random-05`, `random-07`, `random-14`, `random-24`, and `random-26`.

Each body was measured at effort fractions 0.5 and 1.0. At each effort, independent
straight and turning trials used turn fractions +0.25 and -0.25, one nominal warm-up
cycle, and two nominal measured cycles. Step counts were rounded to the nearest frozen
1/120-second physics step. Every positive-turn result was rerun from fresh state and
compared for exact equality.

## Full-effort results

Work is the straight trial's integrated dissipated mechanical energy over two nominal
cycles. Its duration varies with gait frequency, so the raw work column is not a power
comparison.

| Body | Segments | Surge m/s | Yaw at +turn rad/s | Yaw at -turn rad/s | Straight work J | Measured interventions |
|:---|---:|---:|---:|---:|---:|---:|
| root-only | 1 | 0 | 0 | 0 | 0 | 480 |
| swimmer | 4 | 3.843586 | 0.228155 | -0.256704 | 12682.303 | 0 |
| mirrored | 6 | 2.113053 | 0.273363 | -0.298780 | 5211.998 | 0 |
| deep-cap | 10 | 1.137737 | 0.003385 | 0.010943 | 1909.272 | 0 |
| wide-16 | 16 | 0.037534 | 0.002873 | -0.002134 | 5.976 | 0 |
| random-00 | 4 | 0 | 0 | 0 | 0 | 0 |
| random-04 | 3 | 0 | 0 | 0 | 0 | 0 |
| random-05 | 6 | 0.006263 | 0.000034 | 0.000160 | 0.178 | 0 |
| random-07 | 5 | 0 | 0 | 0 | 0 | 0 |
| random-14 | 10 | -0.000100 | -0.000039 | 0.000002 | 121.320 | 0 |
| random-24 | 3 | 0 | 0 | 0 | 0 | 0 |
| random-26 | 3 | 0.008921 | 0.002876 | -0.006940 | 2.562 | 0 |

## What the evidence says

- All 24 body/effort cells were finite and exactly repeatable.
- All reported dissipated work values were nonnegative.
- Twenty cells produced opposite-signed yaw responses or exact zero for opposite
  commands. The four exceptions were `deep-cap` at full effort, `random-05` at both
  efforts, and `random-14` at half effort. These are outcomes, not failed tests.
- `deep-cap` changed from opposite-signed yaw at half effort to same-signed weak yaw at
  full effort. A reduced model cannot assume a universal monotone agility coefficient.
- Four articulated random bodies (`random-00`, `random-04`, `random-07`, `random-24`)
  were exact zero responders under the current hydrodynamic channels. The probe leaves
  that inability visible.
- `random-14` consumed substantial work while its full-effort mean surge was slightly
  negative. Intent and energy expenditure therefore do not imply useful displacement.
- The root-only body was the expected exact motion/work zero, but every measured step
  hit the yaw-inertia floor. That numerical limitation is reported rather than hidden.

## Consequence for the next slice

Milestone 1.3 must consume signed, form-derived response data uniformly. It may
interpolate bounded measured effort points, but it must preserve exact zeros, backward
motion, weak or same-signed turn response, and nonnegative measured cost. It must not
replace these results with free speed/agility parameters or a guaranteed-heading rule.
