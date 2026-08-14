# Reduced-motion held-out comparison

**Date:** 2026-08-09

**Branch:** `restart/original-baseline`

**Measured commit:** `5ada85c`

**Status:** Slice 1.4 exploratory decision; reject cycle means as permanent
terminal targets and select the plan's full-physics/multi-rate fallback for
evaluation, with usefulness not yet demonstrated

## Question

Does the Slice 1.3 response preserve selection-relevant signs, broad scale,
failure cases, and measured cost outside its calibration knots and time window?

This is exploratory model checking, not a confirmatory threshold test. Exact
inability and sign changes are reported directly. Ratios are descriptive; no
universal speed or steerability gate is introduced.

## Method

All runs used CPU float64, still water, and the canonical 1/120-second full
physics. The reduced response was calibrated from effort 0.5 and 1.0, turn
fractions +0.25 and -0.25, one warm-up cycle, and two measured cycles.

Two independent attacks were then run from fresh full-physics state:

1. **Control interpolation:** effort 0.75 and turn fractions +0.125 and -0.125
   on `mirrored`, `wide-16`, `random-05`, and `random-26`. These are neither
   effort nor turn calibration knots.
2. **Horizon stability:** full effort and turn fractions +0.25 and -0.25 on
   `root-only`, `swimmer`, `deep-cap`, and `random-14`, using later windows of
   3 warm-up/4 measured cycles, 7/2 cycles, and, for the three nonzero special
   cases, 15/2 cycles.

Predicted surge and yaw are the reduced model's requested terminal responses.
Predicted cost is its interpolated empirical cost power. Observed values are
independent `probe_motion` results; observed power is integrated turning or
straight work divided by the exact measurement duration.

The comparison used the public `calibrate_reduced_motion`, `probe_motion`, and
`step_reduced_motion` interfaces. No production source or fixture was changed
while gathering results.

## Held-out control interpolation

The table shows positive-turn results. Negative-turn results had the same surge;
both yaw commands are summarized in the final column.

| Body | Surge predicted / observed m/s | +yaw predicted / observed rad/s | Turning cost predicted / observed W | Signed yaw result |
|:---|---:|---:|---:|:---|
| mirrored | 1.34819 / 1.28485 | 0.110741 / 0.159446 | 2698.50 / 1514.59 | both commands agree |
| wide-16 | 0.024055 / 0.022629 | 0.001049 / 0.001027 | 3.00797 / 2.26567 | both commands agree |
| random-05 | 0.003911 / 0.003535 | 0.00000159 / -0.00000972 | 0.026284 / 0.014083 | weak +command flips; -command agrees |
| random-26 | 0.005149 / 0.004318 | 0.000772 / 0.001432 | 0.401861 / 0.307832 | both commands agree |

For the negative commands, predicted/observed yaw was respectively
`-0.118567/-0.167335`, `-0.000713/-0.000822`,
`+0.0000364/+0.0000166`, and `-0.002466/-0.002561`. Negative-turn cost
prediction/observation ratios were between 1.46 and 1.78.

The interpolation preserves broad surge scale: prediction was 1.05x, 1.06x,
1.11x, and 1.19x the observed result. Cost was consistently high but remained
between 1.31x and 1.87x observed. Yaw magnitude was less regular, and the very
weak `random-05` positive response crossed zero. This alone would support a
bounded correction or a declared weak-response domain; it is not the decisive
failure.

## Held-out horizon stability

### Surge

| Body | Calibration 1/2 m/s | Later 3/4 m/s | Late 7/2 m/s | Late 15/2 m/s |
|:---|---:|---:|---:|---:|
| root-only | 0 | 0 | 0 | not rerun |
| swimmer | 3.843586 | 6.731104 | 7.189960 | 7.208583 |
| deep-cap | 1.137737 | 1.976375 | 2.161865 | 1.831625 |
| random-14 | -0.0001004 | -0.0000310 | +0.0000316 | +0.0001587 |

The exact structural zero remains exact. The strong movers' early means are not
their later scale, and `random-14` changes from paid backward motion to weak
forward motion.

### Paired yaw response

| Body | Calibration 1/2 (+ / -) | Later 3/4 (+ / -) | Late 7/2 (+ / -) | Late 15/2 (+ / -) |
|:---|:---|:---|:---|:---|
| root-only | 0 / 0 | 0 / 0 | 0 / 0 | not rerun |
| swimmer | +0.2282 / -0.2567 | -0.1384 / +0.1506 | -0.0093 / -0.0004 | -0.0611 / +0.0614 |
| deep-cap | +0.0034 / +0.0109 | +0.1400 / -0.0524 | +0.2790 / -0.0680 | +0.3381 / +0.0539 |
| random-14 | -0.000039 / +0.000002 | -0.000044 / +0.000009 | -0.000045 / +0.000014 | -0.000041 / +0.000017 |

The swimmer reverses both command-response signs between the calibration and
3/4 windows. `deep-cap` changes from same-signed response to opposite-signed and
back again. A scalar first-order terminal target has one fixed sign from rest;
it cannot reproduce these changes by adjusting a gain, timescale, or interpolation
coefficient.

### Straight empirical cost power

| Body | Calibration 1/2 W | Later 3/4 W | Late 7/2 W | Late 15/2 W |
|:---|---:|---:|---:|---:|
| root-only | 0 | 0 | 0 | not rerun |
| swimmer | 12682 | 38000 | 44046 | 44353 |
| deep-cap | 1238 | 2146 | 2491 | 2442 |
| random-14 | 97.06 | 96.79 | 97.05 | 97.05 |

Early calibration also understates the strong movers' later mechanical cost,
while the weak backward mover's substantial cost is stable.

## Decision

The reduced interpolation is promising over nearby held-out controls, but its
cycle means are not honest permanent terminal responses. The failure changes
direction, steering authority, and cost for already-maintained phenotypes, so it
is selection-relevant rather than cosmetic trajectory error.

No bounded correction is spent. Converting a fixed scalar terminal target into a
state-dependent force/torque response surface or a multi-state transient model is
a second scientific implementation, not a gain or interpolation correction. The
active plan says to stop after one implementation when prospects are not improving
and use the simpler fallback.

Slice 1.5 must therefore use the canonical full mechanics at a measured lower or
multi-rate cadence. It must preserve one `LiveState`, actual full-physics signs and
work, and exact structural inability. Before runner integration it must state and
test how mechanics time maps to ecological time; it may not silently extrapolate
one early burst across an ecological interval. The committed reduced model remains
an unused, reversible experiment and a comparison instrument.

This selects a fallback direction, not a successful fallback implementation.
Slice 1.5 must reject or further simplify it if no explicit mapping both covers
authoritative elapsed time and materially improves complete-world throughput.

## Limits

- These are a deliberately small set of diagnostic forms, not a population-level
  quality claim.
- The late windows establish nonstationarity over the intended approximation
  boundary; they do not identify a universal settling horizon.
- CUDA was unavailable inside the managed sandbox. This comparison is scientific
  CPU float64 evidence; complete-world CUDA performance remains a separate runner
  question.
