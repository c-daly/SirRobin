# S2 implementation review: corrections discovered before measurement

**Date:** 2026-07-12

**Status:** accepted implementation amendments to the S2 execution authority

**Authority:** `2026-07-12-sirrobin-S2-canonical-body-live-locomotion-implementation-plan.md`

## Position

The S2 architecture and physics contract remain sound. Implementation found two internal inconsistencies before
any throughput run. Both are corrected here rather than hidden in code or accommodated by a looser test.

## 1. H2 cannot contain a one-segment root-only body

The plan freezes H1/H2 at 2–16 expressed segments, but the initial literal H2 cycle included `root-only` twice.
That body is valuable as a degenerate A1/D0 diagnostic, but it is not an authorizing H2 body. With the plan's
point-mass yaw approximation, a one-segment body centred on its own COM also has zero yaw moment arm and correctly
activates the inertia guard. Allowing it into H2 would make the zero-intervention gate impossible by construction.

The two occurrences are replaced by the already-frozen multi-segment `random-07` and `random-09` fixtures. The
corpus hash changes before any performance measurement; donor and analytic expected values do not change. A
manifest test now executes the 2–16 lower bound instead of relying on prose.

## 2. A mirrored hinge axis is an axial vector

The development scan originally mirrored attachment and orientation but copied the hinge axis unchanged. Under
reflection across the FLU x-z plane, a hinge axis is an axial (pseudo-)vector and transforms as
`(hx,hy,hz) -> (-hx,hy,-hz)`. Copying it would make mirrored articulated parts move in the same rotational sense
when the reflected physical body requires the opposite sense.

Development now applies the axial-vector reflection to mirrored emissions. This does not alter the frozen donor
transform, mass, or tail fixtures; it closes a live-pose symmetry hole that those static values could not detect.
A half-period/opposite-command live fixture proves the resulting yaw trajectories are mirror images rather than
merely checking that two arbitrary commands happen to have different signs.

## Decision effect

Neither correction changes a threshold after measurement. They make existing definitions executable before Gate
E and preserve the plan's GO/CONDITIONAL/NO-GO semantics.

## 3. The frozen heading controller fails F12

The canonical mechanics pass open-loop sign symmetry, but the plan's donor-shaped closed-loop policy does not
home and settle from a 90-degree command. It initially turns toward the target, then the travel-direction reference
lags the yaw dynamics and the saturated gait bias drives a continuing circuit. At 1,200 f64 steps the swimmer ends
at `yaw=-1.4623 rad` with `turn_bias=+0.1338 rad/depth`: its wrapped target error is `3.0331 rad`, nearly opposite
the requested direction. A 2,400-step f32 CUDA-graph repetition also fails the frozen 15-degree settlement test.
Neither run activates the 8 rad/s emergency backstop.

Exploratory body-heading damping and reduced-authority policies changed the transient but did not settle. They are
not retained: adopting a newly tuned controller after seeing the authorization run would violate the plan's
frozen-policy discipline. The original controller contract remains implemented, the failing detector is retained
as a strict expected failure, and F12 makes S2 **NO-GO**. A successor plan must pre-register a genuinely stable
policy (likely a rate-damped or stopping-distance controller using derived `omega=L/I`) and re-run Gate C before
S3 begins.
