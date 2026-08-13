# SirRobin S2 fallible-steering gate correction plan

**Date:** 2026-07-13

**Status:** historical analysis; superseded as execution authority

**Superseded by:** `2026-07-13-sirrobin-living-loop-recovery-implementation-plan.md`

**Policy:** `../../2026-07-13-sirrobin-test-gate-policy.md`

> **Disposition:** This document correctly identifies exact heading settlement as a phenotype-quality test rather
> than a universal mechanics gate. Its proposed successor authorization sequence is no longer required. The
> recovery branch reuses the valid S2 mechanics and carries steering quality as telemetry while it builds the
> conserved living loop. The historical F12 result remains unchanged.

## 1. Decision being corrected

The original S2 F12 test asked the donor-shaped reactive controller to finish a `+90 degree` command within
`15 degrees` after ten seconds. The swimmer initially turned in the requested direction, remained finite and
below the emergency detector, then overshot and ended almost opposite the target. The frozen authority correctly
recorded that observation as NO-GO under its settlement claim.

Exact heading settlement is not, however, a SirRobin organism requirement. The prior project deliberately treated
turn overshoot as fallible behavior. SirRobin carries the same intent: morphology and physical inertia determine
maneuver quality, and ecological selection receives the consequences. Requiring robot-servo settlement was a
category error.

The subsequent control-effectiveness, local-derivative, cycle-held, caudal-Surface, and paired-Surface studies
remain valid evidence under their own frozen claims. Their exhaustive all-state coverage requirements do not
become S2 requirements, and their diagnostic code remains non-production.

## 2. Corrected S2 steering claim

S2 requires that a bounded reactive intent can cause real signed turning through the canonical physical body. It
does not require every morphology to steer, every command to be safe at a near-emergency starting state, or any
creature to converge exactly on a heading.

For a predeclared articulated reference body and an independently declared articulated holdout:

1. opposite turn requests produce baseline-subtracted angular impulse and yaw displacement with opposite signs;
2. each requested sign initially reduces its wrapped heading error over the declared observation window relative
   to the zero-turn trajectory;
3. the controller changes only bounded gait/appendage commands and never writes yaw, angular momentum, velocity,
   or position;
4. all force, moment, work, finite-state, action-bound, inertia, and solve-validity gates remain green; and
5. no trajectory reaches the unchanged `8 rad/s` emergency detector.

The observation window is two complete discrete gait cycles after the heading request. Response uses an
identically initialized zero-turn baseline. Mixed numerical tolerances establish that the response is nonzero;
there is no arbitrary maneuver-strength floor.

The reference and holdout bodies are selected and committed before execution from already-frozen donor-derived
fixtures. They must have articulated nonzero gait authority, but are not selected from the later actuator-study
rankings. Root-only remains an exact-zero negative control and is not required to turn.

## 3. Phenotype telemetry, not authorization

The corrected run records without gating:

- maximum and final heading error;
- overshoot count and peak overshoot;
- cumulative yaw travel and oscillation count;
- time and energy cost per signed heading change;
- mean forward progress projected onto the desired heading; and
- morphology-specific response magnitude.

Poor values are real phenotype evidence. They are available to S3 behavior and later selection but do not falsify
the locomotion kernel.

## 4. Historical dispositions

- The original F12 remains a valid failure of exact settlement under its old authority.
- The cascaded recovery controller remains rejected because it crossed the real emergency detector. It is not
  reactivated.
- The current bounded donor-shaped controller remains the production policy unless a future ecological need
  justifies a separately tested replacement.
- Diagnostic general-Surface and paired-Surface mechanics remain non-production; this correction does not install
  them or change the hash-bound live kernel.

## 5. Present use

The signed-response claim and telemetry list remain useful guidance for the living-loop mobility contract. They
may support small capability tests or diagnostics when that code is touched. They do not require a new S2 GO
decision, a 100,000-step drift run, profiler attribution, source-hash rebinding, or a two-body authorization corpus
before feeding, lifecycle, observation, or variation work can proceed.

Applicable conservation, finite-state, action-bound, work-accounting, and no-direct-state-write tests remain hard
under the active project policy. Settlement, overshoot, turn time, energetic quality, and morphology coverage are
reported according to the needs of the living world.
