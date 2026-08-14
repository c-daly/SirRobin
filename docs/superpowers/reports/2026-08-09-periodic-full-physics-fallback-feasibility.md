# Periodic full-physics fallback feasibility

**Date:** 2026-08-09

**Branch:** `restart/original-baseline`

**Base commit:** `71ebca3`

**Status:** Independently accepted as a scoped exact-clone swimmer bridge. The
fixture meets the operational target, while varied-phenotype throughput and the
Milestone 1 exit remain unauthorized.

**Uncommitted candidate manifest SHA-256:**
`473398230c46e70f32b0447279be4dcdd3055b2db80d43b987972a99187a13de`

The manifest is the SHA-256 of the ordered `sha256sum` output for
`periodic_motion.py`, `runner.py`, `world.py`, `run_world.py`, and
`test_periodic_motion.py`. The report is excluded so its own prose can record the
binding without becoming self-referential. The measurements below bind to that
manifest plus base commit `71ebca3`; they are not claimed to come from the base
commit alone.

## Candidate mapping

The Slice 1.4 decision prohibited silently extrapolating an early full-physics
burst across an ecological interval. The narrow candidate tested here would:

1. advance the canonical mechanics at 1/120 second through complete gait cycles;
2. compare cycle-boundary dynamic state, body-frame rigid transform, and
   integrated mechanical work;
3. after verified convergence, compose the repeated body-frame transform and
   work across whole remaining cycles; and
4. run the canonical mechanics for any remainder.

This covers every second of authoritative elapsed time. Constant yaw composes
as a circular rigid transform rather than a straight teleport. A cache
miss, changed control, or changed environment would return to full mechanics.
Forms that did not converge would remain on full mechanics.

## Probe

CPU float64, still water, zero turn bias, and the canonical 1/120-second live
step were used on four previously maintained forms: exact-zero `root-only`, the
canonical `swimmer`, difficult `deep-cap`, and weak paid mover `random-14`.

One cycle contained the nearest integer number of full steps for the developed
gait frequency: 60, 60, 92, and 75 steps respectively. At each boundary the
probe recorded:

- translation in the yaw frame at the start of the cycle;
- wrapped yaw change;
- ending velocity in the ending yaw frame;
- ending yaw momentum; and
- the time integral of named dissipated mechanical power.

The diagnostic change value is the largest componentwise
`abs(current - previous) / (1 + abs(previous))`. It is descriptive, not an
acceptance threshold. For weak values the added one makes it less sensitive,
so a small reported value cannot authorize erasing a weak response.

## Results

| Body | Cycle | Body-frame translation m | Yaw delta rad | End body velocity m/s | Yaw momentum | Work J | Max scaled one-cycle change |
|:---|---:|:---|---:|:---|---:|---:|---:|
| root-only | 2 | (0, 0) | 0 | (0, 0) | 0 | 0 | 0 |
| swimmer | 16 | (3.60487, -0.02474) | -0.000115 | (7.04066, -0.27258) | 36.7837 | 22176.8 | 1.47e-3 |
| swimmer | 32 | (3.60486, -0.02404) | 3.25e-8 | (7.04091, -0.27215) | 36.8201 | 22175.8 | 3.13e-6 |
| swimmer | 64 | (3.60486, -0.02404) | 1.12e-12 | (7.04091, -0.27215) | 36.8201 | 22175.8 | 6.20e-12 |
| deep-cap | 64 | (0.94304, 2.27904) | -0.222652 | (0.99655, 3.09346) | -88.9910 | 1353.25 | 6.50e-3 |
| deep-cap | 128 | (0.99499, 2.26615) | -0.222790 | (1.17953, 3.08850) | -84.3671 | 1339.92 | 1.68e-2 |
| deep-cap | 256 | (0.92544, 2.28923) | -0.223035 | (0.73978, 3.12894) | -61.5091 | 1354.08 | 3.00e-3 |
| random-14 | 64 | (0.000255, -0.000177) | -3.95e-5 | (0.000567, -0.000285) | -0.02697 | 60.7082 | 3.62e-5 |
| random-14 | 128 | (0.000272, -0.000296) | -5.97e-5 | (0.000598, -0.000474) | -0.03566 | 60.5078 | 2.32e-4 |
| random-14 | 256 | (0.000271, -0.000431) | -8.66e-5 | (0.000478, -0.000690) | -0.04156 | 60.5378 | 1.83e-4 |

The structural zero is immediately repeatable. The canonical swimmer approaches
a one-cycle orbit tightly by cycles 32–64. The two adversarial forms do not:
their change is nonmonotonic, and materially relevant components continue moving
at cycle 256. `random-14` is particularly dangerous to classify by a generic
tolerance because its real displacement is itself small.

The 256-cycle concurrent CPU probes took about 130 wall seconds for `deep-cap`
and 112 seconds for `random-14`. These are diagnostic costs, not complete-world
throughput evidence. They show that hard forms can remain expensive; they do not
establish the performance of a batched adaptive runner.

A separate nonconcurrent complete-batch probe advanced 128 exact-clone swimmers
through cycle 32 in 16.95 wall seconds and cycle 64 in 32.60 seconds. The
provisional 300 simulated-seconds-per-wall-second target permits 28.8 wall
seconds to cover one shipped 8,640-second ecological interval. Cycle 32 is inside
that budget but its 3.13e-6 descriptive change is not self-authorizing. Cycle 64
is tightly recurrent but 13.2% over before fast-forward and economy overhead.
This is near-boundary feasibility evidence; it requires explicit horizon-scaled
drift telemetry, a canonical comparison, and an actual composed-runner measurement.

## Decision

Reject one-cycle periodic fast-forward as a universal varied- or
mutating-phenotype throughput guarantee. `deep-cap` and `random-14` demonstrate
that no common short convergence horizon can be assumed.

Do not reject one uniform adaptive algorithm. A state- and error-based recurrence
test can be applied identically to every body, fast-forward only recurrent cycles,
and otherwise retain canonical 1/120 mechanics. Different wall-clock work need not
change simulated selection when approximation errors stay inside an accepted,
evidence-backed budget.
The actual risks are an inadequately scaled tolerance erasing weak responses and
nonconvergent forms removing any worst-case throughput bound.

Continue Slice 1.5 only for the immediate exact-clone swimmer world used by the
operational command and the first material lifecycle. The candidate must:

1. use content/state equivalence rather than a body name or amplitude branch;
2. validate recurrence with per-state relative error and explicit projected
   translation, yaw, dynamic-state, and work drift over skipped cycles;
3. restore the interval-start state and execute all canonical steps if validation
   fails;
4. invalidate on changed control or environment; and
5. demonstrate complete-runner throughput rather than isolated kernel speed.

The varied/mutating-population throughput claim remains unauthorized until its
own hard-form and complete-world gate. If the scoped candidate misses either its
error or throughput gate, reject it without starting another locomotion model and
return to the vertical living loop.

## Scoped implementation result

The runner candidate applies one content/state-equivalence and recurrence policy;
it contains no body ID, amplitude, or outcome branch. The periodic path is allowed
only for one-world, all-live exact physical clones with equal body-frame dynamic
state (including identical vertical velocity), equal gait time/control, equal
density, zero ambient flow, and a gait period that is an exact number of canonical
steps. Anything else executes every full-batch 1/120-second step. The canonical
fallback now also has a shape oracle for multiple worlds.

Canonical mechanics is the `HeadlessRunner` default. Periodic acceleration requires
an explicit policy opt-in. The operational command opts in only after constructing
the specifically measured all-live swimmer-clone fixture. Another morphology,
lifecycle composition, or caller requires its own evidence decision before opting
in; eligibility alone is not runtime authorization.

For an eligible clone group, one representative runs canonical mechanics. The
runner requires four consecutive recurrent cycles within `1e-9` relative state and
transform error, up to 64 detection cycles. Every pair in the consecutive window
must meet the complete policy, and authorization reports the componentwise worst
pair. The default policy limits projected translation drift to 0.1 m, yaw drift to
0.001 rad, relative dynamic-state drift to `1e-4`, absolute velocity drift to
0.001 m/s, yaw-momentum drift to 0.001 kg m2/s, and relative work drift to `1e-4`.

These are deliberately named **projections**, not mathematical enclosures. They
linearly extend the observed worst adjacent-cycle change across the unobserved
horizon. They are useful budget and telemetry guards, but neither four observations
nor linear extension proves future contraction. The implementation therefore has
stronger negative controls and a longer canonical comparison, but independent
review must decide whether this empirical approximation is sufficient for the
static exact-clone bridge. It must not be described as a certified error bound.

Failure leaves the authoritative world untouched and executes the complete fallback.
Success composes the planar rigid transform, rotates world velocity consistently,
advances the gait clock across every covered step, and runs one final complete-batch
cycle plus any remainder. Integrated work and projected drift are returned on the
ordinary world tick and accumulated across economy intervals by the command.

### Independent 128-second canonical oracle

A mature swimmer boundary state was advanced for 128 simulated seconds both through
the candidate and through all 15,360 canonical steps. The candidate evaluated 300
representative steps, projected 15,000 steps (250 cycles), and ran 60 final
full-batch steps.

| Quantity | Accepted drift projection | Observed candidate vs canonical error |
|:---|---:|---:|
| position | 7.58e-7 m | 2.91e-8 m |
| yaw | 8.28e-10 rad | 6.70e-11 rad |
| velocity | 3.85e-9 m/s | 1.55e-9 m/s |
| yaw momentum | 1.73e-7 kg m2/s | 3.71e-8 kg m2/s |
| work | 1.55e-9 relative | 1.49e-11 relative |
| gait clock | exact covered-step contract | 4.75e-11 s |

The maximum accepted adjacent-cycle recurrence error was 3.57e-10 relative. The
candidate took 1.896 wall seconds and the complete one-body oracle 75.503 seconds,
a 39.82x speedup. This retained oracle attacks 250 skipped cycles, substantially
more than the original 16-cycle comparison, but it does not validate every possible
future horizon or phenotype.

### Complete operational gate

Command:

```console
env TMPDIR=/tmp UV_CACHE_DIR=/tmp/uv-cache-original \
  UV_PROJECT_ENVIRONMENT=/tmp/venv-original-restart \
  uv run python tools/run_world.py --seconds 8640 \
  --economy-interval 8640 --bodies 128 --device cpu
```

Result:

- 8,640 authoritative simulated seconds in 19.6151 advance-wall seconds;
- **440.476 simulated seconds per wall second**, above the provisional 300 target;
- 1,036,800 mechanics steps covered: 3,780 representative, 1,032,960 periodic,
  and 60 final full-batch steps;
- projected drift totals of 0.008256 m translation, 1.33e-7 rad yaw,
  1.14e-5 relative state, 5.39e-7 m/s velocity, 3.06e-5 kg m2/s yaw momentum,
  and 2.81e-8 relative work;
- exact matter closure, finite state, and synchronized 8,640-second mechanics and
  ecological clocks; and
- 49.0408 GJ integrated named mechanical work across 128 clones.

This demonstrates the accepted bridge's performance and measured accuracy on CPU.
It does not complete Milestone 1: the plan's exit requires bodies that differ for
physical reasons. A varied population, mutation, nonzero ambient flow, or changing
within-interval control remains canonical and needs its own throughput evidence
before becoming an immediate consumer.

## Limits

- This probe tested zero turn bias in still water. Changed control, density, vertical
  state, and ambient flow have direct negative-control fallback tests.
- The authorization tests also attack a malformed consecutive window, projected
  dynamic-state overrun, failed-probe rollback, rotated/translated clone symmetry,
  a partial-cycle remainder, and multi-world fallback ledger shape.
- A longer or multi-cycle orbit may exist. It cannot supply a universal throughput
  bound, but a uniformly validated fast path may still help populations composed
  of forms that recur sooner.
- Current acceptance evidence is CPU evidence. CUDA is not required to clear the
  scoped operational target.

## Verification

- Focused periodic/composition/command gate: 28 passed.
- Complete suite: 130 passed, 5 CUDA-dependent skips, and the one known strict F12
  xfail in 95.22 seconds.
- Ruff: clean.
- Import boundaries: 70 files, 135 dependencies, 7 contracts kept.
- Lock check and `git diff --check`: clean.
