# GPU runtime architecture spike

**Date:** 2026-08-11
**Branch:** `restart/original-baseline`
**Status:** device runtime exercised through Unity; startup, reconnect, and motion
diagnostics added; scientific comparison remains in progress

## Question

Can a cohesive device-oriented architecture make the RTX 5070 useful without
replacing morphology-derived mechanics with a hidden speed/fitness control?

## Result

Yes, enough to continue the redesign. The first candidate motion lane evaluates
canonical hydrodynamic force, torque, actuator power, and dissipation over the
organism's actual gait-time window. It retains current velocity, yaw, yaw momentum,
gait time, turn request, effort, and developed morphology, then performs four state
updates per 0.1-second ecological interval. Two midpoint phase samples are evaluated
as one batch at each stage.

This differs materially from the rejected terminal-target model and the first
full-cycle phase average:

- there is no stored or heritable speed/yaw capability;
- each stage re-evaluates the developed body's canonical hydrodynamic terms;
- actual gait phase is preserved rather than averaged away;
- current dynamic state affects every stage; and
- midpoint transport avoids the large trajectory bias from endpoint-only macro
  integration.

The implementation remains parallel to the current living runner. The candidate now
contains behavior, feeding, maintenance, birth, death, mutation, field ecology,
exact matter closure, named energy ledgers, and read-only observation staging. The
Unity server uses it by default while retaining an explicit reference-runner mode.
The protocol remains `sirrobin-observability/1`; this wiring does not change a
persistence or checkpoint schema.

## Motion fidelity

### Founder from rest, one simulated second

The two-sample/four-stage candidate was compared with all 120 canonical steps.

| Turn bias | Position error (m) | Velocity error (m/s) | Yaw error (rad) | Actuator-work ratio | Dissipation ratio |
|---:|---:|---:|---:|---:|---:|
| 0 | 0.00218 | 0.0565 | 0.000682 | 0.9680 | 0.9859 |
| +0.025 | 0.00416 | 0.0365 | 0.00313 | 0.9713 | 0.9859 |
| -0.025 | 0.00653 | 0.0586 | 0.00180 | 0.9607 | 0.9803 |

The paired turn signs were preserved. The earlier simple full-cycle phase average
had yaw errors up to 0.145 rad and asymmetric work/dissipation ratios down to about
0.85; actual-window sampling removes that failure on this probe.

### Deliberately varied bodies

Eight existing forms were compared at straight, positive-turn, and negative-turn
commands: `root-only`, `swimmer`, `mirrored`, `deep-cap`, `wide-16`, `random-00`,
`random-05`, and `random-15`.

- Structural-zero `root-only` and `random-00` remained exactly motionless with zero
  work and dissipation.
- Weak movers remained weak; they were not promoted to the swimmer's scale.
- Nonzero actuator-work ratios ranged from about 0.961 to 0.999.
- Nonzero dissipation ratios ranged from about 0.980 to 1.000.
- The largest position, velocity, and yaw errors in this set were about 0.0087 m,
  0.0586 m/s, and 0.0057 rad respectively.

This is exploratory coverage, not yet a declared universal response domain.

### Perturbed dynamic state

Three founder trials began with large forward/lateral velocity and positive or
negative yaw momentum instead of rest. One-second errors remained bounded in the
probe: position 0.0092-0.0217 m, velocity 0.0321-0.1241 m/s, yaw 0.00327-0.00763
rad, actuator-work ratios 0.963-0.976, and dissipation ratios 0.976-0.998.

These trials show that the candidate consumes state rather than replaying a fixed
orbit. They also show where the next comparison set should be stricter: high
lateral velocity and yaw momentum produce the largest errors.

## Warm motion throughput

The response stage was compiled as one full graph and replayed four times per
ecological interval. Timing excludes cold compilation and includes state evolution,
phase pose, hydrodynamic force/torque/power, constrained velocity solve, yaw update,
midpoint transport, work ledgers, and failure masks. It does not include feeding,
metabolism, lifecycle, field ecology, or observation.

| Slots | Compiled CPU sim s/s | Compiled RTX 5070 sim s/s | GPU / CPU |
|---:|---:|---:|---:|
| 64 | 15.7 | 26.5 | 1.68x |
| 128 | 8.13 | 23.6 | 2.90x |
| 5,000 | 0.231 | 11.8 | 51.1x |

At 5,000 slots CUDA processed about 591,000 creature-intervals/s. Based on the
earlier isolated canonical evidence of about 2.81 million creature-substeps/s at
10,000 bodies, 120 canonical substeps per simulated second correspond to roughly
4.7 simulated sec/sec. The new 5,000-slot response result is therefore about a
2.5x motion-throughput improvement while retaining much more transient information
than the rejected terminal response. This comparison is an inference across the two
measurements, not a same-command benchmark.

At 64 slots, the GPU is now faster than the CPU even before multiple worlds are
batched. That reverses the current complete runner's result, where CUDA is slower
because host synchronization and trajectory replay dominate.

## Compile granularity finding

Compiling all four stages as one monolithic graph generated a very large fused
Triton module, took several minutes, and exhausted the already crowded `/tmp`
filesystem before compilation completed. Compiling one scientifically cohesive
stage succeeded and replayed efficiently; the fixed four-stage sequence can later
be captured by a CUDA graph or by a small compiled chunk.

This is both a tooling and code-organization result: one giant simulation function
is not the desired GPU architecture. Domain kernels should be cohesive compiled
boundaries composed by a thin session/chunk layer.

The failed compile exposed old temporary environments and caches occupying most of
the 7.7 GB tmpfs. The unused `/tmp/venv-original-restart`,
`/tmp/sirrobin-original-venv`, and `/tmp/uv-cache-original` were removed; they are
reproducible and were not the active project environment. About 4.8 GB became free.

## Device lifecycle transaction

A separate fixed-shape lifecycle kernel now settles, per world and without host
decisions:

- death masks and exact structure/reserve returns;
- funded one-child-per-parent requests;
- deterministic parent ordering by stable ID;
- deterministic lowest-free-slot assignment, including same-step death slots;
- exact parent-to-child material transfer;
- monotonic stable-ID allocation with exhaustion reporting; and
- live parent, generation, and birth-time state.

Randomized four-world/64-slot tests independently census creature material before,
after, and returned by death. A full-graph compile test proves the transaction has no
device-to-host graph break.

Isolated warm replay is cheap on both devices:

| Slots | CPU transactions/s | RTX 5070 transactions/s |
|---:|---:|---:|
| 64 | 16,179 | 4,764 |
| 128 | 15,054 | 5,173 |
| 5,000 | 3,425 | 2,107 |

CPU is faster for this sorting-heavy transaction in isolation. The reason to keep
it tensorized is not an isolated GPU win; it is to keep state resident, eliminate
Python per-creature work, and compose the transaction with the larger GPU chunk.
The benchmark prevents a false claim that every tensor kernel is individually
faster on CUDA.

## Code ownership established

- `organisms/state.py` contains authoritative fixed-capacity identity, live lineage,
  and material state only.
- `organisms/lifecycle.py` contains one lifecycle transaction and its ledger only.
- `organisms/feeding.py`, `metabolism.py`, `mortality.py`, `mutation.py`, and
  `body_cache.py` each own one biological responsibility. Mutation changes genotype;
  the body cache derives the corresponding developed child rather than storing a
  second heritable capability.
- `fields/stencil.py` owns fixed-shape spatial sampling and conservative deposition.
- `physics/ecological_motion.py` contains the device effort-selection contract.
- `physics/phase_response.py` contains the candidate response stage/window and its
  work/intervention ledger.
- `runtime/state.py` and `runtime/config.py` only compose data contracts.
- `runtime/step.py` is a thin transaction graph: it routes explicit domain outputs
  but contains no mechanics, biology, allocation, or conservation equation.
- `runtime/session.py` owns compilation, retry policy, state publication, and the one
  bounded host status read at a chunk boundary.

No new god class was introduced. State containers own no behavior, biological
kernels do not reach through `HeadlessWorld`, and the candidate motion module does
not import the runner, organisms, Unity, or observation code.

## Complete device interval

The parallel interval now advances the following without a host-per-creature path:

1. morphology query and chemical work budget;
2. state- and phase-dependent motion;
3. producer field ecology;
4. deterministic shared-stock feeding;
5. assimilation, actuation settlement, maintenance, starvation, and old age;
6. exact death return and paid deterministic slot assignment;
7. deterministic birth mutation and incremental developed-body cache update;
8. spatial return deposition; and
9. exact aggregate matter closure plus named energy transactions.

`RuntimeSession` keeps an interval candidate private until it copies one aggregate
status tensor to the host. An invalid candidate is never published. The ordinary
successful path has no creature loop or tensor-scalar host decision.

Two speculative common paths preserve exact results while avoiding worst-case work:

- requested effort is evaluated first; insufficient chemical funding causes a
  rerun from the accepted state using all five affordable effort options; and
- shared food is first allocated in one deterministic round; reachable leftover
  stock causes a rerun from the accepted state using the full eight-round allocator.

Tests force both fallbacks and compare them with the robust path. These are not
outcome-tuned approximations: the fast result is accepted only when it is already the
same transaction the robust path would select.

That policy remains available for throughput regimes where requested effort is
usually fundable. It is no longer enabled blindly in the interactive living
configuration: the operational funding census below showed that almost every live
interval needs the exact effort-option solve, making motion speculation net harmful.

## Complete warm throughput

The measurements below include motion, field ecology, feeding, metabolism,
lifecycle, mutation/body-cache maintenance, and matter/energy books. They exclude
rendering, behavior/sensing, and cold compilation.

| Candidate | Slots | RTX 5070 simulated s/s |
|---|---:|---:|
| Eager device interval, before incremental body cache | 5,000 | 0.559 |
| Eager device interval, incremental body cache | 5,000 | 0.767 |
| All domain kernels compiled, robust motion and feeding | 5,000 | 3.225 |
| Compiled, requested-effort fast path | 5,000 | 5.144 |
| Compiled, requested-effort and one-round feeding fast paths | 5,000 | 6.58-6.62 |
| Compiled, requested-effort and one-round feeding fast paths | 128 | 7.72 |

The fresh complete 5,000-slot result is 328,903 creature-intervals/s and closes its
exact matter books. It is 11.8 times the first complete eager CUDA interval, and it
substantially reverses the old complete runner's CUDA regression. It is nevertheless
about 45 times short of the 300 simulated-sec/sec deep-time target. At 6.58
simulated-sec/sec, one simulated day takes about 3.65 wall-clock hours and a
365-day year about 55 wall-clock days.

Cold compilation remains substantial: a new capacity/configuration specialization
can take tens of seconds to minutes and the current Inductor cache is about 555 MB.
Warm throughput, not cold startup, is the number above.

## Operational wiring

`tools/serve_unity.py` now defaults to the cohesive runtime on CUDA and the explicit
`evolution-demo` observation profile described below. The preserved runner remains
available through `--runtime reference`, the prior short-lifecycle calibration is
available through `--profile baseline`, and the device runtime may be exercised on
CPU through `--device cpu`. Construction uses a one-way bootstrap adapter; runtime
kernels do not reach back through `HeadlessWorld`.

Each ecological interval now performs batched producer-field sensing and requests
bounded chemotactic effort, heading, and reproduction. The physical solver still
determines position, velocity, and yaw, while the later transactions alone decide
whether effort and birth are funded. Full pose, field, and population snapshots are
staged to the host every five intervals. Only compact lifecycle and energy event
fields are staged in the intervening intervals, so births and deaths are not lost
when render frames are coalesced. The existing Unity client drains incoming records
and retains only the newest render payload for each Unity update.

A sustained server-backend probe advanced 50 compiled CUDA intervals and formatted
ten Unity snapshots in 1.238 seconds after warm-up: 5.0 simulated seconds at 4.04
simulated-sec/sec with eight live organisms. That result includes behavior, compact
event staging, full observation staging at render cadence, and JSON-ready payload
construction, but excludes socket transmission and Unity rendering.

The sustained probe also exposed an accounting fault that a one-interval test had
missed: hydrodynamic power entering a passively moving zero-gait body was being
classified as organism-supplied actuator work. The device settlement now masks
positive and braking actuator channels whenever requested effort is zero, while
retaining the passive physical trajectory and dissipation. An adversarial synthetic
stage and 25-interval CPU/CUDA backend tests guard the correction.

## First cadence probe

The obvious next shortcut was tested before changing runtime semantics. One founder
was advanced for one second at straight, positive-turn, and negative-turn commands
and compared with the 120 Hz canonical solver.

| Response stage dt | Stage evaluations/sim s | Work ratio range | Dissipation ratio range | Largest yaw error |
|---:|---:|---:|---:|---:|
| 0.025 s (current) | 40 | 0.961-0.971 | 0.980-0.986 | 0.00313 rad |
| 0.0625 s | 16 | 0.909-0.931 | 0.930-0.957 | 0.00519 rad |
| 0.125 s | 8 | 0.801-0.853 | 0.822-0.831 | 0.0566 rad |
| 0.25 s | 4 | 1.469-1.528 | 1.150-1.170 | 0.182 rad |

The 0.0625-second stage is attractive computationally but already fails the current
work and dissipation evidence bounds. Coarser stages rapidly become qualitatively
wrong. No cadence was changed. A useful next approximation must preserve or learn
the missing within-window transient response; merely increasing `dt` would change
selection pressure.

## Live motion diagnosis

Unity observation first exposed passive-looking drift and tight milling. The report
was treated as an observation to test, not as a diagnosis to implement. Two physical
faults were confirmed and corrected in the canonical hydrodynamics used by both
lanes:

- reactive tail velocity is now relative to the body's center-of-mass translation,
  so passive sideslip cannot be counted as gait-generated thrust; and
- rigid-body broadside form drag now opposes lateral center-of-mass motion without
  double-charging joint gait, which remains accounted for by the reactive/wake
  channels.

The current canonical swimmer converges by gait cycle 32 to a body-frame velocity of
`[4.6309791238, -0.0742067209]` m/s and yaw momentum of
`5.5263560730 kg m2/s`. The prior pre-correction periodic fixture was
`[7.0409088616, -0.2721492597]` m/s and `36.8201062592 kg m2/s`. Periodic
acceleration was rebound to the newly measured state without changing any recurrence
or projected-error bound, and it still matches an independent complete canonical run.

The remaining broad circular paths are not a Unity interpolation illusion or
uncontrolled passive inertia. A 30-second streamed trajectory probe and a direct
device-state probe agree:

- three founders accumulated about 0.80-0.93 signed turns and path/displacement
  ratios of 1.86-2.61;
- the same founders spent 35-57% of sampled time gradient-seeking, while richer-food
  intervals used a low-effort forward cruise; and
- the other founders mostly used staggered straight search legs with long pauses and
  accumulated only about 0.09-0.37 signed turns.

The mechanism is the explicit food policy: at or above half the world's current peak
producer concentration, an organism cruises on its current travel heading at 0.1
requested effort; below that boundary it seeks the local gradient at 0.5 effort.
Crossing the boundary can therefore produce repeated outward cruise/inward seek
cycles around a food patch. The heading controller and measured yaw inertia are not
generating the turn in the absence of requests.

This is a crude area-restricted foraging policy, but circular or overshooting
foraging is a permitted fallible organism outcome in the current authority. It has
not been replaced with visually optimal homing. A future hunger, satiation, memory,
or sensory policy may consume the exposed state, but adding one solely to eliminate
circles would tune an outcome rather than repair a demonstrated physical fault.

## Startup and reconnect behavior

The first compiled CUDA interval used to begin after the socket listener opened.
Unity could connect successfully and then see no frames for roughly a minute while
Inductor compiled, making a healthy cold start indistinguishable from a frozen
simulation. The first prewarm repair compiled only the optimistic first interval. It
measured 37.2 seconds and delivered descriptor plus step-zero render record 0.0199
seconds after the listener opened, but a later trigger census proved that the exact
affordable-motion specialization is first needed around interval three and was still
lazy. The 37.2-second figure is therefore withdrawn as a complete cold-start figure.

`RuntimeSession.prewarm_autonomous()` now compiles the private motion and feeding
specializations configured for the session, without assigning any candidate to
authoritative state. The server opens its listener only after all configured paths
and CUDA work synchronize. A truly cold specialization may now make
startup take minutes, but that cost occurs before Unity is told the server is ready;
the first 50 live intervals no longer contain the prior second compilation pause.
Tests prove prewarm does not publish a tick or advance the backend's last interval,
and an adversarial call census proves both speculative and exact paths execute when
that policy is enabled.

An actual warm-cache server smoke measured 41.2 seconds from the prewarm message to
listener readiness with authoritative state still at step zero. A client then
received 300 uninterrupted render records over 30 simulated seconds, disconnected at
sequence 100, resumed with `after_sequence=100`, and finished at sequence 302 without
a conflict or compile stall. Reported sustained rate reached about 0.93 simulated-
sec/sec as paid mutated births grew the live population from 8 to 15. The server was
stopped after the probe; no background listener was left running.

Reconnect freezing had a separate cause. Record identity was derived from simulation
step, so a restarted client/server exchange could reuse an accepted ID for different
content and Unity correctly rejected the conflict. The server now owns one monotonic
stream cursor across client connections, raises it to any valid `after_sequence`,
and emits `render:sequence:<sequence>` identities. A forced-reconnect probe received
305 unique monotonic records over 30 seconds without a conflict.

## Complete small-world cadence

An initial ordered benchmark appeared to show a 50-interval headless chunk at 0.499
simulated-sec/sec, compact event staging at 0.903, and full Unity payload construction
at 0.846. That diagnosis was incomplete. A direct trigger census found that the
optimistic requested-effort path is underfunded on almost every interval from three
onward, while the feeding fast path is resolved. The first measurement in a fresh
process was paying compilation of the robust motion specialization; later modes were
reusing its cache.

The corrected in-process CUDA comparison precompiled both paths before timing:

| Intervals per host boundary | Host boundaries | Wall seconds for 5 simulated s | Simulated s/s |
|---:|---:|---:|---:|
| 1, first measured series | 50 | 6.526 | 0.766 |
| 2, warm repeats | 25 | 5.036-5.076 | 0.985-0.993 |
| 5, warm | 10 | 4.997 | 1.001 |
| 50, warm repeats | 1 | 4.696-4.965 | 1.007-1.065 |

There is no evidenced 2x retry-locality win. Warm window sizes differ only modestly;
host staging is not the dominant cost, and full render-payload formatting in the
earlier probe was about 6% slower than compact event staging. Physics and transaction
work remain the small-world bottleneck.

The comparison did expose a reproducibility edge: window sizes are internally
repeatable, but one-interval versus multi-interval execution differs by about
`3.9e-16` mol in `intake_carry_mol` after 50 intervals. Integer matter books and
population agree. The most likely source is different floating arithmetic between
separately compiled requested-only and five-option motion specializations when a
multi-interval retry replays already-funded intervals, but that causal attribution
is not yet proved. The Unity server always advances one interval per record, so its
operational trajectory is stable; the variation must be resolved or explicitly
bounded before arbitrary chunk size can be claimed scientifically transparent.

These numbers also must not be conflated with the 5,000-slot, two-phase-sample
throughput above. The Unity backend currently uses three phase samples and a small
capacity.

### Removing failed motion speculation from the live configuration

The trigger census found requested-only motion unresolved on intervals 3-7 and 9-50
of the 50-interval probe; feeding was resolved throughout. The live backend was then
run from cloned initial states with identical one-interval host boundaries:

| Motion execution policy | Wall seconds for 5 simulated s | Simulated s/s |
|---|---:|---:|
| Requested-only, then exact on underfunding | 7.303 | 0.685 |
| Exact five-option affordability directly | 2.853 | 1.752 |

Direct exact execution is 2.56 times faster in this measured live regime because it
does not calculate and discard an unaffordable requested trajectory. Both runs ended
with the same integer matter, population, and lifecycle state. Their only tensor
differences were `3.9e-16` mol of intake carry and `4.27e-5` J of maintenance carry,
both far below the next 100 J reserve quantum but retained here rather than rounded
away. The differences arise from the separate compiled floating paths; the direct
five-option solver is the existing authoritative affordability path, not a new
approximation.

`RuntimeUnityBackend` now disables requested-only motion speculation explicitly and
tests bind that operational choice. The general runtime retains both policies for
other population/resource regimes. The next performance change should batch useful
independent work or reduce the exact solver itself; it should not coarsen the physical
cadence already shown to distort work and dissipation.

The same 300-record socket/reconnect probe was repeated through the revised server.
Configured-path prewarm took 39.4 seconds from the warm cache. The exact same births
and mutation records occurred at the same simulated times, and founder trajectory
summaries were identical at printed precision except for a sub-micrometre-scale path
difference in one founder. Sustained streamed rate rose from about 0.93 to 1.36
simulated-sec/sec, a 46% end-to-end improvement after event staging, JSON formatting,
socket transmission, and population growth are included. The listener was again
stopped after the probe.

## Developmental morphology and birth transaction

The next living-loop tranche established the minimum accounting and transaction
boundaries required for hereditary morphology. `DevelopmentState` now partitions
each creature's existing exact `structure_q` among its developed segment slots.
This is not a second matter reservoir: the per-segment `int64` allocation must sum
exactly to the population authority, inactive slots contain zero, death clears the
allocation, and a paid newborn receives a fresh allocation from its developed body.

Morphological mutation now operates on the existing genotype authority rather than
writing mechanics or render geometry directly. The implemented gradual operators
cover independent segment-axis reshaping, attachment-position and attachment-angle
changes, small connected segment buds, and vestigial shrinkage followed by valid
leaf removal. Parametric joint amplitude, swim frequency, and swim-wave changes
remain available. Fixed event opportunities keep the tensor graph static while
allowing zero, one, or several actual mutations at a birth; mutation probability
scales with enabled mutable loci and topology events are weighted rarer than
parameter events.

The birth transaction is now:

```text
propose candidate genotype
  -> develop the complete candidate body
  -> price its structural matter from developed physical mass
  -> ask lifecycle settlement to fund and assign the birth
  -> commit only the accepted candidate into the child slot
```

An invalid, truncated, or unaffordable candidate is not silently replaced with a
parent-shaped clone. Conversely, physical poor quality is not a validity failure:
asymmetry, inability, useless buds, and otherwise fallible bodies remain legitimate
phenotypes for selection to test.

This is not yet a complete juvenile-development model. Current segment matter is
represented and exact, but feeding-driven growth, age- or environment-dependent
expression, metamorphosis, and resorption transactions remain future mechanisms.

## Finite fast-forward

The device Unity backend can now advance a requested finite duration using only
whole authoritative 0.1-second intervals. During fast-forward it suppresses
ordinary render snapshots, aggregates births, deaths, mutation-event count,
dissipation, and light input, accepts cancellation only between exact chunks, and
then stages one current snapshot before ordinary viewing resumes. Stream cadence
can separately coalesce several exact intervals into one Unity frame. Neither mode
scales biological or physical time constants.

The current command surface is startup-oriented (`--fast-forward-seconds` and
`--fast-forward-chunk-intervals`) rather than an interactive Unity control message.
It can exercise exact headless acceleration and later expose the result, but it does
not yet provide an in-viewer fast-forward button.

## Live lifecycle diagnostics and observation profile

The first complete Unity observation reached capacity but later went extinct at
675.9 simulated seconds. The event stream corrected the initial visual hypothesis:
only three founders starved, while the later decline was overwhelmingly deterministic
old age under the viewer's 60-100 second lifespan calibration. The baseline mutation
rate was 0.002 per mutable locus; most resulting events were small parameter steps,
while topology traits retained one-tenth the trait-selection weight of parameter
traits. That calibration was useful for lifecycle testing but did not provide a
useful visual window for turnover and hereditary change.

The observer now reports state and transaction evidence already owned by the device
runtime rather than recomputing lifecycle outcomes. Each render payload includes:

- current age, reserve, generation distribution, producer stock, free slots, and
  clone-funded-parent census;
- cumulative requested, accepted, unfunded, capacity-rejected, and ID-rejected
  births;
- starvation and old-age deaths as separate counts;
- mutated-birth, parameter-event, and topology-event counts; and
- feeding requests, exact producer debits, and exact reserve credits.

Five-second heartbeat events carry the most useful population, generation, funding,
mutation, and death counters. These are host-side observations only. They do not
decide a birth, mutation, death, feeding allocation, or validity result.

The named `evolution-demo` profile is intentionally separate from the biological
kernels and from the retained `baseline` profile. It expands the lifespan window
fivefold to 300-500 seconds and mutation exposure fivefold to 0.01 per mutable locus.
It does not change feeding, maintenance, birth material cost, mutation step sizes,
parameter/topology weighting, mechanics, field ecology, or any conservation check.
The descriptor publishes the selected profile and parameters, so a displayed run
cannot be mistaken for the short baseline calibration or for a universally accepted
biological rate.

A deterministic 700-second RTX 5070 trial then exercised 7,000 complete intervals
under the demo profile. Cold prewarm took 136.1 wall seconds; the full command took
495.1 wall seconds. No candidate chunk was rejected, so the per-interval economy,
feeding, return, matter, finite-motion, and mortality validity checks all remained
satisfied. The bounded outcome was:

| Observation | Result |
|---|---:|
| Initial / final population | 8 / 64 |
| Births / deaths | 126 / 70 |
| Starvation / old-age deaths | 3 / 67 |
| Mutated births / mutation events | 50 / 50 |
| Parameter / topology events | 46 / 4 |
| Final generation range | 1-4 |
| Final clone-funded parents | 42 of 64 |
| Final producer stock | 19,751,596 q in 3 occupied columns |
| Feeding requested / debited | 4,258,370 / 4,258,370 q |

Population remained at capacity while the 67 old-age deaths occurred and were
replaced. That establishes a usable bounded demonstration window and visible
hereditary exposure; it does not establish long-run ecological stability, mutation
rate realism, or evolutionary adequacy. Those remain scientific questions for
longer comparison trials rather than claims embedded in the Unity preset.

## Post-morphology throughput gate

The morphology transaction passes functional and compiled-CUDA tests, but it fails
the complete-world performance gate. Comparable warm, fully compiled CUDA runs on
the RTX 5070 used one untimed warmup interval followed by three measured complete
intervals:

| Slots | Simulated s/s | Creature-intervals/s | Books |
|---:|---:|---:|:---|
| 64 | 0.815 | 522 | closed |
| 128 | 0.755 | 967 | closed |
| 5,000 | 0.793 | 39,671 | closed |

A longer 50-interval check measured 0.960, 1.212, and 1.152 simulated s/s at
64, 128, and 5,000 slots respectively, also with closed books and no invalid
status. The different horizons do not change the decision: the prior comparable
5,000-slot result was 6.58-6.62 simulated s/s, so the current morphology-enabled
candidate is about 8.3 times slower at that operating point.

The cause is architectural rather than GPU absence. Candidate mutation,
development, and structural pricing currently execute over all fixed slots on
every 0.1-second ecological interval, even when the birth-request mask is empty.
A bounded temporary probe fused those three separately compiled stages into one
full graph. Across twenty 5,000-slot repetitions, the fused graph was 1.96 times
faster and produced equal mutation-event masks, developed segment masks, and exact
structure prices. That gain is useful but cannot recover the 8.3-times complete-loop
regression, so it was not integrated as a cosmetic optimization.

The morphology tranche is therefore functionally green but performance-blocked.
The next implementation decision must reduce how much candidate work is evaluated,
for example with a fixed-capacity packed candidate batch or a scientifically
justified reproduction/development cadence. It must preserve deterministic parent
selection, exact candidate pricing, invalid-candidate arrest, and static GPU-friendly
shapes. Merely fusing the current dense all-slot work or coarsening physical motion
is insufficient.

A fresh morphology-enabled server start completed its declared prewarm in 249.0
seconds and opened the listener with authoritative state still at step zero. That
prewarm is not complete, however: the first localhost client triggered another
specialization and did not deliver the requested frame set within a 10-second read
timeout. After that lazy build, a three-frame probe completed in 0.144 seconds and a
subsequent record reported protocol `sirrobin-observability/1`, monotonic identity
`render:sequence:10`, authoritative step 7, and time 0.7 seconds. The server is
functionally advancing, but the claim that all configured paths compile before
accepting a client is false for the current morphology-enabled composition and must
be repaired before startup is described as honest prewarm.

## Tradeoffs and rejected shortcuts

- Exact `int64` matter transactions, deterministic allocation, and developed
  morphology remain authoritative. They constrain fusion but prevent the fast lane
  from silently changing ecological outcomes.
- The phase response is an approximation of 120 Hz mechanics, but it evaluates
  canonical form-derived force, torque, work, and dissipation from actual state and
  gait phase. It has explicit fidelity evidence rather than a stored speed target.
- A fixed-capacity specialization wastes inactive slots and recompiles when capacity
  changes; in return, tensor shapes and allocation stay GPU-friendly.
- Sorting-heavy lifecycle work is slower on the GPU in isolation. Keeping it resident
  still avoids transfers and composes it with the full interval.
- `torch.compile(mode="reduce-overhead")` was tried and rejected for now. Replayed
  CUDA graphs can overwrite live outputs retained by subsequent state. Correct use
  requires explicit persistent double buffers and step-lifetime markers, which is
  more architecture than the measured bottleneck currently justifies.
- TensorFloat32 was also measured separately rather than enabled implicitly. It
  reached 6.74 simulated-sec/sec versus the 6.58-6.62 baseline, only about 2% higher
  and plausibly within run variation. Reduced matrix precision is not justified by
  that marginal result.

## Verification

- Evolution-demo CUDA trajectory: 7,000 exact intervals completed with 126 births,
  70 deaths, 50 mutated births, four topology events, generation four reached, and
  population 64 at 700 simulated seconds. No runtime validity or conservation gate
  rejected a chunk.
- Focused observer/profile CPU gate: five tests passed, covering explicit profile
  disclosure, exact reproduction-rejection diagnostics, mutation-category
  accounting, ordinary device observation, and finite fast-forward aggregation.
- Post-morphology complete CPU gate recovered from the interrupted run: 394 passed,
  with eight expected CUDA-unavailable skips and no failures.
- Focused post-restart CPU integration gate: 51 passed and two CUDA-only tests
  skipped. It covers segment allocation, morphology operators and birth ordering,
  complete living intervals, runtime snapshots, finite fast-forward, and the Unity
  server adapter.
- Direct post-restart RTX 5070 gate: three compiled CUDA tests passed in 850.62
  seconds. They prove a paid morphology-aware birth closes its books, the Unity
  backend advances the compiled runtime, and morphology proposal remains one full
  compiled graph.
- Whole-tree Ruff and `git diff --check` are clean after the recovered candidate.
- Full repository gate after live diagnosis and startup/reconnect repair: 371
  passed and 8 CUDA-only tests skipped in the sandboxed aggregate run. There are no
  failures and no remaining expected-failure marker for the heading controller.
- Direct RTX 5070 live-physics gate: both development/parity and CUDA-graph homing
  tests passed. The corresponding CPU homing contract also passes normally.
- The periodic accelerated swimmer and rotated/translated clone cases pass against
  complete canonical mechanics using the current settled state and their original
  strict error policy.
- Direct complete-runtime gate: one fully compiled CUDA interval committed a paid
  mutated birth, initialized the developed body and motion slot, and closed exact
  matter books.
- Direct server-backend gate: 25 compiled CUDA intervals completed without arrest,
  retained device-resident state, and produced a host render snapshot through the
  same adapter used by the Unity server.
- Cold-start, forced-reconnect, and 30-second streamed-trajectory probes exercised
  the actual listener/backend path and produced unique monotonic records.
- Whole-tree Ruff and `git diff --check` are clean.

## What remains after the server switch

1. Unity Play Mode was exercised interactively, but rendering still has no automated
   frame-level oracle. Visual observations in this report are therefore paired with
   server-side trajectory/state evidence rather than treated as sufficient alone.
2. Longer lifecycle/population comparisons are still needed before the reference
   composition can be retired. Current tests cover exact transactions and bounded
   runs, including the 700-second demo-profile trial, not evolutionary adequacy over
   deep time. The demo profile is observation calibration, not rate authority.
3. The current foraging policy is deliberately minimal and fallible. Population-
   level outcomes should be observed before adding hunger, satiation, sensory memory,
   or another behavioral state; none should be added merely to make paths look ideal.
4. Complete throughput remains far below the 300 simulated-sec/sec deep-time target.
   The next bounded performance target is useful independent batching or the exact
   solver itself, because retry-window localization showed no material warm win and
   coarsening mechanics has already failed the work/dissipation evidence.
5. Move the standalone headless command to `RuntimeSession`; the Unity server is
   switched, while `tools/run_world.py` still exercises the preserved runner.
6. Expand motion holdouts around founder-adjacent mutations, high lateral
   velocity/yaw momentum, longer horizons, and evolving mixed populations before the
   response drives selection.
7. Develop and test a compact within-window response/state model, then use it to
   justify multi-rate motion/field/biology cadences against the canonical runner.
   The first naive coarse-stage probe failed. Kernel fusion alone cannot plausibly
   close the remaining 45x gap; any additional speed must come from fewer
   scientifically justified updates, multiple worlds in one batch, or a lower
   target.
8. Compare longer lifecycle and population trajectories against the preserved
   reference path before retiring duplicated orchestration.
9. Replace dense all-slot morphology candidate evaluation with a bounded execution
   design that restores useful complete-world throughput before treating the
   morphology tranche as checkpoint-ready.

## Decision before follow-up

Continue the GPU-resident rearchitecture, including the cohesive module split. The
Unity server is switched to the device runtime by default, subject to fresh visual
validation, but the reference runner is not retired. The candidate proves that a
GPU-friendly complete transaction is possible and useful; it does not yet prove
deep-time throughput or selection fidelity.

## Bounded birth-work and behavior follow-up (2026-08-13)

Two speculative performance implementations were measured and rejected. A smaller
structural-development graph still ran on every interval and measured only 0.448
simulated s/s at 5,000 slots. A correction first appeared faster when it assumed
the current founder geometry was sufficient to bound future offspring prices, but
that assumption is not valid for the accepted scaled and recursive genotype domain.
The generalized conservative bound measured 1.235 simulated s/s against a 1.670
same-code exact-development control, 26.1% slower. Neither implementation remains
in the runtime. These negative results narrow the next attempt to a genuinely packed
candidate batch or a justified reproduction cadence; another dense all-slot probe
is not warranted.

Behavior was instrumented without changing the policy. Every live interval is now
accounted as seeking, searching, cruising, or idle, and
`tools/diagnose_foraging.py` relates those requests to per-identity effort, intake,
path length, displacement, and yaw. A 30-second compiled-CUDA observation with seed
20260809 covered 2,775 identity-intervals across 13 identities: 623 seeking, 1,083
searching, 321 cruising, and 748 idle, with exact books closed.

That diagnostic reproduces the owner-visible milling mechanism. Founder 1 switched
between 172 seeking and 93 cruising intervals, accumulated 7.09 rad absolute yaw,
and traveled 3.30 times its displacement. Founder 4 switched between 160 seeking
and 83 cruising intervals, accumulated 8.35 rad absolute yaw, and traveled 2.45
times its displacement. Several flat-field searchers instead produced path ratios
near 1.03-1.07. The evidence therefore continues to locate the tight food-patch
paths in the abrupt seek/cruise policy boundary, not in Unity interpolation alone.

## Local reserve-policy comparison (2026-08-13)

The bounded behavior tranche removed the world's peak producer concentration from
organism decisions. Food sufficiency now compares each organism's reserve against a
morphology-scaled structure target and requires producer matter in its local sample.
When a local horizontal gradient exists, both seeking and low-effort cruising request
that local heading. A remote producer spike leaves the local sample, sufficiency,
and requested heading unchanged. Independent negative controls also show that equal
reserves can produce different modes for different structure quantities and that a
reserve-sufficient organism does not cruise in a zero-producer field.

The same 30-second compiled-CUDA diagnostic with seed 20260809 covered 3,062
identity-intervals across 15 identities: 1,342 seeking, 1,083 searching, 60 cruising,
and 577 idle, with exact books closed. This is a comparative characterization, not
an optimization score. Tight paths remain possible: founder 1 accumulated 13.21 rad
absolute yaw and traveled 4.03 times its displacement; founder 4 accumulated 12.68
rad and traveled 3.44 times its displacement. The tranche therefore removes an
unphysical nonlocal input but does not claim to eliminate milling, improve survival,
or optimize population outcomes.

A crash-recovery rerun on 2026-08-14 reproduced those aggregate counts and the
per-founder path and yaw measurements exactly on CUDA. The cold Blackwell compile
was long but completed normally, and the rerun again closed the conservation books.

The established complete-world CUDA regression cell remains healthy. At 5,000
slots, one untimed warmup plus three measured intervals produced 0.654 simulated
s/s; the less noisy 50-interval form produced 1.836 simulated s/s, exact books
closed, and no invalid state. The previous 50-interval observation was 1.152
simulated s/s. This benchmark supplies explicit effort and does not execute the
autonomous policy, so the difference is run variation rather than evidence that the
policy improved throughput.

Verification after independent review completed with 47 focused behavior, Unity-
server, and diagnostic tests passing; focused Ruff and `git diff --check` clean; and
the full Python suite at 420 passed. The reviewer initially required independent
coverage of morphology scaling and local producer presence. Both controls were
added, re-reviewed, and accepted with no Critical or Important findings remaining.

Fresh crash-recovery validation on 2026-08-13 and 2026-08-14 separated the device
gates: the WSL CPU portion passed 413 tests with 7 GPU tests deselected, and the
seven GPU-marked tests then passed on the RTX 5070. Whole-tree Ruff also passed.
The configured import-boundary check kept 6 contracts and reported 1 existing
violation: `sirrobin.physics.ecological_motion` and
`sirrobin.physics.phase_response` import `sirrobin.fields.geometry`. Those modules
are outside this five-file policy change; the result is disclosed rather than
treated as a policy-tranche regression or silently described as green.

Fresh Unity Play Mode validation used the existing read-only viewer against this
worktree's prewarmed CUDA server. The editor compiled without errors, connected to
the observability stream, and accepted live records through at least 15 simulated
seconds; its logs recorded population growth from 8 to 10 and both corresponding
birth events. The server continued cleanly to 27 simulated seconds before the
intentional Play Mode stop disconnected the client. No runtime error appeared in
the Unity console; the only warning was Coplay's unrelated unsupported-toolbar API
notice. The temporary server and editor were stopped after the observation.

## Current decision

The next scientific performance decision remains a genuinely packed candidate
batch or a justified cadence/response model for remaining hot-loop cost. The local
reserve policy should now be observed at population scale before adding hunger,
satiation, sensory memory, or another state. Neither workstream may tune survival,
population, throughput, or visual path shape as a hidden success condition.
