# GPU-resident runtime rearchitecture

**Date:** 2026-08-11
**Status:** owner-directed implementation plan
**Branch:** `restart/original-baseline`

## Purpose

The complete living loop is currently shaped like a small CPU application even
when its tensors are placed on CUDA. Python schedules every ecological tick,
iterates live creatures for feeding, maintenance, births, deaths, and mutation,
reads device scalars for decisions and reporting, and may replay a complete
mechanics interval several times to find affordable effort. That structure cannot
make efficient use of the GPU.

The replacement runtime will keep authoritative simulation state on the device,
advance fixed-shape batches for many ecological ticks without host decisions, and
copy only bounded observations and event batches to the host. The existing
`HeadlessWorld` and `HeadlessRunner` remain the scientific reference while the
replacement is proved. This plan changes the implementation architecture, not the
scientific kernel in the accepted lean living-loop plan.

## Design rules

1. **Tensor state is data, not an object graph.** Mutable authoritative state lives
   in small dataclasses grouped by domain. State containers do not perform biology.
2. **Domain kernels are batched functions.** Motion, feeding, metabolism, field
   ecology, birth, death, and mutation accept tensors/configuration and return
   updated tensors plus named transaction ledgers.
3. **The scheduler only schedules.** It selects declared cadences, invokes kernels,
   swaps buffers, and exposes snapshots. It does not calculate births, settle an
   organism, allocate IDs, or interpret a CUDA scalar.
4. **Static capacity, dynamic masks.** Population, genotype, event, and field shapes
   remain fixed. Birth and death change masks and slot contents, never tensor shape.
5. **No host synchronization in an ordinary chunk.** No `.item()`, `.tolist()`,
   Python loop over creatures, or tensor-valued Python branch is permitted inside
   the device step. Aggregate failure flags are copied only at chunk boundaries.
6. **Exact books stay exact.** Tracked matter and chemical reserve use `int64`
   transactions on the device. Failed closure arrests at the next chunk boundary;
   no correction branch repairs state.
7. **Observation is downstream.** Unity and console reporting receive a staged,
   read-only snapshot at their own cadence. A slow viewer drops stale snapshots;
   it never slows or advances the simulation.
8. **Approximation remains physical and uniform.** The canonical 120 Hz mechanics
   solver becomes an oracle and bounded transient lane. Ordinary ecological motion
   uses one form-derived, state-dependent response model for all organisms.

## Cohesive module layout

```text
src/sirrobin/
  runtime/
    config.py          immutable cadences, capacities, and chunk policy
    state.py           top-level tensor-state composition only
    step.py            thin ordering of domain kernels
    session.py         compiled chunk ownership and snapshot handoff
    events.py          fixed-capacity device event records
  organisms/
    state.py           material, age, live lineage, and allocator tensors
    behavior.py        batched sensing -> bounded intent
    feeding.py         sampled demand and exact shared-stock allocation
    metabolism.py      maintenance and locomotion settlement
    lifecycle.py       death return and paid slot assignment
    mutation.py        deterministic counter-based mutation at birth
  physics/
    ecological_motion.py  device ecological-motion contract and integrator
    phase_response.py     state-dependent phase/harmonic force response
    live_step.py           retained canonical reference solver
  observe/
    world_snapshot.py  device-to-host staging and immutable observations
    unity_stream.py    protocol formatting and latest-snapshot transport
```

These are ownership boundaries, not a requirement to create one class per file.
Files stay small because each contains one state contract or one family of closely
related kernels. Validation of configuration and external inputs happens before a
compiled chunk, rather than being repeated inside every kernel.

## State ownership

- `RuntimeState` groups population, genotype/developed cache, motion, material,
  environment, clocks, allocator, deterministic-random counter, and pending events.
  It has no `advance()` method.
- `RuntimeSession` owns the current `RuntimeState`, a compiled chunk function, and
  snapshot buffers. It does not own biological rules.
- Per-slot live lineage stores current stable ID, parent ID, generation, and birth
  time. Historical lineage is emitted once as a birth event and archived outside
  the causal state; it is not a Python dictionary consulted by the hot loop.
- Developed body and motion response are rebuildable caches written into the child
  slot at a paid birth. Mutation never writes a response directly.

## Device execution

One compiled chunk advances several ecological ticks:

```text
sense and request intent
  -> choose affordable effort on device
  -> advance state-dependent ecological motion once
  -> sample and allocate shared food in one batch
  -> settle locomotion and maintenance in one batch
  -> produce death mask and return material
  -> assign paid births to free slots deterministically
  -> mutate and develop assigned children in place
  -> advance field ecology and clocks
  -> append compact event tensors
  -> compare exact aggregate books
```

The chunk returns state plus small per-world status tensors. It does not return one
Python report object per creature or per tick. Detailed reports are reconstructed
from event tensors only when an observer requests them.

For a small interactive population, phase samples and (when useful) independent
worlds supply the second batch dimension. The intended scientific operating point
remains one large world with thousands of fixed slots; a 64-slot Unity demo is not
allowed to define the architecture.

## Motion and energy

The rejected terminal-target response is not revived. The next response evaluates
canonical form-derived force, torque, work, and dissipation over a small phase basis
while retaining current surge, lateral velocity, yaw momentum, gait phase, effort,
turn request, and developed morphology. A compact harmonic state may retain the
phase-state correlation that simple phase averaging erased.

Affordability is part of the same device solve. The response evaluates bounded
effort knots (including zero and requested effort), chooses the largest affordable
effort without a host branch, and advances exactly once. The selected actuator work
is debited from chemical reserve; passive motion is never charged as muscle work.

Canonical mechanics remains the source of comparison across straight and paired
turns, perturbed velocity/yaw momentum, founder-adjacent mutations, weak and
backward movers, and transient versus settled horizons. Out-of-domain states use a
declared canonical fallback at chunk boundaries, not a silent clamp to a favorable
response.

## Migration sequence

1. **Preserve the reference.** Finish and checkpoint the currently runnable
   lifecycle candidate. Do not mix broad file moves into that checkpoint.
2. **Prove the motion kernel.** Implement the batched phase/state response and
   affordable-effort solve behind a new tensor contract. Benchmark 64, 128, and
   5,000 slots on CPU/CUDA; do not connect it to selection until held-out motion and
   work evidence is adequate.
3. **Build the device transaction slice.** Add tensor population/material state,
   batched maintenance, death return, paid birth allocation, and deterministic
   mutation. Keep matter closure exact over birth/death churn.
4. **Tensorize spatial feeding.** Replace Python intents and cell dictionaries with
   fixed eight-cell stencils, segmented deterministic allocation, and exact integer
   debits/credits.
5. **Compile chunks.** Compose motion, transactions, existing field ecology, clocks,
   status, and fixed event buffers. Remove all ordinary-tick host reads and measure
   complete-loop throughput.
6. **Switch operational surfaces.** Run the headless tool and Unity server from
   `RuntimeSession`. Unity receives only the newest staged snapshot. Keep a command
   to run the reference engine for comparisons.
7. **Retire superseded composition.** Once lifecycle trajectories, exact books, and
   the accepted motion domain are reproduced, remove duplicated old orchestration
   and move enduring domain code to the cohesive layout above.

## Evidence required to switch the living loop

Only four gates block the switch:

- exact tracked-matter closure and nonnegative reservoirs through feeding,
  maintenance, birth, death, and field ecology;
- named chemical input, actuator work, maintenance heat, assimilation heat, and
  dissipation closing within the existing dimensioned tolerance;
- held-out canonical comparisons showing that motion signs, inability, broad work,
  and turn asymmetry are not erased over the declared response domain; and
- complete-world throughput measured headlessly at 64, 128, and 5,000 fixed slots,
  including lifecycle and field work rather than an isolated kernel.

Tests should attack those claims. No registry, schema census, acceptance packet, or
new protocol version is introduced by this rearchitecture.

## Immediate stop conditions

- If the response cannot repair the demonstrated paired-turn and work errors with
  one compact phase-state extension, retain canonical transient bursts rather than
  growing an opaque surrogate.
- If exact deterministic shared-stock allocation cannot compile efficiently, move
  that transaction to fewer, larger device batches; do not fall back to a
  per-creature Python loop.
- If a proposed abstraction requires domain kernels to reach through a session or
  world object, reject it. Pass the state and configuration it actually consumes.
- If a code move does not improve ownership, testability, or the GPU execution
  boundary, defer it until the old composition is retired.

## Progress on 2026-08-11

- Steps 1-4 are implemented in a parallel runtime while the reference remains
  runnable.
- Step 5 has reached one complete interval and multi-interval session execution.
  The session compiles cohesive domain kernels rather than a monolithic world graph,
  keeps candidate state private until one aggregate boundary check, and reruns exact
  robust paths only when optimistic motion funding or food allocation is unresolved.
- The new layout now has data-only state/configuration modules, narrowly owned
  organism/field/physics kernels, a transaction-only interval composer, and a
  scheduler-only session. No replacement world or runner god class was introduced.
- Complete warm CUDA throughput at 5,000 slots improved from 0.559 to 6.58-6.62
  simulated-sec/sec, with exact matter books closed. This is evidence for the
  architecture but remains about 45 times below the 300 simulated-sec/sec target.
- Step 6 is partially implemented. Batched local-field behavior now supplies
  bounded effort, heading, and birth requests; immutable event/render snapshots are
  staged at separate cadences; and the Unity server defaults to `RuntimeSession` on
  CUDA while retaining `--runtime reference`. The live funding census showed that
  requested-only motion was rejected on almost every interval, so this operational
  backend now enters the existing exact five-option affordability solve directly;
  the general session retains speculation for regimes where it is useful. The
  standalone headless command remains to be switched or validated.
- Step 7 remains pending. The reference runner stays available until longer
  lifecycle/population comparisons and the declared motion-domain evidence support
  retiring duplicated orchestration.
- A first naive cadence probe rejected stage durations of 0.0625 seconds and above:
  they materially distorted actuator work and dissipation, with rapidly growing turn
  error. Further performance work must first preserve the missing within-window
  response, then test explicit multi-rate cadences against canonical results. It will
  not add another registry, schema, or acceptance framework.
- A later developmental-morphology slice now partitions exact structure matter by
  developed segment, proposes gradual shape/topology/attachment changes, develops and
  prices the candidate before funding, and commits only accepted births. Zero, one,
  or several mutation events are possible through fixed event slots. Finite exact-
  interval fast-forward is also wired into the device Unity backend.
- That slice is functionally green but fails the complete-world throughput gate.
  A comparable warm 5,000-slot run fell from 6.58-6.62 to 0.793 simulated s/s
  because candidate mutation, development, and pricing execute densely for every
  slot every 0.1 seconds even when no birth is requested. A temporary cohesive-graph
  probe made that subpath 1.96 times faster with equal sampled outputs, which is not
  enough to repair the 8.3-times complete-loop regression. Do not checkpoint the
  morphology tranche as performance-acceptable until candidate work is bounded or
  scheduled at a scientifically justified cadence.
- Live observation now consumes exact lifecycle-ledger outcomes for funded,
  unfunded, capacity-rejected, and accepted births; separate death causes; feeding
  transfer; generation; and parameter/topology mutation counts. A named Unity-only
  `evolution-demo` profile retains the prior calibration as `baseline` while
  expanding lifespan and per-locus mutation exposure fivefold. A 7,000-interval
  CUDA trial retained population 64 through 67 old-age replacements, reached
  generation four, and observed 50 mutated births including four topology events.
  Treat this as bounded demonstration evidence, not long-run ecological or rate
  validation.
