# SirRobin lean living-loop restart plan

**Date:** 2026-08-09  
**Status:** accepted by the project owner on 2026-08-09; active execution plan  
**Branch:** `restart/original-baseline`  
**Baseline:** `3e007af` (`Close the composition seam's review findings`)

## 1. Goal

Restore a scientifically faithful simulation that is useful to run before adding
more biological breadth:

> Morphology-bearing organisms move imperfectly, obtain local food, pay explicit
> maintenance and movement costs, reproduce only by paying for a child, die when
> they cannot continue, return their matter to the environment, and can be watched
> while this happens.

The first success is a small, fast, headless living world. It is not a complete
scientific platform, a polished viewer, or proof of long-term evolution.

## 2. What changes from the previous process

The previous plan's causal principles remain useful. Its delivery process does
not. Work will proceed in small slices rather than multi-domain tranches.

One slice normally:

- adds or changes one causal mechanism or one operational seam;
- touches no more than three production modules unless the owner approves a
  larger scope before implementation;
- adds focused tests for the claim actually introduced;
- ends with the headless world running through one documented command;
- reports simulated-time throughput and any material regression;
- prepares one ordinary commit when the owner requests it, not an acceptance
  packet; and
- is reversible without dismantling later work.

There will be no test registry, content-addressed exit packet, schema census,
task-specific authority document, or independent review for each slice. Git
history, focused tests, milestone notes, and measured runs are sufficient for
ordinary engineering work.

## 3. The small scientific kernel

These are the few rules allowed to block a slice:

1. **Tracked matter closes exactly.** Limiting nutrient moves between explicit
   integer reservoirs. No correction branch repairs a failed balance.
2. **Energy has named boundaries.** Reserve carries a declared chemical-energy
   density. Light is an external input; maintenance and mechanical loss leave as
   heat. Energy is not a second synchronized mutable copy of matter.
3. **Form causes capability.** Motion, intake, maintenance, and child cost derive
   from developed morphology and environmental constants. There are no free
   speed, agility, fertility, fitness, or carrying-capacity controls.
4. **One authority per quantity.** Genotype, creature material, fields, clocks,
   IDs, and allocator state each have one mutable owner. Derived responses and
   observations are rebuildable or read-only.
5. **Intent cannot assign outcomes.** A creature may request bounded effort or
   heading. Physics or a physics-derived response determines displacement and
   cost. Poor motion and inability to turn are valid.
6. **Birth and death are paid transfers.** A child appears only after its full
   configured material is debited. Death returns remaining material exactly once.
7. **Selection is implicit.** Survival and paid reproduction are the only score.
   Extinction and uninteresting outcomes remain valid results.
8. **Headless time is authoritative.** A viewer reads snapshots and never advances
   or repairs the world.

Everything else is a model choice, telemetry, or a research question. It must not
be promoted into a universal gate without a concrete next consumer.

## 4. Lightweight verification and review

### Per slice

- Run the focused tests for the changed mechanism.
- Add one independent invariant test when the slice moves matter, energy, state
  authority, or time.
- Add one negative control when deleting or reversing the safeguard could
  otherwise leave the test green.
- Run a short headless smoke and report simulated seconds per wall second.
- Run the existing full Python suite before committing. The historical F12 xfail
  remains evidence and does not block the living loop.

### Per milestone

- Run the complete headless scenario with owner-visible output.
- Run Ruff and the import-boundary checks.
- Measure complete-world throughput and output volume, not only kernel speed.
- Ask the owner to accept, revise, or stop at the visible milestone result.

Independent review is requested only when a change introduces a
selection-relevant scientific approximation, changes matter/energy causality, or
creates a public persistence contract. The review examines that model and its live
implementation together. Fix Critical findings and concrete correctness defects;
advice that merely asks for more framework, documentation, or hypothetical
generality is not blocking.

### When formal preregistration is appropriate

Preregistration is reserved for a confirmatory scientific experiment whose
thresholds could be tuned after seeing an outcome. It is not required for normal
implementation, exploratory probes, debugging, performance work, or viewer work.

## 5. Execution sequence

Each numbered item below is a separate slice. A slice does not silently absorb
the next item.

### Milestone 0 — operational baseline

**0.1 Preserve and name the restart.** Prepare the resumption decision, this plan,
and the clean `3e007af` baseline identity as one commit when the owner requests it.
Do not merge `main`; its actuator work remains diagnostic donor history.

**0.2 Add one runnable world command.** Add a small tool that constructs the
existing composed world, advances a requested simulated duration, and prints
clocks, population, field totals, creature positions, exact closure, wall time,
and simulated-seconds-per-wall-second. It may begin with the existing cheap
fixture. This is an operational surface, not a stable external schema.

**0.3 Record the starting performance envelope.** Measure 8, 128, and—if
available—5,000 bodies on CPU/CUDA without changing the model. Record the command
and compact results in one milestone note. The provisional usefulness target for
the reduced ecological loop is at least 300 simulated seconds per wall second at
128 bodies on CPU; if evidence shows a better owner-relevant horizon, change the
product target explicitly rather than tuning a test silently.

Keep Python environments, caches, and run artifacts on WSL storage. Measure the
same bounded smoke once from `/mnt/c` and a WSL-native temporary checkout. Move the
active Python worktree only if that comparison shows a material improvement; do
not build a filesystem-mirroring system without evidence.

**Exit:** one command runs the actual baseline and makes its severe cadence cost
visible. No feature work is hidden behind unit tests.

### Milestone 1 — fast, form-derived ecological motion

**1.1 Probe one developed body.** Implement a pure standardized full-physics probe
that measures cycle-averaged surge, yaw response, and mechanical work at bounded
efforts. Return immutable data; do not integrate it into the world yet.

**1.2 Probe varied bodies and state the domain.** Run the same probe on roughly
8–16 deliberately different existing bodies, including poor movers. Confirm signs,
finite outputs, structural zero responses, and repeatability. Do not create a pass
threshold for universal steerability.

**1.3 Add the smallest reduced integrator.** Use the probe output in one uniform
surge/yaw integrator for every organism. Preserve inertia, drag, bounded effort,
overshoot, inability, and measured work. The response is rebuilt from developed
form and cannot mutate independently.

**1.4 Compare and timebox correction.** Compare broad sign, scale, failure cases,
and cost against held-out full-physics runs. Permit one bounded correction. If it
still erases selection-relevant distinctions, fall back to measured full-physics
bursts at a coarser cadence and continue.

**1.5 Put ecological motion in the runner.** Replace million-substep ecological
intervals with the chosen uniform motion path and rerun the operational command.
Do not add feeding in this slice.

**Exit:** the composed world advances at a useful measured rate, bodies differ for
physical reasons, and the failed exact-heading controller is not a dependency.

### Milestone 2 — first complete material lifecycle

**2.1 Add creature material stores.** Add structural nutrient and reserve nutrient
to fixed-capacity creature state and to one whole-world exact ledger. Begin with a
small population and exact-clone morphology.

**2.2 Add one-creature local feeding.** Sample producer stock at continuous
position, debit the actual available stock, and credit reserve plus declared waste.
No independent request clamps or population allocator yet.

**2.3 Add maintenance and starvation death.** Debit a morphology/mass-derived
maintenance cost, route spent matter and heat explicitly, and return all remaining
creature material exactly once on death.

**2.4 Add paid exact-clone birth.** Transfer the full structural and initial reserve
cost before allocating a child ID and slot. Slot exhaustion is an event, not a
carrying-capacity rule.

**2.5 Run the first lifecycle scenario.** One deliberately viable organism must be
able to feed and create a paid child; one deliberately starved organism must die;
the field must receive and recycle returned material. This proves mechanisms, not
general viability.

**Exit:** feeding, maintenance, birth, death, and recycling occur in one runnable
world with exact matter closure and explicit energy boundaries.

### Milestone 3 — population interaction and visibility

**3.1 Add shared-stock contention.** Only now add deterministic integer
apportionment when multiple organisms request the same finite producer stock.
Credits must equal the actual shared debit.

**3.2 Add simple fallible food-seeking intent.** A local producer gradient may
request bounded heading/effort. It may overshoot, circle, fail to turn, or starve.
There is no success repair or settlement gate.

**3.3 Add compact read-only snapshots.** Emit periodic summaries, creature state,
birth/death/feed events, and a downsampled field view. Heavy causal samples are
off by default. Snapshot cadence is independent of simulation cadence. Output rate
and bytes per simulated day are measured.

**3.4 Add save/resume for authoritative state.** Save fields, creature material,
genotypes, clocks, IDs, allocator, and deterministic state. Before a public v1,
format changes may break old developer fixtures; user-owned runs require an
explicit version and migration decision.

**3.5 Connect a minimal viewer only if the headless loop is useful.** Stream the
latest snapshot at a bounded visual cadence. Do not replay an unbounded history on
connect, and do not require archival scientific logging for interactive viewing.

**Exit:** the owner can watch a small population move, feed, reproduce, die, and
alter its environment without changing the simulation trajectory or generating
unbounded telemetry.

### Milestone 4 — first variation

**4.1 Add lineage identity without mutation.** Exact-clone births establish parent
and child lineage records first.

**4.2 Add bounded parameter mutation at paid birth.** Mutate only authoritative
genotype parameters already consumed by development. Rebuild body and mobility
response; never mutate a response cache.

**4.3 Run short replicated exploratory histories.** Report extinction, population,
lineage, morphology, material, and performance distributions without requiring a
preferred outcome.

**Exit:** differential survival and reproduction can occur without a fitness score.
Structural mutation, predators, mating, speciation, land, and richer chemistry
require new evidence from these live runs; they are not pre-authorized here.

## 6. Scope and stop rules

Stop and split a slice before implementation if it requires more than one new
scientific mechanism, a schema plus a mechanism, or unrelated refactoring.

Stop feature work and address runtime if:

- the headless smoke no longer runs;
- complete-world throughput falls materially without an accepted scientific
  reason;
- output grows without a declared bound; or
- the owner cannot observe the mechanism the slice claims to add.

Reject or simplify a mechanism if:

- its material or energy path cannot be stated plainly;
- it needs a target population, morphology, or outcome branch;
- it duplicates an authoritative quantity;
- its safeguard cannot be attacked by a meaningful negative control; or
- one implementation and one bounded correction fail to improve the evidence.

Do not broaden a slice merely because adjacent work is convenient. Record the
adjacent need in the next-slice list and finish the current runnable state.

## 7. Donor policy

`main` contains useful reports, fixtures, developmental pair provenance, and
diagnostic force code. It also contains a large set of rejected actuator paths.
Do not merge it wholesale. Import one donor change only when a current slice has a
specific consumer, and preserve the original result that states its limitations.

The successor SirRobin Living repositories are also donors. Their independent
matter/accounting tests may be adapted when the corresponding mechanism exists.
Their registries, acceptance packets, schema machinery, and always-on durable
observability are not inherited.

## 8. Completion of this plan

This plan succeeds when a small headless world can run at a useful measured rate,
close tracked matter exactly, account for energy boundaries, move through
form-derived fallible mechanics, feed, maintain itself, reproduce by paid transfer,
die and recycle, save/resume, and expose bounded read-only observations.

It does not require a stable population, successful steering, evolved diversity,
scientific validation, a polished viewer, or a large-scale evolutionary result.
Those questions become worth planning only after this loop is routinely usable.
