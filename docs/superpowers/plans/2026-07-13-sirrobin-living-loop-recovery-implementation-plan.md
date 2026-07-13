# SirRobin living-loop recovery implementation plan

**Date:** 2026-07-13

**Status:** active execution authority for `recovery/living-loop`

**Branch baseline:** `0cea74e` (`Implement and evaluate S2 live locomotion`)

**Position paper:** `../../2026-07-13-sirrobin-game-prototype-recovery-synthesis.md`

**Test policy:** `../../2026-07-13-sirrobin-test-gate-policy.md`

## 1. Purpose

This branch will produce the first thin, headless, conserved living loop in SirRobin:

> Thousands of morphology-bearing creatures move imperfectly through a producer field, feed by exact
> transactions, pay body- and movement-derived costs, reproduce only by transferring enough material to a
> real child, die when their reserves are exhausted, and return their material to the same ecological ledger.

The loop must be inspectable without becoming dependent on a renderer. A replay or snapshot observer comes
with the loop; a detachable Unity observer may follow the same read-only contract.

This is a recovery plan, not a continuation of the controller programme. The original S2 exact-heading
settlement failure remains valid evidence about that controller. It does not block use of the implemented body,
signed mechanics, work ledger, or measured CUDA path.

## 2. Decisions already made

1. **Headless is non-negotiable.** The complete world must reset, run, checkpoint, benchmark, and evolve with
   no Unity process. Simulation time belongs to the core.
2. **Python/PyTorch is the current core by pragmatism, not doctrine.** It already holds the conserved economy,
   canonical body, and batched mechanics. Extracted C# code remains an executable scientific donor. Unity is an
   optional observer, not the C# language's only role.
3. **One authoritative state, not one materialization.** Rebuildable body development, mobility responses,
   spatial indices, snapshots, and render meshes are allowed. None may become an independently mutable truth.
4. **Form remains the source of capability.** Locomotion, intake, maintenance, and construction requirements
   derive from developed morphology and physical/environmental constants. There is no free speed, agility,
   efficiency, fitness, or carrying-capacity gene.
5. **Tracked limiting nutrient closes exactly.** Field and creature reservoirs use int64 quanta. Every complete
   world step must conserve their sum exactly and keep every reservoir non-negative.
6. **Energy crosses explicit boundaries.** Light supplies chemical energy to primary production; feeding moves
   chemical energy with biomass; maintenance and locomotion dissipate it as heat. Heat may be a ledger output
   rather than a stored field while nothing consumes it.
7. **Creatures are fallible.** Intent changes bounded actuation. It never snaps position, velocity, yaw, or
   reserve. Overshoot, poor foraging, failed reproduction, and extinction are possible outcomes.
8. **Approximation is part of the design.** Preserve the causal relation that matters to selection, declare the
   operating domain, and measure error where it can change a conclusion. Do not spend full-model cost where a
   cheaper model answers the present biological question.
9. **Every implementation tranche returns to a runnable loop.** Do not perfect a component in isolation while
   the end-to-end organism remains absent.

## 3. Scope

### Included

- one batched world composition root joining fields, economy, locomotion, feeding, metabolism, reproduction,
  death, clocks, ledger closure, snapshots, and telemetry;
- fixed-capacity, GPU-resident creature state with stable IDs and an alive mask;
- a bounded physics-derived mobility experiment, with the current full solver retained as reference and
  fallback;
- continuous-position field sampling and exact shared-stock debits;
- a fixed-stoichiometry initial animal reserve/tissue model;
- exact-clone, full-material reproduction before mutation;
- a deliberately simple fallible food-seeking behavior;
- headless replay/snapshot output and whole-world performance measurement; and
- routine 5,000-creature operation as the scale target, with 10,000 measured as the stretch target.

### Deferred

- mutation, mating, recombination, speciation, predators, combat, and tissue damage;
- ontogenetic growth, variable body composition, detailed digestion, and homeostasis;
- currents, weather, self-shading, sediment, burial, land, and multiple limiting nutrients;
- learned control, perfect navigation, and any exact-heading requirement;
- a production-quality Unity experience; and
- a language rewrite or a second live simulation authority.

The first loop uses a viable reference morphology and exact-clone births to prove mechanism. Variation begins
only after the loop can expose differential survival and reproduction without an authored score.

## 4. Runtime architecture

Modules are named for enduring domains, never for project phases. The intended additions are:

```text
src/sirrobin/
  core/
    world.py             # authoritative composition root
    runner.py            # headless multi-rate schedule
  organisms/
    state.py             # batched creature material/lifecycle state
    feeding.py           # continuous sample -> exact shared-stock transaction
    metabolism.py        # reserve use, maintenance, locomotion heat
    reproduction.py      # paid full-child creation
    mortality.py         # death and field return
    behavior.py          # simple fallible intent
  physics/
    mobility.py          # uniform ecological motion contract
    mobility_probe.py    # full-solver response derivation/validation
  observe/
    world_snapshot.py    # read-only replay/view contract
    world_telemetry.py   # mechanism and research telemetry
```

Names may change to fit the live code, but phase-prefixed packages such as `s3_*` are prohibited. Existing
`economy`, `fields`, `genetics`, and `physics` modules remain the owners of their domains.

### 4.1 Authoritative world state

The composition root owns:

- `EconomyState`: `Nd`, `Bp`, `Bd`, and `Bm` int64 field reservoirs plus their numerical carries;
- `CreatureState`: stable ID, alive flag, genotype/body identity, position, velocity, yaw, age, structural
  nutrient, reserve nutrient, and restart-relevant numerical carries;
- immutable genotype-to-developed-body products or content-addressed rebuildable caches;
- a fixed-capacity slot allocator whose allocation/deallocation order is deterministic on a fixed device; and
- all sub-clocks and schedule counters required to resume the same run.

The first animal chemistry model has two material stores:

- **structural nutrient**: the limiting nutrient incorporated into a developed adult body; and
- **reserve nutrient**: assimilated biomass available for maintenance, locomotion, and reproduction.

Chemical energy is derived from reserve transfer using a declared fixed joules-per-nutrient conversion. It is
not a second synced energy pool. Catabolized reserve nutrient is transferred to a declared matter reservoir
while the associated chemical energy leaves as ledgered heat. Structural nutrient is transferred, never
silently resized, when a creature is born or dies.

The developed body's wet mass and its required structural nutrient are connected by explicit dry-matter and
nutrient-fraction constants. A newborn initially appears at full developed size only after its parent has paid
the entire structural and initial-reserve requirement. That is a declared approximation which avoids minted
adult bodies until ontogeny has a real consumer.

### 4.2 One whole-world matter ledger

The authoritative invariant is, per world:

```text
Nd + Bp + Bd + Bm + creature_structural + creature_reserve = constant
```

The existing field-only `MassLedger` must be generalized or placed under this world ledger before animal-field
transfers begin. There must not be two ledgers with incompatible expected totals. Each operation returns actual
integer debits and credits; the complete tick verifies exact equality and non-negativity.

Shared feeding uses one transaction over all requests. Continuous interpolation weights determine which field
cells a creature samples and debits. When requests exceed stock, deterministic integer apportionment distributes
the available quanta without independent per-creature clamps, overdraft, or minting. The returned actual debit,
not the request, drives assimilation and waste.

### 4.3 Multi-rate headless runner

Mechanics, behavior, metabolism, field reactions, and observation do not need the same cadence. `runner.py`
owns an explicit schedule in simulation time:

- mobility/behavior steps often;
- feeding and animal metabolism at a coarser declared interval;
- the existing ecological reaction/transport kernel at its stable interval; and
- snapshots/telemetry least often.

All cadences are configuration, snapshot, and provenance data. No cadence depends on render frames. Timestep
halving is used when a measured result appears cadence-sensitive; it is not an automatic matrix for every test.

### 4.4 Mobility contract

The ecological runner depends on one uniform mobility interface. It may be backed by:

1. the existing articulated full-physics step at a suitable cadence; or
2. a per-genotype physics-derived response used by a reduced-order surge/yaw integrator.

The reduced response is produced from the developed body by standardized full-solver probes. It may contain
surge/yaw authority, inertia, drag, and work/cost response versus bounded effort. It is immutable, rebuildable,
and keyed by genotype/body content. It is not heritable state, a camera-distance LOD, or a free capability vector.
Every live creature uses the same chosen integration path.

## 5. Acceptance model

### 5.1 Hard engineering gates

These claims are universal over the declared operating domain:

- whole-world limiting-nutrient books close exactly after every complete tick;
- no material reservoir is negative, exceeds its int64 safety domain, or is repaired after the fact;
- a feeding, birth, or death credit cannot exceed its actual debit;
- a birth creates a live state only after the full configured child material is transferred;
- death returns all remaining creature material exactly once and frees the slot exactly once;
- action changes intent/effort only; it does not assign physical outcome state;
- mechanics and derived mobility state remain finite and within declared emergency bounds;
- no hidden carrying-capacity, target-stock, target-morphology, or success branch is reachable;
- snapshot/restart contains every authoritative reservoir, carry, clock, ID, and allocator state; and
- the complete simulation runs without Unity.

### 5.2 Minimum end-to-end capability

A small, deliberately constructed reference scenario must demonstrate that the mechanism exists:

- a capable reference body produces signed movement through its physical mobility path;
- local movement changes access to producer stock;
- at least one creature can debit food, accumulate paid reserve, and create a paid child;
- a starved creature can die and return all of its material; and
- the field economy can subsequently recycle returned material.

This is not a universal viability claim. Holdout bodies, random populations, persistence, abundance, overshoot,
foraging efficiency, and time-to-reproduction are telemetry unless a specific next consumer needs a minimum.

### 5.3 Research telemetry

Record without turning the observation into a correctness gate:

- food encountered, requested, debited, assimilated, respired, and egested;
- maintenance and locomotion cost by body and distance;
- reserve, age, births, deaths, causes of death, and lineage;
- movement speed, turn response, overshoot, and time near food gradients;
- field and creature reservoir totals;
- population and biomass trajectories; and
- complete-world time by subsystem and device.

Extinction, oscillation, inefficient bodies, and failure to discover diversity are admissible findings once the
mechanisms and invariants are sound.

### 5.4 Scale evidence

Correctness tests use the smallest population that exposes their claim. Integration runs then measure the
complete world at 5,000 creatures and attempt 10,000 on the user's CUDA device. Report wall time, simulated-time
advance, memory, and subsystem attribution. No isolated-kernel result, arbitrary real-time multiple, or single
noisy sample may stand in for complete-world usability. Ten thousand is an aspiration, not a reason to reject a
scientifically useful 5,000-creature loop.

## 6. Implementation tranches

### Tranche A — recovery baseline and headless seam

1. Track the recovery position, this plan, the test policy, `AGENTS.md`, and the terse `CLAUDE.md` disposition.
2. Keep later controller/actuator investigations preserved on `main`; do not copy their production or diagnostic
   graph into this branch.
3. Mark the fallible-steering correction as historical analysis superseded by this plan.
4. Repair the root C# oracle project compile glob so both headless oracle projects build independently.
5. Run the Python suite and both C# builds. Preserve the exact-settlement xfail as historical phenotype evidence,
   not a blocker.
6. Add one cheap headless composition smoke test that advances existing mechanics and economy clocks without
   claiming they are biologically coupled yet.

**Exit:** the branch is reproducible, both donor arms are executable headlessly, and there is one named place to
compose the living world.

### Tranche B — bounded mobility inquiry

1. Define the uniform ecological mobility interface without changing genotype authority.
2. Select roughly 64–128 varied existing developed bodies, including root-only and poor movers.
3. Run standardized full-solver surge/yaw/effort probes and derive the smallest useful response representation.
4. Implement a batched reduced surge/yaw step with inertia, drag, bounded authority, and locomotion work/cost.
5. Compare response sign, broad speed/turn/cost range, morphology ordering, failure cases, and complete-world cost.
6. Make one correction if evidence identifies a bounded flaw, then stop the inquiry.

**Decision:** adopt the reduced path only if it gives a useful whole-world benefit without erasing the physical
differences the next ecology uses. Otherwise retain the full solver at a measured multi-rate cadence. Either
result permits Tranche C; there is no demand for exact traces, exact settlement, a universal rank threshold, or
universal controllability.

### Tranche C — creature material and transactional feeding

1. Add fixed-capacity `CreatureState` and include both creature material stores in the whole-world ledger.
2. Define the explicit wet-mass-to-structural-nutrient and reserve-energy unit chain.
3. Implement continuous producer sampling and a deterministic exact shared-stock feeding transaction.
4. Partition the actual food debit into structural/reserve assimilation and field-returned waste/respired matter;
   account for chemical-energy transfer and heat separately.
5. Compose field economy, creature state, movement, and feeding in the headless runner.

**Exit:** moving creatures can alter where matter transfers, and every transfer closes exactly in one live world.

### Tranche D — metabolism, death, and paid reproduction

1. Debit maintenance from a mass-derived basal rate and locomotion from measured/derived physical work.
2. Transfer catabolized nutrient to the declared field reservoir and ledger dissipated heat.
3. Implement starvation death and exact one-time return of structural/reserve material.
4. Implement full-material exact-clone birth, stable ID allocation, and slot exhaustion as an explicit event rather
   than an ecological carrying-capacity rule.
5. Add the simplest bounded food-gradient intent that can be useful but may overshoot or fail.
6. Run the reference feed/reproduce/starve/recycle scenario and a small unscripted mixed run.

**Exit:** one complete lifecycle occurs because local access and material balance permit it, not because a target
population or scripted success branch commands it.

### Tranche E — observation and scale

1. Publish a versioned read-only snapshot/replay contract containing stable IDs, alive state, position, yaw,
   body/genotype identity, material state, lineage, events, and downsampled field views.
2. Build a minimal headless replay inspection path first. Then, if useful, connect a detachable Unity observer
   without moving time or authority out of the core.
3. Profile the complete tick, not only locomotion or economy kernels.
4. Run sustained 5,000-creature measurements and attempt 10,000 on CUDA.
5. Optimize only the measured dominant cost while keeping the loop runnable after each change.

**Exit:** the owner can inspect births, feeding, movement, death, material flow, and failure modes, and the measured
scale is sufficient to choose the next biological experiment honestly.

### Tranche F — first variation, only after the loop

1. Introduce bounded parameter mutation through the existing immutable genotype and exact paid birth.
2. Rebuild developed bodies and mobility responses from genotype content; never mutate a derived cache.
3. Observe differential survival/reproduction without a fitness score or diversity target.
4. Add structural mutation only after parameter variation survives the complete loop and its failure modes are
   visible.

This tranche requires a short successor plan based on live-loop evidence. It is not pre-authorized merely by
finishing the engineering skeleton.

## 7. Working rules

- One implementation plus at most one bounded correction for an exploratory idea. If prospects do not improve,
  take the fallback and move on.
- Exploration does not require preregistration. Freeze fixtures and thresholds only when making a confirmatory
  claim that depends on them.
- Every hard test must name the universal claim or immediate consumer it protects. Otherwise demote it to
  telemetry, a stress diagnostic, or a research experiment.
- Keep diffs small and reversible. Do not combine scientific model changes, infrastructure refactors, and viewer
  work in one tranche.
- Verify claims against current code and complete-world runs. Historical reports remain evidence, not authority
  over a changed question.
- Do not commit or push unless the user asks.

## 8. Completion condition

The recovery programme is complete when the headless living loop described in §1 runs with exact matter closure,
explicit energy accounting, fallible morphology-derived movement, paid feeding/reproduction/death, restartable
state, inspectable output, and measured operation at the intended population scale.

It does not require a stable population, a good controller, evolved diversity, predation, speciation, land
crossing, or a polished viewer. Those become scientifically meaningful questions only after this substrate lives.
