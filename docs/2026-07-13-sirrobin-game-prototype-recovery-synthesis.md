# SirRobin and Game Prototype: recovery synthesis

**Date:** 2026-07-13

**Status:** recovery position; execution is governed by
`superpowers/plans/2026-07-13-sirrobin-living-loop-recovery-implementation-plan.md`

## Position

Both projects remain valuable, but neither current trajectory should continue unchanged.

- **Game Prototype built a living world before its scientific and architectural foundations were
  dependable.** It exposed the right phenomena early—movement, feeding, births, deaths, predation,
  population change, visible bodies, and an inspectable world—but accumulated multiple authorities,
  artificial outcome controls, disabled faithful paths, and a central class that made every correction
  cross-cutting.
- **SirRobin built dependable scientific components before it built a living world.** It proved
  exact nutrient transactions, a canonical genotype-derived body, causal mechanics, and CUDA-scale
  population throughput. It then treated a phenotype-quality failure as a universal blocker and spent
  five investigations deepening that mistake while the ecology, lifecycle, and visible world remained
  disconnected.

The way forward is not to choose one failure over the other. It is to build a **thin, headless, visible,
conserved living loop** on SirRobin's clean state and transaction substrate, while recovering Game
Prototype's empirical virtues: simple behaviors, immediate whole-world observation, short feedback
cycles, and willingness to let imperfect organisms live or die.

The most promising technical inquiry is a **physics-derived reduced-order mobility model**. The full
articulated hydrodynamic solver should remain the reference instrument that turns an immutable genotype
into physical response. It need not necessarily be the substep-level integrator for every organism over
ecological deep time. A per-genotype, rebuildable response model can retain causal form-to-function while
making 5,000–10,000-creature evolutionary runs tractable.

This recommendation keeps SirRobin as the active research core, freezes Game Prototype as an executable
C#/Unity donor and behavioral reference, preserves headless execution, and moves the Unity viewer much
earlier. It does **not** merge the two simulation authorities.

## 1. What the rubble actually says

### 1.1 Game Prototype: alive, observable, and scientifically compromised

The C# project is not empty or merely visual. Its current runtime evidence shows a functioning world:

- 311 embodied organisms at about 50 fps;
- continuing births and deaths;
- predators with locks, grip, and captures;
- moving bodies with measured yaw rates;
- guild, diet, depth, morphology, and lineage telemetry.

That evidence is visible in `Logs/Editor.log:4880-4915`. This matters. A living run exposed bad signs
that component tests alone would not: the same log reports poor on-food behavior, implausible locomotion
costs, and a world whose apparent stability is partly governed.

The scientific foundation is still not authoritative:

- `LifeEconomyConfig.cs:18-41` leaves `grazeStockCouple`, `egestCouple`, and `ventFluxCouple` at zero.
  The comments themselves identify the resulting mints and unit mismatch.
- `NutrientConfig.cs:3-18` leaves nutrient limitation, remineralization return, and vertical mixing at
  zero. The nutrient loop is present as a skeleton but absent from the live world.
- `SimRunConfig.cs:53-60` leaves reproduction build cost and mass-scaled energy capacity off while
  enabling `densityBrake=1`, an explicitly artificial population governor.
- `OceanColony.cs:2098-2145` routes reproduction through carrying-capacity, guild-cap, and authored-form
  terms. This can create a stable-looking population without demonstrating a self-limiting ecology.
- `OceanColony.cs:2427-2451` advances far creatures through a scalar speed and target step, while near
  bodies use a different physical path. That creates two selection regimes.
- `OceanColony.cs:2503-2524` still creates a complete mutated body while the faithful, mass-proportional
  build-cost path is disabled.

The architecture magnifies those problems. `OceanColony.cs` is 3,229 lines, the Life code mixes
simulation time and `MonoBehaviour` ownership with pure C# components, and the organism has both an
open-ended `BodyGenome` and the parallel `eff[]` capability vector. The repository has accumulated many
gated transitions in which correct code exists but is not the running model.

Yet Game Prototype contains irreplaceable assets:

- `SwimEval.cs` and `SwimEvalTests.cs:276-288` contain the original C# physical reference and its exact
  instantaneous work identity.
- `BodyGraph.cs:57-218` contains the open-ended developmental genome and structural mutation ideas.
- `SteerableEbtBody.cs` demonstrates a simulation-only body with optional rendering rather than a
  GameObject as the mechanical authority.
- The runtime lifecycle, pursuit, analytics, inspection, and Unity presentation show what a whole
  experiment needs to reveal to its owner.
- The C# mechanics and development code have already been extracted behind .NET oracle projects in
  SirRobin. The development oracle currently builds outside Unity with small shims. The root SwimEval
  project currently needs its default compile glob scoped because the nested development project introduced
  a second top-level `Program`; that is project-file drift, not a Unity dependency. The valuable boundary is
  **headless core versus optional Unity**, not Python versus C#.

### 1.2 SirRobin: sound components, no living loop, and an overgrown investigation

SirRobin correctly retained the project's scientific north star: abstract the mechanism while keeping
causes honest (`docs/2026-07-11-sirrobin-overview.md:23-37`). It also made several good architectural
decisions:

- exact int64 transactions for the limiting nutrient;
- continuous organism positions and interpolated fields;
- one GPU-vectorized state substrate;
- genotype-to-developed-body causality without an `eff[]` vector;
- clean package boundaries and a headless clock;
- measured mechanics at 5,000 and 10,000 organisms.

S0/S1/S2 therefore produced real value. In particular, the S2 report records finite, work-consistent
mechanics and about 2.0–2.8 million creature-steps/s in the authorizing CUDA cells
(`docs/superpowers/reports/2026-07-12-sirrobin-S2-decision-report.md:13-17,46-63`).

But the parts do not yet make an organism. `src/sirrobin/core/live_world.py:35-53` composes heading,
mechanics, passive transport, and wrapping. `src/sirrobin/economy/step.py:15-81` advances the four
abiotic/producer/microbial reservoirs. Neither composes the other; there is no feeding, creature reserve,
reproduction, death transfer, mutation, mating, or population selection in the production source.

The S2 blocker was misclassified. The swimmer did turn in the requested direction and open-loop mechanics
remained signed and symmetric, but it did not settle within 15 degrees after a frozen episode
(`S2-decision-report.md:65-75`). That is evidence about one controller/phenotype, not evidence that the
mechanical substrate is unusable. Nevertheless, the report prohibited S3 and demanded a successor controller,
new frozen corpus, long drift run, and full benchmark matrix (`S2-decision-report.md:86-92`).

The cost of that category error is measurable. From commit `0cea74e` (the original S2 implementation) to the
preserved controller-study head on `main`, `c0fc62e`, the controller/actuator investigation added:

| Area | Files | Added lines |
|---|---:|---:|
| production source | 13 | 2,037 |
| tests | 23 | 2,258 |
| diagnostic tools | 5 | 3,189 |
| documentation | 18 | 2,197 |
| generated reports | 6 | 242,295 |

None of those five studies connected nutrient, movement, feeding, reproduction, or death. Current Python
totals are approximately 6,561 production lines, 3,979 test lines, and 5,242 tool lines. The diagnostic
apparatus is now almost as large as the simulation and larger than the test suite. That is the clearest
quantitative signature of correctness theater in this repository.

## 2. The best of each project

| Keep | Source | Use going forward | Do not inherit |
|---|---|---|---|
| Causal hydrodynamic equations and work ledger | both; C# remains the original reference | derive mobility and locomotion cost from developed morphology | exact trace matching for every ecological change |
| Open-ended body recipe and structural development | both | immutable genotype → developed morphology → derived capabilities | parallel capability/stat genome |
| A visibly living experimental world | Game Prototype | early viewer, lifecycle telemetry, owner-observed runs | scene or renderer as simulation authority |
| Simple reactive drives | Game Prototype | food-gradient and threat-gradient intent | perfect navigation or authored success |
| Contact pursuit and explicit lifecycle events | Game Prototype | later minimal predation and birth/death events | mean-field kills, duplicate capture, or hidden population governors |
| GPU struct-of-arrays state | SirRobin | one 5k–10k headless world | per-object hot-loop ownership |
| Exact reservoir transactions | SirRobin | matter transfers, feeding debit, reproduction/death partition | universal exactness for continuous mechanics |
| Continuous positions and sampled fields | SirRobin | local uptake and encounters | biology defined by field cell identity |
| Single authority and clean composition | SirRobin | one live state; one-way derived views/caches | a ban on all derived caches or reduced models |
| Small, reversible changes | lessons from both | one end-to-end capability at a time | recursive plans, repeated preregistration, large refactor tranches |

## 3. Correcting the principles without abandoning them

### 3.1 Scientific fidelity has levels

The project should be strict about **causal topology**, not mechanically maximalist about every calculation.

#### Universal invariants

These remain hard:

- tracked matter is neither minted nor deleted;
- energy has explicit inputs, transfers, and dissipative outputs;
- units, signs, and debit/credit identities are sound;
- state remains finite and inventories non-negative;
- actions act through physical capability and do not teleport state;
- no hidden carrying-capacity or target-morphology term creates the desired outcome;
- one state authority exists for each quantity.

The existing exact int64 nutrient transaction system is worth keeping because it is already implemented,
cheap relative to the field work, and catches the prior project's most damaging error. This does not imply that
float mechanics, ecological rates, or every scientific comparison must be exact.

The energy boundary also needs more precise language. An ecological world is not energetically closed: sunlight
and geological flux enter, and degraded heat leaves the modeled system. Scientific fidelity requires an explicit
energy budget across those boundaries; it does not require storing a spatial heat reservoir that no current
consumer uses.

#### Declared approximations

The following are legitimate when their operating domain is stated and the selection-relevant causal relation is
preserved:

- cycle-averaged rather than phase-resolved locomotion;
- reduced-order surge/yaw dynamics rather than all segment forces every substep;
- prescribed mixing rather than Navier–Stokes circulation;
- interpolated Eulerian food fields rather than individual plankton parcels;
- clearance-volume feeding rather than mouth-scale CFD;
- simple chemical-gradient sensing rather than molecular diffusion and receptor biophysics;
- coarse contact/gape predation rather than tissue damage mechanics;
- fixed empirical conversion efficiencies with cited domains;
- statistical outcome checks rather than exact trajectories.

An approximation is acceptable when it preserves **why one organism differs from another**. A cheaper formula
that still derives turning authority from body geometry is useful. A free `agility` gene unrelated to geometry is
not.

#### Phenotypes and research outcomes

Overshoot, foraging efficiency, lifespan, reproductive success, population persistence, speciation, and land
crossing are observations or research results. A deliberately viable reference organism may be used to prove an
end-to-end mechanism, but the general population is allowed to fail.

### 3.2 One authority does not mean one materialization

The useful rule is **one authoritative state**, not “only one representation may ever exist.”

Derived data is allowed when it is:

- a pure or rebuildable function of the authority;
- one-way and never written back as an independent truth;
- invalidated locally when its immutable source changes;
- clearly named as a response model, cache, snapshot, telemetry view, or render asset.

This permits developed bodies, mobility response tables, spatial indices, viewer snapshots, and render meshes.
It still forbids two live creature states that must be synchronized.

### 3.3 Depth-first must mean vertical, not component-perfect

Game Prototype built breadth without closure. SirRobin answered by perfecting components in isolation. The useful
middle is a **walking skeleton**:

```text
light / nutrient
      ↓
producer stock → local feeding → creature reserve/tissue
      ↑                           ↓
remineralization ← detritus ← death / waste / reproduction
                                  ↓
                         movement changes access
```

Build the thinnest complete version of that loop, then deepen the bottleneck the living run exposes. Do not finish
all locomotion research before feeding, or all ecology before a viewer exists.

## 4. The promising mobility direction

### 4.1 Why full articulated substeps should become a question, not a creed

The full PyTorch mechanics proved it can run 5,000–10,000 creatures near real-time or modestly faster. That is a
strong component result, but evolutionary deep time may still make 120 Hz segment-level physics too expensive.
The original design treated any reduced model as a return to Game Prototype's proxy wound. That conflates two
very different designs.

Game Prototype's proxy failed because it was:

- selected per species centroid rather than derived per immutable genotype;
- combined with independent `speed`/`agility`/`eff[]` genes;
- different for far and near creatures;
- activated through gains and periodically re-tethered;
- capable of changing the selection regime with camera distance.

A valid reduced-order model would have none of those properties.

### 4.2 Proposed model

At development or birth:

1. develop the genotype into the canonical body;
2. run a small standardized set of full-physics probes;
3. derive an immutable, per-genotype mobility response, such as surge force/drag, yaw moment/drag, inertia,
   signed control authority, and mechanical work as functions of effort;
4. cache that response by genotype identity or content hash;
5. use one reduced-order rigid-body integrator for **every** organism in the ecological hot loop.

The response may be coefficients or a small lookup table. It is not heritable state and cannot mutate
independently. A cache miss recomputes it from the body. A selected sample can be replayed through the full solver
to monitor approximation error.

This retains inertia, limited authority, overshoot, drag, energetic cost, and organisms that simply cannot turn.
It removes the cost of resolving every segment and gait phase at every ecological movement step. The viewer may
animate segment pose from genotype plus gait phase without becoming a mechanical authority.

### 4.3 Minimal exploratory test

This should be exploratory, not a new authorization bureaucracy:

- use roughly 64–128 deliberately varied existing bodies;
- compare signed response, broad speed/turn/cost ranges, and rank correlation with the full solver;
- report failures and domains; do not require exact episode traces or universal controllability;
- measure complete-world cost, not an isolated kernel only;
- stop after one implementation and one correction. If the approximation is poor, retain full mechanics at a
  lower/multi-rate cadence for the first living loop and revisit only after whole-world profiling.

The question is: **does the reduced model preserve the selection-relevant ordering and physical limitations well
enough to buy meaningful ecological time?** It is not: “can it impersonate every full trace?”

## 5. Headless core and the role of C#/Unity

Headless execution is non-negotiable and language-independent:

- the full simulation must reset, run, checkpoint, benchmark, and evolve with no Unity process;
- simulation time belongs to the core, never the render frame;
- a Unity process is a detachable read client and, later, an action client;
- losing the viewer must not change a seed's scientific trajectory;
- the viewer never holds a second authoritative organism or field state.

For the next living-loop tranche, Python/PyTorch remains the lowest-risk core because the exact economy, canonical
body, and measured CUDA population path already exist. This is a pragmatic choice, not a declaration that C# is
only for presentation. The C# `SwimEval` and `BodyGraph` remain executable headless references; other pure C#
mechanisms may be ported selectively.

Unity should return earlier than the current S7 roadmap suggests, initially as a simple observer:

- periodic snapshots or replay files, not a per-substep dependency;
- stable IDs, alive state, position, yaw, morphology/genome identity, reserve, and lineage;
- downsampled producer/nutrient heatmaps;
- births, deaths, feeding, and reproduction events;
- simple glyphs first, cached body reconstruction later.

An all-C# headless-core rewrite should not be started now. The live ecology is valuable, but its full authority is
still embedded in `OceanColony`, Unity time, and Unity-owned fields. Porting SirRobin's exact economy and CUDA
layout back into that substrate would be another restart before the first restart has produced a life cycle. If a
complete PyTorch living tick later fails the actual 5k–10k service requirement, then a bounded C#/.NET/Burst or
compute-kernel comparison becomes justified by evidence.

## 6. Recommended recovery sequence

### Step 0 — preserve evidence and reset authority

- The later SirRobin head remains preserved on `main` as the complete S2 controller/actuator investigation.
- Branch `recovery/living-loop` starts at commit `0cea74e`, which already contains S0, S1, and the useful S2
  mechanics before the five-study branch.
- Carry forward the recovery synthesis and implementation plan, corrected test policy, startup disposition, and
  any small representation fixes independently required by the next loop.
- Do not delete the failed work and do not keep its diagnostic modules in the production import graph.
- Freeze Game Prototype as an executable donor/reference; do not resume broad refactoring there.

### Step 1 — mobility reduction spike

Answer the question in §4 with one bounded implementation. Keep the full mechanics path available as the
reference. The spike succeeds if it buys a material whole-world speedup while preserving signed physical response
and useful morphology ordering. Exact settlement is irrelevant.

### Step 2 — minimum living loop

Build only the following:

1. **Creature material state.** Add creature-bound tissue nutrient and labile reserve to the exact matter ledger.
   Body scale/mass determines required tissue; the genotype determines shape.
2. **Local feeding.** Sample producer concentration at continuous position. Request uptake from a
   morphology-derived clearance/intake quantity. Debit the producer field transactionally and split every
   limiting-nutrient quantum of the actual debit between creature matter and returned field matter. Account for
   chemical-energy assimilation and dissipated heat separately; energetic loss is not a missing matter credit.
3. **Maintenance and locomotion cost.** Debit labile reserve from a simple mass-derived maintenance rate plus
   measured/derived mechanical work. Route the limiting nutrient released by reserve use to the appropriate
   dissolved or detrital pool; record chemical-energy input and heat loss explicitly.
4. **Death.** When reserve is exhausted or a simple age hazard fires, transfer all tissue and reserve to detritus
   exactly and free the slot.
5. **Reproduction.** A reference organism with enough reserve pays the exact tissue and provision cost of one
   child. Begin with an exact clone if necessary; mutation is not required to prove the lifecycle transaction.
6. **Fallible foraging.** Request heading along a sampled producer gradient. The physical/reduced body may
   overshoot, circle, forage poorly, or starve. No heading-settlement gate exists.

For this first loop, labile reserve may be modeled as organic matter with a fixed limiting-nutrient ratio and
chemical-energy density. Reserve energy is then derived from that one stock rather than stored as a synchronized
float mirror. Primary production adds chemical energy through the explicit light boundary; maintenance and
locomotion discharge it to an explicit heat-loss ledger. Flexible carbon:nutrient stoichiometry, lipid stores, and
spatial heat are later refinements justified only if the biology needs them.

The only hard behavioral capability is that a deliberately viable reference phenotype can feed, pay maintenance,
produce a child, and that a deliberately starved phenotype dies and returns its material. General persistence is
telemetry.

### Step 3 — observe and scale the complete tick

- Begin with a small cohort for readable failures.
- Add the minimal replay/viewer surface immediately after the loop works, not after speciation and weather.
- Run the **complete** loop routinely at 5,000 creatures and attempt the 10,000-creature stretch target on CUDA.
- Set a performance requirement from the ecological clock and desired experiment horizon. Do not reuse an
  isolated mechanics multiple as the whole-world threshold.
- Profile only after the end-to-end run identifies the limiting subsystem.

### Step 4 — introduce variation before additional ecology

- Add parameter mutation and inheritance to the existing genotype.
- Confirm that morphology changes alter intake, movement, maintenance, and reproductive cost through derived
  mechanisms.
- Observe selection distributions; do not require a preferred morphology to win.
- Add structural mutation once the fixed-shape lifecycle is stable.

Only then add predation, mating/speciation, richer circulation, or land. Each addition must close a new transfer
or create a specific missing selection gradient in the already living world.

## 7. Acceptance policy for the recovery

Every proposed hard test must complete this sentence:

> The next consumer cannot function safely or honestly unless ________.

If the blank is “the creature turns neatly,” “the population persists,” “the morphology looks right,” or “the
trace matches the donor,” it is probably telemetry or a research question.

Use the following governance rules:

1. **One plan and at most one adversarial review per tranche.** Further review findings are issues unless they
   identify a violated universal invariant or a demonstrated next-consumer blocker.
2. **Exploration is not preregistration.** Freeze thresholds for a confirmatory scientific claim, not for learning
   what the mechanism does.
3. **Two-attempt stop rule.** If two consecutive studies of the same premise do not increase practical confidence,
   stop refining that premise and reconsider the abstraction or the need.
4. **Every tranche returns to a runnable vertical loop.** Library work may be temporarily isolated, but no sequence
   of component studies may indefinitely block the living world.
5. **Owner-visible behavior is evidence.** Tests support it; they do not replace it.
6. **Approximation error is a budget, not a moral failure.** Report its domain and improve it only when it changes a
   selection-relevant conclusion.
7. **Safety capacity is not carrying capacity.** A fixed tensor capacity may reject/queue births or terminate an
   overloaded experiment explicitly; it must not quietly change reproductive fitness to manufacture stability.

## 8. Near-term definition of success

The next meaningful milestone is not a perfect swimmer, natural speciation, or another controller authority.

It is:

> A headless, reproducible run in which thousands of morphology-bearing creatures move imperfectly through a
> producer field, transactionally feed, pay body- and movement-derived costs, reproduce when they can afford a
> real child, die when they cannot, and return their material to the same nutrient loop—while a lightweight Unity
> or replay observer makes the resulting world inspectable.

That milestone would combine the strongest achievement of each project. It would also create the first substrate
on which the project's genuine research questions—selection on form, ecological diversification, speciation, and
eventual medium crossing—can be asked without either scripted answers or an intractable proof apparatus.
