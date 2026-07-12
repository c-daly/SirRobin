# SirRobin — Restart Brief (understanding, not yet the design)

**Date:** 2026-07-11
**SUPERSEDED IN PART:** Sections 4–5 (the C#/Unity "day-one commitments" and slice mechanics) are
replaced by the grounded Python/PyTorch architecture in
`docs/superpowers/specs/2026-07-11-sirrobin-restart-architecture-design.md`. The north star, fatal-flaw
diagnosis, and salvage *intent* below still stand.
**Purpose:** Consolidate what the prior "game prototype" (codename *Sir Robin*) was trying to be,
why it became a soup, what to salvage, and the day-one decisions the clean restart must force.
This brief *seeds* the brainstorming/design; it is not itself the approved design.

**Source:** synthesis of 5 parallel reads over the prior codebase (`C:\Users\cddal\game prototype`)
+ ~140 design docs. The prior build is the reference/donor — it is not deleted.

---

## 1. North star (the enduring "why" — preserve this)

The sim exists to be a **faithful, consequence-bearing, non-text grounding substrate** in which the
owner's autonomous cognitive core **Sophia (LOGOS framework)** can eventually be embodied and learn
from real survival stakes. Sophia would inhabit a simulated ocean creature via a **Talos** hardware
-abstraction layer over **ROS 2**, with the *same* control/observation interface later transferable
to a physical **TurtleBot3** — the sim creature and the robot are deliberately one "mobile sensate
navigator" archetype — streaming continuous perception→action data into a JEPA-style grounded world
model (CWM-G), enabling eventual sim-to-real transfer. *"Text, used exclusively, is an insufficient
substrate for proper learning."*

Enduring principles:

1. **Faithfulness over veridicality.** Macro-signatures (trophic pyramid, carrying capacity, blooms
   & deserts, speciation, emergent taxonomy, ultimately an unscripted sea-to-land crossing) must be
   *derived* from conserved, closed-loop, relational constraints — never imposed by tunable knobs.
   "Fidelity is the product, not polish."
2. **First-law closure as an invariant, not a toggle.** No channel (production, feeding, predation,
   chemosynthesis, reproduction, egestion) may mint or destroy mass/energy; a single limiting
   nutrient cycles through a closed biogeochemical loop.
3. **Form is function.** Feeding, metabolism, defense, locomotion, predation *emerge* from body
   morphology run through real physics (Lighthill thrust, Lamb added-mass, quadratic drag, Kleiber
   metabolism, Holling response) — never an arbitrary stat vector.
4. **Open-ended evolution, implicit selection only.** Survival is the only score; selection is
   energy→reproduction; niche radiation + real reproductive isolation → speciation.
5. **Anchored measurement discipline.** Frozen unit bridges (300 J/sim-energy, 0.30 W/kg SMR,
   250 kg/sim-mass, muscle η≈0.20) are *measurements* recorded with their arithmetic, never balance
   knobs. Collapse is a diagnostic, never a law to soften.
6. **Research-grade determinism, reproducibility, legibility.** Seed-reproducible; the test/
   observation layer serves triple duty: verification, human legibility, future agent observation.
7. **Depth-first: close the books before breadth.** Prove the conserved loop before adding size,
   predation, temperature, plants, land. Hardness ramps up, never down.

---

## 2. The disease (root cause) and the fatal flaws (symptoms)

**Root cause (process):** *recursive scope explosion.* Implementing one idea kept surfacing
sub-problems that were themselves large projects; being mid-flight forced a "get it done" bulldoze
(a hack / knob / proxy / scalar shim) instead of stopping to re-scope. Each local bulldoze "worked"
but left an unfaithful seam. 535 commits in 22 days + ~57 specs = design outran integration.
Compounded by **no compile-time firewall** (one assembly, pervasive `internal`) so every bulldoze
could recouple into the god-class.

**Fatal flaws (from the audit):**

- **SCI — Matter/energy not conserved; two free-energy sources.** Plankton grows toward a static
  Perlin cap consuming no nutrient pool; remineralized snow is deleted; mean-field predation credits
  the predator with zero prey debit; graze/ chemo channels mint biomass (up to ~600×). *Codex
  quantitative-validity 2/10.* No nutrient state variable exists; the biological pump has only its
  downward half — so every downstream rate was calibrated against a base that mints matter.
- **PROC+ARCH — The faithful path ships DISABLED behind `gain=0` dials** so byte-identical
  determinism goldens stay green (`LocoCostGain=0`, `predConserve=0`, `kleiberExp=1`,
  `nutrientLimitBlend=0`, `microbialLoopGain=0`, …). The harness was pointed at the WRONG invariant:
  byte-identity *rewards never activating fidelity.* The live sim runs the least-faithful config of
  itself and the golden freezes it as the reference.
- **ARCH — Form is not function at the trait level.** A dual genome bolts an arbitrary 15-slot
  `eff[]` feeding/metabolism stat vector onto the faithful body graph; `SpeciesSwimCache` collapses
  every individual to one centroid representative (two same-species bodies swim identically).
- **ARCH — The data-far/physics-near LOD split is the spine, on an unverified premise.** The
  2026-06-30 memo names it "the root of every recurring incoherence"; the perf audit that was to
  decide it was never completed. The proxy launders a capability into a scalar `e.speed` via
  `swimGain01`/`locoCoef`/`formSpeed` Lerps + a `speedFloor..speedCeil` clamp.
- **ARCH — God-class monolith, no boundaries.** `OceanColony.cs` = 3,067 lines implementing 4
  interfaces with ~200 serialized knobs; `AnalyticsScreen.cs` ≈ 2,000 lines; ONE asmdef; config
  smeared across 3 static classes + serialized scene values + live `set_property`.
- **ARCH — No headless/step entrypoint (hard RL blocker).** Sim advances only from Unity
  `FixedUpdate`, reads scene singletons via `FindFirstObjectByType`, depends on
  `Time.fixedDeltaTime==0.02`, holds global mutable statics — so N vectorized RL envs can't coexist
  in one process, and the sim can't run without a live editor scene.
- **SCI — Species concept is cosmetic; speciation can never emerge.** Reproduction is asexual
  clone-and-mutate (no recombination/isolation possible); "species" is a magnitude-blind cosine
  cluster (~43× mass range, same name) that drives nothing; predation had to be *seeded*.
- **PROC — Design outran validated code.** ~57 specs, most "implemented but gated OFF," wrapped in
  R1–R7 / C1–C4 / H1–H4 review scar tissue → analysis-paralysis + a large inventory of inert seams.
- **ARCH — No canonical body.** Measure vs SwimmerBuilder vs SwimEval vs MakeBody describe different
  animals (divergent caps/mass floors; a 180° head/tail disagreement; a mis-oriented fin plate).

---

## 3. Salvage list (proven, mostly-pure — PORT, don't rewrite)

| Asset | Why keep |
|---|---|
| **`SwimEval`/`SwimmerSim`** — deterministic Lighthill EBT locomotion (reactive thrust, Garrick lift, Lamb added-mass; energy closes `P_in=T·U+P_wake`); Unity-light, RNG-free, no-`Time`, oracle-tested | The crown jewel. Reuse as THE locomotion kernel, called *live per-creature* (not just measuring straight-line cruise). |
| **`SimUnits`** — frozen Joule anchor, each constant recorded with its arithmetic + a bright line forbidding retuning | Keeps physical quantities honest; separates measurement from tuning. |
| **`BodyGraph`/`BodyGenome`** — open-ended part-tree genome + single `Measure()→Morpho` developmental walk + structural mutation | True genotype→phenotype development; home for exaptation. Make `_geneRng` instance state, not static. |
| **Field grids** — `WorldSampler` (stateless per-seed world fn), `PlanktonField`, `ZooplanktonField`, `MarineSnowField`, `NutrientField`, `VentMat`, advanced by `TickRows(dt)` | Orthogonal, deterministic, headless-ready. `WorldSampler` is the cleanest thing in the codebase. |
| **`Ent`** — plain-data entity record (no GameObject/Rigidbody) | The natural nucleus of a proper `ColonyState`. Keep data-oriented; no creature class hierarchy. |
| **`Taxonomy` + `TaxonomyNames`** — emergent rank ladder + procedural Latin/Greek names, RNG/clock-free | Purely observational, zero-coupling, drop-in legibility layer. |
| **`SimClock`** — Unity-light single-writer time authority (instance, not static) | Make it THE sole time authority from day one. |
| **The conserved-economy DESIGN** — single limiting nutrient, closed loop (Liebig min × Monod × Redfield × Martin × Ekman); remineralization IS bacterial (BGE→microbe biomass, (1−BGE)→dissolved nutrient) | The principled spine + near-term target. Build it ACTIVE and conservation-tested, not gated off. |
| **Conservation-invariant TEST pattern** (inventory = dissolved + biomass + snow, constant; debit==credit; row-sliced == whole-grid) + determinism tooling (BitConverter identity, FNV-1a goldens, fakes-only EditMode tests) | The CORRECT primary correctness guard — replaces byte-goldens as the top gate. |
| **Talos EntitySpec core/ext manifest + read-only `IColonyView` observation seam** | One control/observation interface spanning sim creature + real robot; one observation investment serving verification + legibility + future agent observation. |
| **Pure math/trait helpers** — `AllocMath`, `NutrientChem`, `DriveMath`, `SpatialHash`, `MutationNoise`, predation pure-statics, the `TraitRegistry` descriptor pattern + `ICapabilities`/`IController` organ seam | Deterministic, unit-testable, portable. Lift them OUT of the god-class into a standalone core. |

---

## 4. Day-one architectural decisions the restart must force

1. **Pure headless Core, Unity as a thin adapter (non-negotiable).** A `ColonyCore` C# library with
   ZERO `UnityEngine` refs, exposing `Step(dt)`/`Reset(seed)`, constructible & steppable with no
   scene. *If it can't run in a console app with no Unity, the restart has already failed.*
2. **Layered asmdefs, one-way dependency direction.** Core (no Unity) → Domain math → World/fields →
   Analytics/Observation (read-only) → Unity Render/IO adapter → Tests. Recoupling = a compile error.
3. **Conservation invariant is the top CI gate, not byte-identity.** Features land ACTIVE & faithful;
   "the books close" is the gate; determinism goldens are a re-capturable regression check.
4. **One unified body substrate + one force law for every creature.** Collapse the LOD proxy; keep an
   aggregate path only behind a flag as an ablation, and *measure perf at full population* before
   re-adding any proxy.
5. **Kill the dual genome — derive function from form.** Feeding (gape ∝ L²), metabolism (Kleiber),
   defense (bulk), locomotion all from `BodyGraph` morphology; heritable "traits" are physical
   descriptors + evolvable couplings, never free benefit/cost knobs.
6. **Instance-scoped clock + RNG, no global statics in Core.** Prerequisite for N parallel envs +
   headless reproducibility; strict RNG draw-order discipline; deterministic build order (no
   background-thread races).
7. **Reproduction/species model that can actually speciate.** Decide up front: a mating/recombination
   model with genetic-distance-gated reproduction (enables isolation) vs. asexual fission as a
   *conscious documented ceiling.* Speciation cannot be bolted on later.
8. **One immutable, validated config record** passed into the Core constructor; the scene carries
   none of it.
9. **RL/embodiment seam defined now, ROS2/Talos built later.** `Step/Reset` env API + read-only
   observation view + Talos core/ext EntitySpec as boundaries on day one. Do NOT build ROS2/Talos
   until the ocean closes its books AND the "Sophia action interface: symbolic vs continuous"
   assumption is verified against real Sophia code (flagged as the highest open risk).
10. **Scope discipline — depth-first, cap the doc-to-code ratio.** Nothing is "done" until it is live
    and its invariant is green. No gated inventory of inert seams. Close the books before breadth.

---

## 5. Proposed build slices (depth-first, in order)

- **S0 — Deterministic pure Core skeleton (no Unity).** `SimClock`, instance RNG, `Ent`/`ColonyState`,
  immutable validated `Config`, `Step(dt)`/`Reset(seed)`, conservation-invariant test harness.
  *Accept:* a trivial world steps bit-identically in a console app and closes an accounting identity.
- **S1 — Conserved single-nutrient economy (keystone).** Liebig×Monod drawdown, microbial-loop
  remineralization (BGE split), Martin profile, basic mixing. *Accept:* total inventory constant to
  tolerance over a long run; blooms/deserts arise from the loop, not Perlin. *Nothing proceeds until
  the books close here.*
- **S2 — One canonical body + live locomotion.** `BodyGraph`→single `DevelopedBody`→`SwimEval.Sim`
  for EVERY creature; feeding/metabolism/defense derived from morphology (kill `eff[]`). *Accept:*
  two differently-shaped individuals swim/feed differently; no scalar proxy; loco energy conserved.
- **S3 — Feeding, metabolism, reproduction on conserved energy.** Holling-II intake with assimilation
  loss egested to detritus; Kleiber metabolism; reproduction paying real construction cost. *Accept:*
  a stable-ish population from energy→reproduction alone; energy conserved end-to-end.
- **S4 — Predation as a staged contest between two bodies.** find→close→seize→consume via
  form+physics; size-selective; strictly conserved. *Accept:* a trophic pyramid + arms race WITHOUT
  seeding a predator tribe.
- **S5 — Speciation, mating, taxonomy.** A reproduction model permitting isolation (recombination +
  genetic-distance gating); `Taxonomy`/`TaxonomyNames` as pure legibility. *Accept:* persistent,
  distinct, niche-adapted lineages that hold apart over time.
- **S6 — Transport, fields, world coupling.** Currents advect nutrients/plankton/detritus; Ekman
  upwelling; `WorldSampler` topology. *Accept:* nutrients upwell, blooms advect, flux attenuates.
- **S7 — Render + observation adapters (Unity).** Thin render layer (avatars that NEVER feed fitness);
  read-only `IColonyView`; rebuilt decoupled HUD. *Accept:* render off changes nothing; same numbers
  headless and in-editor.
- **S8 — RL/embodiment loop (the deferred payoff).** Headless gym-style env; observation/action
  interface; Talos EntitySpec; then ROS2 bridge + sim-to-real TurtleBot3 → CWM-G. *Gated on:* S1–S4
  books closed AND the Sophia action-interface assumption verified.
- **S9 — Plants-as-organisms + the sea-to-land crossing (emblematic endpoint).** Only after the ocean
  is faithful and stable. Hardness ramps up — this is the north-star demo, not an early milestone.

---

## 6. Open questions for the design dialogue (the genuine forks)

1. **Restart strategy:** clean rebuild that PORTS the proven pure pieces (recommended) vs. surgical
   in-place migration in the old repo behind the `IAvatar` seam vs. from-scratch (rewrite even the
   crown jewels)?
2. **Engine posture on this Windows/Unity machine:** are we committed to the pure-C#-Core-first
   workflow (write/test lots of code *outside* Unity), with Unity demoted to an adapter?
3. **Reproduction/speciation:** sexual/recombination from the start (enables true isolation, more
   complex) vs. asexual as a conscious temporary ceiling?
4. **Visible-life tolerance:** the depth-first path means no creatures swimming on screen for several
   slices. How much invisible-foundation time is acceptable before you need to *see* something alive?
5. **Sophia interface reality:** how much is actually known/decided about Sophia's action &
   observation interface today (symbolic vs continuous), since it gates the whole embodiment seam?
