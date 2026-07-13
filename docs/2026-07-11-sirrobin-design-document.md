# SirRobin — Technical Design Document

**Status:** Draft for implementation • Consolidated master spec • **Date:** 2026-07-11
**Register:** Implementation specification (normative). Not a vision essay — every claim is intended to be buildable and testable.

> **2026-07-12 authority amendment.** The consolidated S0 plan plus population-grounded Gate-E revision govern
> S0. The consolidated S1 conserved-nutrient plan governs S1 and supersedes older float-ledger, reservoir,
> energy, parcel-timing, and package-layering prose below. S0 is complete: the original 1,000/90M gate remains
> NO-GO and the pre-registered 5,000/10,000 population gate is GO. S1 is also implemented and records GO in
> `docs/superpowers/reports/2026-07-12-sirrobin-S1-decision-report.md`; S2 is next.

---

## Scope

This document specifies **SirRobin**, a continuously-simulated, mass- and energy-conserving ocean-life evolution simulator, in enough detail for an engineering team to build it. It defines the computational substrate, the world model, the physics/locomotion solver, the genome→development→phenotype pipeline, the conserved ecological economy, the verification discipline, the build roadmap (S0–S9), and the day-one embodiment seam through which an external agent will later drive a creature. It is written for the engineers who will implement the array engine, the CI gates, the S0 spike, and the state contract.

The document assumes familiarity with PyTorch tensor programming, GPU determinism constraints, and undergraduate fluid mechanics/biogeochemistry. It **assumes** the following decisions are settled and does not re-litigate them: PyTorch as the single hot-loop runtime; conservation (not byte-identity) as the primary correctness gate; reactive (non-neural) behavior for the near term; one dense all-ocean world near-term with embodiment and land deferred. Genuinely open items are collected, with owners and blocking status, in the final register. Throughout, each capability is labeled by **risk class** — *solved technique* (established, cited; risk = correct implementation) or *uncertain/gated* (the outcome is a research frontier even though the mechanism is solved; each carries a falsifiable milestone).

---

## Table of Contents

1. [Vision, Scope & Design Principles](#1-vision-scope--design-principles)
2. [Architecture & Substrate](#2-architecture--substrate)
3. [The World Model](#3-the-world-model)
4. [Physics & Locomotion](#4-physics--locomotion)
5. [The Genome & Evolution](#5-the-genome--evolution)
6. [Ecology & the Conserved Economy](#6-ecology--the-conserved-economy)
7. [Verification, Roadmap & the Embodiment Seam](#7-verification-roadmap--the-embodiment-seam)
8. [Consolidated Risk & Open-Questions Register + What's Next](#8-consolidated-risk--open-questions-register--whats-next)

---

## 1. Vision, Scope & Design Principles

### 1.1 Purpose

SirRobin is an ocean-life **evolution simulator** whose terminal purpose is to serve as a **non-text grounding substrate** for an embodied autonomous agent ("Sophia"): a world with genuine causes in which a policy learns from survival consequences rather than from tokens. The near-term deliverable is **not** the agent — it is a world *faithful enough to be worth learning in*, built so as not to foreclose that agent.

The agent couples to the sim through a **Talos** hardware-abstraction layer (HAL) exposed over **ROS2**. The sim publishes senses and accepts actions across a versioned, serializable **state contract**; Sophia's entire neural stack lives *behind* Talos, outside the sim. The same contract is designed to transfer to a physical **TurtleBot3** (sim creature and robot = one "mobile sensate navigator"). The contract schema is specified in §7.4.

**Embodiment is deferred to slice S8**, but the seam is kept open from day one as a hard architectural constraint:

- The canonical population layout — struct-of-arrays torch tensors shaped `(W, N_cap, k)` with a boolean alive-mask (§2.4) — **is** the batched observation tensor the Talos contract exposes. There is no separate "RL representation" to build later.
- The **CORE action** is the 2-vector `{surge_effort ∈ [-1,1], yaw_rate ∈ [-1,1]}` — the differential-drive-executable intersection of a fish and a TurtleBot3 — and it is *validated by the physics, not merely asserted*: the locomotion kernel integrates yaw only and zeroes vertical COM velocity, so `{surge, yaw}` **is the fish's actually-realized degrees of freedom** (§4.4, §7.4).
- **Behavior in the evolving population is reactive** (hardwired drives / gradient-following) for all of S0–S7. Evolved neural "minds" are out of scope until the body and ecology are faithful. Consequence: the genome is **body-only**, and PyTorch is used as a pure vectorized array engine (`inference_mode`/`no_grad` in the hot loop, no autograd).

Nothing in S0–S7 may require Sophia to exist; nothing in S0–S7 may make Sophia impossible to attach at S8.

### 1.2 Fidelity philosophy (an engineering constraint)

**Abstract in MECHANISM, faithful in CAUSALITY.** "Abstract the world" is a directive to simplify *how* a process is computed (few fields, few sources, reduced-order dynamics), never to fake *that* an effect is produced by its cause. The product is causal fidelity; visual or mechanical richness is not.

> Every observable signature (a bloom, a desert, a size refuge, a speciation event, a streamlined body) must be **DERIVED from conserved relational constraints acting on state**, never **IMPOSED by a tunable knob** whose only job is to produce that signature.

This is the specific discipline whose violation destroyed the prior build. There, faithful mechanisms shipped *disabled* behind `gain = 0` / `couple = 0` dials so that byte-identical determinism goldens stayed green — the live simulation ran the *least* faithful configuration, and a scalar `e.speed` capability was laundered through `swimGain`/`locoCoef`/`formSpeed` Lerps and a `speedFloor..speedCeil` clamp into the number that actually drove the world. Those laundering terms are exactly what this constraint forbids.

**Anchored-measurement corollary.** Unit bridges (e.g. the frozen sim-energy↔Joule anchor derived from a real marine-ectotherm SMR band; `N = 300 J`, `η_muscle ≈ 0.20`, `KgPerSimMass = 250`) are *measurements*, not balance knobs. They are frozen behind a bright line and never retuned to hit a behavioral target. A collapse or an implausible equilibrium is a **diagnostic that fidelity was dropped upstream**, to be root-caused — never softened by adjusting an anchor.

### 1.3 Scope

**In scope, near-term (S0–S7):** one **small, dense, periodic (wrap-around), all-ocean** world with no land; a single-limiting-nutrient conserved economy; one canonical body per creature with live morphology-derived locomotion; feeding / metabolism / reproduction / predation on conserved energy; emergent speciation via innovation-aligned crossover and spatial assortative mating; abstracted currents/weather transport; a headless telemetry surface plus a remote/replay viewer. **GPU from day one**, batched over one world's population. The lever is **density, not size**, because density drives the encounter / compete / hunt / **mate** interactions that make ecology — and especially speciation — emerge.

**Deferred (with the seam kept open):** RL/embodiment (S8); plants, land, rivers, sediment, coastline, and the water↔land crossing (S9). All land-dependent features defer *together*. Many-parallel-worlds batching defers to S8. Evolved neural cognition is out of near-term scope entirely.

**Explicitly out of scope as emergent phenomena** (seeded or parameterized instead): origin of life, chemistry-from-physics, geology-from-physics. A proof of *unbounded* open-endedness is an unsolved research problem; the design target is **sustained adaptive radiation**, not a proof of unboundedness.

### 1.4 Design principles (normative rules)

Each principle is a rule the implementation MUST satisfy. These are the **canonical home** for the cross-cutting laws; later sections reference them by number rather than restating them.

| # | Rule (normative) | Rationale | Concrete prior-build failure prevented |
|---|---|---|---|
| **P1** | **Conservation.** No channel mints or destroys mass/energy. Every process moves conserved quantity between **tracked reservoirs**; the books close to a *relative* tolerance every tick. This is the **top CI gate** — an invariant, *not* byte-identity. | You cannot calibrate or learn in a world that violates first-law closure. | Two free-energy sources (uncoupled grazing + uncoupled vent flux); a one-way pump where dead biomass "remineralized into nothing"; a static-Perlin producer cap consuming no pool. |
| **P2** | **Form-is-function.** Every capability — thrust, drag, metabolic rate, feeding rate, defense, ground reaction — is **derived from morphology run through real physics**, never read from a stat vector. | Shape must be the *reason* a creature can act, so selection acts on real bodies. | The dual genome: an open-ended body graph bolted to a parallel 15-slot `eff[]` stat vector, mutated as independent non-morphological knobs. |
| **P3** | **Single canonical representation per quantity.** Exactly one stored representation of any quantity. A second is permitted **only** for a genuinely *distinct kind*, behind a strictly **one-way** interface, with **zero synchronization code**. | "More than one way of representing the same thing is the beginning of the problem." | Four builders each developing a different body from one genome; scalar `e.speed` re-tethered by glue; `SpeciesSwimCache` centroid re-tether. |
| **P4** | **Clean abstraction boundaries.** A consumer queries **WHAT** a subsystem provides (sampled at a position/time), never **HOW** it is produced. Interfaces are designed around output categories; implementations start trivial and grow faithful with zero downstream change. | Leak-free boundaries keep complexity *behind* them. | A 3,067-line `OceanColony` god-class in one undivided assembly; the LOD proxy bleeding into the economy's cost/feeding terms. |
| **P5** | **Continuous, not a discrete grid that leaks into biology.** Creatures and reactive parcels are point-entities at **continuous float positions** sharing one spatial hash; field sampling is **interpolated** (smooth value + gradient); uptake is a continuous rate (∝ local conc × dt). A grid may *store* smooth abiotic fields, but discreteness must not reach the biology. | Ecology needs smooth gradients to climb; the world must not feel quantized. | Cell-wise `grazeClaim`/`snowClaim` shares, cell-bucketed `preyMass`/`predPressure` — "a discrete system when I really needed continuous." |
| **P6** | **Start simple, grow on demand.** World richness tracks what the biology demonstrably needs. A subsystem is added only when a gradient the evolution needs demands it. | Complexity added speculatively is complexity that rots. | Terrain/atmosphere/geology built before the nutrient books closed. |
| **P7** | **Depth-first — close the books before breadth.** One conserved loop working, tested, and closing to tolerance before the next layer is built. Slices are gated: nothing proceeds until the current slice's invariant is green. | Fast, honest feedback on a closed loop. | Breadth-first accretion of half-finished systems tuned live against each other with no closed invariant. |
| **P8** | **Implicit selection only.** The *only* fitness signal is survival-to-reproduction. No fitness term references a designer's target morphology, behavior, or outcome. | Emergence must be earned by the world's gradients, not steered toward an authored answer. | `formSelectionStrength` — a birth-chance multiplier keyed to proximity to hand-drawn morphological archetypes. |

**P3 anti-pattern (the diagnostic tell).** The sin is not ">1 representation" absolutely; it is **>1 representation OF THE SAME QUANTITY**, whose unmistakable signature is **synchronization glue** — code whose job is to keep copy A consistent with copy B. *If you are writing code to keep two representations of one thing consistent, you have already lost.* Reviews and CI treat sync glue as a defect. This converges with P5: representing all discrete/reactive stuff (creatures, plankton, dissolved-nutrient parcels) as point-entities in one continuous coordinate space sharing one spatial hash satisfies P3 and P5 simultaneously. The one admitted distinct kind is the smooth abiotic background (temperature/currents/light) behind a one-way read-only interface with no sync.

### 1.5 The process risk these rules mitigate: recursive scope explosion

The prior build did not fail on any single equation; it failed on a **process pathology**. Each new idea silently spawned a hidden sub-project (a nutrient cycle, a genome upgrade, a predation model, a terrain system), each "secretly a project," each **bulldozed into place** on top of an unclosed foundation. Complexity compounded recursively until the codebase became soup — a god-class, dual genomes, laundered scalars, disabled-faithful paths — none removable without breaking the others.

P1–P8 are precisely the mitigations:
- **P4 + P3** bound each sub-project *behind an interface* so it cannot leak; a subsystem can be trivial-now / faithful-later without touching consumers.
- **P6 + P7** forbid opening a new sub-project until the current one's books close, so scope cannot recurse.
- **P1 + P8** give every slice a single, unambiguous acceptance gate (conservation closes; selection is survival-only), so "done" is measurable, not negotiable.

A new capability is admitted only when (a) its interface is defined around output categories, (b) the prior slice's invariant is green, and (c) it earns its keep via a specific gradient the evolution needs. "Each secretly a project" is a *warning*; such projects are added **one at a time, late, each only when it pays for itself.**

### 1.6 The emblematic goal: the bidirectional water↔land crossing

The long-horizon target is **not** a one-way "sea-to-land" finish line. It is: *life crosses the water–land interface freely, in BOTH directions, repeatedly.*

- **sea → land:** a swimming lineage evolves ground-reaction locomotion (fin → leg).
- **land → sea:** a terrestrial/amphibious lineage RETURNS to water and re-adapts (legs → flippers, streamlining) — the whale/dolphin/seal/ichthyosaur/sea-turtle pattern.

Design consequences (these constrain the Genome §5, Physics §4, and World §3 sections coherently):
1. **Evolution is not a ratchet.** The boundary is a permeable, two-way corridor. The genome must stay **reversible / exaptation-friendly**: no irreversible ratchets; a `Segment↔Surface` flip plus neutral drift plus parametric re-adaptation must let a walking limb drift back toward a swimming surface (§5.5, §5.9).
2. **Exaptation is the lever.** The *same* limb structure is re-purposed medium-to-medium (leg ↔ flipper) with no new body plan.
3. **Convergence is a directly observable falsifiable signature that physics drives form.** Independent lineages re-deriving streamlined aquatic forms is a measurable outcome, not a designed one (§5.9).
4. **The physics supports return "for free."** Locomotion is architected as **additive medium-dependent force-contributors on articulated bodies** (§4.5): one physics, position decides the medium, no mode switch. There is no separate "walking mode" to write, so re-entry to the sea costs no new machinery. The World presents the boundary as a continuous medium gradient, not a switch (§3.11).

This remains **S9-era and a research frontier** (uncertain/gated). Capability (genome + physics) is necessary but not sufficient; whether the crossing *occurs* also requires the ecological gradient. The vehicle is built to *reach* the endpoint; whether it *arrives* is the honest long-horizon question.

### 1.7 The namesake: the sea robin

The project's emblematic creature — "**Sir Robin**" ← **sea robin** — is the searobin/gurnard (family Triglidae): a fish that **walks along the seafloor on modified lower pectoral fin-rays** that serve simultaneously as leg-like appendages *and* as chemosensory organs, while still swimming with its fins. It is the concrete emblem of three core theses at once:

- **Form-is-function (P2):** the same appendage does swimming thrust and ground-reaction walking with **no mode switch** — exactly the additive medium-dependent physics of §4.5.
- **In-water fin→limb exaptation:** fin→leg is a *smooth, in-water co-option* — evidence the transition is a gradual ramp, not a cliff, needing neither air-breathing nor leaving the water.
- **An achievable near-term milestone that de-risks S9:** a benthic fish that walks on re-purposed fins is reachable *inside the water*, well before the full crossing. Its acceptance criteria are specified in §5.10 and §4.6.

---

## 2. Architecture & Substrate

This section specifies the computational substrate: the array engine, package layering and enforced boundaries, the canonical in-memory representation, the environment API, and the determinism/precision posture. These decisions dissolve the prior build's fatal architecture wound and are the most expensive to revisit later.

### 2.1 Substrate: one unified vectorized array engine

**Decision.** The Core is a single **PyTorch** program. All world state — creature kinematics, articulated-body physics, abiotic fields, genome tensors, reactive resource parcels — lives in `torch.Tensor` objects advanced by batched tensor ops. There is no second numerical runtime in the hot loop. `numpy` is permitted only as an offline oracle cross-check. **JAX and NVIDIA Warp are escape hatches**, invoked only if an S0-measured throughput/determinism wall cannot be cleared in torch (Warp is the lighter hatch, for a single branchy per-body kernel, before any full JAX rewrite).

**Rationale.** The prior build's root architectural wound was a data-far/physics-near **LOD proxy**: a scalar `e.speed` synced to a real swimmer body by re-tether glue, existing *only* because running faithful physics for the whole population seemed unaffordable. That one unverified premise spawned every laundered scalar downstream. Vectorizing the true per-body physics for all `N` creatures as batched tensor ops **removes the reason the proxy exists**. Because this is an *optimistic* performance premise, it is not assumed — **S0/SpikeSwim verifies it with measured throughput gates before the architecture is bet on it** (§7.2).

torch (not numpy) is the substrate because: (a) `device=` portability — identical code runs CPU or CUDA by one knob; (b) `torch.compile` / CUDA-graph capture kill per-op dispatch overhead once the batched math is correct; (c) a single stack that also hosts any future neural minds. The population is **reactive** for now, so the hot loop runs under `torch.inference_mode()` with no autograd.

### 2.2 Package layering and enforced import boundaries (P4)

Python has no compile-time firewall, so boundaries are encoded by `import-linter` contracts. The capability
graph has independent siblings rather than a fictitious total ordering:

```text
numerics <- physics
numerics <- fields <- economy
numerics <- genetics
core later composes public sibling contracts; observe reads public contracts
```

| Package | Owns | May import | Must NOT import |
|---|---|---|---|
| `numerics` | dtype/device policy, quaternion & 3×3 solves, segmented reductions, RNG manifest, compensated-sum ledger | (stdlib, torch) | anything below |
| `physics` | articulated-body dynamics; additive force-contributors; `DevelopedBody`, `MediumSample`/`FluidSample` contracts | `numerics` | `fields`, `genetics`, `core` |
| `fields` | grid geometry, generic `FieldSample`, interpolation, conservative transport | `numerics` | `physics`, `economy` internals, `genetics`, `core` |
| `economy` | exact conserved reservoirs, transactions, reactions, ledger, snapshot | `numerics`, public `fields` | `physics`, `core`, `observe` |
| `genetics` | genome tensors, development scan → `DevelopedBody`, mutation/crossover/distance | `numerics`, `physics` (contracts only) | `fields`, `core` |
| `core` | later `Colony` composition, whole-tick orchestration, spatial hash, sibling adapters | public contracts above | `observe` |
| `observe` | telemetry (parquet/CSV/heatmaps), the Talos state contract, the Unity/replay viewer feed | all above | — |

A violation (e.g. `physics` importing `core`) fails an import-linter check. `physics` owns its mechanical
`FluidSample`/future `MediumSample`; `fields` owns generic `FieldSample`. The future `core` composition root
samples fields and constructs the mechanical input, so neither sibling imports the other. This replaces the C#
assembly-definition firewall lost in the prior `OceanColony` god-class.

### 2.3 The entity model: point-entities in continuous space, on continuous fields (P5)

Two distinct **kinds** of thing (this is the one sanctioned P3 exception — genuinely distinct kinds):

- **Discrete, reactive, structured things — creatures and resource parcels — are point-entities in a *continuous* coordinate space, all sharing ONE spatial hash.** Position is a float vector, never a cell index. They differ only in payload and interaction rules; diffusion, feeding, and predation all reduce to one primitive: *local interaction between nearby entities*, resolved by neighbor queries against the shared hash.
- **The smooth abiotic environment — nutrient background, temperature, light, later currents/terrain/atmosphere — is a set of *continuous fields*.** A field is mathematically a mass-weighted density `M(t)·p(x)` (a scalar mass tracked exactly + a shape that only redistributes); it is stored concretely as a high-resolution Eulerian grid **queried by interpolation** (smooth value + gradient at the exact entity position).

**Continuity is a hard law (P5).** Field reads are **interpolated** (no cell-edge jumps); uptake is a **continuous rate** `∝ local_conc · dt` (not a cell-share chunk); encounters are **continuous** via the spatial hash. The grid only *stores* smooth fields; it never bounds an interaction. The spatial hash quantizes nothing — it is a neighbor-lookup accelerator invisible to physics. The two "grids" (field grid vs spatial hash) are architecturally independent and never conflated (§3.2).

*The field discretization for reactive resources (Eulerian-interpolated grid vs. Lagrangian parcels-as-entities vs. hybrid) is an **S1 decision measured empirically** (§3.1 fork, §6.2). The escape hatch that keeps dense grazing free is putting consumption on the parcels (an eaten parcel is simply gone — depletion holes are never stored). The abiotic property-of-space fields (temp/currents) remain the one grid-backed kind.*

### 2.4 The canonical batched representation

Struct-of-arrays torch tensors, static shape, boolean masks. Two levels: **creature-level**
(fixed-capacity population) and **segment-level** (fixed per-creature slots). The capability-based module and
tensor names are durable; milestone labels such as S0/S1 never become runtime namespaces.

#### 2.4.1 Creature level — `(W, N_cap)` with alive-mask and dead-slot recycling

```
alive        : bool   [W, N_cap]           # the single source of truth for "exists"
pos          : f32    [W, N_cap, 2]         # continuous world position (periodic box)   [m]
heading      : f32    [W, N_cap]            # yaw                                          [rad]
lin_vel      : f32    [W, N_cap, 2]         # surge velocity in world frame               [m/s]
ang_vel      : f32    [W, N_cap]            # yaw rate                                     [rad/s]
energy       : f32    [W, N_cap]            # metabolic reserve (conserved)               [J]
nutrient     : f32    [W, N_cap]            # structural nutrient bound in tissue         [mol]
genome_ptr   : i64    [W, N_cap]            # index into the genome tensors (§5.2)
age          : f64    [W, N_cap]            # biological age                              [s]
species_tag  : i64    [W, N_cap]            # observational read-out only, never gates mating
```

- `W` = number of worlds (near-term `W = 1`; the dimension is retained so S1 per-world fields and S8 vectorized-envs need no re-layout).
- `N_cap` = fixed maximum population per world. `k` extra scalar channels are appended per slice as slices land; the shape family stays `(W, N_cap, ·)`.
- **Births/deaths mutate `alive` and recycle dead slots** — a death frees its index; a birth claims a free index by fixed-order `cumsum`/gather/where (never a nondeterministic atomic append). Tensor shapes and addresses therefore stay **static**, required for `torch.compile` / CUDA-graph stability and mirroring what a JAX rewrite would force.
- **Dead rows are masked, never index-selected out.** Every reduction applies `alive`; boolean `index_select`
  is forbidden in the hot loop because it changes shape.
- **No periodic compaction occurs in the canonical path.** Dead rows are overwritten in place on birth; a
  flattened arena is considered only in a later measured optimization slice.

#### 2.4.2 Segment level — fixed `[W,N_cap,17]` slots plus mask

Each body has 17 fixed slots: slot 0 is a finite identity sentinel and slots 1..16 are real segment capacity.
`seg_mask` is the sole source of segment existence. The fixed layout deliberately pays masking cost to preserve
static shapes and addresses; S0 measures that cost on heterogeneous H1/H2 populations. Flattened/CSR storage is
a deferred optimization, authorized only if profiler attribution and lifecycle-inclusive measurements justify it.

```
seg_mask       : bool[W,N_cap,17]
seg_localPos   : f32 [W,N_cap,17,3]
seg_localRot   : f32 [W,N_cap,17,4]
seg_abc        : f32 [W,N_cap,17,3]
seg_mass_sim   : f32 [W,N_cap,17]       # structural sim-mass; multiply by 250 once
seg_ma         : f32 [W,N_cap,17,3]     # added-mass principal terms [kg]
seg_parent     : i16 [W,N_cap,17]       # creature-local; every root points to sentinel 0
seg_depth      : i8  [W,N_cap,17]       # 0..MaxDepth=5; sentinel excluded by mask
seg_ampDeg     : f32 [W,N_cap,17]
seg_phase      : f32 [W,N_cap,17]
seg_isSurface  : bool[W,N_cap,17]
seg_isTail     : bool[W,N_cap,17]
```

Plus per-body scalars `tail_slot`, `swimFreq`, and `swimWave` on `[W,N_cap]`.

- **Per-body reductions** (COM, the 6 unique added-mass `M_eff` entries, axial quadratic drag, yaw inertia) are masked sums over the fixed slot axis. No `body_id`, `index_add_`, `scatter_add_`, compaction, or pointer repair exists in the canonical path.
- **Pose** resolves as a bounded **6-pass depth-scan** (MaxDepth=5, one extra pass for the root): each pass is a masked batched quaternion compose of child-onto-parent. Fixed pass count ⇒ static shape ⇒ no data-dependent loop bounds. This is the *same* gather→compose→scatter kernel shape as the genome development scan (§5.3), so the two share an implementation.
- H1/H2 occupancy timings quantify the fixed-slot cost. A later flattened prototype must beat this complete
  lifecycle-inclusive baseline before it may replace the canonical layout.

#### 2.4.3 The observation tensor IS this layout

The `(W, N_cap)` + alive-mask creature structure **is** the batched observation tensor the Talos state contract exposes (§7.4). There is no second population representation for the RL seam — the sim publishes senses by slicing the same canonical tensors. Designing a separate observation buffer would be exactly the duplicated-and-synced copy P3 forbids.

### 2.5 The single-representation law as an architectural constraint (P3, restated concretely here)

The prior build had no canonical body (four builders each produced a different animal from one genome), a **dual genome** (form in `BodyGraph` + a parallel 15-slot `eff[]` stat vector kept in sync at every birth), and the LOD proxy. Each was one quantity represented twice with glue. Consequences enforced here:

- Capabilities (feeding, locomotion, defense, energetics) are **derived from morphology through physics on demand** (P2), never stored as a stat vector. The `eff[]` vector is **deleted, not migrated** (an explicit S2 acceptance gate — "kill `eff[]`").
- The genome is the **single** source of body structure; the `DevelopedBody` is a pure fixed-shape *function* of it (deterministic and re-derivable, not stored-and-synced).
- A render mesh, the observation tensor, and the taxonomy species-tag are all **derived read-outs**, never authoritative stores.
- The one sanctioned second representation is the smooth abiotic background field (§2.3), admitted behind a one-way read interface.

### 2.6 Environment API and sim-owned time

The Core exposes a Gymnasium-shaped interface. Time belongs to the sim, not to any wall clock or engine frame.

```python
class Colony:
    def reset(self, seed: int) -> Observation:
        """Deterministically construct initial state from (seed, Config).
        Re-seeds torch/numpy/python RNG; resets the SimClock to t=0.
        Returns the initial Observation (the §2.4 canonical tensors)."""

    def step(self, dt: float, action: Action) -> Observation:
        """Advance one simulation step of `dt` sim-seconds under `action`.
        Pure function of (prior ColonyState, dt, action) — no global mutable
        state, no wall-clock read. Returns the post-step Observation."""

    def state(self) -> ColonyState:          # fully serializable snapshot
    def load(self, s: ColonyState) -> None:  # resume/migrate/checkpoint
```

- **`Observation` / `Action`** are dict-of-tensors with a leading `(W, N_cap)` batch dim + alive-mask, in nested CORE/EXT structure (§7.4). SI units throughout; body frame FLU per ROS REP-103 (x-forward, y-left, z-up); radians; seconds.
- **Sim-owned time (`SimClock`).** Simulation time flows from a single clock the Colony owns: `Now` (f64 accumulated sim-seconds — the biological timeline), `Dt` (f32 sim-seconds this step), `Step` (i64), `Scale` (sim-seconds per driver-second). Exactly one writer (`step`), every consumer read-only; all rate integrations are `rate · Dt`. **Biological time is a free quantity the sim sets, never inherited from a render/engine frame** — the prior build aged creatures in literal wall-clock seconds (a "45-second lifespan" no one chose). `Now` is f64 because biological time reaches months–years (~10⁷ s), where f32 resolution exceeds one second. The clock state `(Now, Dt, Step, Scale)` is part of the serializable snapshot; a run is reproducible from `(seed, Config, Scale-schedule)`.
- **No global mutable state.** The Core runs headless (no GameObjects, no `FixedUpdate`, no scene singletons). `ColonyState` is fully serializable for snapshot/checkpoint/resume/migrate on any CPU/GPU node.

### 2.7 Determinism posture

(This is the **canonical home** for determinism; §4.8, §5.3, §6.6, §7.1–7.2 reference it.)

- **Primary correctness gate = conservation invariants (P1), not byte-identity.** The books close to a *relative* tolerance robust to sub-tolerance float noise. The prior build pointed its harness at byte-identity — faithful mechanics shipped disabled behind `gain=0` while goldens stayed green — an inversion explicitly rejected.
- **Tier 1, exact bookkeeping:** hard gate. Checked int64 reservoir transfers and book closure are exact.
- **Tier 2, reproducible discrete decisions:** hard gate. Counter-addressed integer randomness and fixed-order
  lifecycle claims reproduce for a fixed logical schedule.
- **Tier 3, bit-identical floating trajectories:** informational diagnostic. Same-device eager deterministic
  reruns report divergence and cost, but float identity does not authorize the scientific mechanics. Cross-device
  tolerance-based physical/oracle gates remain mandatory.
- The hot path contains no atomic segment reductions: fixed-slot masked axis sums replace `body_id` scatter.
- Every new heritable-but-inert gene **appends** to the RNG draw manifest, short-circuited to zero draws while inert, so adding latent genome capacity never shifts the deterministic RNG stream.

### 2.8 Precision — deliberate hybrid

(Canonical home; §4.8, §7.1.3 reference it.) Never blanket float64 (consumer RTX FP64 is a 32–64× throughput cliff). The hot loop is **float32** with *relative* conservation tolerances; **float64 is used in exactly three places:**

1. the one-time `LambK` added-mass k-factor precompute (the donor computes it in double; matching the oracle requires it);
2. the **energy/validation-arm ledger reductions and f64 diagnostics**, accumulated by compensated/pairwise summation — conserved **mass** reservoirs are exact **int64 quanta** and close by exact `==` (§6, INV-MASS), *not* a float64 ledger;
3. the S0 oracle-match configuration, to isolate "is the math correct" from float32 transcendental divergence.

Acceptance thresholds (measured in S0, carried forward): mixed dimensioned per-step force and `R_step`
tolerances; the executable 100,000-step prefix ratio `D_k=|ΣR|/Σ|scale|` stays below `1e-3` (f32) or
`1e-6` (f64) after 100 steps, with monotone signed bias reported separately; oracle single-step force terms
`<1e-4` relative with absolute floors; episode aggregates `<1e-3` relative.

### 2.9 Device knob, scale, and honest performance risk

- **`device=` is a knob on the immutable `Config`.** The identical batched code runs CPU or CUDA. **GPU from day one, scope = ONE large world** batched over its population; many-parallel-worlds batching is deferred to S8. GPU's payoff over multicore CPU **rises with population size**, so **S0/SpikeSwim measures the CPU↔GPU crossover `B*` at the real population size** and the device is chosen on numbers.
- **Start small and dense (P6/P7).** The near-term world is a small, dense, periodic, all-ocean box holding a modest-but-viable population (dense hundreds-to-low-thousands: enough standing variation to avoid drift/extinction). The lever is **density, not size**. Scale world and population up together, deliberately, once the loop works.
- **Honest performance framing (engineering risk, stated plainly).** The prior build hit ~3 fps at fewer than ~5,000 organisms — *not* evidence the physics is slow: profiling attributed the dominant spike to `InfiniteWorld.Update` (45–69 ms of terrain-chunk streaming + PhysX collision-mesh baking, "unrelated to creatures"), with the faithful physics math ~100–200× cheaper than the per-frame budget. Dying at a *low* count is the textbook signature of fixed **per-object overhead**. The original S0 floor of **9.0e7 creature-steps/s** at 1,000 live remains a recorded NO-GO. The subsequently pre-registered population-grounded gate requires 600k at 5,000 and 1.2M at 10,000; all H1/H2 cells passed. This authorizes locomotion viability, not a complete future tick budget. H0 never authorizes (§7.2).

---

## 3. The World Model

### 3.1 Scope and stance

The world is a **thin, conserved stage** whose only job is to generate the selective gradients evolution needs; sophistication is spent on the biology, not the abiotic model. Every subsystem obeys "abstract in mechanism, faithful in causality" (§1.2). The world is **field-first**: one static structural field is canonical, and terrain, geological sources, mineral distribution, and (later) land/rivers/sediment are all *derived readouts* of it, never placed by hand.

Three storage roles are distinguished without duplicating authoritative state: continuous-position point
entities; mutable conserved Eulerian reservoirs changed only by economy transactions; and exogenous abiotic
drivers sampled through one-way interfaces. S1's nutrient and biomass grids are the second role, not read-only
backgrounds. Concentration is always derived from int64 inventory and cell volume.

**Roadmap placement:** S0 had no world-field implementation. The exact conserved nutrient/light column is
implemented in **S1**; currents/weather/horizontal transport arrive at **S6**; the full land cascade and coastal
corridor are deferred together to **S9**. Everything below is designed for the S9 destination but implemented
trivially first, so the generator climbs without downstream change (P4/P6).

### 3.2 Coordinate system, units, and the field-grid vs spatial-hash distinction

World frame is ENU-consistent SI: **x east, y north, z up**, metres, seconds, radians, kelvin. Surface `z = 0`; seafloor `z = −H(x,y)`, `H > 0` = depth. The horizontal domain is a **periodic box** of side `L` (wrap-around in x and y). Positions are `float32` in the hot loop.

Two grids hide under the word "grid"; they are architecturally independent and must never be conflated (this conflation quantized the prior build's biology):

| | **Field grid (Eulerian)** | **Spatial hash** |
|---|---|---|
| Purpose | Store/evolve continuous abiotic fields | Neighbor lookup for point-entities |
| Resolution | `G × G` horizontal, `B` vertical bands | Cell size = interaction radius `r_int` |
| Quantizes biology? | **No** — biology reads via interpolation (P5) | No — invisible to physics |
| Canonical for | temp, current, light, nutrients, `h(x)` | nothing (pure index structure) |
| Tensor | `(W, G, G, B)` per field | `(cell_id → entity slot)` buckets |

Field storage tensors carry a leading world dimension `W` (near-term `W = 1`). A 2D field (structural field, elevation) is `(W, G, G)`; a depth-resolved field (nutrients, marine snow, temperature) is `(W, G, G, B)`; a purely analytic field (initial light, initial `h`) needs no storage.

**Initial parameters (tunable; pinned in `WorldConfig`):** `L ≈ 4096 m`, `G = 256` (→ ~16 m cells), `B = 16` depth bands, `H ≈ 300 m`. `r_int` is set by the ecology, not the field grid. These are placeholders until S0 hardware numbers land; fields are stencil workloads (the GPU's best case), so `G` can rise cheaply.

**Continuous sampling contract (INV-W3).** Every field exposes exactly one read primitive, and **all biological reads go through it**:

```
Field.sample(x: Tensor[..., 3]) -> (value: Tensor[...], grad: Tensor[..., 3])
```

`value` is (bi/tri)linear-interpolated at the exact continuous position; `grad` is the analytic gradient of that interpolant (smooth value and gradient, no cell-edge discontinuity). Uptake is a **continuous rate** `∝ local_conc · dt`. No consumer may index a field cell directly.

### 3.3 The canonical structural field

One static scalar field `Φ(x,y)` is the single source of truth from which terrain shape and all site placement derive. It **must carry internal structure to key off** — smooth featureless noise is forbidden, because ridges/faults/basins must be *readable*. Two admissible generators (a `WorldConfig` choice; both satisfy the contract):

- **Voronoi "plates".** `K` sites `s_k` scattered (periodically) in the box. For a query `x`, let `d1 ≤ d2` be the nearest and second-nearest (toroidal) site distances. **Edge-ness** `E(x) = clamp((d2 − d1)/w_edge, 0, 1)` (small near plate boundaries); **plate id** `argmin_k`. Plate edges = ridge/fault lines; interiors = basins.
- **Ridged multifractal noise.** `Φ = Σ_i a_i · (1 − |noise_i(f_i·x)|)`, whose sharp crests are ridges.

`Φ` is **static** (mass-free — carries no conserved quantity, never ticks); geological sources cycle on top of it, giving disturbance dynamism without world drift. It is queried, never exposed: no consumer sees sites, seeds, or octaves (INV-W2, INV-W4).

### 3.4 The terrain / geology interface (the one load-bearing decision)

The **interface**, not the generator, is the load-bearing terrain decision (P4). Consumers query *what geology provides at a position/time*, never *how* it is computed. Time is behind the interface: consumers re-sample "now" each tick, so moving hotspots and growing deltas need no special handling.

```python
class Geology(Protocol):
    def elevation(self, x_hz: Tensor[..., 2]) -> Tensor[...]: ...          # seafloor height z_bed [m], up positive
    def mineral(self, x_hz: Tensor[..., 2], element: int) -> Tensor[...]: ...  # dissolved conc sourced by geology
    def heat_sources(self) -> SourceSet: ...                                # ALL point-sources (active or dormant)
    def active_sources(self, t: float) -> SourceSet: ...                    # subset erupting/venting at t
    def river_sources(self, t: float) -> SourceSet: ...                     # river-mouth inputs (empty until S9)
```

`SourceSet` is a struct-of-arrays: `pos (M,3)`, `type (M,)`, `output (M,)` (heat/chemical/sediment flux, SI), `active (M,) bool`, `age (M,)`.

| Category | Query | Destination implementation | **S0/S1 trivial stub** |
|---|---|---|---|
| Geometry | `elevation(x)` | Voronoi base + hotspot deposition + subsidence | analytic `h(x)`: flat, or smooth bowl |
| Minerals | `mineral(x, e)` | derived from `Φ` (crust age, vent proximity) | uniform constant per element |
| Heat/chem sources | `heat_sources()` / `active_sources(t)` | vents on plate edges + hotspots, cycling | empty `SourceSet` |
| River/sediment | `river_sources(t)` | flow-accumulation over `h`, coastal mouths | empty `SourceSet` |

**Acceptance criteria (testable at S0):**
- **INV-W4 (opacity):** a CI import-boundary check asserts consumers import only the `Geology` protocol; no plate/seed/trajectory symbol is reachable.
- **Substitution test:** swapping the stub generator for the Voronoi generator changes *no* line in fields/ecology/physics and passes their unchanged tests.
- **Time-transparency test:** advancing `t` so an `active_sources` set changes (or a hotspot moves) requires no consumer-side branch.

### 3.5 Geological sources and the moving-hotspot speciation engine

Sources are a small, sparse set of discrete point-sources (the easy sparse regime). Their *placement* is a derived readout of `Φ` (vents concentrate along plate edges where `E(x)` is small, plus a few hotspots), so placement is geologically faithful, never arbitrary. Sources feed the **existing** fields via additive **source terms**:

- **Chemical + heat →** temperature and reduced-chemical (Fe, S) fields → a sunlight-independent **chemosynthetic niche** (a second food web on the seabed, before any land).
- **Deposition → elevation field** (secondary layer atop the `Φ`-derived base): active volcanoes/hotspots add mass to `h(x)`, building seamounts and islands.

**Moving hotspots** are the premier speciation engine and a specific requirement. A hotspot is a discrete source with a slowly-updating position `p(t)` sweeping across the *static* `Φ` (mathematically equivalent, for the track it lays, to a fixed hotspot under a drifting plate — isolating the useful island-chain effect without dragging the world or its biota). Trajectory is a `WorldConfig` choice: constant slow velocity (linear chains) or slow random walk (curved tracks).

**Aging/subsidence is mandatory — it is what makes the mechanism work, not merely look right.** Once the hotspot moves on, the island it left must **subside and erode** back toward the seafloor, giving the age progression (young high islands near the hotspot, old drowned seamounts trailing away) that drives serial allopatric isolation. The elevation update over a deposit column:

```
dh/dt = deposit(x,t)                      # active source adds mass (conserved vs sediment reservoir)
        − subsidence_rate · (h − h_base)  # relaxation back toward the Φ-derived base
        + κ_erode · ∇²h                    # erosion = slope-diffusion (mass moves to sediment reservoir)
```

This yields the Hawaii/Galápagos cycle — colonize the new island, then get cut off as it sinks — generating repeated adaptive radiation at near-lowest cost, staying field-first-clean. **Conservation tie-in (P1):** deposition, subsidence, and erosion are all *transfers* between the elevation field and a tracked geological/sediment reservoir, never mint/deletion.

### 3.6 Minerals and iron limitation

The mineral distribution is a derived readout of `Φ` plus vent proximity: rock type / crust age / distance-to-vent set where Fe, S, P concentrate. This is biologically load-bearing because **iron is a real limiting micronutrient** (HNLC / iron hypothesis: macronutrients replete, blooms fire only where Fe is available). Fe is sourced from deep vents and scavenged onto sinking particles, so surface water stays Fe-starved unless upwelling lifts it — making the vent system load-bearing for surface productivity and patterning *where* the ocean is productive → patchy selection → niches and speciation. The nutrient chemistry itself (Redfield/Liebig/Monod/Martin) is specified in Ecology (§6); the world model's responsibility is to *supply the geologically-patterned mineral source fields* via `mineral(x, element)`.

### 3.7 The land cascade (deferred to S9)

All land-dependent features defer together, added one at a time behind the same interface (field-first makes them safe to add late as further derived readouts):

- **Land** emerges from `sea_level` vs bathymetry: `is_land(x) = elevation(x) > sea_level`.
- **Rivers** via standard **flow-accumulation** over `h`: route each cell downslope, accumulate upstream area; high-accumulation channels are rivers; their coastal outlets populate `river_sources(t)` with nutrient + sediment + freshwater flux. River mouths become the richest ocean zones (estuaries/deltas), add a **freshwater↔salt gradient** (a real speciation axis), and the intertidal delta setting for the crossing.
- **Sediment/burial → a TRACKED reservoir (P1).** Burial is a **transfer** into a conserved sediment reservoir, never deletion (fixing the prior build's #1-class bug where buried matter vanished). Closed total = biotic + dissolved + marine-snow + sediment + geological. River sediment deposits at the coast via the same deposition-builds-elevation mechanism, closing the geology↔hydrology loop.

### 3.8 Atmosphere, currents, and the circulation ladder

Atmosphere/winds/weather/currents/clouds are simply **more coupled fields** read through the same `sample(x)` interface, evolved by abstracted fluid dynamics (advection + diffusion + coupling), no new paradigm. This is the causal engine of ocean fidelity: solar + rotation → winds → surface currents + mixing; currents **transport** nutrients/heat/plankton (upwelling blooms, gyres, fronts, dead-zone deserts) and disperse larvae (population connectivity vs isolation = a direct speciation lever).

This is the **compute-heaviest** part of the model, so it is abstracted hardest and the abstraction boundary saves it: ecology only reads current/temp/light *fields at a position*, so the circulation model climbs a ladder unnoticed by any consumer:

- **L0 — Prescribed (cheap; the S6 entry point).** Analytic wind-driven gyres from a stream function `ψ(x,y,t)`, giving a divergence-free horizontal current `u = −∂ψ/∂y, v = ∂ψ/∂x`; day/season solar forcing on light; a few hand-placed upwelling zones. No PDE solve.
- **L1 — Reduced fluid (emergent).** Barotropic quasi-geostrophic vorticity on the periodic box:
  ```
  ∂q/∂t + J(ψ, q) = F − r·∇²ψ + ν·∇⁴ψ,   q = ∇²ψ + β·y,   solve ∇²ψ = q − β·y each step
  ```
  Gyres, eddies, and fronts emerge from vorticity advection; GPU-friendly (advection + an FFT Poisson solve on the periodic box — deterministic on a fixed device). Shallow-water is the admissible alternative. **Upwelling** is the honest Ekman term: vertical velocity `w_e ∝ curl(τ/ρf)` from wind-stress curl — required because a divergence-free horizontal current produces zero upwelling by construction.
- **L2 — Fuller dynamics (multi-layer / primitive equations)** only if ever wanted, behind the same interface.

**Interlock with geology:** currents deflect around bathymetry, upwell along ridges, carry vent plumes and (later) river inputs. **Conservation in transport (INV-W5):** advection is implemented in **flux form** (conservative) with a positivity/flux limiter, so a transported field's total mass is exactly redistributed, never created, never negative.

### 3.9 Light and the vertical column (initial-world richness)

Even the all-ocean initial world carries a **vertical light/depth axis**. Surface irradiance attenuates by Beer–Lambert:

```
I(x, z) = I0(x, t) · exp(−k_att · (−z)),   z ≤ 0,   I0 = solar(lat, season, diel) · cloud_factor
```

with `k_att` in m⁻¹. This yields photic → twilight → deep → seafloor zonation: surface-weighted production → sinking detritus (the biological pump) → depth-structured niches. Light is analytic (no storage) in the initial world; it becomes a stored, cloud/current-coupled field at S6. Depth zonation plus available vents means both the **photic** and **chemosynthetic** food webs exist before any land.

### 3.10 The initial world specification (S0–S1)

A **small, dense, periodic, ALL-OCEAN box, no land** — density (not size) drives interactions; small runs fast enough to close the books and tune (P7).

- **Domain:** periodic box side `L ≈ 4096 m`; depth `H ≈ 300 m`; vertical bands `B = 16`.
- **Bathymetry:** stateless analytic `h(x)` behind the `Geology.elevation` stub — `H(x,y) = H0` (flat) or `H0·(1 − a·bowl(x,y))` (smooth bowl if a depth axis is wanted at S1).
- **Minerals/heat/sources/rivers:** stubbed — `mineral` uniform, `heat_sources`/`river_sources` empty.
- **Fields present:** analytic light `I(x,z)`; the S1 conserved nutrient reservoir + marine-snow pump (§6). No currents until S6 (or an optional prescribed L0 drift).
- **Field tensors:** `(W=1, G=256, G=256, B=16)` per depth-resolved field; `(1, 256, 256)` for 2D fields.
- **Determinism:** field solvers use deterministic reductions and FFT-based Poisson; no wall-clock time enters any update.

### 3.11 The coastal / intertidal zone — a two-way corridor (S9-era)

When land arrives (S9), the coastal/intertidal zone is a **bidirectional corridor** (§1.6), not a one-way finish line. The world must present the water↔land boundary as a **continuous medium gradient**, not a discrete switch, because the additive-force medium-dependent physics (§4.5) resolves swimming vs walking vs amphibious crawl purely from *position relative to the waterline*. The world's obligations:

- **A smooth waterline field / medium query:** `medium(x)` returns a continuous submersion fraction across the intertidal band, not a boolean, so contact/buoyancy/drag blend continuously.
- **A survivable ramp, both directions:** near-shore bathymetry must provide a gentle intertidal gradient (no fitness cliff) so the corridor is traversable sea→land (fish grows legs) *and* land→sea (an amphibious lineage re-adapts).
- **Two-way niche structure:** estuarine/delta productivity and the freshwater↔salt gradient give reasons to cross in both directions repeatedly.

The world does not *cause* the crossing — that emergence is the hard S9 frontier — but it must not foreclose it: the boundary is a permeable, continuous, two-way medium gradient.

### 3.12 World-model invariants (acceptance gates)

- **INV-W1 — Conservation (P1).** Every conserved reactive field stores its tracked mass as **int64 mass quanta** and closes **exactly** (`==`) per world, net of explicit tracked sources/sinks (§6, INV-MASS) — not a float64 relative-tolerance ledger. Static fields (`Φ`) carry no mass. Elevation change is matched 1:1 against sediment/geological reservoir transfer.
- **INV-W2 — Single canonical representation (P3).** Render mesh, slope, normals, `is_land`, waterline are all *derived* from `h`/`Φ`; none stored and synced.
- **INV-W3 — Continuity (P5).** All biological reads go through `sample(x)`; no consumer indexes a field cell; uptake is a continuous rate.
- **INV-W4 — Interface opacity (P4).** Consumers import only the `Geology`/`Fields` protocols (CI import-boundary check).
- **INV-W5 — Positivity.** Reactive concentration fields are `≥ 0` after every step (flux-limited advection; clamped remineralization deposit).
- **INV-W6 — Determinism (§2.7).** Reproducible-within-seed on a fixed device; deterministic field reductions, FFT Poisson, no wall-clock input.

---

## 4. Physics & Locomotion

### 4.1 Scope, position in the stack, and the one load-bearing invariant

The `physics` package is pure numerics: it consumes articulated bodies (geometry + pose) and fluid/medium samples, and produces **forces, accelerations, and an energy ledger**. It knows nothing of genes, creatures, feeding, or economy — those live upstream and reach physics only through two frozen contracts: the **`DevelopedBody`** (what a body *is*, produced by `genetics`, §5) and the **`MediumSample`** (what the environment *is at a position/time*, produced by `fields`, §3). This is P4 made concrete; the import-linter makes `physics` importing `core`/`fields`-internals a build failure.

The section specifies, in dependency order: (1) the `DevelopedBody` geometry (§4.2); (2) the hydrodynamic force set (§4.3–4.4); (3) the additive-contributor articulated-body core (§4.5–4.7); (4) determinism, energy closure, and oracle validation (§4.8).

**The governing physics invariant (a CI gate, ranked with conservation, P1).** Every force contributor that removes mechanical energy from a body must deposit it into a tracked sink (wake KE, drag dissipation, ground-friction heat), and the per-step algebraic identity `P_in = Σ(useful power) + Σ(dissipated power)` must close to tolerance. Mechanical work is drawn from a creature's metabolic reserve at `ΔE_metabolic = ΔW_mech / (η_muscle·N)` with **η_muscle ≈ 0.20** and the frozen energy anchor **N = 300 J / sim-energy** (§1.2, §4.8). This is the physics-side face of first-law closure.

### 4.2 The `DevelopedBody` — the geometry the solver consumes

`genetics` develops each genome by a fixed-depth batched scan into a `DevelopedBody`: the canonical fixed-slot
struct-of-arrays tensor (§2.4.2), shared across the whole population. Development is a fixed-shape pure
function; all topology growth happens in `Mutate()` (§5.5), never here, so the scan batches on GPU. Segment
`j`'s canonical fields (SI, body frame FLU per ROS REP-103 — x-forward, y-left, z-up):

| Field | Shape `[S_total, …]` | Units | Meaning |
|---|---|---|---|
| `center` | `[S,3]` | m | rest center in body frame |
| `rest_rot` | `[S,4]` | quat | rest orientation; local +z = long axis |
| `local_pos` | `[S,3]` | m | attach offset vs parent |
| `local_rot` | `[S,4]` | quat | rest local rotation vs parent |
| `abc` | `[S,3]` | m | ellipsoid semi-axes `(a,b,c)`; `c` = half-length |
| `volume` | `[S]` | m³ | displaced ellipsoid volume `(π/6)(2a)(2b)(2c)` |
| `mass` | `[S]` | sim-mass | inertial mass `max(0.1, 2a·2b·2c·ρ_gene)` |
| `area` | `[S,3]` | m² | anisotropic drag reference areas `(areaX,areaY,areaZ)` |
| `m_add` | `[S,3]` | kg | added-mass tensor diagonal `(maX,maY,maZ) = k_i·ρ_w·V` |
| `fin_ma_perp` | `[S]` | kg | Surface-fin broadside added mass (0 for Segments) |
| `parent_idx` | `[S]` | int | parent index within body (−1 = root) |
| `depth` | `[S]` | int | graph depth (root = 0) |
| `amp_deg` | `[S]` | deg | joint gait amplitude, clamped to `AmpMax = 58°`; 0 if root |
| `phase` | `[S]` | rad | gait phase = `−depth·swimWave` |
| `is_surface` | `[S]` | bool | Surface (thin plate/fin) vs Segment — the **exaptation flip bit** |
| `is_tail` | `[S]` | bool | posterior-most, largest +z reach |
| `has_joint` | `[S]` | bool | `parent_idx ≥ 0` → carries an actuator |

Plus per-body `tail_slot`, `swim_freq`, and `swim_wave`. Per-body reductions (COM, the six unique `M_eff`
entries, drag, yaw inertia) are masked sums over slot axis 17. Pose resolves as a bounded **6-pass depth scan**
(MaxDepth=5), each pass a masked batched quaternion compose (parents precede children).

**Rationale.** The ellipsoid `(a,b,c)` triple is the single geometric primitive: it sets displaced volume (buoyancy), the anisotropic drag areas, *and* the Lamb added-mass tensor. One representation, three physical consequences — P2 with no stat vector. The `is_surface` bit is deliberately a single flip: a drag-only fin and a thrust-bearing swimming surface differ only in this bit and their `(a,b,c)`, which makes fin↔limb exaptation a *small, reversible* genome edit rather than a new part type (§4.7, §5.5).

**Migration note / engineering risk.** The donor stored box vs ellipsoid inconsistently (inertial mass from box volume, displaced volume from the ellipsoid `π/6`). The SirRobin port uses the **ellipsoid volume for both** inertia and displacement; this shifts the added-mass baseline and therefore requires **re-recording oracle fixtures** (§4.8; open item #3, §8).

### 4.3 The hydrodynamic force set — `SwimEval` re-ported to torch

`SwimEval` is deterministic Lighthill/Lamb elongated-body theory (EBT) — the **crown jewel** and the first force
contributor. Its validated operation order is re-expressed as batched tensor operations over fixed
`[B,17,...]` slots and `[B,...]` bodies. Fixed timestep **Dt=1/120 s**, water density
**ρ_w=1000 kg/m³**.

**(a) Reactive trailing-edge thrust (added-mass half).** All thrust is a *single boundary term* — the lateral fluid-momentum flux shed at the trailing edge — not a body-interior volume integral. At the tail tip, with forward speed `U`, lateral tail-tip velocity `V_t`, and tail-backbone slope `s`:

```
W_t     = V_t + U·s                         # crossflow shed into the wake
T_react = 0.5·m_t·(V_t² − U²·s²)            # Lighthill reduced form  [N]
P_wake  = 0.5·m_t·U·W_t²                    # irrecoverable shed-wake KE   [W], ≥0
P_in    = m_t·U·V_t·W_t                     # rate body works on fluid    [W]
m_t     = maX_tail / (2·c_tail)            # transverse added mass per unit length  [kg/m]
```

with exact instantaneous closure `P_in ≡ T_react·U + P_wake`. For a traveling wave `h=a·sin(kx−ωt)` this yields `⟨T_react⟩ = ¼ m_t a²k²(c²−U²) > 0` for `U<c`, `→0` as `U→c` — thrust needs slip; coherence shows up as *efficiency* (lower wake for equal thrust), never as a penalty term. `U`, `V_t`, `s` are sampled at the **tail tip** (`center_tail + rot_tail·ẑ·c_tail`), not the segment center.

**(b) Garrick circulatory lift (finite-wing half) — Surface caudal fins only.** A thin high-aspect-ratio caudal fin carries bound-circulation (Kutta–Joukowski) lift the added-mass term omits. Additive, physically distinct:

```
α    = clamp(atan2(V_t, U) − asin(s), ±FinStallAoA)     # FinStallAoA = 0.35 rad
C_L  = a_L·α ,   a_L = 2π·AR/(AR+2)                      # finite-wing lift slope
q    = 0.5·ρ_w·U²·S                                       # KJ dynamic pressure on forward throughflow
C_Di = FinProfileCd + C_L²/(π·e·AR)                      # FinProfileCd = 0.02, e = 0.9
T_fin = q·C_L·sinβ − q·C_Di·cosβ ,   P_fin = q·C_Di·Q    # energy closes P_in = T·U + P_fin
```

Aspect ratio `AR = span/chord` is the efficiency lever (lunate tuna vs eel); lift `→0` at rest (no self-start artifact).

**(c) Lamb added-mass tensor (one-time precompute, float64).** Per segment,
`k_i=α_i/(2−α_i)` where
`α_i=abc·∫₀^∞ dλ/((axis_i²+λ)·Δ(λ))`,
`Δ=√((a²+λ)(b²+λ)(c²+λ))` (Lamb §114). Production and analytic gain1 use committed GL256
quadrature under `λ=L·t/(1−t)`, including the full Jacobian, corroborated against GL512/SciPy. The untouched
gain0 arm retains the donor's composite-Simpson behavior only for historical conformance. `m_add,i=k_iρ_wV`.

**(d) Anisotropic quadratic form drag (dissipative), axial-only.** `F_drag,j = −0.5·ρ_w·Cd·areaZ_j·|v_z|·v_z` along each segment's local +z, with **Cd = 0.1** (streamlined). Lateral broadside drag is deliberately *not* charged — a slender undulator's lateral resistance is *reactive* (added mass), already carried by `m_t`/`P_wake`; charging a `v²` form drag on the 2–5 m/s tail-wag would double-count. Dissipated power `W_drag = Σ_j max(0, −F_drag,j·u_j) ≥ 0`.

**(e) Added-mass effective-mass matrix + semi-implicit COM integration.** The reactive force is a momentum exchange with entrained fluid, so added mass is folded into the LHS effective-mass tensor (semi-implicit) to stay stable when `m_a ≳ M_body`:

```
M_eff    = M_body,kg·I₃ + Σ_j R_j·diag(maX,maY,maZ)_j·R_jᵀ # 6 unique entries, masked slot sum
F_stream = (T_react + T_fin)·f̂ + Σ_j F_drag,j              # + gravity/buoyancy/contact at S9 (§4.5)
dv_xz = solve(M_eff[xz,xz], (F_stream·Dt)_xz)
v_xz += dv_xz ; v_y = 0 ; x += v·Dt                          # constrained horizontal solve
```

`SolveSym3` is the donor's closed-form symmetric cofactor solve (control op order for the oracle; do **not** dispatch to `linalg.solve`). Momentum-conservation guard (CI): with `Cd=0`, no undulation, an initial drift must coast exactly.

**Frozen constants (measurements, never balance knobs — §1.2):** `Dt=1/120 s`, `ρ_w=1000`, `KgPerSimMass=250`, `AmpMax=58°`, `Cd=0.1`, `FinProfileCd=0.02`, `e=0.9`, `FinStallAoA=0.35 rad`, `η_muscle≈0.20`, `N=300 J`.

### 4.4 Gait — rhythmic joint actuation (no neural brain)

Population cognition is **reactive** (§1.1); there is no evolved brain in the hot loop. Locomotion is driven kinematically by a central-pattern-generator-style joint gait encoded in the genome:

```
θ_j(t) = amp_deg_j · sin(2π·swim_freq·t + phase_j) ,   phase_j = −depth_j·swim_wave     [deg]
```

A coherent head→tail phase gradient (`swim_wave` monotone in depth) is a traveling wave → net thrust; a standing/incoherent gait nets ~0 thrust while still shedding wake KE → COT→∞ (selected against, emergently). Steering (S2+, `StepLive`) adds a latched **DC bias** `θ_j += turnCmd·depth_j`; the asymmetric wake produces a *real* yaw torque `τ = Σ r×F`, integrated as **angular momentum** `L_yaw += τ·Dt`, `ω = L_yaw/I_yaw`, with physical quadratic yaw form-drag `τ_drag = −Cyaw·ω·|ω|`, `Cyaw = 0.5·ρ_w·YawCd·Σ_j areaX_j·r_j³` (broadside `YawCd≈1`). The turn *rate* is drag-set physics, not a clamp; `turnCmd` is the only "placeholder brain" seam (a P-controller on heading error), bounded so the biased peak stays within `AmpMax`.

The realized locomotion DOF is **2: {surge, yaw}** — the kernel zeroes vertical COM velocity and integrates yaw only — which is exactly the CORE action contract `{surge_effort, yaw_rate}` shared with the TurtleBot3 (§7.4), making fish↔robot action-transfer risk ≈ 0.

### 4.5 The load-bearing architecture — additive force contributors on one articulated-body core

**This is the single most important architectural decision in the section**, dictated by the S9 endpoint (destination-shapes-foundation, biting early at S2). Locomotion physics is a set of **additive force contributors summed on ONE articulated-body dynamics core**:

```
F_total(body, medium, t) = F_hydro + F_gravity + F_buoyancy + F_contact/friction + …
```

- **The core** owns articulated-body state (per-segment pose from the depth-scan, COM linear state, yaw angular-momentum state), the `M_eff` assembly, the `SolveSym3` semi-implicit integrator, and the energy ledger. It exposes one interface to each contributor:

  ```python
  class ForceContributor(Protocol):
      def accumulate(self, body: DevelopedBody, pose: Pose, vel: State,
                     medium: MediumSample) -> ForceTorquePower
      # returns per-body (F[B,3], τ_yaw[B], P_dissipated[B]); contributors are summed, never switched
  ```

- **Hydrodynamics (`SwimEval`, §4.3) is the FIRST and, at S2, ONLY contributor.** It fully occupies the interface; the core is built and validated with exactly one contributor plugged in.
- **Gravity, buoyancy, contact/friction, and articulated-body land dynamics ADD later** (S9-era) as additional `ForceContributor` implementations, *without rewriting the core*:
  - `F_gravity = −m·g·ẑ_world`
  - `F_buoyancy = +ρ_medium·V·g·ẑ_world` (medium density from `MediumSample`)
  - `F_contact` = compliant/penalty ground reaction along the terrain normal + Coulomb-style tangential friction against `elevation(x,y)` from the geology interface (§3.4), active only where a segment penetrates the substrate.

**The medium sets which contributors dominate — there is NO mode switch.** Contributors are *always all summed*; position relative to the waterline decides which are large:

| Position | Dominant contributors | Emergent regime |
|---|---|---|
| Below waterline, off-bottom | `F_hydro`, `F_buoyancy≈F_gravity` | swimming |
| At the bottom / benthic | `F_hydro` + `F_contact` (fin-rays push off substrate) | walking-in-water (**sea robin**) |
| Above waterline, on land | `F_gravity`, `F_contact/friction`; `F_hydro`→~0 | walking |
| Intertidal / surf zone | all contributors comparable | amphibious crawl |

Because `F_hydro` scales with the surrounding fluid's momentum and `F_contact` scales with substrate penetration, each **automatically vanishes** where physically irrelevant — *without any `if medium==LAND` branch anywhere in the solver*. The transition is continuous in position, which makes it lawful and learnable.

**Rationale for "additive-first" over a swimming-only kernel later refactored:** the donor was hydrodynamics-only and would have required a rewrite to add gravity/contact. Committing to the summation architecture at S2 — even while only `F_hydro` exists — costs one interface indirection and eliminates the S9 rewrite risk entirely. `SwimEval` becomes the *swimming specialization* of a general articulated-body-with-pluggable-forces model.

### 4.6 The sea robin — the worked example and near-term milestone

The searobin/gurnard (Triglidae, §1.7) walks the seafloor on modified lower pectoral **fin-rays** serving simultaneously as leg-like struts *and* chemosensory probes, while still swimming with its fins. It is the canonical demonstration that the additive-force architecture is correct.

**Under the one force sum, the SAME appendage produces both behaviors with no mode switch:**
- In the water column, a fin-ray (a Surface segment) undulating with the gait sheds a trailing-edge crossflow `W_t` → contributes `T_react` (and, if AR is high, `T_fin`) → **swimming thrust**.
- Against the bottom, the same fin-ray penetrating the substrate generates `F_contact` + friction → **walking traction**.

Both are `F_hydro + F_contact` evaluated at the fin-ray's current position/velocity; the only thing that changes is whether the ray is in open fluid or touching the seabed. The contact contributor turns on smoothly as the ray nears the substrate; nothing decides "this is now a leg."

**Why it is the right first milestone.** The full S9 crossing additionally requires air-breathing, desiccation tolerance, and a survivable intertidal fitness gradient (the hard long-horizon part). The sea robin needs **none** of those. It therefore **de-risks the physics half of S9 independently of the ecology half**. **Acceptance (see also §5.10):** a benthic lineage evolves fin-rays that measurably contribute *both* a positive `P_wake` (swimming) *and* a sustained `F_contact` impulse (walking) within one individual's locomotion, with no genome flag distinguishing "leg mode."

### 4.7 Bidirectional water↔land — why "return to the sea" is free

The requirement (§1.6) is a **permeable boundary crossed in both directions, repeatedly**. The additive-force architecture supports this natively and imposes one requirement on the genome:

- **One physics, medium = position, no mode switch (§4.5)** ⇒ a limb yielding ground-reaction on land yields thrust/drag in water *by the same force sum*. A lineage drifting from bottom-walking back into the water column simply finds `F_contact→0` and `F_hydro` growing; nothing is un-built or re-coded. **Return to the sea is "free"** — it costs no new mechanism, only re-selection of gait/shape parameters.
- **Convergence is a form-is-function signature.** Because streamlining minimizes `F_drag` and maximizes reactive-thrust efficiency under *the same* hydrodynamics for *any* returning lineage, independent land→sea lineages should re-derive streamlined body plans and flipper-like Surfaces — a directly observable, physics-driven convergence (detected as phenotype-space clusters distant in genotype space, §5.9).
- **Genome requirement — reversibility, no ratchets (§5.5, §5.9).** The `is_surface` flip (Segment↔Surface = leg↔flipper/thrust-surface), parametric `(a,b,c)` re-adaptation, and neutral drift must let a walking limb drift *back* toward a swimming surface. No irreversible developmental step may exist that a return path cannot traverse. This is why the exaptation lever is a **single bit + three continuous axes**, not a distinct part class.

### 4.8 Determinism, energy closure, and oracle validation

**Precision & determinism** follow §2.7–2.8: hot loop float32 with mixed tolerances; f64 analytic/oracle
configuration and Lamb precompute; no atomic segment scatter; exact integer bookkeeping and discrete decisions
are hard gates; float trajectory identity is a same-device diagnostic only.

**Energy-closure acceptance criteria (CI gates, from S0/SpikeSwim):**
- **Instantaneous:** `|P_in − (T_react·U + P_wake + P_fin)| / max(|P_in|,ε) < 1e-6` (float32).
- **Long-horizon (10⁵ steps):** retain every prefix and gate
  `D_k=|Σ_{i≤k}R_i|/Σ_{i≤k}max(|ΔKE_i|,|Wimp_i|,|WM_i|,E_ATOL)` below `1e-3` (f32) or
  `1e-6` (f64) after step 100. Report monotone signed bias separately.
- **Metabolic coupling:** loco energy debited as `ΔE = ΔW_mech/(η_muscle·N)`; no path may bank mechanical work at implicit η=1.

**Oracle validation against the C# donor.** The C# `SwimEval` is demoted to an **offline fixture generator** (Unity-light — only `Vector3`/`Quaternion`/`Mathf`; driven via `ReconstructForTest`/`LambKForTest`/`CoastTest`/`MomentumLedger`). Frozen fixtures across H1/H2 (ragged 2–16-segment, mirror-paired, ~40% fin-tail) genomes guard the port:
- Lamb `k`-factors: `< 1e-6` abs (float64).
- Single-step force terms (`tReact`, `pWake`, `pFin`, all 6 `M_eff` entries, `dv`): `< 1e-4` rel (float64).
- 8 s episode aggregates (`cruiseSpeed`, `costOfTransport`, `reactiveRatio`): `< 1e-3` rel.
- **Gate on aggregates + short-horizon forces**, *not* on a long-horizon bit-trace: the gait is chaotic, so trajectory RMS divergence over many steps is expected.

**Batching / layout.** Population `(W,N_cap)` plus `alive`; segments use fixed `(W,N_cap,17,...)` slots plus
`seg_mask`; per-body reductions are masked slot-axis sums. Wrap the step in `torch.inference_mode()`. Every
data-dependent branch (fin-active, degenerate-body guard, medium selector) is a `torch.where`, never a Python
`if` over the batch.

**Staging.** S0 ports the one-shot frozen-heading `Step` (verify the vectorization thesis). S2 ports `StepLive` (yaw-integrating, heavier — **re-measure throughput against `StepLive` before committing S2**) and stands up the additive-contributor core with `F_hydro` as the sole contributor. Gravity/buoyancy/contact contributors and the sea-robin milestone land in the S9-era extension, on the *already-additive* core — no rewrite.

---

## 5. The Genome & Evolution

This section specifies the genotype→development→phenotype pipeline: the encoding each creature carries, the batched developmental transform that turns it into a physical body, and the variation/heredity/selection operators that make the population evolve and speciate. Sophistication lives in the *body and its evolution*, not in a brain; everything downstream reads morphology run through physics (P2); there is no stat vector.

**Terms.** *Genotype* = the heritable graph a creature carries. *Development* = a pure, fixed-shape function that expands the genotype into a *DevelopedBody* (§4.2). *Phenotype* = the DevelopedBody plus the capabilities Physics derives from it. *Innovation id* = a globally monotone integer minted when a structural gene first appears, marking homology (NEAT historical marking). *P* = population capacity (fixed slot count); a boolean `alive_mask[P]` recycles dead slots so all shapes stay static.

### 5.1 Pipeline and invariants

```
Genotype (Sims part-graph + NEAT ids)      # heritable, variable topology, FIXED tensor capacity
        │  development = pure fixed-shape fn (batched over P, no data-dependent loop bounds)
        ▼
DevelopedBody (masked segment tensor)       # the SINGLE canonical body; contract with Physics (§4.2)
        │  Physics: additive force-contributors (hydrodynamics first; gravity/buoyancy/contact later)
        ▼
Capabilities (thrust, drag, feeding gape, mass, buoyancy, ground reaction) — DERIVED, never stored as genes
```

Load-bearing invariants (CI-gated):

- **I-GENOME-1 (development is pure & fixed-shape).** All growth happens in `Mutate()` *between* generations. Development never changes segment count; it only fills/masks a fixed-capacity tensor. Make-or-break for GPU batching and determinism; violating it is a build-stop.
- **I-GENOME-2 (single canonical body, P3).** The DevelopedBody is the only body. No parallel `eff[]`, no cached scalar speed. The tell of a violation is synchronization glue.
- **I-GENOME-3 (form-is-function, P2).** Every capability is a function of DevelopedBody geometry evaluated by Physics. No capability is read from a gene.
- **I-GENOME-4 (homology preserved).** Innovation ids are monotone and never reassigned; silencing/re-expressing a gene preserves its id. This makes crossover and distance well-defined and makes exaptation reversible.
- **I-GENOME-5 (append-only RNG manifest, §2.7).** Every new heritable gene appends to the mutation draw manifest, short-circuited to zero draws while inert, so adding a gene never re-baselines determinism for existing genes.

### 5.2 Genotype tensor representation

Struct-of-arrays, fixed capacity, boolean masks. A heterogeneous population is *same shape, different masks/values*. Capacities are sized from the realistic ceiling (donor mean ≈6 segments; data cap 28; physics cap 16), not the theoretical max.

| Tensor | Shape | dtype | Meaning |
|---|---|---|---|
| `node_f` | `[P, N_max, F_node]` | f32 | node parameter fields |
| `node_type` | `[P, N_max]` | i8 | part class {Segment, Surface, +reserved plant-organ slots} |
| `node_iid` | `[P, N_max]` | i64 | node innovation id |
| `node_mask` | `[P, N_max]` | bool | valid node |
| `edge_f` | `[P, E_max, F_edge]` | f32 | edge parameter fields |
| `edge_src`,`edge_dst` | `[P, E_max]` | i16 | local node indices |
| `edge_iid` | `[P, E_max]` | i64 | edge innovation id |
| `edge_mask` | `[P, E_max]` | bool | valid edge |
| `body_g` | `[P, F_body]` | f32 | body-level genes |

Suggested capacities: `N_max = 24`, `E_max = 48`, `S_max = 32`, developmental depth `L = 6` (pin against the S0/H1 raggedness profile; `S_max` is post-mirror-expansion).

**`F_node` fields** (`F_node ≈ 13` floats; type/id in `node_type`/`node_iid`):

| Field | Units | Notes |
|---|---|---|
| `log_a, log_b, log_c` | log(m) | ellipsoid semi-axes, **log-scale** → ratio-symmetric mutation; kills the additive-mass ratchet |
| `density_gene` | sim-mass·m⁻³ | segment structural density; multiply integrated sim-mass by 250 kg/sim-mass exactly once |
| `port_intake, port_sense` | {0,1} | mouth / chemosensor flags (Sense feeds the reactive drive and the Talos EXT chem sense) |
| `jFreq, jPhase, jAmp` | Hz, rad, rad | hinge actuation (gait) |
| `hinge_axis (2)` | rad | evolvable hinge axis as two angles (identity default) |
| `expressed` | {0,1} | neutral-drift toggle: silenced node has zero mass/drag/upkeep, drifts freely, can re-express |

**`F_edge` fields** (`F_edge ≈ 10`):

| Field | Units | Notes |
|---|---|---|
| `attach (3)` | fraction of parent semi-axis | attachment point on parent surface |
| `orient_quat (4)` | — | child orientation; renormalized each compose |
| `scale` | — | multiplicative size factor down the edge |
| `mirror` | {0,1} | build a bilateral ±side pair of the child subtree |
| `recursion_r` | int | terminal recursion count: a **self-edge (src==dst) with r>1** = serial repeated segment/limb |

`F_body` = `{swimFreq (Hz), swimWave (rad/segment), sessility∈[0,1], tropism∈[0,1], hungerGain, …}`. Bilateral pairs are represented by the edge `mirror` flag, not by duplicated nodes, so a pair never changes node count during development.

**Innovation registry.** A single monotone `next_iid` counter per run (host-side). Structural mutations draw ids from it; a within-generation cache maps `(operation, structural-key)`→id so the *same* structural mutation arising in the same generation gets the same id (NEAT convention).

### 5.3 Development: batched fixed-depth frontier unroll

Development is a bounded instance-frontier traversal, unrolled a **fixed** `L` layers — the same gather→compose→scatter kernel as the S0/§4.2 pose depth-scan. State per active frontier instance: `(node_idx, world_transform 4×4, remaining recursion counters, side_sign ∈ {+1,−1}, out_slot)`. Frontier tensor shape `[P, W_front]` (fixed width).

```python
def develop(node_f, node_type, edge_f, edge_src, edge_dst, masks) -> DevelopedBody:
    frontier = seed_root_instances()             # root node, identity transform, side=+1
    dev = zeros([P, S_max, F_seg]); dev_mask = zeros([P, S_max], bool)
    for layer in range(L):                        # FIXED L; no data-dependent bound
        nf   = gather(node_f, frontier.node_idx)  # [P, W_front, F_node]
        slot = precomputed_unique_slot(layer, frontier)   # deterministic, no scatter_add atomics
        write_segment(dev, dev_mask, slot,
                      center = frontier.world_transform[...,:3,3],
                      quat   = mat_to_quat(frontier.world_transform),
                      a,b,c  = exp(nf.log_abc),
                      mass   = density * (4/3)*pi*a*b*c,
                      is_surface = (node_type==SURFACE),
                      gait_phase = -depth * body_g.swimWave,
                      ports, expressed)           # multiply by masks; never boolean index_select
        child = compose(frontier.world_transform, edge_transform(edge_f))   # reflection matrix on mirror/side
        frontier = enqueue(child, decrement_recursion(),
                           mask = frontier_budget_ok & segment_budget_ok)
    return DevelopedBody(dev, dev_mask, body_g)
```

`F_seg` is the Physics contract of §4.2. Physics reads masked segments only by **multiply-by-mask, never boolean index-select** (keeps shapes static). **Determinism** per §2.7: fixed `L` and fixed frontier width make `S_max` static; precomputed unique slot indices for a plain scatter (no atomics); `use_deterministic_algorithms(True)`; fixed dtype/device/op-order.

**Acceptance (P0, migration gate).** The torch batched unroll must reproduce the C# donor `Measure()` aggregates — `{mass, area, intake, bulk, avgDensity, parts}` plus shape descriptors `{length, girth, fineness, propArea, asymmetry, compactness}` — within **rel tol 1e-4 (float64)** on a representative genome sample **before any new capability is layered on**. The donor's exact DFS traversal order and mirror double-count are preserved so the float-sum-order golden holds.

### 5.4 Form-is-function: capabilities derived from morphology (P2)

Capabilities are computed by the Physics layer from the DevelopedBody, never read from genes:

- **Locomotion.** DevelopedBody → `SwimEval` (§4.3) → per-segment thrust/drag/added-mass → realized surge and yaw. Physics is the additive-force model (§4.5): hydrodynamics first; gravity/buoyancy/contact/ground-reaction add later on the *same* articulated body with **no mode switch**.
- **Feeding.** Intake-port surface area sets gape/uptake; filter vs bite emerges from surface geometry and placement (§6.4).
- **Mass / buoyancy.** From `Σ density·volume` and displaced volume — sets sinking, cost of transport, neutral-buoyancy niches.
- **Defense.** Armored (bulk/density) vs fast (agility from compactness + streamlining) emerge as two viable body plans, a disruptive axis.

Because a *fin surface* yields hydrodynamic thrust/drag in water and *ground-reaction impulse* on contact from the same additive-force model, one appendage does both jobs without a switch — the physics substrate for the sea robin (§5.10) and the bidirectional crossing (§5.9).

### 5.5 Mutation operators

Every structural op mints a monotone innovation id; every new heritable-but-inert gene appends to the RNG manifest (I-GENOME-5). Parametric jitter touches one node/region (the graph's locality advantage).

| Operator | Effect | Innovation id? | Evolvability role |
|---|---|---|---|
| Parametric jitter | masked Gaussian/Laplace on param slices; **log-scale** size drift | no | locality, smoothness |
| add-node | split an edge, insert node | mint node id | complexification |
| add-edge | connect two nodes; self-edge with r>1 = serial repeat | mint edge id | complexification + reuse |
| ±recursion `r` | tune serial-segment count | no | cheap regularity |
| toggle `mirror` | bilateral pair on/off | no | symmetry |
| **flip Segment↔Surface** | drag fin ↔ thrust fin ↔ ground strut | no (id preserved) | **exaptation lever (reversible)** |
| toggle `expressed` | silence → drift → re-express | no (id preserved) | neutrality → exaptation |
| toggle Intake port | grow/lose a mouth (+ `EnsureMouth` repair) | no | feeding innovation |
| prune | remove a non-root subtree | no | anti-bloat (metabolic pressure) |

**No operator is a one-way ratchet:** Segment↔Surface flip is symmetric, expressed toggle is bidirectional, log-scale size drift is ratio-symmetric, add-node/prune are inverse-capable, and silenced ids persist so re-expression restores homology. This is the mutation-side guarantee of reversibility (§5.9, §1.6).

### 5.6 Crossover — aligned by innovation id

The donor had *no* crossover (asexual clone-and-mutate, which can never speciate). Innovation-number bookkeeping is **non-optional**: naive positional alignment hits the competing-conventions problem and yields non-viable monsters (*solved technique*).

```python
def crossover(parent_a, parent_b, fitter):        # batched over mating pairs
    for table in (nodes, edges):
        align a,b BY innovation id (gather/sort by iid — batchable)
        matching genes  -> pick or blend params (per-gene random parent, or mean for numerics)
        disjoint/excess -> inherit from the FITTER (or spatially closer) parent
    repair:                                        # deterministic validity pass
        drop dangling edges (src/dst not present)
        re-run EnsureMouth; enforce N_max/E_max/S_max caps
    return child_genotype
```

Because edges reference nodes by *stable global id*, a full graph recombines coherently. The repair pass is deterministic and part of the fixed-shape contract.

### 5.7 Compatibility distance

NEAT compatibility distance over innovation-aligned genes (a masked reduction, batchable pairwise within spatial buckets):

```
δ(a,b) = c1·E/N + c2·D/N + c3·W̄
   E  = # excess genes (id beyond the other genome's max id)
   D  = # disjoint genes (non-matching id within range)
   N  = max(gene_count_a, gene_count_b)   (or 1 if both small)
   W̄  = mean absolute param difference over MATCHING genes
```

Node and edge tables contribute jointly. Starting coefficients `c1 = c2 = 1.0`, `c3 = 0.4`; species threshold `δ_t` tunable (fixed, or auto-tuned to a target species count — open decision). δ drives **mating**; it is a *genotypic* metric, kept separate from the *phenotypic* cosine used only for naming (§5.11).

### 5.8 Spatial assortative mating and emergent reproductive isolation (P8)

Reproductive isolation is **emergent, not imposed** — no species label is ever assigned to gate mating (that would re-impose the thing meant to emerge). Mating is a masked pairwise op within existing SpatialHash buckets:

```
P(mate | a,b) = 1[ dist(a,b) < r_mate ] · sigmoid( (δ_t − δ(a,b)) / w )
```

`r_mate` = mating radius (spatial structure supplies isolation-by-distance), `w` = choosiness width. Under **disruptive/ecological selection**, local drift + assortment let a panmictic cloud split into non-interbreeding clusters (de Aguiar 2009 topopatric; Dieckmann & Doebeli 1999 sympatric). Optionally couple the mating cue to a heritable trait already under ecological selection (a "magic trait," Servedio 2011) for the fastest sympatric split. **"Species" are read out** for analytics as connected components of the who-can-breed-with-whom graph — never used to gate.

Tuning tension (a research-frontier risk, §5.13): too-strong assortment fragments into non-viable singletons; too-weak stays panmictic. The S5/Phase-2 milestone must *prove* a split.

### 5.9 Evolvability properties and the reversibility requirement

| Property | Level | Mechanism |
|---|---|---|
| Regularity / modularity | medium (high w/ P3-CPPN) | self-edge recursion, duplicate-diverge, mirror |
| Locality / smoothness | high | jitter on one node changes one region |
| Complexification | high & genuine | NEAT topology growth, homology preserved |
| Exaptation | high | Segment↔Surface flip; express/silence toggle |
| Neutrality / robustness | medium-high | expressed bit + inert appended genes = neutral networks |
| Pleiotropy / epistasis | low-medium | rises where the optional CPPN is added |
| Open-endedness | **unsolved** | expect to need novelty/MAP-Elites (Phase 4); still a frontier |

**Reversibility / exaptation-friendliness is a hard requirement** (bidirectional crossing, §1.6, §4.7): the genome must carry **no irreversible ratchets**. A walking limb (a Segment chain producing ground reaction) must be able to drift back toward a swimming surface (leg→flipper) through: distal Segment→Surface flips, parametric flattening (`log_b` shrinks), joint-amp/phase re-tuning, and neutral drift of silenced nodes that later re-express in the aquatic context. The medium-dependent additive-force physics makes the *return to water* "free" (§4.7).

**Convergent aquatic re-adaptation across independent lineages is an observable target:** detect phenotype-space clusters (low fineness, high propulsive surface, streamlined) that are *distant in genotype space* (different innovation-id lineages) — a directly measurable signature that physics, not the encoding, drives form.

### 5.10 The sea robin milestone (near-term, de-risks S9)

The sea robin (§1.7, §4.6) is the fin→limb exaptation made flesh, **in water, gradual**, with **no air-breathing and no leaving the water**.

**Milestone (achievable before S9):** a *benthic lineage emerges* whose ventral pectoral Surfaces produce measurable ground-reaction impulses supporting a walking/probing gait on the seafloor while the same body still swims. It requires only: (a) the graph encoding (proven-capable — Sims evolved walkers and swimmers), (b) reactive rhythmic gait already in the joint genes, (c) the additive contact-force set active near the seafloor (§4.5), (d) a benthic gradient the ecology rewards (buried prey reachable by fin-ray probing; Sense port on the ventral rays).

**Milestone acceptance criteria:** in a single deterministic run, ≥1 established lineage in which (1) ventral Surface segments spend a measured fraction of the gait cycle in seafloor contact producing net upward + forward ground-reaction impulse; (2) the same lineage retains nonzero swimming thrust from the same appendages; (3) the walking capability is heritable and improves under a benthic-foraging gradient. Report the impulse budget and gait phase from telemetry; do not assert from code inspection.

### 5.11 Observational emergent taxonomy (naming, not selection, P8)

Taxonomy is **observation only — it never feeds selection or gates mating.** Two distinct axes:

- **Genotypic δ (§5.7) drives mating** and defines biological species as connected components of the interbreeding graph.
- **Phenotypic cosine drives naming.** A per-creature morphology fingerprint (DevelopedBody shape descriptors + body-level genes) clusters into an emergent Linnaean ladder (species → genus → … → kingdom), each rank the same lineage+centroid+threshold mechanism budding when a clade's centroid drifts past its rank threshold. Names are procedurally generated Latin/Greek morphemes keyed to standout traits. **Establishment is a one-way latch** (set when a clade first reaches its rank's min-pop, cleared only by extinction) so names don't flicker. Kingdoms are recognized life-form domains (Animalia/Plantae/Bacteria/Virus), not emergent clusters.

Keeping δ (mating) and cosine (naming) as *separate* axes is deliberate; never let the naming axis leak into selection.

### 5.12 Phased path P0–P4

| Phase | Content | Gate / acceptance |
|---|---|---|
| **P0** (refactor, zero behavior change) | Port C# `Measure()`/`Develop()` to the batched fixed-iteration torch unroll producing the DevelopedBody. Dovetails with S0/SpikeSwim. | Hard-diff vs C# aggregates (§5.3). |
| **P1** (topology upgrade, still asexual) | tree→node+edge tables; children→edges; recursion `r`; innovation ids; boxes→ellipsoid a/b/c. Genome now *alignable*. | Develop still matches; ellipsoid readout validated vs box baseline (Lamb terms may need re-tuning — §5.13). |
| **P2** (speciation — the payoff) | NEAT distance + innovation-aligned crossover + spatial assortative mating. | **Falsifiable milestone:** one deterministic run where a population splits into two non-interbreeding clusters under disruptive selection. Do not proceed until demonstrated or its absence root-caused to the ecology. |
| **P3** (generative modifier, optional / plants) | HyperNEAT-lite CPPN indexed by developmental coordinates (depth, serial index, side) → graded parameter offsets. Required for plants (S9). | Deferrable for animals if recursion-indexed deltas already give enough along-body gradient. |
| **P4** (open-endedness pressure, research) | Novelty search / MAP-Elites over a behavior descriptor, *if* fitness-only search plateaus. | A frontier, not a switch. |

### 5.13 Engineering risk

**Solved technique (low risk):** the encoding (Sims recursive graph), NEAT innovation markings, aligned crossover, compatibility distance, and batched padded-tensor development are established and cited (Sims 1994; Stanley & Miikkulainen 2002; TensorNEAT/EvoX 2024; PyTorch-NEAT). The migration is an in-place evolution of the donor `BodyGraph`, not greenfield.

**Research frontiers (gated, falsifiable):**
- **Emergent speciation may not occur.** The binding constraint is *ecological niche diversity*, not encoding power. Mitigations: magic-trait coupling; Kleiber + prune pressure so new structure must earn fitness before more DOF; gate on the P2/S5 split test. **Falsifier:** if no disruptive-selection setup produces a durable split, the ecology (not the encoding) is the fault.
- **Assortment tuning is bimodal-fragile** (fragment vs panmixia). Sweep `w`, `r_mate`, `δ_t`.
- **Ellipsoid reinterpretation** (a/b/c vs box) may shift the Lamb added-mass terms. Validate against the box baseline in P1; re-tune if the oracle tolerance is breached (open item #3, §8).
- **Open-endedness is unsolved.** Aim for *sustained adaptive radiation*, not a proof of unboundedness.
- **GPU determinism tax** on the development scatter. Mitigated by precomputed unique slots (no atomics) and deterministic-algorithm mode.

**What would invalidate the approach:** development that cannot be kept fixed-shape (I-GENOME-1 fails → batching collapses); a P2 that cannot produce a split under any ecological setup; or a reversibility failure where a walking lineage cannot drift back to swimming (an irreversible ratchet slipped into an operator), breaking the bidirectional requirement.

---

## 6. Ecology & the Conserved Economy

This section specifies the biogeochemical and trophic layer: a closed, conserved material loop that makes primary production nutrient-limited, and the feeding/metabolism/reproduction/predation transfers that ride on it. It maps to slices **S1** (conserved single-nutrient economy — the keystone; books must close first), **S3** (feeding/metabolism/reproduction), **S4** (predation), **S6** (transport/upwelling). Every mechanism obeys P1 (conservation) and P2 (form-is-function). The design is a torch re-port of the prior build's *validated equations* (Redfield, Liebig, Monod, Martin, Kleiber, Holling-II, the BGE microbial split) with its *architecture inverted*: the prior build minted matter; this layer's purpose is to make that structurally impossible.

### 6.1 Reservoirs, currencies, and the two ledgers

**Two independently conserved currencies.** Do not conflate them; the prior build's fatal bug was measuring one in units of the other.

| Currency | Symbol | Unit | Source | Sink | Ledger closure |
|---|---|---|---|---|---|
| **Limiting nutrient** (canonical mass currency; P-like — no gaseous phase ⇒ airtight) | `N` | mol (µM = mmol·m⁻³ in fields) | geology only (vents/weathering; **off at S1**) | burial only (sediment; **off at S1**) | **exactly constant** in the S1 closed box |
| **Energy** | `E` | J | the **sun** (external; the one open input) | respiration → heat; burial | `dE/dt = P_sun − P_heat − P_burial` |

**Single canonical representation (P3).** `N` is the canonical stored quantity in every biomass pool, measured in the **same currency everywhere** (mol nutrient-equivalent, biomass converted via Redfield), so every transfer is 1:1 and conservation is `Σ reservoirs = const`. Energy content of a *producer/detritus* pool is a **derived readout** `E = c_cal · B` (caloric anchor), never a second stored-and-synced scalar. The one genuinely distinct kind is a **creature's energy reserve** (lipid/glycogen store, C-rich, decoupled from structural nutrient) — stored explicitly on the entity because it is a different quantity.

**The closed inventory.** S1 contains exactly four reservoirs:

```
I_N,S1(t) = Σ Nd_q + Σ Bp_q + Σ Bd_q + Σ Bm_q
```

Later slices extend the registry only when their mechanisms land: creature-bound `struct_N` arrives with real
creatures, and `Sed` arrives with an enabled burial transfer. Neither exists as an S1 placeholder.

| Reservoir | Representation | Tensor / SoA | Kind |
|---|---|---|---|
| `Nd_q` dissolved nutrient | conserved Eulerian **reservoir field** | `(W, Gx, Gy, Gz)` int64 mass quanta | abiotic nutrient |
| `Bp_q` primary-producer biomass | conserved co-grid reservoir field | `(W, Gx, Gy, Gz)` int64 mass quanta | biotic |
| `Bd_q` detritus (marine snow / POC) | conserved co-grid reservoir field | `(W, Gx, Gy, Gz)` int64 mass quanta | biotic |
| `Bm_q` microbial biomass | conserved co-grid reservoir field | `(W, Gx, Gy, Gz)` int64 mass quanta | biotic |
| `struct_N`, reserve `E` | **point-entities** (creatures), SoA | `nutrient[W,N_cap]`, `energy[W,N_cap]`, `mass[W,N_cap]`, `pos[W,N_cap,3]`, `alive[W,N_cap]` | discrete |
| `Sed` sediment | seafloor field | `(W, Gx, Gy)` | burial sink |

Fields are stored at high grid resolution (GPU-cheap stencils); **creatures sample fields by interpolation** — smooth value + gradient at the exact continuous position, continuous uptake rate `∝ local conc × dt`, **never a per-cell share** (P5).

### 6.2 Primary production — Liebig-minimum × Monod drawdown

Growth is the product of a max specific rate, a light term, and the **single most limiting** nutrient factor (Liebig 1840, *not* a sum), each a saturating **Monod** function (Monod 1949; Dugdale 1967).

```
light:   γ_L(x)   = I(x) / (I(x) + K_I),        I(z) = I_0(t) · exp(−k_d · z)      # PAR, depth-attenuated
nutrient factor:  f_N(x) = Nd(x) / (Nd(x) + K_N)                                    # Monod uptake
Liebig:  f_lim(x) = min( γ_L(x), f_N(x) [, f_Fe(x) …] )                             # min, per element
growth:  Gp(x)    = μ_max · f_lim(x) · Bp(x)                                        # [mol N · m⁻³ · s⁻¹]
```

**Drawdown (conserved, the load-bearing edit, P1).** Realized new biomass debits the dissolved field 1:1:

```
ΔBp = Gp · dt ,   ΔNd = −ΔBp        # paired transfer, debit==credit exactly
ΔBp ← min(ΔBp, Nd)                  # strict guard: never grow more than the nutrient present
```

Contrast the prior build: production grew toward a **static Perlin cap** (`baseCap`, set once, never updated), consuming **no pool** — a stateless magic well that minted matter. Here `Bp` grows only by moving nutrient out of `Nd`; a bloom mechanically empties its own nutrient and self-terminates (§6.8).

**Rate anchor (units, not a balance knob — §1.2).** Calibrate `μ_max`, `c_cal`, and grid so cell-integrated production matches real net primary productivity: global mean **≈140–150 gC·m⁻²·yr⁻¹** (Field et al. 1998; oligotrophic ~50, upwelling ~300–400), phytoplankton carbon **≈45 kJ·gC⁻¹** (Platt & Irwin 1973), Redfield **C:N:P = 106:16:1** (Redfield 1934/1958). Freeze these as recorded derivations.

**Representation decision.** S1 implements only Eulerian `Bp_q/Bd_q/Bm_q` fields and records interpolation and
synthetic point-depletion resolution. The parcel fork is deferred until real dense grazing exists; S1 does not
build or authorize an unused second representation.

### 6.3 Bacterial remineralization — the BGE split and the microbial loop

Remineralization is **not** a decay-to-nutrient sink; it is bacterial decomposition that **builds biomass** (Azam et al. 1983). The decomposition flux splits by **bacterial growth efficiency** (del Giorgio & Cole 1998):

```
R(x)  = k_remin(z) · Bd(x)                 # decomposition flux [mol N · s⁻¹]
ΔBm   = + BGE · R · dt                      # BGE ≈ 0.2 → microbial biomass
ΔNd   = + (1 − BGE) · R · dt                # respired remainder → dissolved nutrient
ΔBd   = − R · dt
INVARIANT:  ΔBm + ΔNd = −ΔBd  (with BGE + (1−BGE) = 1)   # nothing minted, nothing lost
```

Microbes are then grazed by detritivores/microbivores (§6.4), closing the detrital loop. This fixes the prior build's leak (`MarineSnow.Tick: s −= remin·s·dt` released matter **discarded**).

**6.3.1 Martin depth profile.** `F(z) = F(z₀)·(z/z₀)^(−b)`, `b ≈ 0.858` (Martin et al. 1987). Remineralization at depth is `−dF/dz`; implement as depth-dependent `k_remin(z)` on the sinking `Bd` field (advection velocity `w_sink` downward).

**6.3.2 Vertical mixing.** Conservative turbulent diffusion returns deep nutrient toward the photic zone: `∂Nd/∂t = ∂/∂z( K_z(z) · ∂Nd/∂z )`. Discretize **flux-form** (fluxes between cells; `Σ` invariant to machine tolerance), double-buffered (read snapshot, write scratch) so there is no read-after-write gradient bias.

**6.3.3 Detritus reservoir.** `Bd` must **accumulate** (labile turnover days–weeks, not the prior ~9 s half-life). A single-pool relaxed decay + high ceiling gives deposit-feeders a standing stock; a labile/refractory two-pool split is deferred.

**6.3.4 The biological pump (emergent).** Surface production → sinking `Bd` → deep remineralization (Martin) → deep `Nd` enrichment → surface depletion; mixing + Ekman upwelling (S6) close the return. Vertical zonation is the S1 all-ocean world's primary adaptive axis, not prescribed — it falls out of the pump.

### 6.4 Feeding — Holling type-II, form-derived, one coupled act

A grazer at continuous position samples the local edible density `D` (interpolated from `Bp`, `Bm`, or `Bd`). Intake is **one act** that simultaneously depletes the field and feeds the animal — energy gained and biomass removed can **never** diverge (the prior build's decoupled `grazeRate`/`grazeImpact` was the boom's fingerprint).

```
C_form        = f(gape_area, filter_surface, size)        # capacity DERIVED from morphology (P2)
I_bio         = I_max · C_form · D/(D + K_half) · dt       # Holling type-II (Holling 1959)
I_bio         ← min(I_bio, D_local)                        # never remove more than present
field debit:  ΔD = −I_bio                                  # SAME biomass leaves the field
assimilated:  I_assim = AE · I_bio → creature reserve      # AE ≈ 0.3–0.4 (Lindeman ~10% trophic)
egested:      E_eg    = (1 − AE) · I_bio → detritus Bd      # assimilation loss closes into the pump
INVARIANT:    I_bio = I_assim + E_eg                       # ingested == assimilated + egested
```

Holling-II saturation makes per-capita intake bounded above (gut) and → 0 smoothly as `D` → 0, so the field self-limits with **no arbitrary refuge floor**. `C_form` from the body means the number of grazers a patch supports = production ÷ *real* per-capita intake — a bounded, emergent number, not an appetite constant.

### 6.5 Metabolism, reproduction, and predation

**6.5.1 Metabolism — Kleiber allometry (Kleiber 1932).**

```
P_basal  = B0 · M^α           [W],   α ≈ 0.79   (fish/ectotherm range 0.75–0.90; measurable, not 0.75-canonical)
P_active = P_basal + P_loco   where P_loco is the hydrodynamic muscle power from SwimEval (§4, S2)
```

Metabolism debits the creature's **energy reserve** and emits heat (energy-ledger sink). It does **not** touch the nutrient ledger — `struct_N` stays in the body until death routes it to `Bd`. An optional Q10/Arrhenius temperature multiplier (`~2–3` per 10 °C) on `P_basal` and `k_remin` is deferred to the temperature-field slice. Sublinear α gives efficient giants (the allometric size refuge) as an emergent, not tuned, outcome.

**6.5.2 Reproduction — a real construction cost (P1).** Producing offspring pays for the offspring body out of the parent's reserves; nothing is minted (the prior build's `energyCap = 16` flat tank for copepod and whale caused a fecundity cliff).

```
E_build = e_tissue · M_off        # energy to build offspring tissue
N_build = r · M_off               # structural nutrient, drawn from parent → child
eligibility: reserve_E ≥ E_build + reproThresh(M)      # mass-scaled bar (no cliff)
transfer:    parent.(E, struct_N) −= (E_build, N_build) ;  child.(E, struct_N) += (…)   # debit==credit
```

Reserve cap and reproduction bar **both scale ∝ M** (shared pivot) so headroom is mass-independent. This is a provisional single-reserve-breeding model; income-breeding / ontogenetic growth dissolves the knife-edge later.

**6.5.3 Predation — a staged contest between real bodies (S4).** `find → close → seize → consume` in continuous space over the shared spatial hash, reading only capabilities **derived from form + physics** (P2) and controller intents — never `carnivory` or an `eff[]` stat.

| Stage | Requirement | Derived-from-form inputs |
|---|---|---|
| **find** | two-sided, multi-modal, non-dominant | `Detect(modality)` range vs opponent `Signature`; modalities {vision↔light, smell↔chemical, lateral-line↔flow, electro↔electric} |
| **close** | keyed on **relative velocity/heading** + a burst gear | pursuit from body drag/yaw-torque physics; gait throttle (SwimEval) |
| **seize** | two-sided: `GripRate(pred)` vs `Evade(prey)`; overpower is a **size ratio** | gape, jaw force `F_crack`, predator:prey mass ratio |
| **consume** | energy+nutrient transfer gated by `CanCapture ∧ CanDigest` | `CanCapture = Reach ∧ GripRate>0 ∧ Overpower≥1` |

**Strict conservation:** the consumed prey's `(E, struct_N)` is debited and credited to the predator (assimilated `AE`) + detritus (egested `1−AE`) in one paired transaction; **prey debit == predator credit + egesta**. "Predator" is a *realized role* (the controller chose to hunt and capture succeeded), not a `carnivory ≥ 0.5` cutoff (P8).

### 6.6 Conservation invariants — the top CI gate (P1)

S1 nutrient conservation is verified as **exact int64 predicates**, the primary correctness gate, not
byte-identity or a float reduction. Later currencies must define their real reservoirs before claiming closure.

```
INV-MASS (per element):
   I_N(t) = Σ(Nd_q+Bp_q+Bd_q+Bm_q)
   S1 closed box: I_N(t) == I_N(0) exactly, per world

INV-TRANSFER (per transfer op):  every debit is paired with an equal credit
   debit_q == Σ credits_q exactly; all operands int64 and nonnegative

INV-ENERGY:
   | E_tot(t) − E_tot(0) − ∫₀ᵗ (P_sun − P_heat − P_burial) dt |  <  τ_energy
```

Implementation: transfers use one availability-limited integer debit and exact integer credit apportionment.
`close_books()` runs after every authorizing step, checks each world independently, and rejects negative or
`>=2^62` state. Concentration is derived in float64 for rates and is never stored as a synchronized mirror.

**Prior-build failure modes this structurally forecloses:** (a) two free-energy sources (sunlight cap-well **and** vent flux both unbounded); (b) static Perlin cap production consuming no pool; (c) decoupled graze knobs; (d) remineralization matter discarded; (e) fecal pump routing an *energy* fraction as if it were *biomass*. Each is now impossible.

**S1 acceptance gate:** exact closure holds after every step of the 10^6-step closed soak; the uncapped
`d_dd=0` pulse still rises, draws down nutrient, and crashes; and the same-horizon `dt_eco/2` run converges on
the frozen bloom metrics. Row-sliced and whole-grid transaction order agree exactly. **Nothing proceeds past S1
until these are green (P7).**

### 6.7 Geology as source and sink; iron patterning

Geology is the ultimate **source** and **sink** of the element, which *strengthens* the top invariant (burial is a transfer, not a deletion). All terms are **additive, gated, keyed to geometry**, queried through the terrain/geology interface (§3.4); the core loop never changes when a term is added (P4/P6).

- **Sources (deep):** hydrothermal vents emit nutrient + Fe + heat into the fields at depth (a sunlight-independent chemosynthetic niche, before any land). Weathering/rivers at the coast defer with all land features to **S9**.
- **Sink:** detritus reaching the seafloor transfers to the **sediment reservoir** `Sed` (burial), balanced against sources so the pool reaches a non-conservative steady state.
- **Iron limitation (grow-on-demand, P6).** S1 runs a **single** nutrient. A second field `Fe(x)`, derived from geology (§3.6), enters Liebig as an additional `min` term `f_Fe`. Because Fe is sourced only at depth and scavenged onto sinking particles, the surface stays Fe-starved unless upwelling lifts it → **HNLC deserts** appear where Fe-limited (Martin & Fitzwater 1988; Boyd et al. 2007). Add Fe only when the evolution needs the gradient.

### 6.8 Emergent dynamics (not tuned, P1/P8)

Carrying capacity, blooms, deserts, and boom-bust are **derived** from the conserved loop, not imposed. There is **no `carryingCap`/`nicheCapFrac` knob** — its presence would flag a non-conserved base.

| Phenomenon | Mechanism (emergent) |
|---|---|
| **Carrying capacity** | production ÷ real per-capita Holling-II intake; local density-dependence from shared-cell depletion within a tick |
| **Self-terminating bloom** | nutrient pulse → production spike → drawdown empties `Nd` → production collapses → biomass declines |
| **Oligotrophic desert** | persistent low `Nd` (gyre interior / deep / Fe-starved) → chronic `f_lim` low |
| **Boom-bust** | predator–prey + resource lags on a conserved currency (Lotka-Volterra-like, grounded in real transfers) |
| **Biogeography** | upwelling provinces (eutrophic) vs downwelling gyres (oligotrophic) as distinct habitats (S6 currents) |

**Acceptance (S1→S3):** activating the conserved loop must **not** require re-introducing a cap knob; a bloom must draw its own nutrient down and crash without a hand-coded ceiling; the drifter population must plateau at production/intake. Breakage on activation is a **diagnostic lead** — the response is to re-anchor `μ_max`/production to real NPP, **never** to re-soften depletion and re-hide a mint (§1.2).

### 6.9 Interfaces, tensor shapes, and engineering risk

**Signatures (torch, `inference_mode`, f32 hot loop / exact int64 mass-quanta reservoirs / f64 validation arm):**

```python
def primary_production(Nd, Bp, light, cfg) -> (dBp, dNd)          # dNd == -dBp; clamp dBp<=Nd
def remineralize(Bd, depth, cfg)          -> (dBd, dBm, dNd)      # dBm+dNd == -dBd; BGE split
def vertical_mix(Nd, Kz, dz, dt)          -> Nd_new               # flux-form, conservative, double-buffered
def graze(pos, C_form, field, cfg, dt)    -> (I_assim, egesta, field_debit)   # Holling-II, clamped
def metabolism(mass, temp, cfg)           -> P_basal              # Kleiber M^alpha
def reproduce(parent, M_off, cfg)         -> (child, parent_debit)            # construction cost
def predation_step(bodies, caps, hash, cfg) -> transfers         # find->close->seize->consume, conserved
def close_books(reservoirs)               -> ledger_residual      # f64; asserts INV-MASS/ENERGY
```

Field ops are batched stencils on `(W,Gx,Gy,Gz)`; creature-field coupling uses interpolated gather at `pos[W,N_cap,3]` and paired `index_add_` scatter for the field debit (deterministic slots).

**Engineering risk:**
- *Proven technique:* the equations and the conserved-transfer pattern are textbook and were validated in the prior build; the port is mechanical.
- *Uncertain (measure in S1):* the Eulerian-field vs Lagrangian-parcel fork under dense grazing (quantization risk); closed-loop **tuning fragility** (a conserved loop can oscillate — mitigate with source/burial damping and staged activation, one dial at a time with the ledger watched); grid resolution resolving HNLC vs macronutrient provinces.
- *Would invalidate the approach:* if the ledger cannot close to `<1e-9` without ad-hoc correction terms (an unpaired transfer remains hidden), or if the conserved loop cannot reach a stable non-trivial standing stock without a cap knob (the base is not really self-limiting). Both are S1 go/no-go signals.

**References:** Redfield 1934/1958; Liebig 1840; Monod 1949; Dugdale 1967; Martin et al. 1987; Ekman 1905; Holling 1959; Lindeman 1942; Kleiber 1932; Azam et al. 1983; del Giorgio & Cole 1998; Field et al. 1998; Platt & Irwin 1973; Martin & Fitzwater 1988; Boyd et al. 2007; Tyrrell 1999.

---

## 7. Verification, Roadmap & the Embodiment Seam

This section specifies how SirRobin proves it is correct, the order it is built in, and the day-one seam through which an external agent will later drive a creature. Every gate carries a numeric threshold or an explicit "TBD-on-hardware" marker.

### 7.1 Verification discipline

#### 7.1.1 The primary gate is conservation, not byte-identity (P1)

The top-level correctness gate is exact int64 conservation for quantized reservoirs. Float mechanics and
independent rate corroboration use dimensioned mixed tolerances. Bit-for-bit float replay is a secondary,
same-machine-same-device diagnostic — never the primary pass/fail gate.

**Rationale (the prior-build failure this prevents).** The donor's CI gate was **byte-identity** of a seeded run against a stored golden (FNV/BitConverter hash). That invariant is satisfiable by code that does nothing — so faithful mechanics shipped **disabled behind `gain=0` dials** to keep the golden green, and every new term shipped "ramped, defaulting to today's behavior." Fidelity therefore shipped *dark*. Byte-identity answers "did the numbers change?"; a conservation gate answers "do the books close?" and is by construction **robust to sub-tolerance float noise**, so it creates no pressure to gate mechanics off.

**The invariant, precisely.** Every tracked quantity Q (nutrient mass; energy) lives in a fixed set of named reservoirs. Let `R_i(t)` be reservoir masses and `X_ext(t)` the net external flux (zero for a closed box; nonzero only for declared sources/sinks, each itself a *transfer*):

```
closure_residual(t) = | Σ_i R_i(t) − Σ_i R_i(0) − ∫₀ᵗ X_ext dτ |  /  max(Σ_i R_i(0), ε)   <   τ_cons
```

For S1 mass reservoirs this expression is implemented in integer quanta and must equal zero after every step of
the 10^6-step soak. There is no tolerance or drift budget to consume. Mixed-tolerance residuals remain
appropriate for independent float rate and mechanical equations, not conserved mass state.

**Invariant CI-CONS-1 (no free channel).** Every code path that changes any `R_i` must, in the same transaction, apply the equal-and-opposite change to another `R_i` (or to `X_ext` with a declared source). Enforced as a test pattern (§7.1.4). The two-free-energy-sources bug and the "marine-snow silently deleted" bug are exactly the class this forbids.

#### 7.1.2 Determinism posture

Per §2.7, exact bookkeeping and reproducible discrete choices are hard requirements. Float replay is compared in
a same-device eager deterministic diagnostic and its maximum divergence and tax are reported; bit identity is
not a scientific authorization gate. Cross-machine and CPU↔GPU bit identity remain out of scope.

#### 7.1.3 Precision — deliberate hybrid

Per §2.8: three float64 islands (`LambK` precompute; global ledger; S0 oracle-match config), everything else float32. Energy closure gated at `< 1e-6` (f32 per-step algebraic) / `< 1e-3` (f32 10⁵-step KE budget) / `< 1e-6` (f64 budget).

#### 7.1.4 Test culture: fakes-only, tolerance-invariant, telemetry-first

- **Tolerance/invariant tests plus independent frozen fixtures.** Assertions use dimensioned mixed tolerances,
  the executable prefix budget, and committed values emitted independently before production. Tests never
  generate their own expected results. Long-horizon trajectory divergence under a chaotic gait is expected and
  is not a bit-trace gate.
- **Fakes over mocks, boundaries over internals.** Tests exercise a subsystem through its published contract (`DevelopedBody`, `MediumSample`, the state-contract dict) using in-memory fakes; they never reach into another package's internals. Enforced by the import-boundary linter (§2.2).
- **Oracle fixtures from the C# donor (§7.5).** Frozen conformance fixtures — a LambK grid, single-step force terms, 8-second episode aggregates across H1/H2 genomes — guard the port. Recorded *values*, checked to `τ_rel`, not a live C# dependency.
- **Telemetry-first observability.** Every foundation slice ships a cheap headless observation surface before it ships a render: parquet/jsonl dumps of every gate metric plus profiler attribution, CSV/heatmap plots of reservoir balances, per-cause mortality, lifespan, morphospace occupancy. **No claim of correctness or performance is ever made from code inspection** — only from a telemetry artifact. 3D render is deferred to S7.

### 7.2 S0 — "SpikeSwim": the verification spike, in full

S0 verifies the project's optimistic performance premise ("faithful full-population physics is affordable when vectorized") **before the architecture is bet on it** (§2.1) — the measure-first discipline whose *absence* was the root architectural wound of the prior build.

**What it builds.** A standalone PyTorch program — **no Unity, no ecology, no feeding, no steering** — that
ports the donor's `Sim.Step` (one-shot, **frozen-heading**; *not* `StepLive`) to a batched tensor kernel over
`B=W·N_cap` fixed-slot bodies. The donor-shaped episode is 3 s warmup plus 5 s measurement at `dt=1/120`.
A separate 100,000-step configuration emits the executable prefix drift budget. A fixed lifecycle schedule
measures dead-slot recycling without compaction or an invented ecological death model.

**Canonical layout under test:** struct-of-arrays `(W,N_cap,k)` plus `alive`, with fixed
`(W,N_cap,17,...)` segment slots plus `seg_mask`, exactly as §2.4 specifies. Slot 0 is the identity sentinel;
pose is a bounded six-pass depth scan. H0 measures the homogeneous denominator, while the frozen H1/H2
corpora authorize and expose masking/heterogeneity cost. A flattened layout is not built inside S0.

**Sweeps.** `B ∈ {1, 64, 256, 1024, 4096, 16384, 65536, 262144}` on CPU and CUDA. Heterogeneity: **H0** homogeneous 6-seg (isolates vectorization), **H1** realistic ragged 2..16 mean~6 with mirror pairs and ~40% fin tails, **H2** skewed worst-case (mostly 2–3 seg + rare 16). Acceleration ladder: **r0** eager → **r1** `torch.compile(mode='reduce-overhead')` → **r2** explicit CUDA-graph capture. Two dtype configs: `float64` (oracle-match) and `float32` (throughput/invariant).

**The four quantitative acceptance gates:**

| Gate | Metric | Threshold |
|---|---|---|
| **A** | Scaffold and representation | Fixed slots, static lifecycle, checked exact transfers, artifact integrity |
| **B** | Physical/oracle fidelity | Untouched gain0 donor plus independent analytic gain1; H1/H2 mandatory; zero authorization regularization |
| **C** | Mechanical consistency | Mixed-tolerance force identities, discrete `R_step`, executable 100,000-step prefix budget |
| **D** | Reproducibility posture | Exact discrete decisions hard-gated; float replay reported as a diagnostic |
| **E** | Throughput | Historical 1,000/90M gate: NO-GO. Revised authority: H1/H2 non-OOM with zero regularization and minima ≥600k at 5,000 live and ≥1.2M at 10,000 live: GO. |

The complete equations, frozen H0/H1/H2 distributions, tolerances, benchmark protocol, and GO/CONDITIONAL
GO/NO-GO classes are normative in
`docs/archive/plans/2026-07-12-sirrobin-S0-consolidated-implementation-plan.md`. This design document does
not restate or weaken them.

**Falsifiers (any one trips → cheaply kill or revise the thesis before pouring commits on it):**
- **F1** — Ragged heterogeneity defeats batching: H1 flattened slower than a per-body Python loop at single-world N, or H2 padding craters.
- **F2** — Depth-scan / reduction > 50% of step time.
- **F3** — Per-op launch overhead dominates at realistic N even at r2 (→ narrow GPU to many-worlds; keep S0–S2 on CPU).
- **F4** — Determinism tax > 2× on GPU.
- **F5** — Oracle un-portable without a de-vectorized loop (aggregates diverge > 1e-3).
- **F6** — float32 semi-implicit energy drift breaches the gate (monotone, not oscillating).
- **F7** — Churn/compaction swamps the step.

**META-FALSIFIER (the load-bearing one).** *A green H0 number does NOT authorize the architecture.* H0 hides exactly the two risks that matter — ragged heterogeneity (F1) and population churn (F7). **The gate that authorizes the build is H1/H2 clearing (d) while (a)/(b)/(c) hold.** Any report that leads with an H0 number is non-conforming.

### 7.3 Build roadmap — S0…S9, depth-first (P7)

Nothing proceeds past a slice until its acceptance criterion (AC) is green on telemetry.

| Slice | What it lands | Acceptance criterion (AC) |
|---|---|---|
| **S0 SpikeSwim** | Batched frozen-heading Lighthill step; the four gates of §7.2. | All four gates pass at **H1/H2** (not just H0); go/no-go decision recorded from telemetry. |
| **S1 Conserved single-nutrient economy** *(keystone)* | Exact four-reservoir loop: Liebig×Monod drawdown, producer loss, BGE split, microbial turnover, Martin sinking, `Nd/Bp/Bm` mixing. | **Books close exactly** every step over 10⁶ steps; uncapped `d_dd=0` bloom/crash passes; `dt_eco/2` convergence passes; no terminal reservoir trap. **Nothing proceeds until green.** |
| **S2 One canonical body + live locomotion** | `BodyGraph → DevelopedBody → Sim.StepLive` (yaw-integrating P-controller — **re-measure throughput against StepLive before committing**). Feeding/metabolism/defense **derived from morphology** — `eff[]` deleted. | Every live creature swims via the one canonical body; `eff[]` gone; StepLive throughput re-cleared. Genome P0/P1 AC: batched torch develop-walk reproduces C# `Measure()` aggregates within tolerance (§5.3). |
| **S3 Feeding / metabolism / reproduction on conserved energy** | Holling-II intake + assimilation loss → detritus; Kleiber metabolism; real morphology-derived juvenile construction cost. | A cohort feeds, grows, reproduces, dies with **energy books closed** end-to-end; population sustains without minting energy. |
| **S4 Predation as a staged contest** | find → close → seize → consume, all conserved; **no seeded predator**. | A predatory strategy arises implicitly; transferred mass fully accounted (prey tissue → predator + detritus). |
| **S5 Speciation / mating / taxonomy** | NEAT-innovation-aligned crossover + compatibility-distance gating + spatial assortative mating; observational taxonomy. | **Falsifiable milestone:** a single deterministic run where a panmictic population splits into two non-interbreeding clusters under disruptive selection. Do not proceed until demonstrated *or* its absence root-caused to the ecology (niche diversity), not the encoding. |
| **S6 Transport / currents / upwelling** | Advect nutrients/plankton/detritus; Ekman transport; circulation ladder (§3.8) behind the field interface. | Advected fields still close the books; patchy productivity and larval dispersal emerge; downstream ecology consumes them through the unchanged field query. |
| **S7 Render / observation surface** | Unity as a remote/replay viewer over the state contract; richer telemetry. | Render **never feeds fitness** (read-only); viewer reproduces a checkpointed run. |
| **S8 RL / embodiment loop** | Talos state contract → ROS2 → TurtleBot3 (§7.4). | **Gated (§7.4.4):** books closed (S1–S4 green) **AND** Sophia's action-interface assumption verified against real code. A CORE-only policy drives a sim fish end-to-end via the contract. |
| **S9 Plants + bidirectional water↔land crossing** | Plant kingdom (L-system/CPPN); land/rivers/sediment cascade (§3.7); the emblematic endpoint. | Emergent, unscripted crossing of the water/land interface **in both directions** (sea→land limb, land→sea re-streamlining), on the additive medium-dependent physics with **no mode switch**. Research frontier. |

**Near-term de-risking milestone — the sea robin.** Before the full S9 crossing is attempted, the reachable proof point is an emergent **benthic fish that walks on re-purposed lower pectoral fin-rays** while still swimming (§4.6, §5.10). It needs neither air-breathing nor leaving the water; it exercises the exact mechanism S9 depends on. Expected reachable once S2 (unified additive locomotion) + S3/S4 (a benthic foraging gradient) are green. Treat its emergence as the milestone that **de-risks the water↔land frontier**.

### 7.4 The RL / embodiment seam — the sim↔Talos state contract

The embodiment loop is the project's reason-for-being but is **deferred in execution and kept as a day-one seam**: the canonical population representation (§2.4) *is* the observation tensor, so the contract adds a schema, not a re-layout.

#### 7.4.1 Shape and framing

A versioned **dict-of-tensors** with a leading `(W, N_cap)` batch dim + alive-mask, nested **CORE/EXT** (mirrors `gymnasium.spaces.Dict({'core':…, 'ext':…})`) for both observation and action. SI units; ROS REP-103 body frame FLU; radians; seconds. Every exchange carries a `Header`:

```
Header = { contract_version: SemVer, tick: int, sim_time_s: float64,
           world_id: int, agent_id: int,
           embodiment ∈ {SIM_FISH, TURTLEBOT3}, ext_present: bool }
```

#### 7.4.2 CORE — the differential-drive-executable intersection of fish and TurtleBot3

CORE is exactly what a differential-drive base can execute, so a CORE-only policy transfers to the physical robot unchanged.

- **Action OUT (the entire CORE action is this 2-vector):**
  - `surge_effort ∈ [-1, 1]` → `Twist.linear.x`
  - `yaw_rate ∈ [-1, 1]` → `Twist.angular.z`
- **Obs IN:** `lin_vel(3)`, `ang_vel(3)`, `orientation(4)` (or `heading(1)` in a 2-D profile), `range_egocentric(K)` (robot LaserScan / fish nearest-neighbour+terrain on the same K beams), `flow_rel(3)`, `energy(1)` (fish metabolic reserve / robot battery SoC), `contact`.

**Validated by the crown jewel, not merely asserted compatible.** SwimEval's cruise path **zeroes vertical COM velocity** and `StepLive` **integrates yaw only** (§4.4) — so the 2-DOF `{surge, yaw}` CORE **is the fish's actually-realized locomotion DOF today**, not a lossy down-projection. Heave/pitch/roll belong in EXT precisely because the kernel does not integrate them. ⇒ fish↔robot **action-side transfer risk ≈ 0**; the residual risk is entirely **observation-side** (§7.4.4, open Q #5).

#### 7.4.3 EXT — sim-only richness, never load-bearing for transfer

Fish: **chemical gradients** {food, predator kairomone, …} on the same K beams — the fish's *primary* navigation sense, deliberately EXT because it has **no robot analogue**; plus light/depth/temp/marine-snow, per-segment proprioception, gape, heave/pitch/gait actions, feeding strike. Robot: camera, IMU, wheel ticks, joint states. **Design rule:** a behaviour using only CORE obs + CORE action is, by construction, transferable to the TurtleBot3 unchanged.

#### 7.4.4 Serialization and gating

- **Serialization ladder.** In-process (Sophia in the same Python) → pass torch tensor dicts directly (zero serialization; pydantic shape/dtype check in debug only). At a process/network boundary → canonical IDL is **Protobuf proto3** (field-number evolution, additive-only within a major; `reserved` for removals; SemVer-major bump for any breaking reshape). ROS2 face = generated `.msg` + `geometry_msgs/Twist` matching CORE 1:1 (pin a distro). msgpack/JSON for telemetry/checkpoints. A **conformance test freezes a CORE fixture and asserts pydantic + Gymnasium space + `.msg` agree field-for-field.**
- **The gate on building S8 (two conditions, both required):**
  1. **Books closed** — S1–S4 conservation invariants green.
  2. **The Sophia action-interface assumption is verified against real Sophia code** — whether Sophia emits *symbolic/parametric intents* or *continuous vectors*. This is the **highest strategic risk in the project.** Mitigation is structural: keep the contract **continuous-first**; push any symbolic→continuous decode into a **Talos-side adapter, never the sim schema**. If Sophia is symbolic, `Talos.compile(intent)→cmd_vel` is the adapter; if continuous, that step collapses to pass-through. Do not build S8 until this is verified against live code, not the placeholder `ExecutorShim`.

### 7.5 Salvage plan — running code vs offline oracle/fixtures

The C# donor is **not called live**. Two disposition classes:

**(A) Re-ported as running torch code** (the equations and design, re-expressed vectorized):
- `SwimEval`/`SwimmerSim` — the deterministic Lighthill/Lamb locomotion (the crown jewel), re-ported as the **first force-contributor** of an additive articulated-body model (§4.5).
- `BodyGraph`/`BodyGenome` + `Measure()` development → the bounded recursive part-**graph** + NEAT innovation ids + batched fixed-depth develop-scan (§5).
- `SimUnits` (frozen Joule anchor + bright-line) — measurements, never balance knobs (§1.2).
- The conserved-economy **design** (single-nutrient closed loop, BGE split, Martin profile) — stateful conserved reservoirs (§6, S1).
- The field machinery (`WorldSampler`/`PlanktonField`/`NutrientField`/vents) — behind the field-first interface (§3).
- `Taxonomy`/`TaxonomyNames` — observational emergent taxonomy (§5.11, S5).
- The Talos EntitySpec CORE/EXT read-only observation seam → §7.4's state contract.

**(B) Demoted to offline oracle / fixture generator** (reference values, not running code):
- The C# `SwimEval` donor becomes a tiny headless console harness (or the existing `ReconstructForTest`/`LambKForTest`/`CoastTest`/`MomentumLedger` seams). It emits **frozen fixtures**: a LambK grid, single-step force terms, 8 s episode aggregates across H1/H2 genomes.
- Conservation-invariant tests move from C# BitConverter/FNV goldens to **pytest + tolerance invariants**.

**Explicitly superseded (do not salvage as architecture):** C#/Unity as host; the data-far/physics-near LOD proxy; the dual genome (`eff[]`); static non-conserving Perlin fields; byte-identity-as-gate; the 3,067-line `OceanColony` god-class; asexual clone-and-mutate + post-hoc cosine "species." The durable salvage is **validated equations + recorded oracle values**; the C# *text* does not run in SirRobin. *(The prior build's 74-commit master-sequence — small individually-verifiable commits, runtime-verify every step — is retained only as **process** discipline.)*

---

## 8. Consolidated Risk & Open-Questions Register + What's Next

### 8.1 Risks (with mitigation)

| # | Risk | Sev | Mitigation |
|---|---|---|---|
| RK-1 | **F1 — ragged heterogeneity defeats batching** (the vectorization thesis fails at H1/H2). | High | Measure H1/H0 and H2/H0 taxes on the canonical fixed-slot path; profile before proposing another layout. |
| RK-2 | **F3 — per-op launch overhead dominates** at realistic N even at r2. | High | Acceleration ladder r0→r1→r2; if r2 fails, **narrow GPU to many-worlds, keep S0–S2 on CPU** via `device=`. |
| RK-3 | **Wrong-invariant regression** — pressure to gate mechanics off to keep a run green. | High | Conservation is the *only* correctness gate; bit-identity is same-device regression only; no mechanic ships behind a green-keeping `gain=0`. |
| RK-4 | **Sophia action-interface unknown** (symbolic vs continuous) — highest strategic risk to S8. | High | Contract continuous-first; symbolic decode in a Talos adapter, never the sim schema; **S8 gated on verifying live Sophia code.** |
| RK-5 | **F6 — float32 long-run energy drift** breaches closure. | Med | Gate every retained prefix with `D_k`; report signed monotone bias separately; semi-implicit integration measured in S0. |
| RK-6 | **F4 — GPU determinism tax** > 2×. | Med | Precomputed-unique-slot deterministic scatter; CPU-first fallback; deterministic segment reductions. |
| RK-7 | **C#→torch port bugs the algebraic energy identity won't catch** (a sign/order error that still balances). | Med | Conformance fixtures (LambK grid + single-step forces + aggregates) are the real guard, not the energy identity alone. |
| RK-8 | **F7 — lifecycle churn cost** swamps the step. | Med | Fixed-capacity in-place dead-slot recycling, measured with a committed schedule; no compaction. |
| RK-9 | **Emergent speciation may not occur** (S5) — binding constraint is ecological niche diversity, not encoding power. | Med | Magic-trait coupling option; gate on the two-species-split test; root-cause absence to ecology before touching the encoding. |
| RK-10 | **Expressive genome manufactures morphospace the ecology can't reward** (complexity without emergence). | Med | Kleiber + prune pressure; new structure must earn fitness before more DOF is unlocked. |
| RK-11 | **Ellipsoid a/b/c reinterpretation shifts Lamb added-mass terms.** | Low-Med | Validate vs the box baseline; re-tune Lamb coefficients if the tolerance is breached (open Q #3). |
| RK-12 | **F2 — pose depth-scan / reductions dominate** the step (> 50%). | Med | Profiler attribution in S0; shared gather→compose→scatter kernel for pose and development; fixed 6-pass bound. |
| RK-13 | **Closed ecological loop oscillates / is tuning-fragile** at S1. | Med | Source/burial damping; staged one-dial-at-a-time activation with the ledger watched; re-anchor to real NPP, never re-soften depletion. |

### 8.2 Open questions (needing a decision or external verification)

| # | Question | Blocking for | Owner |
|---|---|---|---|
| 1 | **Target hardware** (dev CPU core-count; GPU model/VRAM/FP64 tier; cluster shape). Pins the S0 throughput floor from placeholder to real. | S0 gate (d) | **Needs owner input** |
| 2 | **Sophia's action interface** (symbolic / continuous / both) — verify against live code, not the placeholder shim. | S8 (not earlier) | Owner + Sophia team |
| 3 | **Ellipsoid vs box readout** — does a/b/c require re-tuning SwimEval's Lamb terms, and at what tolerance vs the oracle? | S2 / genome P1 | Physics |
| 4 | **Determinism numeric definition** — resolved by the three tiers in §2.7; float identity is diagnostic. | S0 | Architecture |
| 5 | **CORE-only foraging sufficiency** — the chemical gradient (the fish's primary sense) is EXT with no robot analogue. If a CORE-only policy cannot forage, the shared task becomes navigate-to-goal, not forage. | ~S8 (research) | RL |
| 6 | **StepLive throughput** — S0 ports the lighter frozen-heading `Step`; the live yaw-integrating `StepLive` must be **re-measured before S2 is committed.** | S2 | Physics |
| 7 | **Field discretization for reactive resources** — Eulerian-interpolated grid vs Lagrangian parcels vs hybrid, under dense grazing. | S1 | Ecology |
| 8 | **ROS2 distro** (pins `Twist` vs `TwistStamped`, `sensor_msgs` versions); whether the physical robot needs a chemical-sense analogue. | S8 | RL / infra |
| 9 | **Sophia in-process vs out-of-process** — decides whether Protobuf serialization is ever in the hot path. | S8 | Architecture |
| 10 | **Differentiate the world, ever?** The biggest torch-vs-JAX fork. "Never differentiate" keeps torch + `inference_mode` ideal; "differentiate" is the strongest JAX pull. Recommendation: **assume no; flag, don't decide now.** | (architectural) | Architecture |
| 11 | **Precise S5 two-species-split acceptance test** — disruptive-selection setup, population size, spatial radius, run length constituting a valid split. | S5 | Evolution |
| 12 | **Realistic `N_cap` and max-segments-per-body** — pin against the S0/H1 raggedness profile. | S0 / S2 | Architecture |

### 8.3 What's Next

**Build S0 / SpikeSwim first.** It is the single gate that authorizes (or kills) the whole architecture, at the lowest cost, before any commits are poured onto an unverified premise. Concretely:

1. Stand up the capability packages (`numerics`, `physics`, `validation`, `benchmarks`) and import-linter contract (§2.2); milestone names do not namespace runtime modules.
2. Port donor `Sim.Step` (frozen-heading) to the canonical fixed-slot kernel (§2.4.2, §4.3) with the six-pass pose scan and both independent oracle arms.
3. Wire the C# `SwimEval` donor as an **offline fixture generator** (§7.5-B) and freeze the LambK / single-step-force / 8 s-aggregate fixtures across H0/H1/H2 genomes.
4. Run the sweep of §7.2 and produce a **telemetry artifact** reporting all four gates, the H1/H2 taxes, the CPU↔GPU crossover `B*`, and profiler attribution.
5. Record the go/no-go from the H1/H2 numbers (the meta-falsifier: a green H0 does not authorize the build). Only on green does S1 (the conserved single-nutrient economy — the keystone whose books must close before any breadth) begin.

Pin open questions #1 (hardware) and #6 (StepLive re-measurement) as prerequisites to reading the S0 result as authoritative; everything else defers to its slice.
