Confirmed against the design doc: currents/flow are S6 (design:284, 385, 412), light is analytic at S1 (design:403, 412), and the `{electro↔electric}` modality (design:960) has **no backing field anywhere** in the roadmap. That grounds the S4 sensory trim. Here is the revision.

---

## FIXES: Roadmap consistency, evolutionary sequencing & the import firewall (#16, #17, #18)

These three fixes are coupled: the DAG (#16) is the spine, the S4 resequencing (#17) is the largest edit to that spine, and the import firewall (#18) is the CI contract that the DAG's layer order asserts. I give each as concrete replacement text with the exact plan lines it supersedes.

---

## Finding #16 — One authoritative dependency DAG

**Problem restated.** Three artifacts disagree: the slice table (plan:556–564) says `S5 gating dep = S4` and `S6 gating dep = S1 fields`; the ASCII graph (plan:720–740) routes `S3→S5` and `S5→S6→S7→S8`; the critical-path prose (plan:742) says "`S3→S5`, S4 branches off S3" **and** "S6/S7 parallelizable once S5 lands." So S5's parent is simultaneously S4, S3, and (via the graph) S3+S4; S6's parent is simultaneously S1 and S5. Unresolvable as written.

**Resolution — the single source of truth is the artifact-level dependency table below; the ASCII graph and the critical-path prose are regenerated from it.** Dependencies are stated as *named produced artifacts a slice consumes*, not slice numbers, so they cannot drift.

### 16.1 REPLACES the slice-table "Gating dep" column (plan:556–564)

| Slice | Consumes (artifact-level dependency) | Rationale |
|---|---|---|
| **S0** | Scaffold: `numerics`, oracle harness, import-linter, det conftest | go/no-go spike |
| **S1** | S0 GO gate; `core/ledger` (f64 reservoir book); `fields.contracts` | keystone nutrient economy |
| **S2** | S0 (`physics.pose` 6-pass kernel, `physics.swim_step`, oracle fixtures); S1 (`core/ledger` energy reservoir, for the Kleiber metabolism read S2.12/§524); `fields.contracts` protocol layer | canonical body + StepLive |
| **S3** | S2 (`genetics.develop`→`DevelopedBody`, morphology-derived `capabilities`, `step_live`); S1 (`Nd/Bp/Bd` reservoirs + `transfer`/`close_books`); `core.spatialhash` (real neighbor query, promoted from S0 stub) | close the energy loop; **lands the asexual evolutionary engine (§17.2)** |
| **S4** | **S3 only** (feeding economy, `reproduce`, the S3 asexual mutation/inheritance/lineage engine, energy ledger, `core.spatialhash`); the S1 sensory field set (`light`, `Nd/Bp/Bd`) | seeded-predator mechanism (engineering); **not** unseeded emergence |
| **S5** | **S3 only** (population + genome + asexual `reproduce`/mutation engine, on which sexual crossover is layered); genome phase P2 | speciation/mating engine. **S4 is NOT a dependency**; S4's ecology is an *optional enrichment input* to the emergence experiment RX‑2, not a build gate |
| **S6** | **S1 fields only** (`fields.scalar_field`, `sample(x)->(value,grad)`, flux-form advection + positivity limiter primitives) | currents/weather/transport. **Buildable any time after S1; NOT gated on S5** |
| **S7** | `observe` read-only surface + the versioned `SimulationSnapshot` (the #14 checkpoint); ≥ S2 (bodies to render). Renders S1–S6 content opportunistically as each lands | viewer; a consumer, never a producer |
| **S8** | S1–S4 invariants green; `observe/contract` CORE/EXT schema; **external Sophia interface verification** (out-of-band) | embodiment, dual-gated |
| **S9a** | S2 additive contributor core; S3 (energy); S4 (benthic foraging gradient) | sea-robin walk de-risk |
| **S9b** | S9a; S6 (currents/medium richness); genome reversibility guard | full water↔land crossing |

**Two contradictions explicitly killed:** (1) `S5` now consumes **S3, not S4** — S4 becomes a parallel branch off S3 that S5 does not wait on. (2) `S6` consumes **S1 only** and is decoupled from S5 — the "parallelizable once S5 lands" claim is deleted.

### 16.2 REPLACES the ASCII graph (plan:720–740)

```
SCAFFOLD(S-1) ──> S0 ──(GO: H1/H2 clear + a/b/c/oracle hold)──> S1 ──┐
   │ numerics·oracle·import-linter·f64 ledger·SimClock·RNG(keyed)    │  (Nd/Bp/Bd reservoirs,
   │                                                                 │   transfer/close_books)
   ├──> genome P0 (develop-scan, shares S0 pose kernel) ──┐          │
   └──> fields protocol layer (Field/Geology contracts) ──┴──> S2 <──┘
                                                            │ (canonical body, StepLive,
                                                            │  kill eff[], P1 ellipsoid, capabilities)
                                                            ▼
                                                           S3 ──────────────────┐
                                        (feeding·metabolism·reproduction·        │ asexual engine +
                                         ASEXUAL MUTATION+INHERITANCE+LINEAGE)    │ lineage validated
                                              │                    │             │
                                    (S4 branch)│          (S5 branch)│            │
                                              ▼                     ▼            │
                                             S4                     S5           │
                                 (SEEDED predator,          (mating/crossover,   │
                                  mechanism-validated)       genome P2, split)   │
                                              ┆                     ┆            │
                                     RX-1 unseeded          RX-2 species split   │
                                     emergence (research,   (research gate,      │
                                     blocks nothing)         blocks nothing)     │
                                                                                 │
   S1 fields ──> S6 (currents/transport, INDEPENDENT of S5) ──┐                  │
                                                              ├──> S7 (viewer) <──┘ (needs ≥S2 + SimulationSnapshot)
   observe surface + SimulationSnapshot ───────────────────► │
                                                              ▼
                             S1–S4 green + external Sophia verify ──> S8 (embodiment, dual-gated)

   {S2 additive core, S3, S4} ──> S9a (sea-robin) ──{+ S6}──> S9b (two-way crossing)
```

Solid `──>` = hard artifact dependency; `┆` = research experiment that consumes the slice but gates nothing downstream (§17.3).

### 16.3 REPLACES the critical-path paragraph (plan:742)

> **Critical path:** Scaffold → S0 → S1 → S2 → **S3**. From S3 the tree forks into two independent branches that share no build dependency: **S4** (seeded predation) and **S5** (mating/speciation) — neither waits on the other, and S5 does **not** depend on S4. **S6** (currents/transport) depends only on the S1 field layer and may be built at any point after S1, in parallel with the S2–S5 spine; it is **not** gated on S5. **S7** needs only the `observe` surface plus the `SimulationSnapshot` checkpoint and at least S2 (bodies); it renders later slices opportunistically as they land. **S8** slips independently behind its dual gate. **S9a** is reachable once {S2, S3, S4} are green and should be pulled forward as the frontier de-risk; **S9b** additionally needs S6. **Cross-cutting:** genome P0→P4 shadows the spine (P0 with S0, P1 with S2, **the asexual mutation/inheritance/lineage engine with S3**, P2 crossover with S5, P3/P4 deferred); the fields protocol layer exists before S1 consumes it, its rich generator deferred to S6/S9 with zero downstream change (P4/P6).

---

## Finding #17 — S4 demands evolution before an evolutionary engine exists

**Problem restated (plan:574–580).** S4's acceptance requires an *unseeded* predator to arise, yet (a) no asexual mutation operator, rates, inheritance, or lineage tracking is specified anywhere before S4 (crossover is deferred to S5; the S3 `reproduce` at plan:570 clones "juvenile body from genome" but never defines the *mutation* that makes offspring differ from parent); and (b) S4's detection couples to four modalities `{vision↔light, smell↔chemical, lateral-line↔flow, electro↔electric}` when the backing fields for two of them do not exist at S4 — **flow arrives at S6** (design:284, 385) and **no electric field exists anywhere** (design:960 pairs `electro↔electric` with a field the roadmap never builds).

The fix has four parts: (1) land + validate the asexual engine at S3; (2) trim the S4 sensory list to fields that exist; (3) make the S4 *engineering* acceptance a **seeded** predator; (4) reserve unseeded emergence as a separate research experiment with its own falsifiable milestone that gates nothing.

### 17.1 REPLACES the S3 component list (adds to plan:570) — the asexual evolutionary engine lands here

Insert into S3 **Components**, and add to the S3 acceptance (plan:572):

**Asexual reproduction engine (new S3 sub-components, each with a validation gate):**

- **`mutate(genotype_soa, rng_keys) -> genotype_soa`** — operates on the `genetics/genotype.py` node/edge SoA (§S2.4), static-shape (never changes `S_max`; capacity is padded, `alive`-masked). Three operator classes with fixed per-event rates drawn from the **counter-based keyed RNG** (cross-cutting resolution #13, keyed by `(seed, step, stable_entity_id, gene_iid, event_kind, draw_index)`):
  - *Parametric:* per-gene-parameter Gaussian perturbation on the **log-scaled** morphology params (`log_a/log_b/log_c`, amp_deg, phase, swim_freq) — `θ' = θ + N(0,σ_type)`, per-type `σ` in `core/config.py`, drawn with `event_kind=PARAM_MUT`. Log-scaling kills the additive ratchet (§5.2).
  - *Structural-add:* with rate `p_add`, activate a padded-but-inert node slot (assign a fresh monotone innovation id from `genetics/innovation.py`, `event_kind=STRUCT_ADD`). Because the slot pre-exists in padded storage and its draw is keyed by `gene_iid`, activating it **does not shift any other organism's RNG stream** (this is exactly what the keyed RNG buys us over the removed manifest, #13).
  - *Structural-toggle:* with rate `p_toggle`, flip a `Segment↔Surface` type bit or a `mirror` edge bit (`event_kind=STRUCT_TOGGLE`) — the reversibility the S9 guard later requires.
- **Inheritance:** `reproduce` (plan:570) is amended: child genotype = `mutate(clone(parent_genotype))`; child `DevelopedBody` = `genetics.develop(child_genotype)` — the *real* developed body, never a copied parent body, never a flat tank (already required at plan:570).
- **Lineage tracking:** `ColonyState` gains `stable_id [W,N_cap] i64` (monotone, never reused) and `parent_id [W,N_cap] i64` (from cross-cutting resolution #13). On birth into a recycled free slot, `stable_id ← next_stable_id++`, `parent_id ← parent.stable_id`. `species_tag` (plan:180) remains observational and never gates anything.

**Validation gates that must be green before S4 begins (added to S3 acceptance, plan:572):**

| Test | Asserts | Threshold |
|---|---|---|
| `test_mutation_shape_static` | `mutate` never changes `S_max` or any tensor shape; only pre-allocated slots activate | shape identical |
| `test_mutation_stream_stable` | activating an inert gene in organism A leaves organism B's keyed RNG draws byte-identical (the #13 property) | `max_abs(Δ)==0` |
| `test_inheritance_heritable` | over N asexual generations, a parametric trait's parent→offspring regression slope > 0 with no drift injection beyond `σ` | slope ∈ (0,1], variance matches `σ²` accumulation |
| `test_lineage_wellformed` | every live non-founder has a `parent_id` that existed; `stable_id` never reused across 1e6 births/deaths | exact |
| `test_selection_shifts_mean` | under a seeded fitness differential on one trait, cohort trait-mean moves in the selected direction over G generations, books still close | directional, `<τ_energy` |

Only when these five are green is the S4 detection/predation mechanism built on top.

### 17.2 REPLACES the S4 detection modality list (plan:578) — trim to fields that exist at S4

At S4 the existing environment is: analytic **light** (S1, design:403), the **chemical scalar fields** `Nd/Bp/Bd` sampled via `sample(x)->(value,grad)` (S1), and the **spatial hash** (`core.spatialhash`, real at S3). Flow does not exist until S6; no electric field exists at all. So S4 `find` supports exactly three modalities:

| Modality | Backing artifact (exists at S4) | Detection query |
|---|---|---|
| **Proximity / near-field mechanoreception** | `core.spatialhash` neighbor query | range-limited neighbor distance; the always-available floor sense |
| **Vision** | analytic `light` field (S1) | light-attenuated line-of-sight radius over spatial-hash neighbors: detection range `∝ f(I(x,z))` from `light.sample` |
| **Chemoreception (smell)** | `Nd/Bp/Bd` scalar fields (S1) via `sample(x)->(value,grad)` | gradient-ascent on the standing chemical/biomass field (patch- and carcass-plume finding), not per-individual scent (no per-entity emitted field exists yet) |

Replace the S4 `find` component (plan:578) with: *"`find` = two-sided `Detect(modality)` range vs opponent `Signature` across the modalities whose fields exist at S4 — `{proximity↔spatial-hash, vision↔light, chemoreception↔scalar-field-gradient}`; no dominant modality."*

**Deferred modalities, each behind a named field prerequisite (add to S4 as an explicit deferral note):**
- **Lateral-line ↔ flow** — requires the S6 current/velocity field; wired into `Detect` as a *post-S6 predation enrichment*, changing no consumer code (P4, behind `sample`).
- **Electroreception ↔ electric** — requires a dedicated bioelectric source field that **no slice currently builds**; it is out of scope for the engineering roadmap and reserved to research experiment RX‑1's enrichment set, gated behind first specifying that field. It must not appear in the S4 build.

### 17.3 REPLACES the S4 acceptance (plan:580) — seeded predator is the engineering gate; unseeded emergence is research

**S4 Acceptance (engineering, falsifiable) — SEEDED predator mechanism validation.** Seed a functional predator genome (a carnivory-capable morphology + intent) into a run and assert the *mechanism* is correct and conservative — this is the go/no-go for S4:
1. A seeded predator executes `find→close→seize→consume` reading **only** form- and physics-derived capabilities across the three S4 modalities (never a `carnivory` flag; P8) — verified by the `test_no_stat_vector`-style guard extended to hunting.
2. Every kill is **one paired transaction**: prey `(E, struct_N)` → predator credit `(AE·)` + detritus `((1−AE)·)` → `Bd`; `INV-TRANSFER < 1e-6`, energy books close end-to-end (`<τ_energy`, drift bounded-oscillating over ≥1e6 steps).
3. Predator/prey capabilities differ **only** through morphology-through-physics (`close` from real drag/yaw-torque + burst gear; `seize` two-sided `GripRate` vs `Evade`, overpower = mass ratio).
4. Telemetry: trophic occupancy, kill/attempt ratio, per-modality detection stats, mass-flow ledger.

**"Done" is falsified if:** a seeded predator cannot capture-and-digest without minting mass/energy, or any capability reads a stat flag rather than morphology.

**RESERVED — research experiment RX‑1 (unseeded predator emergence). Blocks nothing; not an engineering gate.** In a run seeded with **no** predator, does a predatory strategy arise implicitly under the S4 mechanism? This is a falsifiable *research milestone* (M‑S4R below), not a build gate: it consumes the validated S4 mechanism and (optionally) the S4/S6 ecology enrichment, and its failure is root-caused to encounter economics (world *dense not large*, design:2.9), never to a hunt-reward knob (P8 forbids it). Success signature: emergent trophic pyramid + arms-race trace.

### 17.4 REPLACES the S4 row of the slice table (plan:559) and the milestone/at-a-glance tables

- Slice table (plan:559) "Done is falsified if…": *"a **seeded** predator cannot capture-and-digest with prey mass fully accounted (prey debit == predator credit + egesta), or any hunting capability reads a stat flag."* (The "never arises unseeded" clause is moved to RX‑1.)
- Milestone table (plan:754) — split M‑S4 into two rows:

  | ID | Milestone | Observable = done | Risk class |
  |---|---|---|---|
  | **M‑S4** | Predation mechanism (engineering) | **seeded** predator hunts via form+physics only; prey mass fully accounted, books closed | solved (impl risk) |
  | **M‑S4R** | Unseeded predator emergence (research) | predatory strategy arises with **no** predator seeded; trophic pyramid + arms-race trace | 🟥 research frontier / blocks nothing |

- Engineering-vs-research table (plan:631) — S4 row becomes: *"Mechanism (build): 🟩 solved (seeded predator, validated). Emergent outcome (bet): unseeded predator + arms race — 🟥 RX‑1 (research experiment, gates nothing)."*
- Class column of the slice table (plan:559): S4 build class is **🟩** (the mechanism is engineering); the 🟦/🟥 tail lives entirely in RX‑1.

---

## Finding #18 — The import-linter firewall is incomplete

**Problem restated (plan:114–140).** The `layers` contract is fine (one-way ordering). But the single `forbidden` contract lists only `source_modules = {physics, genetics}` against `forbidden_modules = {fields.geology, fields.light, core, observe}`. It therefore does **not** stop: `core` from importing `physics.swim_step` or `fields.nutrient` internals; `observe` from importing any concrete module in any layer; or any consumer from importing `fields.scalar_field`/`fields.chem`/`fields.detritus` (only `.geology`/`.light` were named). The stated invariant — "cross-layer access via `contracts.py` only" (plan:33) — is unenforced.

**Resolution.** Adopt one uniform rule and enforce it exhaustively: **the only cross-layer import target of a layer L ∈ {physics, fields, genetics, core} is `L.contracts`; every other module in L is private to L.** `numerics` is the exception — it is the shared leaf-utility floor (dtype/quat/solve/reduce/ledger/rng/units), importable by all layers directly (the `layers` contract already prevents it from importing upward). This requires two small module-map additions (below) plus a fully enumerated `setup.cfg`.

### 18.1 Module-map additions (REPLACES/augments plan:78–95) so "contracts-only" is realizable

- **`genetics/contracts.py`** (new): public surface — `Genotype` dataclass + `develop(genotype, cfg) -> DevelopedBody` entry + any Protocol `core` needs. `genotype.py`/`develop.py`/`innovation.py` become private.
- **`core/contracts.py`** (new): public read-only surface — the `ColonyState` view type + a `ReservoirSnapshot` Protocol that `observe` reads. `config/clock/colony/state/spatialhash/ledger/economy/parcels` become private to `core`.
- **`physics`**: fold the `ForceContributor` Protocol + a `build_hydro_contributor(cfg) -> ForceContributor` factory into `physics/contracts.py`; `force.py`, `lamb.py`, `reconstruct.py`, `pose.py`, `swim_step.py`, `step_live.py`, `capabilities.py` are private. Consumers (e.g. `core`) obtain the concrete kernel only through the factory typed as the Protocol — so `core` never names a concrete physics module.
- **Composition-root exception:** the top-level **driver** packages (`spikeswim/`, `scripts/`) are *not* layers; they are allowed to reach concretes (the S0 driver runs `physics.swim_step` directly). They are constrained separately (18.3) to stay physics/numerics-only.
- Intra-layer imports remain free (e.g. `core.colony` may import `core.state`); the contracts-only rule is **cross-layer only** (matches plan:33).

### 18.2 REPLACES the entire `[importlinter]` block (plan:116–140)

```ini
[importlinter]
root_package = sirrobin
include_external_packages = False

# 1) One-way layering. numerics is the leaf; observe the top. No upward import.
[importlinter:contract:layers]
name = SirRobin one-way layering
type = layers
layers =
    sirrobin.observe
    sirrobin.core
    sirrobin.genetics
    sirrobin.fields
    sirrobin.physics
    sirrobin.numerics

# 2) physics internals are private: only sirrobin.physics.contracts crosses layers.
[importlinter:contract:physics-internals-private]
name = physics internals reachable only via physics.contracts
type = forbidden
allow_indirect_imports = True
source_modules =
    sirrobin.fields
    sirrobin.genetics
    sirrobin.core
    sirrobin.observe
forbidden_modules =
    sirrobin.physics.force
    sirrobin.physics.lamb
    sirrobin.physics.reconstruct
    sirrobin.physics.pose
    sirrobin.physics.swim_step
    sirrobin.physics.step_live
    sirrobin.physics.capabilities

# 3) fields internals private (fixes the .geology/.light-only omission).
[importlinter:contract:fields-internals-private]
name = fields internals reachable only via fields.contracts
type = forbidden
allow_indirect_imports = True
source_modules =
    sirrobin.genetics
    sirrobin.core
    sirrobin.observe
forbidden_modules =
    sirrobin.fields.geology
    sirrobin.fields.light
    sirrobin.fields.scalar_field
    sirrobin.fields.nutrient
    sirrobin.fields.chem
    sirrobin.fields.detritus

# 4) genetics internals private.
[importlinter:contract:genetics-internals-private]
name = genetics internals reachable only via genetics.contracts
type = forbidden
allow_indirect_imports = True
source_modules =
    sirrobin.core
    sirrobin.observe
forbidden_modules =
    sirrobin.genetics.genotype
    sirrobin.genetics.develop
    sirrobin.genetics.innovation

# 5) core internals private: observe may touch only core.contracts.
[importlinter:contract:core-internals-private]
name = core internals reachable only via core.contracts
type = forbidden
allow_indirect_imports = True
source_modules =
    sirrobin.observe
forbidden_modules =
    sirrobin.core.config
    sirrobin.core.clock
    sirrobin.core.state
    sirrobin.core.colony
    sirrobin.core.spatialhash
    sirrobin.core.ledger
    sirrobin.core.economy
    sirrobin.core.parcels

# 6) Driver packages (spikeswim, scripts) are composition roots, NOT runtime layers.
#    They may reach physics/numerics concretes but must never touch the upper layers.
[importlinter:contract:drivers-are-physics-only]
name = spikeswim driver stays physics/numerics only
type = forbidden
allow_indirect_imports = True
source_modules =
    sirrobin.spikeswim
forbidden_modules =
    sirrobin.fields
    sirrobin.fields.*
    sirrobin.genetics
    sirrobin.genetics.*
    sirrobin.core
    sirrobin.core.*
    sirrobin.observe
    sirrobin.observe.*
```

Notes that make this hold up under harsh re-review:
- **`allow_indirect_imports = True`** on contracts 2–5 is load-bearing: it means only a *direct* `import sirrobin.fields.nutrient` from `core` fails, while the legitimate chain `core → fields.contracts → (internally) fields.nutrient` passes. Without it every contracts module would trip its own contract.
- The `layers` contract (1) already forbids *upward* imports, so lower layers are not listed as sources in the internals contracts (e.g. `physics` is never a source against `fields`, because `physics→fields` is already a layers violation). Sources are only the *strictly higher* layers.
- Contract 6 uses the `.*` one-segment wildcard alongside the bare package name to catch both the package and its submodules (import-linter 2.1). The internals contracts (2–5) **enumerate** rather than wildcard, because `forbidden` has no exclusion syntax and `contracts.py` must stay importable — enumeration is the only way to allow exactly `L.contracts` while blocking every sibling.

### 18.3 Enumeration-drift guard (REPLACES the belt-and-braces note at plan:142 and hardens plan:106)

Because contracts 2–5 enumerate current modules, a *new* private module added later would silently escape the firewall. Close that procedurally in `tests/test_import_boundary.py`:

```python
def test_every_private_module_is_firewalled():
    # For each layer L in {physics, fields, genetics, core}: every .py in L
    # except contracts.py and __init__.py MUST appear in that layer's
    # *-internals-private forbidden_modules list in setup.cfg.
    # Fails CI when a new internal module is added but not registered.
```

Plus the existing programmatic run of all six contracts via `import-linter`'s API so a violation fails an ordinary `pytest` (not just the `lint-imports` CI job), and an assertion of interface opacity (INV-W4): no `plate`/`seed`/`hotspot`/`octave`/`swim_step`/`nutrient` symbol is reachable from a consumer through anything but a `*.contracts` module. CI job `boundary` (plan:240) runs `lint-imports --config setup.cfg` and this test, fail-fast, before `conservation`.

**G‑SCAF‑2 (plan:262) is strengthened:** the injected-violation probe must now include, in addition to `physics→core`: (a) `core→physics.swim_step` (concrete-internal reach), (b) `observe→core.colony` (top layer reaching a concrete), and (c) `core→fields.nutrient` (the previously-unguarded field internal) — each must independently fail `lint-imports`.

---

### Files/lines this revision edits (all in `C:\Users\cddal\SirRobin\docs\superpowers\plans\2026-07-11-sirrobin-implementation-plan.md`)
- **#16:** slice-table dep column (556–564) → §16.1; ASCII graph (720–740) → §16.2; critical-path prose (742) → §16.3.
- **#17:** S3 components/acceptance (570, 572) → §17.1 (asexual engine + 5 validation gates); S4 `find` modalities (578) → §17.2 (trim to 3 existing-field senses, defer lateral-line/electro); S4 acceptance (580) → §17.3 (seeded engineering gate + RX‑1 research reservation); slice-table S4 row (559), milestone table (754→M‑S4/M‑S4R), engineering-vs-research table (631) → §17.4.
- **#18:** module map (78–95) → §18.1 (add `genetics/contracts.py`, `core/contracts.py`, fold ForceContributor into `physics/contracts.py`); `[importlinter]` block (116–140) → §18.2 (6 exhaustive contracts); belt-and-braces/opacity note (106, 142) and G‑SCAF‑2 (262) → §18.3 (drift guard + strengthened violation probes).
