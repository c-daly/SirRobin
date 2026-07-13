# SirRobin — Developer Reference

**Status:** **living reference — kept current with the code as it evolves; the single up-to-date
developer-facing map of the Core** · **Last updated:** 2026-07-12 · **Audience:** developers building or
extending the Core.

This is the orientation document: the *premise* (what SirRobin is and the invariants you must hold), the
*architecture* (how the parts fit and where the boundaries are), and the *technical reference* (the
developer-facing API surfaces). It is a **map, not the territory** — the authoritative sources are:

- **The code** — `src/sirrobin/…` — is authoritative for implemented APIs. The consolidated S0 plan, the
  population-grounded Gate-E revision, and both S0 decision reports are authoritative for S0 acceptance.
- `docs/superpowers/plans/2026-07-12-sirrobin-S1-conserved-nutrient-implementation-plan.md` is the execution
  authority for S1 and supersedes older S1 reservoir, energy, parcel, and package-layering prose.
- `docs/superpowers/plans/2026-07-12-sirrobin-S2-canonical-body-live-locomotion-implementation-plan.md` is the
  active S2 authority and supersedes older S2 coordinate, tail-direction, energy, action, capacity, and
  throughput prose.
- `docs/2026-07-11-sirrobin-design-document.md` — the master technical design (cited *design §X*); the source for Parts I–II and the **target** for `[designed]` surfaces not yet coded.
- `docs/superpowers/specs/2026-07-11-sirrobin-restart-architecture-design.md` and `…-genome-encoding-design.md`.
- `CLAUDE.md` — the working contract and the nine non-negotiable laws.

> **Reading the status tags.** **S0 / "SpikeSwim" is complete under the revised population-grounded gate** —
> every production module the accepted plan lists exists under `src/sirrobin/`, with a frozen 192-body corpus,
> both oracle arms, tests, and decision reports. The original 1,000-creature/90M Gate E remains a historical
> NO-GO; the pre-registered 5,000/10,000-creature replacement is GO. **S1 is implemented and records GO. S2 is
> implemented but records NO-GO on controller falsifier F12.**
> Surfaces are tagged **[S0]** or **[S1]** (coded now — Part III cites source symbols), **[designed]**
> (specified in the design doc / plan, no code yet, built in a
> later slice), or **[frontier]** (a research goal, not solved technique). Parts I–II (premise, architecture)
> hold as written; **Part III is grounded in the actual code.**

---

# Part I — Premise

## 1. What SirRobin is

SirRobin is a small, faithful, GPU-vectorized **ocean-life evolution simulation**: a world where a creature's
*form* is the *reason* it can swim, feed, fight, and reproduce, and where niches, speciation, and an eventual
**bidirectional sea↔land crossing** *emerge* from conserved physical laws rather than being scripted.

Its long-horizon purpose is a **non-text grounding substrate** for an embodied agent ("Sophia"): a world with
real survival stakes in which a cognitive core can be embodied via a hardware-abstraction layer ("Talos") over
a simple state contract, later transferable to a physical TurtleBot3. That embodiment is the *reason-for-being*
but is **deferred in execution** — kept as a day-one seam (the state contract, §II.7), built last.

It is a **clean-architecture restart** of a prior build (`C:\Users\cddal\game prototype`), which became
un-maintainable "soup." The prior build is a **donor, not a template**: salvage its proven equations
(`SwimEval` hydrodynamics, `BodyGraph` genome, field grids, `SimUnits`, taxonomy) and its hard-won lessons —
**not** its architecture (C#/Unity, the LOD proxy, the dual genome, byte-identity-as-gate all superseded).

## 2. The organizing thesis: faithful vectorization

The prior build's fatal wound was a **data-far LOD proxy** — a scalar stand-in for real physics — that existed
only because faithfully simulating every creature seemed unaffordable. That one unverified assumption spawned
every laundered scalar and arbitrary knob that followed.

The thesis: **step the true per-body physics for the entire population as batched PyTorch tensor operations on
the GPU.** This dissolves the reason the proxy existed — faithful *and* affordable — and yields the throughput
deep-time evolution needs (10⁸–10⁹ ticks/run). S0 establishes locomotion-component viability at the target
population; it does not yet establish the complete future tick budget. Because this is an optimistic premise, it must be
**verified first, by measurement**, before the architecture is bet on it. That verification is S0/SpikeSwim
(§II.8). *Measure, don't assert* is a law, not a slogan (design §7.1).

## 3. The nine non-negotiable laws

These are invariants. Hold them like conservation laws — a change that violates one is a bug, not a tradeoff
(`CLAUDE.md`; design §1.4).

1. **Fidelity is the product.** Abstract in *mechanism*, faithful in *causality*. Signatures are *derived* from
   conserved, relational constraints — never imposed by a knob.
2. **Conservation — the books close.** Nothing mints or destroys mass/energy; everything moves between tracked
   reservoirs. This is the top CI gate.
3. **Form is function.** Capability is derived from morphology through real physics — never a stat vector.
4. **Single canonical representation.** One representation per quantity. If you are writing code to keep two
   copies of one thing in sync, you have already lost. (A second representation is allowed only for a genuinely
   *distinct kind*, behind a strictly one-way interface, with zero sync code.)
5. **Clean abstraction boundaries.** Query *what* a subsystem provides, never *how*. This is what lets any piece
   start trivial and grow arbitrarily faithful later without touching its consumers.
6. **Continuous, not a discrete grid that leaks into biology.** Continuous positions, interpolated field
   sampling, continuous encounters.
7. **Start simple, grow on demand.** The world's richness tracks what the biology needs; it never runs ahead.
8. **Depth-first.** Close one loop's books — make its tests green — before the next layer exists.
9. **Implicit selection only.** Survival→reproduction is the only score. Design the *gradient*, never the
   *outcome*.

## 4. The emblem: the sea robin

The **sea robin** (gurnard) — a fish that *walks* the seafloor on modified pectoral fin-rays while still
swimming — is the project's namesake and emblematic milestone. It is the fin→leg exaptation made flesh, *in
water and gradual*: proof that one appendage can do both swimming thrust and ground-reaction walking with **no
mode switch**. It is the concrete target for the unified, medium-dependent locomotion physics (§II.4) and an
achievable near-term milestone that de-risks the full water↔land crossing (design §1.7, §4.6).

## 5. What is solved technique vs. research frontier

Be honest about which is which (design §1.2, §5):

- **Solved technique:** the vectorized faithful physics, the conserved economy, the genome
  encoding/crossover/compatibility-distance, form-derived capability.
- **Research frontier [frontier]:** speciation-with-isolation, open-ended evolution (sustained novelty), and
  the sea↔land crossing. These are *gated by the ecology* (they need real disruptive selection and niche
  diversity) — the engineering enables them; it does not guarantee them. Say so.

---

# Part II — Architecture

## 1. Substrate: one vectorized array engine

The Core is **Python / PyTorch**, headless, GPU-capable, cross-platform. One substrate carries the world/physics
math (torch tensors are numpy-like) and, later, any neural minds (same stack, gains autograd on demand). The hot
loop runs under `no_grad` on detached tensors — torch as a pure batched array engine, not an autodiff graph
(design §2.1 — population cognition is reactive, not learned, for now).

The identical tensor program runs on CPU or CUDA via a `device=` knob; the device choice is made from
**measured** CPU↔GPU crossover at the real population size, not asserted (design §2.9). GPU's payoff rises with
population — the near-term target is **one dense world**, not many parallel worlds (that is deferred to the S8
RL phase).

## 2. The entity model: discrete points on continuous fields

Three storage roles share one coordinate space without duplicating authoritative state:

- **Discrete point-entities** — creatures and, only if later grazing evidence requires them, resource parcels — live at continuous float
  positions in one continuous coordinate space and share **one spatial hash** for neighbor queries. Diffusion,
  feeding, and predation all reduce to one primitive: *local interactions between nearby entities*. The spatial
  hash quantizes nothing physical; it is invisible to the biology.
- **Conserved Eulerian reservoir fields** — S1's `Nd_q/Bp_q/Bd_q/Bm_q` int64 tensors — are mutable only through
  exact economy transactions. Concentration is a derived sample, never a mirrored float field. Consumers receive
  read-only samples; economy owns the state-changing transfers.
- **Exogenous/abiotic driver fields** — temperature, currents, structural geology, and similar
  properties-of-space — are sampled through one-way interfaces. S1 begins only with analytic light and prescribed
  vertical mixing; geology, currents, temperature, and structural terrain remain deferred.

Fields are sampled by **interpolation** (smooth value + gradient at the exact position, no cell-edge jumps) and
uptake requests are calculated from a **continuous rate** (proportional to local concentration × dt) before
being committed as exact integer quanta, so the storage grid does not become a per-cell biological rule
(Law 6). The grid only *stores* the field; it is a discretization detail behind the field interface, not a
structure the creatures can feel.

## 3. Package layering and the import firewall

Python has no compile-time assembly firewall, so boundaries are encoded as **import-linter contracts**. They
are currently checked locally; a repository CI workflow has not yet been added.

```text
numerics  <- physics
    ^      <- fields <- economy
    |                    |
    +--------------------+

core (later composition root) depends on public sibling contracts
observe depends only on public read contracts
```

- **numerics** — dtype policy, quaternions, the constrained solver, counter-RNG, the int64 transfer primitive.
  Pure math; imports nothing above it.
- **physics** — articulated bodies + fluids + geometry through *real* physics. Defines the mechanical contracts
  (`DevelopedBody`, `FluidSample`, `ForceContributor`) and consumes them. **Knows nothing of creatures, genes,
  or economy** — it is the single sealed place to later solve cross-cluster determinism.
- **fields** — grid geometry, interpolation, and conservative transport. It owns generic `FieldSample`, not
  body mechanics or reservoir transactions.
- **economy** — exact nutrient reservoirs, reaction requests, transaction ordering, ledger, and snapshot. It
  depends on public `fields` and `numerics`, never on `physics`.
- **genetics** — the genome and development; *produces* `DevelopedBody`s. Neither fields nor genetics reach into
  physics internals; they hand physics its inputs and query *what* it provides.
- **core** — later composition root for the colony/world and whole-tick orchestration. When bodies couple to
  fields, core samples a field and constructs the physics-owned `FluidSample`/`MediumSample`; neither sibling
  imports the other. Downstream consumers see only the public read-only contract surface (`core.contracts`).
- **observe** — telemetry, plots, and the viewer/embodiment read surface. Depends one-way on `core.contracts`;
  **render can never feed fitness** (an equivalence gate makes any leak a CI failure).

The payoff of clean boundaries (Law 5): every subsystem can start *trivial* (S1 = a closed all-ocean nutrient
column with analytic light and prescribed vertical mixing) and grow *arbitrarily faithful* later (Voronoi
structural field + moving hotspots + rivers) with **zero downstream changes**, because consumers query the
output *categories*, never the mechanism (design §3.4 — the abstraction boundary is the one load-bearing
terrain decision).

## 4. The load-bearing physics decision: additive force-contributors

Locomotion is architected as **additive force-contributors on one articulated-body core** (design §4.5).
The body core owns articulated state and assembles the effective inertial matrix; each force law
is a *contributor* that adds into one force sum. `SwimEval`'s hydrodynamics (Lighthill/Lamb undulatory thrust,
wake, added mass, drag) is the **first and, at S0, only** contributor.

This is what makes the **bidirectional water↔land crossing** free rather than a rewrite: gravity, buoyancy, and
ground contact/friction are *additional contributors* that add later, and the **medium (position relative to
the waterline) decides which dominate** — water→swim, land→walk, intertidal→amphibious crawl, continuously. One
appendage gives thrust/drag in water and ground-reaction on land with no mode switch, so a limb can drift
leg↔flipper and lineages can return to the sea (design §4.7). The additive seam is stood up at **S2**; only the
hydrodynamic contributor exists at S0.

## 5. The evolutionary engine: recipe, not blueprint

Genes are a **developmental recipe**, not a blueprint. The genome is a **bounded recursive directed part-graph**
(Karl Sims 1994) with **NEAT innovation-number markings** on every node/edge, **developed by a fixed-depth
batched scan** into a `DevelopedBody` segment tensor that feeds the physics unchanged (design §5). It is a
*migration* of the prior `BodyGraph` (tree→graph + innovation ids), not a rewrite.

- Development is a **fixed-shape pure function** — all growth happens in `Mutate()`, so `develop()` batches on
  the GPU over a heterogeneous population.
- The **innovation ids** are the load-bearing addition: they make genomes *alignable* → crossover + NEAT
  compatibility-distance + spatial assortative mating → **emergent speciation [frontier]** (impossible for the
  prior build's post-hoc cosine "species").
- **Form is function (Law 3):** feeding, metabolism, defense, and locomotion are *read off the developed
  morphology through physics* — there is no stat vector. The prior build's `eff[]` is deleted; killing it is
  S2's headline gate.

## 6. The conserved economy: the books close

Every tracked conserved quantity lives in a **named reservoir**; every state change is a **transfer between
reservoirs** — never a raw `R += x` and never an undeclared source (design §6; Law 2). The system boundary is
explicit: a closed slice has no external flux, while a later open slice represents sunlight, geology, heat, or
burial through named boundary transactions. Conserved reservoir totals are **int64 fixed-point quanta**, so
`close_books()` is exact integer equality rather than a float tolerance.

At **S0** only a *fake* mass reservoir exercises the transfer primitive. **S1 conserves nutrient mass only**
across exactly `Nd_q`, `Bp_q`, `Bd_q`, and `Bm_q`; concentration and chemical-energy equivalents are derived
diagnostics, never stored mirrors. A native energy ledger waits for real reserve and heat reservoirs. S0's
mechanical `R_step` is a discrete work-consistency residual, not total physical-energy conservation.

## 7. The embodiment seam: a state contract

The sim↔Talos boundary is a **state contract** — a defined, versioned, serializable schema of state (senses out,
actions in), not a chatty API (design §7.4). Sophia's entire neural stack lives *behind* Talos, outside
the sim. Because it is a serializable state schema, the same interface may later transfer to a physical
TurtleBot3. S0 realizes constrained horizontal x/z translation with a frozen heading; it has no yaw state or
angular integration. The `{surge, yaw}` action contract and its performance remain S2/S8 risks, alongside
observation transfer. This seam is built at **S8**, but it constrains the read surface (`core.contracts`, `observe`)
from day one.

## 8. The verification spike: S0 / SpikeSwim

S0 is the measured gate for the **locomotion component** of the vectorization thesis. It ports the
donor's one-shot, **frozen-heading** `Sim.Step` to batched torch over realistic heterogeneous H1/H2 populations
and answers one bounded question by *measurement*:

> Can the canonical fixed-slot body representation run the donor-grounded frozen-heading locomotion step, with
> correct mechanics and independent corroboration, at a rate that makes locomotion a viable component of the
> later whole tick?

S0 has **no ecology, genome mutation, steering, or metabolic ledger**. It is graded on five gates — scaffold
integrity (A), physical/oracle fidelity against two independent arms (B), force/power + discrete mechanical
consistency (C), reproducibility posture (D), and throughput/affordability on mandatory H1/H2 populations (E) —
producing a telemetry-backed **GO / CONDITIONAL GO / NO-GO** decision. The original 90M gate produced NO-GO;
the later owner-pinned 5,000/10,000 population gate produced GO. The full whole-tick affordability question
remains later authority. See the S0 plan, revision, and reports for the acceptance history.

## 9. Determinism posture (relaxed, three tiers)

The determinism target is three tiers (design §2.7; relaxed 2026-07-12):

- **Tier 1 — exact conservation** (int64-quanta reservoirs). **Hard gate.** Exact regardless of float order or
  GPU atomics.
- **Tier 2 — statistical reproducibility** (integer Philox counter-RNG + stable entity IDs). **Hard gate.**
  Reproducible stochastic *decisions* for valid ablations.
- **Tier 3 — bit-identical float replay.** **Dropped as a gate** — an optional same-device diagnostic only.
  `torch.compile`/CUDA-graphs may reassociate freely; `use_deterministic_algorithms` is not required; atomic
  float scatter is permitted; there is no compile-parity gate on the float physics.

Tolerance-based oracle, momentum, force-power, and energy checks remain **hard gates** — relaxing Tier 3 relaxes
float *replay*, never physical *correctness*.

## 10. The build roadmap (what exists, what's next)

Depth-first — each slice closes its books before the next exists. S0 and S1 are implemented and measured; S2
is next.

| Slice | Scope | Status |
|---|---|---|
| **S0** | SpikeSwim — vectorized frozen-heading locomotion; verify the thesis by measurement | **GO under the revised population-grounded gate (original 90M floor = historical NO-GO)** |
| **S1** | Conserved single-nutrient economy (keystone; books must close) | **complete; GO** |
| S2 | One canonical body + live locomotion for every creature; the additive-force seam; prevent `eff[]` | **implemented; NO-GO (F12 controller settlement)** |
| S3 | Feeding / metabolism / reproduction on conserved energy | designed |
| S4 | Predation as a staged contest between bodies (no seeded predator) | designed |
| S5 | Speciation / mating / taxonomy | designed / frontier |
| S6 | Transport / currents / upwelling | designed |
| S7 | Viewer + observation surface (render never feeds fitness) | designed |
| S8 | RL / embodiment loop (Talos/ROS2; gated on books closed + Sophia interface verified) | designed |
| S9 | Plants + the bidirectional sea↔land crossing | frontier |

---

# Part III — Technical Reference (developer-facing API, as implemented)

**Primary sources: the code (`src/sirrobin/…`, cited `file:line`) and the accepted plan (the consolidated S0
implementation plan, cited *plan §*).** The design doc is the target only where a later accepted correction or
slice plan does not supersede it. **S0 and S1 are complete and measured.** New public contracts use ENU world
coordinates and ROS REP-103 FLU body coordinates, with Unity-order `(x,y,z,w)` quaternion storage
(`numerics/quat.py:1`), radians, and seconds. **S0 is a frozen exception:** its donor-conformance tensors retain
the donor XZ-horizontal/Y-up frame behind the S2 boundary adapter. `B` =
batch (bodies), `S` = `s_slot` = **17** (slot-0 sentinel + 16 real segments).

## III.1 Actual module layout & the import firewall

```
src/sirrobin/
  numerics/   dtype · quat · solve_constrained_xz · solve_donor · transfer · flux [S0/S1]
  physics/    config · contracts · lamb · pose · force_reactive · force_fin ·   [S0, present]
              force_drag · mass_matrix · swim_step · energy
  fields/     contracts · geometry · grid · sample · transport                   [S1, present]
  economy/    config · contracts · state · ledger · reactions · step · snapshot [S1, present]
  core/       contracts (ArtifactRef) · clock (SimClock)                        [S0, present]
  observe/    telemetry · economy                                                [S0/S1, present]
  validation/ corpus · drift · economy  benchmarks/ lifecycle · episode · locomotion
tools/  build_corpus · build_fixture_manifest · build_s0_report · gain1_oracle · profile_locomotion ·
        run_benchmark_cell · run_determinism_diagnostic · run_drift   (offline; never imported by production)
oracle/ SirRobinOracle.csproj + shims + fixtures/ (C# donor + frozen JSON)
tests/  flat test_*.py       pyproject.toml · setup.cfg (import-linter)
```

Deltas from the plan's §3 tree (all deliberate): **`config` lives in `physics/`, not `core/`** — so `physics`
can import it without breaking the `physics ⊄ core` rule; **`physics/energy.py`** and **`benchmarks/episode.py`**
are extras (the Gate-C energy gates + the episode harness); the plan's **`spikeswim/` package is split** into
`validation/` (frozen evidence) + `benchmarks/`. S1 adds a real economy snapshot but no Philox sampler or full
`Colony` snapshot; report generation remains offline under `tools/`. There is still no `genetics/` package.
Boundaries are **configured and checked with `setup.cfg` import-linter** (not enforced by CI — no workflow
exists yet): `numerics` is the leaf; `physics` and `fields/economy` are independent siblings; `economy` may
import public `fields` and `numerics`; `fields` imports only `numerics`; neither imports `core`. Package
`__init__.py` are docstring-only markers — import concrete
modules: `from sirrobin.physics.swim_step import SwimKernel`.

## III.2 Config & time

**`LocomotionConfig`** (`physics/config.py:10`) — the single frozen config + unit anchor; every value part of
`sha256()` (`:60`). The dataclass retains the original S0 experiment defaults
`worlds=1, n_cap=1024, n_live=1000, s_max=16, max_depth=5, dt=1/120,
rho_water=1000, rho_neutral_gene=4, kg_per_sim_mass=250, drag_coeff=0.1, fin_profile_cd=0.02, fin_span_eff=0.9,
fin_stall_aoa=0.35, ellipsoid_mass_gain=1, fin_plane_gain=1, kappa_max=1e6, lam_floor_kg=1e-9, eps_spd=1e-6`,
the Gate-C tolerances (`p_atol_f64/f32`, `e_atol_f64/f32`, `rtol_f64/f32`), and the historical
`throughput_floor=9.0e7`,
`vram_cap_bytes=11 GiB`; `s_slot = s_max+1 = 17`.

> **Historical vs. current throughput authority.** `n_cap=1024`, `n_live=1000`, and `throughput_floor=9.0e7`
> (90M) are the **historical defaults of the original failed experiment**, preserved for config-hash provenance
> — they are **not** current policy. The revised gate derives the floor from the live population as
> `floor = N_live × 120 Hz × 1×`:
>
> | Live population | Capacity | Hard floor (1× real-time) |
> |---:|---:|---:|
> | 5,000 | 5,120 | 600,000 creature-steps/s |
> | 10,000 | 10,240 | 1,200,000 creature-steps/s |
>
> Measured minimums (min of 5 reps, CUDA r2 rung, RTX 5070, zero regularization, non-OOM): H1@5k 1.68M (2.80×),
> H2@5k 1.51M (2.52×), H1@10k 2.44M (2.03×), H2@10k 2.38M (1.99×) — i.e. **~2× real-time at 10,000 creatures**.
> The 5× and 10× throughput classes are **optimization targets, not met and not gates**.

**`validate()`** (`:43`) enforces the frozen invariants: donor
caps `s_max=16, max_depth=5`; `dt=1/120`; **`kg_per_sim_mass == rho_water/rho_neutral_gene`** (`250 == 1000/4` —
the mass-unit anchor's pinned provenance); `ellipsoid_mass_gain ∈ {0,1}` and `fin_plane_gain ∈ [0,1]`.

> **On the two "gain" knobs.** `ellipsoid_mass_gain` and `fin_plane_gain` are **not** the prior build's
> fidelity-disabling dials. `ellipsoid_mass_gain` selects the mass model for the two oracle arms — `0` = donor
> box mass (gain0 conformance), `1` = corrected ellipsoid `π/6` (gain1) — and `fin_plane_gain ∈ [0,1]` blends
> the surface-fin plate exaptation. Both are constrained and validated; neither disables physics.

**`SimClock`** (`core/clock.py:6`) — `now: float, step: int`; `advance(dt)` (`:11`) raises on `dt≤0`, else
`now += dt; step += 1`. Sim-owned time; physics reads no wall-clock. (The design doc's fuller `Colony`
Environment API with `reset/step/state/load` is **[designed]** — no `Colony` exists yet; §III.8.)

**`EconomyConfig`** (`economy/config.py`) — the hash-bound S1 configuration: one `64×64×32` grid over
`640×640×160 m`, `dt_eco=8640 s`, and `q_mass=1e-9 mol` nutrient per integer quantum. It derives transport
substeps from mixing/sinking CFL constraints, enforces a positive deep remineralization floor whose residence
time fits the validation horizon, rejects reaction steps above the frozen rate bound, and caps configured
inventory below `2^62`. `with_half_timestep()` constructs the authorizing convergence configuration without
changing physical rate anchors.

## III.3 The body & step contracts (`physics/contracts.py`)

**`BodyBatch`** (`:14`) — the S0 body representation (the code's analogue of the design doc's `DevelopedBody`),
a fixed-slot SoA over `[B, S=17, …]`, slot 0 an unused sentinel. Fields (tensors): `alive[B]bool ·
stable_id[B]i64 · seg_mask[B,S]bool · local_pos[B,S,3] · local_rot[B,S,4]quat · abc[B,S,3] (semi-axes) ·
density_gene[B,S] · amp_deg[B,S] · phase_rad[B,S] · is_surface[B,S]bool · is_tail[B,S]bool · parent[B,S]i64 ·
depth[B,S]i64 (init −1) · fin_span[B,S] · fin_chord[B,S] · swim_freq[B] · swim_wave[B] · f_hat[B,3] · n_hat[B,3]
· x_com[B,3] · v_com[B,3] · gait_time[B]f64`. Built by **`from_rows(rows, config, *, dtype=f32, device="cpu")`**
(`:43`) from corpus dicts.

> **Key difference from the design doc's `DevelopedBody`.** `BodyBatch` stores the **raw morphology inputs**
> (`density_gene`, `fin_span`, `fin_chord`, `local_pos/rot`, `abc`) and **derives** mass, added mass, and pose
> at runtime (`mass_matrix`, `pose`) — rather than the design doc's *precomputed* `DevelopedBody` (materialized
> `mass`, `area`, `m_add`, `center`, `rest_rot`). Because S0 has no `genetics` development stage yet, storing
> inputs and deriving outputs is single-source-clean (Law 4). Shared geometry fields/conventions match the
> design (`abc`, `local_pos/rot`, `is_surface`, `is_tail`, `parent`, `depth`).

Companion dataclasses: **`Pose`** (`pos, rot`; `:117`), **`MassProperties`** (`mass_sim, mass_kg, added_mass,
matrix`; `:123`), **`StepLedger`** (`:131`) — the full per-step diagnostic record (`u, vt, slope, t_react,
p_reactive_in, p_wake, p_wake_dissipated, t_fin, p_fin_in, p_fin, p_drag, f_drag, f_stream, m_before, m_after,
dv, j_reg, regularized, delta_ke, work_impulse, work_delta_m, r_step`). Note the two wake fields: **`p_wake`**
is the *signed* wake-energy flux used by the reactive-channel identity, while **`p_wake_dissipated`** is the
nonnegative wake power actually dissipated (`u≥0` only), matching the donor's work accounting.

## III.4 `numerics` API

- **`dtype`** — `HOT_DTYPE=f32`, `REFERENCE_DTYPE=f64`, `INDEX_DTYPE=i64`; `require_float_dtype(dtype)`
  (`dtype.py:10`).
- **`quat`** (`(x,y,z,w)` Unity order) — `identity`, `conjugate`, `multiply` (Hamilton), `normalize` (clamped),
  `rotate(q,v)`, `angle_axis_deg`, `euler_unity_deg` (Unity z→x→y) (`quat.py:10-58`).
- **`solve_constrained_xz(matrix, impulse, body_valid, *, kappa_max=1e6, lam_floor=1e-9, eps_spd=1e-6) ->
  SolveResult`** (`solve_constrained_xz.py:18`) — the production solver. `SolveResult{dv, j_reg, regularized,
  condition_estimate}`. Extracts the 2×2 x/z sub-form; **division-free degenerate predicate** `(lam_max <
  lam_floor) | (det < lam_max²/kappa_max)` (`:39`); Tikhonov `reg = eps_spd·lam_max` on the degenerate diagonal;
  **booked reaction `j_reg = −reg·dv`** (the sign the review pinned; `:56-63`); **`dv_y ≡ 0`** (`:55`); asserts
  finiteness of inputs and outputs with device-side `torch._assert_async` (`:32,66-67`), and `raise ValueError`
  only on a wrong-shaped `[B,3,3]` matrix (`:27`). Matches plan §5.2.
- **`solve_sym3_donor(matrix, rhs)`** (`solve_donor.py:6`) — gain0 reference only: the untouched donor 3×3
  cofactor solve (`det<1e-12` axial fallback). Not the production path.
- **`transfer`** — the int64 conservation primitive (Gate A, fake-reservoir scaffold). `transfer_quanta(src_q,
  dst_q, requested_q, mask) -> (new_src, new_dst, shortfall)` (`transfer.py:10`): caps `effective=min(request,
  src)`, raises `OverflowError` on int64 destination overflow, no overdraft. `close_books(*reservoirs,
  expected_total) -> bool` (`:38`): `False` if any reservoir negative, else exact `sum == expected_total`.

## III.5 `physics` API

- **`numerics.ellipsoid_added_mass` / `physics.lamb` compatibility API** — ellipsoid added mass by pinned
  256-pt Gauss–Legendre: `lamb_coefficients(abc)` (sum scaled to
  exactly 2), `lamb_factors` (`α/(2−α)`), `added_mass(abc, rho_water)` = `factor·ρ·V`, `V=(4/3)π·abc.prod`
  (`lamb.py:14-42`). `donor_lamb_factors`/`donor_added_mass` = the gain0 Simpson-2048 arm.
- **`pose`** — `resolve_pose(body, time_s, *, apply_gait=True) -> Pose` (`pose.py:19`): gait flex
  `θ = amp_deg·sin(2π·swim_freq·t + phase_rad)` about local +Y, **fixed 6 depth passes** propagating parent
  transforms. `tail_slots`, `gather_slots`, `tail_tip` (tail center + rotated local `+z·c`).
- **Force channels** (pure functions):
  - `reactive_channel(mt, u, vt, slope) -> (thrust, p_input, p_wake, wt)` (`force_reactive.py:6`): `wt=vt+u·slope`,
    `thrust=½·mt·(vt²−u²·slope²)`, `p_wake=½·mt·u·wt²`, `p_input=mt·u·vt·wt`.
  - `fin_channel(lift_slope, aspect_ratio, area, u, vt, slope, active, *, rho_water, profile_cd, span_eff,
    stall_aoa) -> (thrust, p_input, p_wake)` (`force_fin.py:10`): Garrick finite-wing; `u_cl=max(u,0)`, AoA
    clamped to `±stall_aoa`, induced drag `cdi=profile_cd + cl²/(π·span_eff·AR)`.
  - `drag_channel(segment_velocity, rotation, area_z, mask, *, rho_water, cd) -> (force[B,3], dissipated[B])`
    (`force_drag.py:10`): axial-z-only quadratic drag `−½·ρ·cd·area_z·|v_z|·v_z`, per-segment nonneg dissipation.
- **`mass_matrix`** — `prepare_mass_data(body, config) -> StaticMassData{seg_mass_sim, added_mass,
  fin_perpendicular_mass}` (`:23`): `box_mass=8·abc.prod·density_gene` (≥0.1), `mass_scale=1+ellipsoid_mass_gain·
  (π/6−1)` (gain0=box, gain1=ellipsoid), plus the surface-fin plate-mass exaptation blend by `fin_plane_gain`.
  `mass_properties(body, pose, config, static) -> MassProperties` (`:52`): **`M_eff = Σ_seg R·diag(m_add)·Rᵀ +
  mass_kg·I₃`** (`:59-61`), `mass_kg = mass_sim·kg_per_sim_mass` — the body-inertia term `mass_kg·I₃` **is**
  included (the `M_eff` correctness the review pinned; body mass and added mass never conflated).
- **`SwimKernel`** (`swim_step.py:26`) — the complete frozen-heading step; the S0 "driver" (there is no `Colony`
  yet). `__init__(body, config)` precomputes static mass + the tail slot, derives the **frozen heading** `f_hat`
  (from `rest_com − tail_center`, y zeroed) + `n_hat`, and the reactive/fin constants (**mutates `body.f_hat`/
  `body.n_hat` in place**). **`step() -> StepLedger`** (`:73`): poses/masses at `t0,t1`; tail-tip velocity →
  `u`(surge)/`vt`(lateral)/`slope`; reactive+fin+drag → `f_stream=(t_react+t_fin)·f̂ + f_drag`;
  `solve_constrained_xz(mass1.matrix, f_stream·dt, valid)`; `v1=v0+dv` (`v_y=0`); then integrates **in place**
  (`v_com.copy_` / `x_com.add_` / `gait_time.copy_`, `:147-149`) so the buffers keep static addresses for
  CUDA-graph replay (the r2 rung).
- **`energy`** — Gate C (`energy.py`): `step_closes(ledger, config) -> bool[B]` (`:29`): `|r_step| ≤ atol +
  rtol·max(|ΔKE|,|W_imp|,|W_M|,atol)` — the dimensioned mixed tolerance (fixes divide-at-rest).
  `prefix_budget(residuals, scales, *, warmup=100) -> PrefixBudget` (`:36`): the executable 10⁵-step drift gate
  `|Σr|/Σscale` + a `monotone_bias` flag — the concrete replacement for "bounded-oscillating". `R_step = ΔKE −
  v_mid·(F_stream·dt + J_reg) − ½v₀ᵀΔM v₀` is computed in `SwimKernel.step` (`swim_step.py:136-145`).

## III.6 `validation`, `benchmarks`, `observe`

- **`validation/corpus.py`** — `load_corpus(path)` / `validate_corpus(corpus)` (`:22,28`): enforces schema
  `"sirrobin.locomotion.corpus.v1"`, **exactly 192 bodies**, unique ids, **per-class H0/H1/H2 segment-count
  histograms** (`EXPECTED_HISTOGRAMS`, `:11` — 64 bodies per class, 192 total), slot/parent ordering,
  `depth≤5`, H1/H2 mirrored/fin/tilted counts, **and worst-case-diversity constraints on H2** (fin/non-fin
  interleaving, the eight full 16-segment bodies spread across the class with 8 distinct scale/aspect
  signatures) — this **pins the H0/H1/H2 authorizing populations** so a benchmark can't be gamed with a
  favorable corpus. `verify_sidecar(path)` checks the `.sha256`.
- **`benchmarks/lifecycle.py`** — `apply_fixed_churn(body, step, *, period=1000, fraction=0.02) ->
  LifecycleEvent|None` (`:20`): deterministic slot-recycling churn (no RNG); `tensor_addresses(body)` for
  static-buffer stability.
- **`benchmarks/episode.py`** — `run_episode(kernel, *, warmup_steps=360, measure_steps=600) -> EpisodeResult
  {cruise_speed, cost_of_transport, reactive_ratio, mechanical_work, regularization_count}` (`:22`): the
  donor-shaped 3 s + 5 s episode.
- **`benchmarks/locomotion.py`** — `benchmark_cell(corpus, class_name, *, capacity, live, device,
  rung="r0-eager", steps=600, warmup=20, repetitions=5, churn=False) -> BenchmarkResult` (`:48`): measures
  `live·steps/elapsed` over the **acceleration ladder `rung ∈ {r0-eager, r1-compile, r2-cudagraph}`**
  (`torch.compile(mode="reduce-overhead")` for r1; explicit CUDA-graph capture for r2), reports CUDA peak
  memory, `status="oom"` on OOM; `write_result(...)` emits `sirrobin.locomotion.benchmark.v1` with provenance.
- **`observe/telemetry.py`** — `RunManifest{config_hash, corpus_hash, device, dtype, torch_version, hardware,
  seed}` + `TelemetryWriter(path, manifest).write(kind, payload)` (JSONL, sorted, flushed).

## III.7 The S0 data-flow spine (as implemented)

```
oracle/fixtures/corpus.json ──validate_corpus──▶ rows ──BodyBatch.from_rows──▶ body   [validation/corpus.py; physics/contracts.py:43]
   ▼
SwimKernel(body, config)  ── derives frozen f_hat/n_hat, static mass ──                 [swim_step.py:27]
   │  .step():
   │    resolve_pose(t0), resolve_pose(t1)                       6-pass FK               [pose.py:19]
   │    mass_properties → M_eff = ΣR·diag(m_add)·Rᵀ + mass_kg·I                          [mass_matrix.py:52]
   │    tail_tip kinematics → u, vt, slope                                              [swim_step.py:81-90]
   │    reactive_channel + fin_channel + drag_channel → f_stream                         [force_*.py]
   │    solve_constrained_xz(M_eff(t1), f_stream·dt, valid) → dv, j_reg                   [solve_constrained_xz.py:18]
   │    v1 = v0 + dv (v_y=0); in-place integrate x_com; R_step                            [swim_step.py:130-149]
   ▼
StepLedger ──▶ energy.step_closes / prefix_budget  (Gate C)                              [energy.py:29,36]
           ──▶ benchmarks (episode aggregates, throughput)                               [benchmarks/]
           ──▶ observe.TelemetryWriter (JSONL + manifest)                                [observe/telemetry.py]
transfer.transfer_quanta / close_books  — int64 conservation scaffold (Gate A)          [transfer.py]
```

Validated against **two independent oracle arms** (plan §2.2): **gain0** (untouched C# donor,
`oracle/SirRobinOracle.csproj` + Unity shims → `gain0_donor.json`, `test_donor_gain0.py`) and **gain1**
(`tools/gain1_oracle.py`, numpy/scipy, no torch/donor imports → `gain1_analytic.json` + `quadrature_gl256.json`,
`test_oracle_independence.py`). Three oracle/corpus gaps the plan review flagged (H0/H1/H2 definition,
`KgPerSimMass` provenance, frozen fixture values) are now closed in code.

## III.8 S1 conserved-nutrient API (as implemented)

- **`EconomyState` / `EconomyCarries`** (`economy/state.py`) — the only authoritative nutrient state:
  `nd_q/bp_q/bd_q/bm_q[W,Gx,Gy,Gz]` as int64 quanta, plus named float64 sub-quantum reaction carries, a
  sinking-face carry, and one outgoing mixing carry per source cell before face apportionment. Step, ecological
  time, and buffer parity are scalar tensors so compiled execution does not recompile on Python metadata
  changes. `validate()` enforces shapes, dtypes, carry intervals, nonnegativity, and `<2^62`.
- **`MassLedger`** (`economy/ledger.py`) — freezes expected totals per world and returns an exact Boolean closure
  result per world. There is no relative mass tolerance.
- **`numerics.flux`** — `commit_flux` converts nonnegative f64 requests plus bounded carry into availability-
  limited integer transfers without turning shortfall into future debt; `deterministic_fraction` performs the
  exact one-debit/two-credit BGE split; `apportion_integer` uses stable largest-remainder allocation for competing
  face debits.
- **Fields** — `GridGeometry`; generic `FieldSample(value_mol_m3,gradient_mol_m4)`; `sample_reservoir` for
  trilinear sampling with periodic x/y and closed z; public positions use ENU world coordinates with surface
  `z=0`, water `z<0`, and internal `depth=-z`; `ScalarGrid` is the read surface plus a synthetic exact
  `deplete_at` probe; and conservative `sink_vertical` / `mix_vertical`. Fields own no economy state and import no
  economy or physics package.
- **`reaction_step`** (`economy/reactions.py`) — computes all local requests from the same old snapshot, then
  commits producer maintenance, mortality, primary production, BGE decomposition, and microbial turnover.
  Production cannot create the first producer quantum; conservative `Bp` mixing provides recolonization.
- **`EconomyKernel.step`** (`economy/step.py`) — reactions → closed-bottom `Bd` sinking → same-snapshot
  `Nd/Bp/Bm` mixing → tensor clock/parity advance → exact per-world closure ledger. Transport substeps are frozen
  in the config hash.
- **`save_snapshot/load_snapshot`** (`economy/snapshot.py`) — safetensors state plus JSON schema/config metadata;
  restores every reservoir, carry, clock value, and parity bit. A negative test proves omission of carries changes
  committed continuation.
- **Validation/evidence** — `validation/economy.py` owns the hard uncapped bloom and half-timestep comparison.
  Frozen NumPy-only fixtures live under `oracle/fixtures/economy/`; the S1 GO report is
  `docs/superpowers/reports/2026-07-12-sirrobin-S1-decision-report.md`.

The authorizing results are exact closure over a one-million-step production soak; a full bloom/crash at
`d_dd=0`; all half-step aggregate differences below 0.3%; and a non-OOM full grid on the RTX 5070. Compiled CUDA
is fastest at a median 109.1 ecology steps/s with about 38 MiB peak allocation. These results authorize the
material cycle only—not grazing, self-shading, burial, a complete biological pump, or creature energy.

## III.9 S2 canonical body and live-locomotion API (implemented; NO-GO)

- **`GenotypeBatch`** (`genetics/genotype.py`) — fixed `[W,N,24]` node and `[W,N,48]` edge tensors with stable
  IDs, expression/type/axes/density/port/joint fields, attach/rotation/scale/mirror/recursion edges, and swim
  frequency/wave genes. `validate()` enforces the fixed capacities, live root, innovation IDs, endpoints, and
  finite geometry. `from_donor_rows` is fixture ingestion, not the hot development path.
- **`develop(genotype) -> DevelopedBody`** (`genetics/develop.py`) — fixed 16-emission batched DFS stack scan,
  deterministic positive-then-mirror ordering, depth ≤5, axial-vector hinge reflection, ellipsoid mass/areas,
  GL256 Lamb added mass, and aft local-`−x` tail selection. Output is immutable `[W,N,17,...]`; slot 0 is the
  sentinel. Development imports `physics.contracts` only; import-linter forbids physics implementation imports.
- **Canonical contracts** (`physics/contracts.py`) — `DevelopedBody`, `FluidSample`, `ForceTorquePower`,
  `HydrodynamicDiagnostics`, `LiveState`, and `LiveStepLedger`. `LiveState` stores ENU COM position, water-relative
  velocity, yaw, yaw angular momentum, f64 gait time, desired heading, gait turn bias, and heading-latch state.
  It deliberately stores no angular velocity, heading quaternion, cached speed, or capability score.
- **Live mechanics** — `resolve_live_pose` performs the fixed FLU depth scan; `hydrodynamic_contribution` assembles
  reactive tail thrust, surface-fin lift, axial segment drag, force moments, `M_eff`, pose-dependent yaw inertia,
  and quadratic yaw drag in ENU; `sum_contributions` provides the typed additive seam. `step_live` solves ENU x/y,
  integrates linear velocity plus `L_yaw`, derives `omega=L/I`, wraps yaw, and emits exact linear/angular
  discrete-work ledgers. The 8 rad/s value is an async assertion, never a clamp.
- **Core composition** — `update_heading_controller` implements the frozen donor-shaped desired-heading policy;
  `advance_live_world` composes controller → physics → passive uniform-current transport → periodic x/y wrapping,
  while leaving depth unchanged. `save_live_snapshot/load_live_snapshot` stores genotype plus every live-state
  field and re-develops the body; `DevelopedBody` is never serialized as a second authority.
- **Morphology** — `query_morphology` derives segment count, structural mass, displaced volume, projected areas,
  and intake-face area from the canonical axes. Intake flags do not switch hydrodynamics.
- **Measurement** — `benchmarks.live` supports eager, compile, and safe CUDA Graph rungs over literal H1/H2
  cycles at 5,000/10,000. CUDA Graph clears every hard floor with zero intervention, but CPU/eager CUDA do not.
  The decision remains **NO-GO** because the strict CPU and CUDA F12 tests prove the heading controller does not
  home and settle. See `docs/superpowers/reports/2026-07-12-sirrobin-S2-decision-report.md`.

## III.10 Not yet built — `[designed]` (the design doc is the target)

These are in Parts I–II and the design doc but have **no code** yet; a later slice builds them. Do not treat
their signatures as final.

- **`Colony` / Environment API** (`reset/step/state/load`) — the top-level driver; S0 drives via `SwimKernel`
  and S1 via `EconomyKernel`; whole-tick composition remains later. **[designed, S2+]**
- **Later genetics** — mutation, crossover, compatibility distance, mating, and the shared innovation registry
  remain unimplemented. S2 lands genotype storage and development only; emergent speciation remains a frontier.
  **[designed, S3/S5]**
- **Later abiotic fields** — geology, structural `Φ`, currents, temperature, terrain, and the core adapter from
  generic `FieldSample` to physics-owned `MediumSample`. S1 implements only conserved nutrient/biomass grids,
  analytic light, and prescribed vertical transport. **[designed, S2/S6/later]**
- **Later trophic/energy economy** — creature structural nutrient and real reserve/heat reservoirs, feeding,
  metabolism, reproduction, sediment/burial, and predation arrive only with the mechanisms that use them.
  Derived chemical-energy diagnostics are not authoritative reservoir state. **[designed, S3/S4/later]**
- **The Talos state contract** — the versioned CORE/EXT dict-of-tensors + `Header`; the **design-target** CORE
  action is the 2-vector `{surge_effort, yaw_rate}` → `geometry_msgs/Twist`. S0 is frozen-heading with no yaw;
  live yaw (`StepLive`) is an **S2** performance-and-fidelity risk, and the claim that a CORE-only policy
  transfers to a TurtleBot3 unchanged is **unverified**. `core/contracts.py` currently holds only
  `ArtifactRef{path, sha256}` (design §7.4). **[designed, S8]**
- **Creature energy coupling** — S2 emits hydrodynamic and discrete-work diagnostics only. It has no reserve/heat
  reservoirs and performs no metabolic debit; the complete transaction begins in S3. **[designed, S3]**
- **`rng` (Philox), full `Colony` snapshot** — S1 has an exact economy snapshot, but no biological sampler or
  composed world/creature snapshot yet. **[designed, S3/later]**

---

*Maintenance: Part III is grounded in the code (`file:line`) plus the accepted slice authorities — the
consolidated S0 plan, the population-grounded **Gate-E revision**
(`docs/archive/plans/2026-07-12-sirrobin-locomotion-gate-E-revision.md`), the **S1 conserved-nutrient
implementation plan**, the **S2 canonical-body/live-locomotion plan**, and the S0/S1/S2 decision reports
(reflecting S0/S1 GO and S2 NO-GO as of 2026-07-12). **This is a
living document: keep it current as the code and slices land — update it with the code, don't let it fall
behind.** Exact line numbers are **navigational, not stable API identifiers**: they will drift and small
surfaces (e.g. `StepLedger` fields) may change between edits, so re-verify against `src/` before relying on a
specific `file:line`. Promote `[designed]` items to their slice as they land.*
