# SirRobin — Consolidated conserved-nutrient implementation plan

**Status:** executed authority — GO recorded in `docs/superpowers/reports/2026-07-12-sirrobin-S1-decision-report.md`
**Date:** 2026-07-12
**Scope:** exact nutrient bookkeeping, Eulerian nutrient/biomass fields, local producer and microbial
reactions, conservative vertical transport, restartable state, and evidence-backed acceptance
**Decision produced:** **GO / CONDITIONAL GO / NO-GO** for beginning the canonical-body/live-locomotion slice

## 0. Authority and purpose

This plan consolidates the nutrient-economy decisions from:

- `docs/2026-07-11-sirrobin-design-document.md`, especially §§2.3, 2.7–2.9, 6, and 7;
- `docs/archive/2026-07-11-restart-brief.md`;
- `docs/superpowers/specs/2026-07-11-sirrobin-restart-architecture-design.md`;
- `docs/archive/plans/2026-07-11-sirrobin-implementation-plan.md`;
- the Rev-2, Rev-3, and Rev-4 correction plans;
- `docs/2026-07-12-sirrobin-developer-reference.md`;
- the original and population-grounded S0 decision reports.

Once accepted, this document supersedes the S1 portions of the earlier implementation and correction plans.
Later corrections govern when they conflict with earlier prose: in particular, Rev-3's exact int64-quanta
reservoirs supersede the design/Rev-2 f32/f64 reservoir state, and Rev-4's separation of mass, metabolic
energy, and f32 mechanics supersedes every mixed-currency equation.

The prerequisite is satisfied: the population-grounded S0 report records GO at 5,000 and 10,000 creatures.
This slice answers one bounded question:

> Can a closed, single-limiting-nutrient world generate a non-trivial producer–detritus–microbial cycle while
> every nutrient quantum remains attributable, every transport step stays nonnegative, and no carrying-capacity
> or free-source mechanism is needed?

This is a material-cycle test. It is not yet a creature ecology.

---

## 1. Reconciled scope and frozen decisions

### 1.1 What this slice builds

The live nutrient inventory contains exactly four stored reservoirs, all int64 mass quanta:

```text
I_mass_q = sum(Nd_q) + sum(Bp_q) + sum(Bd_q) + sum(Bm_q)
```

| Reservoir | Meaning | Stored shape | Stored unit |
|---|---|---|---|
| `Nd_q` | dissolved inorganic limiting nutrient | `[W,Gx,Gy,Gz]` | int64 quanta |
| `Bp_q` | primary-producer nutrient-equivalent biomass | `[W,Gx,Gy,Gz]` | int64 quanta |
| `Bd_q` | detrital nutrient-equivalent biomass | `[W,Gx,Gy,Gz]` | int64 quanta |
| `Bm_q` | microbial nutrient-equivalent biomass | `[W,Gx,Gy,Gz]` | int64 quanta |

One mass quantum is frozen at:

```text
q_mass = 1e-9 mol nutrient
stored_mol(cell) = reservoir_q(cell) * q_mass
concentration(cell) = stored_mol(cell) / cell_volume_m3
```

Concentration is derived when rates or observations need it. There is no second authoritative concentration
field and no f32/f64 mirror of reservoir state.

The processes are:

```text
Nd -> Bp                primary production
Bp -> Nd                producer maintenance/respiration
Bp -> Bd                producer mortality
Bd -> Bm + Nd           bacterial decomposition, BGE split
Bm -> Nd                microbial maintenance/turnover
Bd(z) -> Bd(z+1)        sinking, closed at the bottom boundary
Nd(z) <-> Nd(z+1)       vertical turbulent mixing
Bp(z) <-> Bp(z+1)       conservative producer dispersal/recolonization
Bm(z) <-> Bm(z+1)       conservative microbial dispersal
```

Every arrow is an exact integer debit/credit transaction. Local reaction rates and transport coefficients are
f64 calculations that request transfers; they never mutate reservoirs directly.

### 1.2 Decisions that supersede older S1 prose

1. **Reservoir state is int64, not f32 or f64.** `close_books` is exact `==`; tolerance language applies to
   independent rate/transport corroboration, not bookkeeping.
2. **S1 conserves nutrient mass only.** No creature reserve exists yet, so there is no meaningful stored
   metabolic-energy ledger. `E_chem=e_N*biomass` and absorbed/respired energy may be emitted as derived
   diagnostics, but they never enter an equality and are never stored as a synced biomass-energy pool. The
   native energy ledger begins when real reserve and heat reservoirs land.
3. **The microbial pool has explicit turnover.** Without `Bm -> Nd`, BGE continuously traps nutrient in `Bm`
   because S1 has no microbivores. A first-order microbial maintenance/turnover flux is required for a
   non-trivial closed cycle; its anchor is frozen before dynamics testing and replaced, not double-counted,
   when real grazing lands.
4. **Producer loss is mandatory.** Production alone can plateau but cannot crash. S1 includes measured
   producer maintenance `Bp -> Nd` and mortality `Bp -> Bd`; the consumer-plateau claim remains deferred.
5. **Eulerian fields are the only implementation in this slice.** S1 measures interpolation and synthetic
   point-depletion resolution. It does not build an unused parcel implementation. The parcel fork is decided
   at the feeding slice using real dense-grazing telemetry.
6. **Sediment is absent, not a zero-valued placeholder.** Burial is off in the closed box; bottom detritus
   remains `Bd` and can remineralize. `Sed` first appears with an enabled burial transaction.
7. **Creature `struct_N` is absent.** There is no ColonyState in this slice. Adding empty creature reservoirs
   would create speculative schema and weaken the single-source-of-truth rule.
8. **Horizontal currents, Ekman upwelling, geology, vents, iron, temperature, grazing, predation, metabolism,
   reproduction, and mutation are deferred.** Analytic depth-dependent light and prescribed vertical mixing
   are the only environmental drivers.
9. **Milestone names never namespace runtime code.** Runtime packages are `economy`, `fields`, `numerics`,
   `observe`, and later `core`; no `s1` package or class name is introduced.
10. **Suspended living pools are not immobile.** `Bp` and `Bm` use the same conservative vertical
    finite-volume mixing mechanism as `Nd` (with reservoir-specific frozen diffusivity anchors if required).
    This prevents `Bp=0` from becoming a permanent sterile-cell absorbing state while preserving the rule that
    production cannot spontaneously generate producers.
11. **Self-shading is deferred explicitly.** S1 uses prescribed analytic light
    `I0*exp(-k_att*z)` without biomass-dependent attenuation. Bloom termination must therefore be demonstrated
    from nutrient drawdown and baseline physical losses, never credited to a self-shading feedback that is not
    implemented.

### 1.3 Closed boundaries

The S1 box is periodic horizontally and no-flux vertically for nutrient inventory:

- no geological source;
- no river/weathering source;
- no burial/export sink;
- no advective boundary export;
- no external nutrient injection after reset.

Light is an analytic driver, not nutrient. It can change reaction rates but cannot change `I_mass_q`.

### 1.4 Proposed validation world — freeze before implementation

The older documents do not pin world dimensions, grid resolution, or ecological timestep. They are
load-bearing because concentration units, transport CFL limits, soak duration, and memory depend on them.
Phase 0 must accept or replace this proposed validation configuration before any expected values are emitted:

| Quantity | Proposed value | Reason |
|---|---:|---|
| `W` | 1 | near-term single world |
| `Lx,Ly,Lz` | `640,640,160 m` | small, dense all-ocean validation volume |
| `Gx,Gy,Gz` | `64,64,32` | 10 m horizontal, 5 m vertical cells |
| boundary x/y | periodic | master design |
| boundary top/bottom | no nutrient flux | closed S1 inventory |
| `dt_eco` | `8640 s` (0.1 day) | 1e6 steps = 100,000 ecological days |
| production/reaction evaluation | once per ecological step | rates are expressed per second |
| transport | subcycle only if frozen CFL checks require it | no silent timestep clipping |

The config validation must prove, from frozen `Kz_max`, `w_sink_max`, reaction-rate bounds, and cell sizes:

```text
mixing CFL:  2*Kz_max*dt_transport/dz^2 <= 1
sinking CFL: w_sink_max*dt_transport/dz <= 1
```

If either fails, the number of transport substeps is a derived integer in the config hash. It is never chosen
at runtime from observed field values.

Transport CFL stability is necessary but not sufficient. The nonlinear reaction step has a separate empirical
convergence gate: the frozen authorizing pulse is rerun over the same physical horizon at `dt_eco/2`, with all
rate anchors unchanged. Peak `Bp`, minimum `Nd`, late-window mean `Bp`, and integrated transfer totals must each
agree within `5%` relative mixed tolerance, and peak/crash event times must agree within one coarse
`dt_eco`. Both timesteps must independently close exact integer books. Failure is a numerical-model failure,
not an ecological cycle.

### 1.5 Scientific anchors — provenance before tuning

The documentation pins only part of the parameter set: `BGE≈0.2`, Martin `b≈0.858`, producer maintenance
`m_resp≈0.10/day`, and baseline producer mortality `d0≈0.05/day`. It does not pin `mu_max`, `K_I`, `K_N`,
`d_dd`, microbial turnover, `k_remin(z)`, `w_sink`, `Kz(z)`, light attenuation, or initial inventory.

Phase 0 creates `oracle/fixtures/economy/anchor_manifest.json` with, for every anchor:

- symbol, numeric value, SI unit, and valid range;
- source/provenance and conversion derivation;
- whether it is a measurement, numerical stability choice, or validation initial condition;
- config hash contribution;
- the behavioral tests it influences.

All anchor values are frozen before bloom/standing-stock results are observed. Changing an anchor after a
dynamics run creates a new manifest version and invalidates prior dynamics evidence. Closure failures may
never be repaired by anchor changes.

---

## 2. Canonical numerical contracts

### 2.1 Exact transfer primitive

The landed `numerics.transfer.transfer_quanta` is the starting point, not yet the complete S1 ledger. S1
promotes it from fake-reservoir scaffold to a currency-aware transaction surface:

```python
transfer_quanta(src_q, dst_q, requested_q, mask)
    -> (src_after_q, dst_after_q, shortfall_q)
```

Preconditions remain hard:

- all reservoirs/requests are int64 and nonnegative;
- source and destination shapes match;
- request is capped at source availability;
- destination overflow is checked before addition;
- no reservoir or cumulative meter may exceed `2^62`;
- malformed live state raises rather than returning a plausible zero.

S1 adds a `MassLedger` that owns the registry and initial per-world inventory. Concrete reaction code receives
transaction methods, not writable registry internals. `close_books()` reduces per world and asserts:

```text
sum(Nd_q,Bp_q,Bd_q,Bm_q) == I0_mass_q
all reservoirs >= 0
all reservoirs < 2^62
```

The equality is exact on CPU and CUDA. Since every term is nonnegative and the checked total is below `2^62`,
no intermediate int64 reduction can overflow.

### 2.2 Deterministic flux commitment

Each nonnegative directed reaction channel owns an f64 sub-quantum carry:

```python
acc = requested_mol + carry_mol
n_q = floor(acc / q_mass)
carry_after = acc - n_q*q_mass
```

The carry remains in `[0,q_mass)`, belongs to the source until a quantum commits, and is snapshot state. It is
not included in stored inventory. Signed physical fluxes are split into two named nonnegative directions.

Availability is resolved after commitment:

```text
effective_q = min(n_q, source_q)
shortfall_q = n_q - effective_q
```

The caller must define the shortfall consequence; no process may ignore it. For substrate-limited reactions,
the realized flux is `effective_q`. A persistent nonzero shortfall in a supposedly CFL-safe transport path is
a failed detector, not routine clipping.

### 2.3 Exact split transactions

Bacterial decomposition is one source debit with two credits. Independent rounding of all three legs is
forbidden. First commit the total `Bd` decomposition quanta, then partition that exact integer:

```text
n_total = committed decomposition from Bd
n_Bm    = deterministic_fraction(n_total, BGE, split_carry)
n_Nd    = n_total - n_Bm
Bd     -= n_total
Bm     += n_Bm
Nd     += n_Nd
```

`deterministic_fraction` owns a bounded f64 fractional carry so long-run `n_Bm/n_total` corroborates BGE
without ever violating `n_Bm+n_Nd==n_total`.

### 2.4 Conservative face transport

Mixing and sinking operate on cell inventories, never on a separately stored concentration field. Fluxes are
computed from f64 derived concentrations and face geometry. Each undirected face produces one signed request,
converted into exactly one directed integer transfer.

For a cell with multiple outgoing faces, the transport kernel must use deterministic integer apportionment:

1. compute all nonnegative outgoing physical requests from the same old-state snapshot;
2. commit the cell's total outgoing request with one bounded carry;
3. cap the committed total at source availability;
4. allocate the integer budget across faces by floor plus stable largest-remainder order `(axis,face_index)`;
5. debit the source once and credit each neighbor by its allocation;
6. assert allocated credits sum exactly to the source debit.

This avoids sequential directional bias, negative cells, and the multi-face overdraft that independent clamps
would hide. Horizontal wrap faces are unique; vertical boundary faces request zero. Transport reads old state
and writes scratch, then swaps once.

### 2.5 Units

All rate functions use explicit physical quantities:

```text
inventory_q                 [quanta]
inventory_q*q_mass          [mol]
concentration               [mol m^-3]
specific rates              [s^-1]
reaction flux               [mol s^-1 cell^-1]
diffusive face flux         [mol s^-1]
light                       [W m^-2] or documented PAR equivalent
dt                          [s]
```

Public positions use the project ENU world frame: the water surface is `z_world=0`, the ocean occupies
`z_world<0`, and `depth_m=-z_world`. Reservoir storage is ordered internally from shallow to deep by positive
depth. `FieldSample.gradient_mol_m4[...,2]` is the derivative with respect to world z, so its sign is the
negative of the internal depth derivative.

No function may accept an ambiguous `amount` or `density` without its unit in the contract/docstring.

### 2.6 Determinism posture

Exact bookkeeping does not imply identical ecological trajectories across devices. Every CPU/CUDA/eager/
compiled run must close its own int64 books exactly. Stable integer apportionment and tie-breaking are exact
for identical committed requests. The f64 rate calculations that decide when a sub-quantum carry crosses a
boundary are validated with mixed tolerances; CPU↔CUDA or eager↔compiled state identity is an informational
diagnostic, consistent with the project's three-tier determinism policy. Same-device snapshot continuation,
with the same execution rung, remains exact.

---

## 3. Reaction and transport model

### 3.1 Analytic light and nutrient limitation

```text
I(z)      = I0 * exp(-k_att*z)
gamma_L   = I/(I+K_I)
f_N       = C_Nd/(C_Nd+K_N)
f_lim     = min(gamma_L,f_N)
G_prod    = mu_max*f_lim*Bp_mol
```

`liebig_min` is literal `min`, not a gain/ramp. Production requests `Nd -> Bp` and realizes no more than
available `Nd`. `Bp=0` remains zero under reactions, so validation initial conditions seed nonzero producer
biomass explicitly rather than relying on spontaneous generation. Conservative `Bp` mixing may subsequently
recolonize a sterile cell from a neighbor; no reaction may create the first producer quantum in an all-zero
producer world.

### 3.2 Producer losses

```text
P_resp = m_resp * Bp_mol                         Bp -> Nd
M_prod = (d0 + d_dd*C_Bp) * Bp_mol              Bp -> Bd
```

The concentration-dependent mortality term is a physical crowding/loss rate, not a hard ceiling: it has no
target stock and no branch on population level. Its value is frozen in the anchor manifest. Because it is
mathematically capable of imposing a logistic equilibrium, it cannot authorize the uncapped-loop claim. The
complete authorizing pulse is also run with `d_dd=0`, and that ablation must still rise, draw down `Nd`, peak,
and crash through nutrient exhaustion plus baseline loss. If only `d_dd>0` terminates the bloom, S1 is NO-GO.

All reaction requests for a step are computed from the same old-state snapshot. Competing debits from `Bp`
are resolved by a documented stable priority only after total availability is known. The proposed priority is
maintenance, then mortality; production credits cannot be spent until the next ecological step.

### 3.3 Detrital decomposition and microbial turnover

```text
R_detritus = k_remin(z)*Bd_mol                   Bd -> Bm + Nd
T_microbe  = m_microbe*Bm_mol                    Bm -> Nd
```

The BGE split applies to realized detrital decomposition. Microbial turnover is required while no microbivore
exists; the future feeding slice replaces the provisional term or retains a separately sourced maintenance
component, never adds grazing on top without an explicit double-count audit.

`k_remin(z)` has a strictly positive frozen floor recorded in the anchor manifest. Configuration validation
must prove that the implied bottom-cell residence time is finite and within the declared validation horizon.
This prevents a closed-bottom column from turning its deepest `Bd` cell into a quasi-terminal reservoir by
construction.

### 3.4 Sinking and Martin corroboration

`Bd` sinks downward with `w_sink` through conservative face transfers. The bottom is closed: material remains
in the bottom `Bd` cells. Depth-dependent `k_remin(z)` is derived so the column's expected surviving detrital
flux corroborates the Martin profile:

```text
F(z)/F(z0) = (z/z0)^(-b), b≈0.858
```

The implementation may use an equivalent discrete attenuation, but the mapping from `b`, `w_sink`, layer
depths, and the frozen `k_remin` floor must be committed in the independent oracle. `martin_weights` must be
nonnegative and its partition must close exactly in the integer transaction arm. Gate D corroborates the
attenuation shape and conservative transport only; it does not claim a complete biological pump. Burial/export
is absent in S1 and a functioning open-boundary pump remains later work.

### 3.5 Vertical mixing and producer recolonization

Mixing uses the conservative finite-volume face form:

```text
J_{k+1/2} = -Kz_{k+1/2} * A_xy * (C_{k+1}-C_k)/dz
Delta R_k = dt*(J_{k-1/2}-J_{k+1/2}), R in {Nd,Bp,Bm}
```

There is no post-update positivity clamp that mints mass. Positivity comes from the stable timestep plus exact
availability-limited face transfers. A clamp activation counter is therefore a stop-and-audit detector and
must remain zero in authorizing runs.

`Bp` mixing is a required recolonization mechanism: a zero-`Bp` cell adjacent to a populated cell must receive
producer quanta under a nonzero gradient, while an all-zero `Bp` world remains identically zero. `Bm` is mixed
for the same suspended-pool consistency. `Bd` remains a sinking particulate pool in S1; adding turbulent
particle diffusivity is a later refinement and may not be implied by S1 results.

### 3.6 Step ordering

The canonical ecological step is frozen:

1. derive concentrations and light from state at `t_n`;
2. compute all local reaction requests from the same snapshot;
3. commit/resolve local transactions in order: producer maintenance, producer mortality, primary production,
   detrital decomposition/BGE split, microbial turnover;
4. sink `Bd` using old post-reaction transport state;
5. vertically mix `Nd`, `Bp`, and `Bm` from the same old post-sinking transport snapshot;
6. advance the ecological clock;
7. run exact `close_books` and emit the step ledger/diagnostics.

Changing this order changes the numerical model and requires a config/schema version bump. Row-sliced or
compiled implementations must reproduce this logical order.

---

## 4. Field and observation contracts

### 4.1 Eulerian storage

`fields.grid.ScalarGrid` owns geometry and readonly access to one int64 reservoir tensor. Economy code owns
transactions; field code owns interpolation and transport geometry. Public sampling returns derived values:

```python
FieldSample(value_mol_m3: Tensor, gradient_mol_m4: Tensor)
sample(position_m) -> FieldSample
```

Sampling is trilinear, periodic in x/y, and clamped only to the closed vertical domain. Exact grid-center and
linear-ramp fixtures are committed. Padded/invalid positions fail validation rather than wrapping silently in z.

### 4.2 Transactional point depletion probe

S1 includes a synthetic `deplete_at(position, requested_q)` validation surface that distributes one integer
debit across the eight interpolation neighbors using deterministic largest-remainder weights, returning the
realized debit. The sum removed is exact and no cell becomes negative.

This is infrastructure evidence, not biological grazing. The report records:

- smallest feature with a nonzero sampled gradient;
- depletion footprint in cells/metres;
- requested versus realized quanta;
- hotspot recovery under mixing/reaction;
- cost relative to field stepping.

The data informs the later Eulerian/parcel decision; it cannot authorize parcels or claim dense grazing works.

### 4.3 No horizontal ecology claim yet

Depth-dependent light, sinking, and mixing can authorize vertical zonation: depleted photic-zone `Nd`, deeper
remineralization, and a nonuniform producer profile. Persistent gyre deserts, fronts, upwelling provinces, and
horizontal biogeography require S6 currents/geology and are not S1 acceptance criteria.

---

## 5. Package and import structure

Create capability packages only:

```text
src/sirrobin/
  numerics/
    flux.py                 # commit_flux, exact fraction, integer apportionment
    transfer.py             # checked exact transfers (promoted, not duplicated)
  fields/
    contracts.py            # FieldSample and read protocols
    geometry.py             # GridGeometry, coordinates, boundary policy
    sample.py               # trilinear sampling
    transport.py            # conservative face-flux kernels
  economy/
    config.py               # EconomyConfig and anchor/config hashes
    contracts.py            # state and step-ledger public contracts
    ledger.py               # MassLedger registry and exact close_books
    reactions.py            # pure physical rate requests
    state.py                # canonical four-reservoir state + carries
    step.py                 # ordered composition root for one ecology step
    snapshot.py             # lossless restart of state/carries/clock/config
  observe/
    economy.py              # readonly telemetry projections
oracle/fixtures/economy/
  anchor_manifest.json
  reaction_cases.json
  column_cases.npz
  bloom_config.json
tools/
  economy_oracle.py         # numpy-only independent fixture generator
  run_economy_soak.py
  build_economy_report.py
tests/
  economy/
  fields/
  numerics/
docs/superpowers/reports/
```

There is no `s1` runtime namespace. `economy` may depend on public `fields` and `numerics`; `fields` may depend
on `numerics`; neither may import `physics`, `core`, or `observe`. `physics` remains independent of economy and
fields. `observe` reads public contracts only. The import-linter gains explicit sibling/forbidden contracts;
a strict total layer chain is not used to pretend that physics and economy depend on one another.

Phase 0 ratifies this sibling structure in the master design. Generic scalar sampling remains owned by
`fields` as `FieldSample`. Mechanical medium inputs remain owned by `physics` as a physics-facing
`FluidSample`/future `MediumSample` contract. When S2 couples bodies to fields, the composition root in `core`
samples fields and constructs the physics contract; neither sibling imports the other and no duplicate
authoritative field state is introduced.

The full `core.Colony` composition root is deferred. `economy.step` is the narrow S1 driver and later becomes
one contributor to the whole tick.

---

## 6. Fixtures-first verification

### 6.1 Independent oracle

Before torch economy code is implemented, `tools/economy_oracle.py` uses only Python/NumPy and writes literal
fixtures for:

- Monod and light limitation over zero, half-saturation, saturated, and extreme inputs;
- each local reaction and the combined same-snapshot request set;
- exact BGE integer partition including long-run fractional carry;
- one-dimensional mixing columns with no-flux boundaries;
- sinking with closed bottom accumulation;
- Martin attenuation over the frozen depth bands;
- transport apportionment ties and near-empty cells;
- trilinear values/gradients and periodic x/y boundaries;
- a short bloom/crash trajectory on a tiny column.

The fixture manifest records source hashes, config/anchor hashes, dtype, units, and array-order conventions.
Production tests consume committed values and are forbidden from generating expected results.

### 6.2 Gate A — configuration, units, and boundaries

| ID | Assertion |
|---|---|
| A1 | Proposed validation world or its accepted replacement is frozen and hash-bound before fixtures. |
| A2 | Every anchor has a value, unit, provenance, range, and classification. |
| A3 | `q_mass`, cell volume, concentration conversion, and all rate units pass dimension tests. |
| A4 | Mixing/sinking substep counts satisfy frozen CFL inequalities. |
| A5 | Import boundaries pass; injected economy→physics and fields→core imports fail. |
| A6 | Config/anchors/fixtures/snapshots round-trip losslessly. |

### 6.3 Gate B — exact bookkeeping

| ID | Assertion |
|---|---|
| B1 | Every local and transport transaction preserves total mass exactly. |
| B2 | `close_books` is exact per world after every authorizing step. |
| B3 | Reservoirs never go negative and never exceed `2^62`. |
| B4 | Every carry remains in its declared half-open interval and restores identically. |
| B5 | BGE and face apportionment credits sum exactly to their single debit. |
| B6 | An injected raw reservoir write is caught by an AST/runtime audit. |
| B7 | A one-million-step closed soak has zero book failures—there is no drift tolerance to consume. |

### 6.4 Gate C — reaction fidelity

| ID | Assertion |
|---|---|
| C1 | Monod/Liebig/light functions match independent f64 fixtures with mixed tolerance. |
| C2 | Production never exceeds available `Nd`; zero substrate and zero seed behave correctly. |
| C3 | Producer maintenance/mortality and microbial turnover match frozen rate fixtures. |
| C4 | BGE long-run realized split matches the anchor within one mass quantum per committed event. |
| C5 | Changing tensor iteration/order cannot change integer closure or stable tie results. |

Initial oracle tolerances, frozen before production:

```text
f64 rate/value: atol 1e-12 in native SI unit + rtol 1e-10
f32 diagnostic: atol 1e-7  in native SI unit + rtol 1e-5
integer state: exact ==
```

### 6.5 Gate D — field and transport fidelity

| ID | Assertion |
|---|---|
| D1 | Trilinear value and analytic gradient fixtures pass at centers, faces, wraps, and arbitrary points. |
| D2 | Mixing a constant field is identity; a two-layer gradient moves mass down-gradient for `Nd`, `Bp`, and `Bm`. A sterile `Bp` cell is recolonized from a populated neighbor while an all-zero producer world remains zero. |
| D3 | Mixing and sinking conserve exact quanta and remain nonnegative. |
| D4 | No-flux top/bottom and periodic x/y boundaries are exercised independently. |
| D5 | Whole-grid and row-sliced implementations agree exactly when they preserve the same committed requests/order; eager/compiled each close exactly and rate/aggregate differences stay within frozen mixed tolerances. |
| D6 | Martin column survival, including the positive remineralization floor, matches independent expected values within the rate tolerance; this corroborates attenuation, not a complete biological pump. |
| D7 | Clamp/intervention/transport-shortfall counters are exactly zero on authorization configurations. |

### 6.6 Gate E — emergent closed-loop behavior

The dynamics corpus is frozen before the production run and contains at least:

1. uniform low nutrient/seed biomass;
2. surface nutrient pulse with seed producers;
3. deep nutrient inventory with low surface nutrient;
4. no-light control;
5. no-mixing control;
6. BGE endpoint controls (`0` and `1`) as non-authorizing mechanism checks;
7. `d_dd=0` authorizing anti-cap ablation;
8. `dt_eco/2` authorizing reaction-convergence run over the same physical horizon.

The authorizing pulse must satisfy all of:

- `Bp` rises above its initial value;
- `Nd` declines during growth;
- `argmax(Bp)` is not the final sample;
- final `Bp < 0.70*peak(Bp)` during the crash window;
- `Nd` partially recovers after its minimum;
- `Bd` and `Bm` are each nonzero during the cycle;
- no reservoir becomes a monotone terminal nutrient trap;
- the late-window amplitude decreases or reaches a nonzero statistically stationary band;
- exact books close at every sampled step;
- no carrying-capacity symbol or target-stock branch is reachable.

The `d_dd=0` run must independently satisfy the rise, drawdown, non-final peak, crash, nutrient-recovery,
nonzero-detritus, nonzero-microbe, and exact-closure requirements above. It is a hard GO gate, not diagnostic
telemetry.

The `dt_eco/2` run must preserve the same qualitative outcome and meet the frozen convergence metrics in §1.4.
A cycle that disappears or materially shifts under timestep halving is numerical and cannot authorize S1.

Vertical-zonation acceptance requires a surface/deep difference with the expected sign under light+sinking+
mixing and loss of that difference in the appropriate controls. Horizontal deserts and consumer plateaus are
explicitly not claimed.

### 6.7 Gate F — restart, representation evidence, and affordability

| ID | Assertion |
|---|---|
| F1 | Snapshot at an adversarial carry state, restore, and continue gives exact integer/carry/clock replay. |
| F2 | Partial snapshot omitting carries diverges in a negative test. |
| F3 | Synthetic point depletion removes exactly the returned amount and produces finite continuous samples. |
| F4 | Eulerian resolution metrics are emitted; parcel adoption remains undecided until real grazing. |
| F5 | Authorizing grid is non-OOM under the 11 GiB project cap on the RTX 5070. |
| F6 | No Python per-cell loop, per-step host synchronization, or dynamic-shape selection exists in the hot step. |
| F7 | Eager/compiled timing, cell-updates/s, peak memory, and profiler attribution are recorded. |

S1 does not invent a hard whole-tick throughput floor. Its grid dimensions are provisional validation
dimensions and the later whole-tick gate includes real creatures, fields, and feeding. A structural hot-path or
OOM failure is still NO-GO.

---

## 7. Implementation sequence

### Phase 0 — documentation and anchor freeze

1. Accept or replace the proposed validation world and ecological timestep.
2. Freeze the scientific/numerical anchor manifest with unit derivations.
3. Amend the master design and developer reference so final int64 mass state supersedes f32/f64 reservoir text,
   S0 records both historical and revised Gate-E decisions, and the fields/economy/physics sibling layering plus
   future core-owned medium adapter is the single current architecture.
4. Pin S1's four-reservoir inventory, microbial turnover, `Bp`/`Bm` mixing, positive deep remineralization
   floor, mass-only scope, no-flux bottom, self-shading deferral, and parcel deferral.
5. Add `safetensors` to dependencies for deterministic tensor snapshots; JSON stores config/schema metadata.
6. Freeze schema versions and the full gate matrix before dynamics are observed.

**Exit:** repository search returns one current S1 answer for reservoirs, units, timestep, grid, energy scope,
microbial fate, boundaries, and acceptance.

### Phase 1 — independent evidence and numerical primitives

1. Extend checked transfer tests to per-world registry semantics and the `2^62` bound.
2. Implement/test `commit_flux`, deterministic fractional split, and integer apportionment.
3. Write and freeze the independent oracle fixtures and manifest.
4. Add negative fixtures that would mint under independent rounding or multi-face clamping.

**Exit:** Gates A and the primitive subset of B/C/D are green without economy production code.

### Phase 2 — canonical fields and exact ledger

1. Implement `GridGeometry`, reservoir-backed scalar grids, and trilinear sampling.
2. Implement `MassLedger`, state validation, exact per-world `close_books`, and transaction audit.
3. Implement lossless snapshot/restore including every carry and buffer parity bit.
4. Implement transactional synthetic point depletion.

**Exit:** Gates A, B1–B6, D1, and F1–F4 are green.

### Phase 3 — local reaction channels

Implement one channel at a time, each fixture- and ledger-gated before the next:

1. primary production;
2. producer maintenance;
3. producer mortality;
4. detrital decomposition and exact BGE split;
5. microbial turnover;
6. combined same-snapshot reaction transaction.

**Exit:** Gate C green; combined local reactions close exactly and cannot overdraw.

### Phase 4 — transport

1. conservative integer face apportionment;
2. vertical sinking with closed bottom;
3. vertical mixing with frozen subcycling;
4. Martin-profile mapping and corroboration;
5. row-sliced/compiled equivalence and intervention counters.

**Exit:** Gate D green on tiny fixtures and the authorizing grid.

### Phase 5 — full closed loop and soak

1. Compose the canonical step order.
2. Run mechanism controls before the authorizing pulse.
3. Run bloom/crash and vertical-zonation corpus.
4. Run one-million-step closed soak with exact closure checked every step on device and sampled to telemetry.
5. Snapshot/restore at multiple carry/buffer phases during the soak.

**Exit:** Gates B7, E, and F1/F2 green.

### Phase 6 — performance and decision report

1. Benchmark eager and compiled field steps on CPU/CUDA.
2. Record peak memory, substep count, cell-updates/s, compile warmup/failure, and profiler attribution.
3. Emit representation-resolution evidence without implementing parcels.
4. Generate metric tables from telemetry.
5. Classify every falsifier and record GO/CONDITIONAL GO/NO-GO.

**Exit:** Gate F green and the S1 decision report is committed.

---

## 8. Task DAG

| ID | Task | Depends on | Acceptance |
|---|---|---|---|
| T00 | Reconcile docs and accept validation config | — | one current authority |
| T01 | Freeze anchor/config/schema manifests | T00 | A1–A4 |
| T02 | Extend import firewall/package scaffold | T00 | A5 |
| T03 | Independent economy oracle | T01 | fixture hashes frozen |
| T04 | Exact flux carry/fraction/apportionment | T01,T03 | B1,B4,B5 |
| T05 | MassLedger and audit surface | T02,T04 | B2,B3,B6 |
| T06 | Grid geometry and interpolation | T02,T03 | D1,D4 |
| T07 | Snapshot with carry/buffer parity | T05,T06 | A6,F1,F2 |
| T08 | Transactional point depletion probe | T04,T06 | F3,F4 |
| T09 | Primary production | T03,T05,T06 | C1,C2 |
| T10 | Producer loss channels | T03,T05,T06 | C3 |
| T11 | Decomposition/BGE/microbial turnover | T03,T04,T05,T06 | C3,C4 |
| T12 | Combined local reaction step | T09–T11 | B1–B5,C5 |
| T13 | Sinking + Martin mapping | T03,T04,T06 | D3,D4,D6 |
| T14 | Vertical mixing | T03,T04,T06 | D2–D4 |
| T15 | Canonical economy step | T07,T12–T14 | D5,D7 |
| T16 | Dynamics/control corpus | T15 | E |
| T17 | One-million-step soak/restart | T15,T16 | B7,F1,F2 |
| T18 | CPU/CUDA benchmark/profiler | T15–T17 | F5–F7 |
| T19 | Generated decision report | T18 | all risks classified |

Critical path:

```text
T00 -> T01 -> T03 -> T04 -> T05
                 \-> T06 -> T09/T10/T11 -> T12
                         \-> T13/T14 ------> T15 -> T16 -> T17 -> T18 -> T19
T05/T06 -> T07/T08 -------------------------/
```

No dynamics work starts before anchor/fixture freeze. No full soak starts before all single-step channels and
transport gates are green. No next milestone begins before T19 records an accepted decision.

---

## 9. Falsifier register

| ID | Falsifier | Detector | Required response |
|---|---|---|---|
| R1 | Int64/cumulative overflow is reachable | config bound + runtime `<2^62` check | reduce inventory horizon or change quantum in a new schema; never wrap |
| R2 | Sub-quantum carry biases rates | long-run fraction/rate fixtures | audit commitment and split carries; do not enlarge q to hide cost |
| R3 | Independent rounding mints mass | exact split/face negative fixtures | replace with single-debit partition before proceeding |
| R4 | Transport creates negatives or uses clamps | intervention/shortfall counters | stop; fix CFL or apportionment; no post-hoc renormalization |
| R5 | Microbes become a terminal nutrient trap | reservoir time series/control | audit turnover/BGE units; do not add a deletion sink |
| R6 | Bloom cannot decline without target-stock logic | full authorizing pulse with `d_dd=0` | NO-GO; audit physical loss anchors and scope, never add a carrying cap |
| R7 | Apparent stability is extinction | late-window stock/flux gate | reject; require nonzero active cycle |
| R8 | Horizontal desert claim needs missing currents | control/claim audit | defer claim to transport/world slice |
| R9 | Eulerian depletion is visibly quantized | point-depletion telemetry, later real grazing | open a separate parcel experiment; do not carry two live representations |
| R10 | Row slicing/amortization changes results | exact integer-state comparison | fix logical ordering/buffer parity before performance work |
| R11 | Restart omits carries or parity | adversarial snapshot test | schema failure; block soak/GO |
| R12 | CUDA/compile path violates closure or materially changes rates | eager/compiled/CPU/CUDA closure plus mixed-tolerance fixture/aggregate comparisons | isolate lowering or retain eager; exact books never waived |
| R13 | Hot field step is host-bound or OOM | profiler/memory gate | fuse/vectorize in its own tranche or revise grid; no Python cell loop |
| R14 | Energy scope is silently reintroduced | registry/AST audit | remove stored biomass energy; defer native energy ledger to reserve slice |
| R15 | S0 physics/import boundaries regress | full existing suite + import-linter | stop and repair regression before S1 acceptance |
| R16 | Anchor tuning follows observed bloom | manifest chronology/hash audit | invalidate run; freeze a new experiment before rerunning |
| R17 | Apparent ecological cycle is an explicit-step artifact | same-horizon `dt_eco/2` convergence gate | NO-GO; reduce/freeze timestep or change the integrator in a new schema before rerunning |
| R18 | Local producer extinction is absorbing | `Bp` mixing/recolonization fixtures and pulse spatial trace | repair conservative producer transport; never add spontaneous generation |
| R19 | Closed bottom becomes a quasi-terminal `Bd` trap | positive `k_remin` floor, residence-time validation, long-soak pool trace | revise the sourced floor or horizon before dynamics; never delete bottom stock |

No risk is retired because this plan names a mitigation. It is clear only when its detector is green in
committed evidence.

---

## 10. Decision classes and definition of done

### GO

- Gates A–F are green.
- Exact mass closure holds after every step of the one-million-step soak.
- No negative, overflow, clamp, ignored-shortfall, or regularization event occurs.
- The authorizing bloom grows, draws down nutrient, declines, and enters a nonzero late regime without a cap.
- The same uncapped behavior passes with `d_dd=0`, and the same physical-horizon run converges at `dt_eco/2`.
- Vertical zonation passes only the claims available from light/sinking/mixing.
- Snapshot/restart is exact for integer state, carries, clock, and buffer parity.
- Eulerian evidence is recorded without prematurely claiming the parcel fork resolved.
- Existing S0 CPU/CUDA/oracle/import tests remain green.

### CONDITIONAL GO

All correctness, conservation, dynamics, and restart gates are green, but one non-structural performance path
(for example `torch.compile`) fails while a non-OOM eager/CUDA path remains scientifically usable. The report
states the operating constraint and queues a bounded optimization. Conditional GO cannot waive exact closure,
nonnegativity, non-trivial dynamics, anchor freeze, or snapshot completeness.

### NO-GO / REVISE

Any of the following is terminal for this plan:

- exact books fail even once;
- a reservoir becomes negative or overflows;
- the loop requires a carrying-capacity/target-stock branch;
- bloom termination requires `d_dd>0`, or the claimed cycle fails the `dt_eco/2` convergence gate;
- `Bm` or another pool becomes an unintended terminal sink;
- the bloom only grows/plateaus or collapses to trivial extinction;
- transport relies on mass-minting clamps or changes with row slicing;
- restart cannot reproduce committed transfers;
- the authorizing grid is OOM or requires a Python per-cell loop;
- the implementation stores biomass mass and chemical energy as synchronized authoritative copies.

The durable output is not merely a nutrient-field implementation. It is evidence that the project's first
ecological loop is closed, causal, restartable, and sufficiently non-trivial to support later morphology-driven
feeding without reintroducing the prior build's mint or cap.
