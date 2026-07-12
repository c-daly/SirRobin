# SirRobin — Consolidated S0 / SpikeSwim Implementation Plan

**Status:** executed S0 authority; outcome **GO** for the revised 5,000–10,000-creature real-time objective,
with the original 90M stretch target retained as failed evidence in the original report and the revised result
in `docs/superpowers/reports/2026-07-12-sirrobin-S0-population-gate-revision-report.md`. The owner replaced only
the ungrounded throughput/population requirement through the pre-registered
`2026-07-12-sirrobin-locomotion-gate-E-revision.md`; the original result remains historical evidence.
**Date:** 2026-07-12
**Scope:** S0 scaffold and frozen-heading locomotion kernel only
**Decision produced:** locomotion-kernel **GO / CONDITIONAL GO / NO-GO** from committed evidence

## 0. Authority and purpose

This document consolidates the S0-relevant decisions from:

- `docs/2026-07-11-sirrobin-design-document.md`;
- `docs/2026-07-11-restart-brief.md`;
- `docs/superpowers/specs/2026-07-11-sirrobin-restart-architecture-design.md`;
- `docs/superpowers/plans/2026-07-11-sirrobin-implementation-plan.md`;
- the Rev-2, Rev-3, and Rev-4 correction plans;
- `docs/superpowers/plans/2026-07-12-sirrobin-S0-kernel-spec.md`;
- `docs/superpowers/reviews/2026-07-12-codex-review-of-s0-kernel-spec.md`.

The original 1,000-creature/90M throughput experiment remains preserved in the decision report as failed
stretch evidence. After that measurement, the project population objective was explicitly revised to 5,000–
10,000 creatures; this is a new pre-registered authorization criterion, not a retroactive retuning of the old
experiment.

Once its Phase 0 documentation reconciliation is accepted, this plan is the single execution authority for
S0. It supersedes the S0 portions of the earlier implementation/correction plans and the standalone kernel
spec. It does **not** supersede later-phase ecology, genome, RNG, feeding, or embodiment decisions except where
an S0 scaffold must keep their boundary open.

S0 answers one bounded question:

> Can the chosen canonical fixed-slot body representation execute the donor-grounded, frozen-heading
> locomotion step over realistic heterogeneous H1/H2 populations with correct mechanics and independent
> corroboration, at a measured rate that makes locomotion a viable component of the later whole tick?

S0 is a **necessary-component test**, not authorization of the complete simulator. End-to-end throughput is
decided later by `G-E2E`, after S2 supplies `StepLive`, development, a field, a spatial hash, and the fixed
feeding stub, and is re-confirmed after real S3 feeding.

The plan deliberately does not broaden S0 into ecology, mutation, steering, metabolic reserve accounting, or
live evolutionary development.

---

## 1. Frozen decisions

### 1.1 Decisions retained from the project direction

| Area | S0 decision | Reason / provenance |
|---|---|---|
| Runtime | One Python/PyTorch Core; no Unity in the live path | Master design §§2.1, 7.2 |
| Device | Identical tensor program on CPU and CUDA; choose from measurements | Master design §2.9 |
| World scope | One near-term world; multi-world batching is not required for an unconditional S0 GO | Rev-2 §6.2–6.3 |
| Hot precision | f32 production; f64 oracle/reference configuration; f64 Lamb precompute | Master design §2.8 |
| Creature layout | `[W,N_cap,...]` plus `alive`; `W=1`, `N_cap=10240`, authorizing `N_live=10000` (lower cell 5120/5000) | Rev-2 §4.1 and Rev-3 §9 |
| Segment layout | `[W,N_cap,S_slot,...]`, `S_slot=17`: slot 0 identity sentinel plus 16 real slots | Rev-3 §2 |
| Parent convention | Creature-local index; root parent = sentinel slot 0; no negative gather index | Rev-3 §2 |
| Pose | Fixed six-pass depth scan over depths 0..5 | Donor cap and all plans |
| Lifecycle | Static buffers; dead-slot recycling by `cumsum`/gather/where; no compaction or reallocation | Rev-2 §4.4, Rev-3 §2.3 |
| Locomotion | Donor one-shot frozen-heading `Sim.Step`; no yaw/steering | Master design §7.2 |
| Production timestep | `Dt = 1/120 s` | Master design §§4.3, 7.2 |
| Length unit | One simulation-length unit is exactly 1 metre; all positions and ellipsoid semi-axes are stored in metres | Master design §§2.4, 4.2 |
| Structural mass unit | `seg_mass_sim` is sim-mass; `1 sim-mass = 250 kg` | Donor `SwimEval.cs:771,1023`; `SimUnits`; restart brief §1 |
| Gene density unit | `density_gene` is sim-mass/m³; physical tissue density is `250*density_gene kg/m³` | Required reconciliation of donor mass convention with SI geometry |
| Added-mass unit | `m_add = k rho_water V` in kg, with `rho_water=1000 kg/m³`, `abc` in m, and `V` in m³ | Master design §4.3 |
| Physics mass | `M_eff = M_body,kg I + sum R diag(m_add) R^T`; body mass and added mass are never conflated | Master design §4.3 |
| Horizontal constraint | Solve the x/z principal submatrix of `M_eff`; enforce `v_y=0` separately | Rev-2 #2, Rev-4 D1 |
| Reservoir scaffold | Exact bounded int64 fake-mass transfers, clearly labelled scaffold-only | Implementation plan §1.8, Rev-3 §1 |
| Performance floor | Revised real-time floor `F_loco_S0 = 1.2e6 creature-steps/s` at 10,000 live creatures; 5×/10× are reported targets | Revised population objective |
| Later whole-tick floor | `F_sci = 2.31e7 creature-steps/s`; not an S0 gate | Rev-2 §6.2, Rev-3 §9.1 |

`N_cap=10240` is the primary static buffer size; the scientific population anchor is 10,000 live creatures.
The 5120/5000 cell remains a lower-population affordability point. Dead lanes are included in all authorizing
timings because padding and fixed capacity are part of the chosen architecture.

The unit chain is therefore explicit:

```text
abc_sim * (1 m/sim-length)                    -> abc_m                       [m]
V = (4/3) pi a_m b_m c_m                     -> displaced volume            [m³]
mass_sim = density_gene * V                   -> structural mass             [sim-mass]
M_body_kg = 250 kg/sim-mass * sum(mass_sim)  -> body inertia scalar         [kg]
m_add_i = k_i * 1000 kg/m³ * V               -> per-axis added mass         [kg]
M_eff = M_body_kg I + sum R diag(m_add) R^T  -> effective inertial matrix   [kg]
```

The Phase 0 doc edit corrects any surviving table that labels `density_gene` directly as kg/m³ while also
applying `KgPerSimMass`; both conventions may not coexist. `KgPerSimMass=250` is not a new calibration. It is a
frozen donor convention cited at `SwimEval.cs:771,1023`, carried by donor `SimUnits`, recorded in
`docs/2026-07-11-restart-brief.md:43,101-102`, and frozen in
`docs/2026-07-11-sirrobin-design-document.md:53,519`. T04 records the donor source revision and exact source
lines in the fixture manifest so this provenance remains auditable after the donor is removed.

### 1.2 Canonical-layout reconciliation

The master design originally said flattened/CSR segments were canonical and padded storage was an ablation.
Phase 0 reconciled it to the later Rev-2/Rev-3 decision that preserves static addresses and CUDA-graph capture:

- fixed padded slots are canonical for S0 through S2;
- flattened/arena storage is a deferred optimization, not an S0 implementation;
- S0 quantifies the padding cost analytically and through a same-layout occupancy experiment;
- a flattened prototype is built only in a later dedicated optimization slice if telemetry shows both:
  `segment-axis work > 30%` of step time and a credible recoverable padding cost larger than the arena's
  lifecycle/capture cost.

Phase 0 updates the master design and architecture spec so the repository has one normative answer before code
begins.

### 1.3 Determinism reconciliation

Rev-3 contains an explicit owner decision relaxing bit-identical floating replay. That later decision governs:

1. **Tier 1 — exact bookkeeping:** hard gate. Integer transfers and `close_books()` are exact inside their
   validated int64 range.
2. **Tier 2 — reproducible discrete decisions:** hard gate. Counter-based integer RNG reference vectors and
   lifecycle slot claims reproduce for a fixed key/schedule. S0 itself performs no mutation or mating.
3. **Tier 3 — bit-identical float trajectory:** optional diagnostic, not CI authorization. Run the same-device,
   no-compile deterministic smoke test and report divergence/tax, but do not constrain the production hot loop
   to achieve bit identity.

Tolerance-based oracle, momentum, force-power, and energy checks remain hard gates. Relaxing Tier 3 does not
relax physical correctness.

### 1.4 Explicit non-decisions

S0 does not decide:

- StepLive/yaw performance;
- the complete end-to-end tick budget;
- ecology reservoir architecture beyond validating the bounded int64 transfer primitive;
- metabolic reserve, heat, feeding, growth, death, or reproduction transactions;
- genome mutation rates or evolutionary outcomes;
- whether flattened storage is ever worthwhile;
- whether GPU is inherently preferred to CPU.

### 1.5 Authorizing corpus — frozen statistical content

The corpus is part of the acceptance contract, not benchmark input chosen after implementation. T05 writes the
literal rows described here to `oracle/fixtures/corpus.json`; T06/T07 and every production test consume that
same file. Once T06 begins, changing any row creates a new corpus version and invalidates all old fixtures and
benchmark comparisons.

Every class contains exactly 64 logical bodies before replication to benchmark batch size:

| Class | Exact segment-count histogram | Required morphology/branch content | Purpose |
|---|---|---|---|
| H0 | `{6:64}` | One six-segment axial, untilted, non-fin body repeated 64 times; identical gait and scale; no mirrored branch | Isolate vectorization and establish the homogeneous denominator; never authorize |
| H1 | `{2:6,3:8,4:8,5:8,6:8,7:8,8:6,10:4,12:4,16:4}`; mean `6.4375` | 32/64 contain at least one bilateral mirrored branch; exactly 26/64 have active Surface fin tails; 32/64 contain nonzero pitch or roll so `M01`/`M12` are exercised; sizes/aspect ratios and gait values cover each quartile of the donor-valid range | Realistic ragged heterogeneous authorization class |
| H2 | `{2:28,3:28,16:8}`; mean `4.1875` | All eight 16-segment bodies are tilted anisotropic; exactly 32/64 have active fin tails; fin/non-fin and 2/3/16-segment rows are interleaved rather than grouped; the eight full bodies occupy distinct scale/aspect-ratio octiles | Skewed branch/mask/occupancy stress authorization class |

Additional anti-gaming rules:

- H1/H2 contain both coherent traveling-wave and deliberately inefficient/incoherent gait cases; at least 8 of
  each class are zero/near-zero net-thrust controls.
- At least 8 H1 and all 8 full H2 bodies have tilted anisotropic matrices for which the donor's unconstrained
  3-D-then-zero path and the correct constrained 2-D solve measurably differ.
- Fin-tail rows cover the full four quartiles of aspect ratio and angle-of-attack inputs inside the donor-valid
  range; no fixture relies only on the axial H1 body from the older kernel spec.
- Geometry, density, gait frequency, gait wave, amplitude, pose, fin parameters, and flags are literal values in
  `corpus.json`, not regenerated from a seed during tests.
- A separate `corpus.sha256` stores the SHA-256 of the exact `corpus.json` bytes; the fixture manifest stores
  that hash, row-order hash, generator source hash, donor source revision, and config hash. Tests fail on any
  mismatch.
- Benchmark replication tiles the 64 rows in their committed interleaved order and truncates only after complete
  cycles where possible. It may not sort by segment count, fin branch, or morphology.

The literal morphology values are frozen during T05 **before** either oracle or kernel is run. T05's review
checks the histograms and coverage rules above directly; performance results can never be used to select or
discard a row.

### 1.6 Conscious S0 infrastructure scope

Full Philox distribution machinery and full-world `SimulationSnapshot` are **deferred**, not silently absorbed
into S0:

- S0 uses no mutation, mating, ecological death roll, or stochastic reproduction, so it does not implement or
  gate the later Philox allocator/distribution transforms.
- S0 lifecycle churn consumes a committed deterministic event schedule from the fixture corpus. Exact equality
  of claim masks, stable IDs, and event records tests the lifecycle without pretending to test biological RNG.
- S0 artifacts/configs must serialize losslessly, but a complete `SimulationSnapshot` waits until S1 introduces
  persistent world reservoirs. Building it now would create a misleading empty schema and speculative breadth.
- The three-tier determinism policy in §1.3 remains the later architectural rule; only the tiers exercised by S0
  appear in S0 acceptance.

---

## 2. S0 acceptance contract

S0 is complete only when all correctness gates are green, the performance evidence exists, and a decision
report classifies every falsifier. A fast wrong kernel is a failure; a correct slow kernel is a valid NO-GO or
conditional result, not an invitation to weaken a gate.

### 2.1 Gate A — scaffold and representation integrity

| ID | Assertion |
|---|---|
| A1 | Import firewall passes; an injected upward/private import fails it. |
| A2 | `S_slot=17`; slot 0 is immutable finite identity; real capacity is exactly 16. |
| A3 | Every root parent is 0; no segment/lifecycle gather can receive a negative index. |
| A4 | Empty and dead bodies have zero forces and acceleration; mixed batches do not leak padded values. |
| A5 | Static addresses/shapes survive lifecycle churn and CUDA graph replay; no compaction or dynamic selection. |
| A6 | Fake mass transfers conserve exactly inside a checked range; source and destination remain nonnegative; overflow attempts fail loudly. |
| A7 | Config, corpus, fixture, and benchmark artifacts round-trip losslessly; no full-world snapshot or resumed-execution claim is made in S0. |

The fake ledger validates reusable infrastructure. It is not evidence that locomotion conserves metabolic or
total physical energy.

### 2.2 Gate B — physical/oracle fidelity

The oracle is two independent arms, each with a distinct purpose:

1. **gain0 untouched-donor conformance** protects the port of the historical behavior and operation ordering;
2. **gain1 analytic corroboration** validates the corrected ellipsoid-mass/added-mass path without relying on a
   patched donor.

Required checks:

| ID | Assertion |
|---|---|
| B1 | Lamb coefficients/factors match committed analytic fixtures; sphere gives `k=0.5`; coefficient sum is 2 within f64 tolerance. |
| B2 | In the f64 torch reference configuration, gain0 pose, tail-tip kinematics, force terms, all six `M_eff` entries, and `dv` match f64 untouched-donor traces; the f32 production configuration separately passes the same single-step fixtures with mixed tolerances. |
| B3 | The f32 production kernel's gain0 960-step cruise speed, cost of transport, reactive ratio, and mechanical work match committed donor aggregates. |
| B4 | gain1 inertial mass, Lamb coefficients, reactive force, fin force/power, tilted `M_eff`, constrained `dv`, and momentum match committed independent values. |
| B5 | H1 and H2 both pass; H0 cannot authorize B. |
| B6 | Zero-force coast preserves velocity within tolerance; the all-sphere momentum identity closes. |
| B7 | No production/oracle import circularity; no fixture is emitted by a patched donor. |
| B8 | Regularization count is exactly zero throughout the H0/H1/H2 authorization corpus. |

Thresholds retained unless the fixture preflight proves a stricter stable value:

- Lamb coefficient/factor: f64 absolute `< 1e-6`;
- gain0 f64-torch-vs-f64-donor trace comparison: relative `< 1e-4`;
- single-step f32-production-vs-committed-fixture: relative `< 1e-4`, with a mixed absolute floor for zero terms;
- f32 production episode aggregates: relative `< 1e-3`.

### 2.3 Gate C — force/power and discrete mechanical consistency

#### C1. Donor force-law identities

Per body and step:

```text
U_cl = max(0,U)
W_t  = V_t + U s

P_reactive = m_t U V_t W_t
P_reactive = tReact U + pWake

P_fin_input = F_n V_t
P_fin_input = tFin U_cl + pFin

p_in = P_reactive + P_fin_input
```

Use the mixed criterion

```text
abs(lhs-rhs) <= P_ATOL(dtype) + 1e-6 * max(abs(lhs), abs(rhs), abs(each constituent))
```

with initial frozen floors:

```text
P_ATOL(f64) = 1e-10 W
P_ATOL(f32) = 1e-6  W
```

The oracle preflight may tighten these floors before kernel implementation. It may not loosen them after seeing
production failures without a documented unit/rounding analysis.

#### C2. Discrete `R_step`

Let `M_n` and `M_{n+1}` be the x/z submatrices of the full effective inertial matrices at consecutive poses:

```text
M_{n+1}(v_{n+1}-v_n) = F_stream Dt + J_reg
DeltaKE = 0.5 v_{n+1}^T M_{n+1} v_{n+1} - 0.5 v_n^T M_n v_n
W_imp   = v_mid dot (F_stream Dt + J_reg)
W_M     = 0.5 v_n^T (M_{n+1}-M_n) v_n
R_step  = DeltaKE - W_imp - W_M
```

The vertical constraint is checked separately and does no work because both endpoint vertical velocities are
zero.

Per-step scale and gate:

```text
S_step = max(abs(DeltaKE), abs(W_imp), abs(W_M), E_ATOL(dtype))
abs(R_step) <= E_ATOL(dtype) + RTOL(dtype) * S_step

RTOL(f64) = 1e-6       E_ATOL(f64) = 1e-12 J
RTOL(f32) = 1e-3       E_ATOL(f32) = 1e-8  J
```

This is an integrator-consistency gate, not by itself validation of the hydrodynamic force model.

#### C3. Executable 100,000-step drift gate

Replace the undefined phrase "bounded-oscillating" with a prefix-budget test:

```text
C_k = sum_{i=1..k} R_i
A_k = sum_{i=1..k} max(abs(DeltaKE_i), abs(W_imp_i), abs(W_M_i), E_ATOL)
D_k = abs(C_k) / A_k
```

After the first 100 steps:

```text
max_k D_k < 1e-3  # f32
max_k D_k < 1e-6  # f64
```

Emit the full `C_k`, `A_k`, and `D_k` curves. A monotone signed residual is reported as numerical bias even if
it remains below threshold; crossing the bound fails. This catches endpoint cancellation without asserting an
oscillation that the driven system has no obligation to exhibit.

### 2.4 Gate D — reproducibility posture

Hard requirements:

- exact int64 transfer/reference tests;
- the committed fixed schedule produces identical lifecycle claim masks, stable IDs, and discrete event records;
- fixture and benchmark manifests include seed, config hash, source revision, hardware, device, dtype, and
  software versions.

Informational diagnostic:

- two same-device, eager, deterministic-mode float runs over 960 steps;
- report maximum absolute state/ledger divergence and deterministic-mode performance tax;
- never convert this diagnostic into the primary scientific gate.

### 2.5 Gate E — throughput and affordability

The authorizing configuration is f32, `W=1`, `N_cap=10240`, `N_live=10000`, fixed padded slots, and the H1/H2
corpora. Both H1 and H2 must have at least one non-OOM measurement at the authorizing population. An all-OOM
H1 or H2 authorization sweep is a failure.

Throughput is always computed from live scientific work, not allocated padding:

```text
creature_steps_per_second = N_live * completed_steps / measured_wall_seconds
```

Dead creature lanes, sentinel lanes, and unused segment slots remain part of the measured cost but never inflate
the numerator.

The revised real-time S0 floor requires:

```text
min(5 measured repetitions) >= F_loco_S0 = 1.2e6 creature-steps/s
```

for both H1 and H2 at 10,000 live creatures on at least one `(device,rung)`. Also report 5× and 10× target
fractions, plus:

- CPU and CUDA curves and crossover `B*`;
- H1/H0 and H2/H0 heterogeneity tax;
- analytic padding ratio and same-layout mean-occupancy/full-occupancy timings;
- lifecycle churn on/off cost;
- deterministic-mode diagnostic tax;
- peak allocated memory and OOM cells;
- profiler attribution for the winner and every tripped falsifier.

H0 is a baseline only. It cannot authorize S0.

---

## 3. Build artifacts and module boundaries

Create only the S0 subset of the intended repository:

```text
pyproject.toml
setup.cfg
src/sirrobin/
  numerics/
    dtype.py
    quat.py
    solve_donor.py
    solve_constrained_xz.py
    transfer.py
  physics/
    config.py
    contracts.py
    lamb.py
    pose.py
    force_reactive.py
    force_fin.py
    force_drag.py
    mass_matrix.py
    swim_step.py
  core/
    clock.py
    contracts.py
  observe/
    telemetry.py
  validation/
    corpus.py
  benchmarks/
    lifecycle.py
    locomotion.py
tools/
  gain1_oracle.py
oracle/
  SirRobinOracle.csproj
  fixtures/
    corpus.json
    gain0_lamb.json
    gain0_trace_H1.npz
    gain0_trace_H2.npz
    gain0_aggregates_H1.json
    gain0_aggregates_H2.json
    gain1_analytic.json
    quadrature_gl256.json
    quadrature_gl32_negative.json
    manifest.json
tests/
  scaffold/
  numerics/
  physics/
  oracle/
  validation/
  benchmarks/
runs/                         # ignored raw benchmark output
docs/superpowers/reports/     # committed S0 decision report
```

Milestone names such as S0/SpikeSwim never appear as runtime module namespaces. Durable evidence loading lives
under `validation`; capability benchmarks live under `benchmarks`; locomotion mechanics live under `physics`.
`physics` may depend only on `numerics` and its own contracts. The C# oracle and independent Python oracle are
offline tools and are never imported by production.

No S1 field/economy implementation, full-world snapshot, Philox sampler, S2 `StepLive`, genetics
implementation, viewer, ROS, or feeding code is created in this slice.

---

## 4. Canonical tensors and invariants

### 4.1 Body storage

For each tensor below, the leading shape is `[W,N_cap,S_slot]`, viewed `[B,S_slot]` without copying:

```text
seg_mask        bool
seg_local_pos   f32 [...,3]
seg_local_rot   f32 [...,4]
seg_abc         f32 [...,3]
seg_mass_sim    f32
seg_area_z      f32
seg_m_add       f32 [...,3]
seg_amp_deg     f32
seg_phase       f32
seg_is_surface  bool
seg_is_tail     bool
seg_has_joint   bool
seg_parent      i16
seg_depth       i8
```

Slot 0 is finite neutral identity. Every real segment is in 1..16. `seg_mask` is the sole source of truth for
segment existence. `alive` is the sole source of truth for creature existence.

Padded/dead lanes must contain finite values. Multiplication by a false mask is not accepted as protection for
NaN or infinity.

### 4.2 Per-body state

```text
alive           bool [W,N_cap]
stable_id       i64  [W,N_cap]
generation      i32  [W,N_cap]
x_com           f32  [W,N_cap,3]
v_com           f32  [W,N_cap,3]      # y == 0
f_hat           f32  [W,N_cap,3]      # frozen unit vector
n_hat           f32  [W,N_cap,3]      # frozen unit vector
swim_freq       f32  [W,N_cap]
tail_slot       i16  [W,N_cap]
gait_time       f64  scalar or [W]
```

`M_body_kg` is derived, never independently stored as an authoritative second copy:

```text
M_body_kg = KgPerSimMass * masked_sum(seg_mass_sim)
KgPerSimMass = 250
```

The fixture/config hash includes `KgPerSimMass`, density conventions, `Dt`, water density, all fin/drag
constants, condition thresholds, tolerances, and mass-model stage.

### 4.3 Lifecycle

Death changes `alive`; birth claims free slots through `cumsum`, clamps every eager gather index, overwrites the
entire child payload including all segment lanes, then sets `alive`. No slot compaction occurs. Lifecycle
churn uses a fixed committed schedule in S0 so performance and graph stability are measured without introducing
an ecological death model.

---

## 5. Complete frozen-heading step contract

The port follows the untouched donor trace for operation ordering. Before implementation, Task T04 records that
ordering as a fixture. The intended mathematical step is:

1. Preserve pose/tail-tip state at `t_n` and `M_n`.
2. Advance gait time by `Dt`; resolve pose at `t_{n+1}` with the six fixed passes.
3. Compute the tail tip from segment center plus rotated local long-axis half-length.
4. Compute tail velocity, signed forward `U`, transverse `V_t`, slope `s`, and `W_t`.
5. Compute reactive thrust/input/wake terms.
6. Compute the surface-fin circulatory channel with `U_cl=max(0,U)`.
7. Compute per-segment axial drag and dissipated drag work.
8. Construct `M_{n+1}` from body inertia plus rotated Lamb added mass.
9. Form `F_stream=(tReact+tFin)f_hat+F_drag`.
10. Solve the constrained x/z update; keep `v_y=0`; update COM position semi-implicitly.
11. Emit all oracle and energy terms without recomputing them downstream.

If the untouched donor trace contradicts the proposed old/new-pose ordering, stop T04 and amend this section.
Do not silently choose an order that makes a test convenient.

### 5.1 Effective mass

```text
M_eff = M_body_kg I_3 + sum_j R_j diag(m_add_j) R_j^T
```

The production horizontal solve uses:

```text
M_xz = [[M00,M02],
        [M02,M22]]
P_xz = (F_stream Dt)_xz
```

The donor gain0 conformance path retains the untouched donor 3x3 cofactor solver and floor only for fixture
comparison. It is not the production constrained solver.

### 5.2 Production solve and intervention policy

Use one production function with mutually exclusive branches:

- `INVALID`: dead, empty, zero body mass, nonfinite input, or construction-bound violation; output zero only
  for dead/empty, reject malformed live bodies;
- `EXACT`: valid and `kappa <= KAPPA_MAX`; solve the true determinant;
- `REGULARIZED`: adversarial protection path; `(M+rI)dv=P`, `J_reg=-r dv`.

The division-free condition predicate and eager denominator masking from the standalone kernel spec are
retained. Matrix input bounds are validated before the solve.

Regularization is a numerical intervention, not a physical damping channel. Every activation is counted and
logged with signed `v_mid dot J_reg`. Unit/adversarial tests must exercise it, but B8 requires zero activations
for every authorizing H0/H1/H2 body and episode. If it activates there, S0 stops for a mass/unit/matrix review.

The f32 determinant-safety claim is verified against an adversarial f64 eigen/reference corpus; the plan does
not assert that `EPS_SPD > eps_f32` alone proves safety.

---

## 6. Fixtures-first implementation sequence

### Phase 0 — reconcile documents and freeze the experiment

1. Amend the master design and restart architecture design to make fixed `[B,17]` slots canonical through S2.
2. Replace their S0 determinism language with the three-tier posture in §1.3.
3. Replace the old S0 acceptance table with Gates A–E from this plan.
4. Record `N_cap=10240`, `N_live=10000`, the lower 5120/5000 cell, hardware pins, VRAM cap 11 GiB, and
   the revised `1.2e6` real-time floor; retain 5×/10× as explicit acceleration targets.
5. Correct the unit tables to use the single chain in §1.1, including `density_gene` in sim-mass/m³ and
   `KgPerSimMass=250 kg/sim-mass` exactly once.
6. Mark older S0 plan/spec documents historical/superseded; do not delete them.

**Exit:** a repository-wide search finds one current S0 answer for layout, determinism, energy normalization,
oracle coverage, and throughput authorization.

### Phase 1 — scaffold without claiming physics

Build package/import structure, immutable config, dtype policy, clock, lossless artifact schemas, telemetry
manifest, bounded int64 transfer, sentinel layout, empty-body smoke path, and fixed lifecycle schedule.

**Exit:** Gate A green. The fake ledger is labelled scaffold-only in code, tests, and telemetry.

### Phase 2 — commit independent evidence before the kernel

1. Freeze one canonical `corpus.json` consumed by both oracle arms and later production tests.
2. Attempt the Unity-light untouched-donor console first. Record the donor revision, dependency audit, and
   extraction result before implementing production physics.
3. If the shim console cannot be built cleanly, use a donor-controlled Unity batch/test runner that emits the
   identical offline fixture schema. This is slower and less portable but remains an untouched-donor arm.
4. If neither runner is possible, accept pre-existing donor artifacts only when their donor revision, config,
   corpus, and content hashes are independently verifiable. Otherwise gain0 is blocked.
5. Record H0/H1/H2 reconstruction, trace, coast, momentum, and episode artifacts from the surviving untouched
   donor path.
6. Build `tools/gain1_oracle.py` with no torch, production, or donor imports.
7. Commit GL256 nodes/weights, the required `1/t^2` Jacobian, GL512/SciPy convergence evidence, and literal
   expected outputs. GL32 is retained only as a negative regression: implementation preflight measured
   `8.3e-5` relative error on the required 10:1 prolate case, while GL256 reaches the `<1e-8` gate.
8. Include isotropic, prolate/oblate, tilted anisotropic, reactive-only, surface-fin, multi-segment H1, and
   skewed H2 cases.
9. Run fixture schema/provenance tests before production physics exists.

**Exit:** all expected numbers are committed and independently reviewable; no test generates its own expected
values. Analytic gain1-only evidence may support continued numerical prototyping if donor extraction is blocked,
but it cannot complete S0 or authorize S1 because it does not replace untouched behavioral traces and 960-step
aggregates. The S0 status remains **BLOCKED** until an equivalent independent behavioral arm exists; an owner may
approve continued S0 prototyping, not waive Gate B or relabel the result GO/CONDITIONAL GO.

### Phase 3 — numerics and body mechanics bottom-up

Implement and gate, in order:

1. quaternion primitives and Unity-order reference cases;
2. Lamb analytic production precompute;
3. sentinel-safe six-pass pose;
4. body mass and full `M_eff` assembly;
5. untouched-donor solver for gain0 comparison;
6. production constrained solver and adversarial regularization tests.

**Exit:** fixture comparisons for pose, mass, Lamb, matrix, and solve are green before force integration begins.

### Phase 4 — forces and full step

Implement reactive, fin, and drag channels as separate pure functions. Assemble `swim_step` only after each
channel passes its fixture and force/power tests. Then add COM integration and the complete `StepLedger`.

**Exit:** Gates B1, B2, B4, B6, B7, B8 and C1/C2 green on small H1/H2 batches.

### Phase 5 — episode and long-horizon mechanics

Implement 360 warmup steps plus 600 measured steps, then gain0 aggregate comparisons. Implement the separate
100,000-step energy configuration and prefix-budget telemetry.

**Exit:** B3/B5 and all of Gate C green in f32 production and f64 validation configurations.

### Phase 6 — lifecycle and reproducibility diagnostics

Add fixed churn on/off runs, graph-address stability, discrete claim-mask reproducibility, and the optional
same-device float smoke comparison.

**Exit:** Gates A5 and D green; all diagnostic values emitted without becoming hidden authorization gates.

### Phase 7 — staged benchmark funnel

Correctness must be green before timing.

| Stage | Purpose | Timed cells | Configuration |
|---|---|---:|---|
| 1 | Rung probe | 5 | authorizing B, H1, f32: CUDA r0/r1/r2; CPU r0/r1 |
| 2 | CPU/GPU crossover | 16 | B ladder × CPU/CUDA, H1, chosen rung |
| 3 | Authorizing cells | about 8 | B in `{1024,2048,4096}` × H1/H2 on best device plus H0 baselines |
| 4 | Padding occupancy | 2 | canonical layout at H1 mean occupancy vs all 16 real slots occupied |
| 5 | Lifecycle cost | 2 | churn off/on at authorizing B, H1 |
| 6 | Determinism diagnostic | 2 | production vs deterministic eager diagnostic at authorizing B |
| 7 | Profiler | about 4 | winner plus every falsifier-tripped cell |

Frozen protocol:

```text
repetitions             = 5
warmup_steps            = 360
timed_steps             = 600
cell_process            = fresh subprocess
CUDA timing             = synchronize around each 600-step window, never each step
compile warmup           = discard until two consecutive timings differ <2%, max 10 attempts
compile warmup failure  = record and fall back to eager for that device/B
cell timeout            = 180 s, recorded as timeout
VRAM cap                = 11 GiB; record peak allocation
statistics              = median + IQR + minimum of five
authorization statistic = minimum of five
```

`B*` is the smallest batch where CUDA and CPU IQRs do not overlap and CUDA's median is faster. Otherwise report
the crossover as unresolved within noise.

### Phase 8 — evidence-backed decision

Generate, do not hand-compose, the metric tables in the S0 report from committed telemetry. The prose decision
may interpret the evidence but may not replace or omit it.

Decision classes:

- **GO:** Gates A–E green; both H1 and H2 clear `1.2e6` at 10,000 live creatures; no authorizing
  regularization; no load-bearing correctness falsifier trips.
- **CONDITIONAL GO:** both 5,000-creature H1/H2 cells clear real time, but either 10,000-creature cell does
  not. Work proceeds under a 5,000-creature cap while a measured optimization addresses 10,000.
- **NO-GO / REVISE:** a correctness gate fails, H1/H2 cannot produce a non-OOM authorizing result, no rung/device
  clears the floor, or the profiler identifies a structural bottleneck requiring a different kernel/layout.

S1 may begin only after the report records an accepted GO or an owner-approved conditional scope. No threshold
is retuned in the report-writing phase.

---

## 7. Task DAG

| ID | Task | Depends on | Acceptance |
|---|---|---|---|
| T00 | Documentation reconciliation | — | One current S0 authority; contradictions removed |
| T01 | Package/import scaffold | T00 | Import probes and CPU/CUDA allocation green |
| T02 | Config, clock, manifest, artifact schemas | T01 | Immutable config; lossless artifact serialization |
| T03 | Fixed-slot layout and lifecycle primitives | T01 | A2–A5 green |
| T04 | Untouched-donor trace-order audit | T00 | Step ordering recorded and cited |
| T05 | Canonical H0/H1/H2 corpus | T04 | Exact §1.5 histograms/coverage; literal rows and hashes frozen before timing |
| T06 | gain0 donor extraction + fixture generator | T05 | Untouched donor path recorded; frozen trace/aggregate files committed, or S0 marked blocked |
| T07 | gain1 independent oracle | T05 | Literal outputs + convergence + import audit green |
| T08 | Bounded int64 transfer scaffold | T01 | A6 including overflow/negative preconditions |
| T09 | Quaternion numerics | T05 | Independent literal reference cases within tolerance |
| T10 | Lamb precompute | T07,T09 | B1 green |
| T11 | Pose scan | T03,T06,T09 | Sentinel/root/mixed-batch pose fixtures green |
| T12 | Body mass and `M_eff` | T10,T11 | Unit-chain/apply-250-once assertions plus gain0/gain1 matrix fixtures green |
| T13 | Donor and constrained solvers | T07,T12 | Reference/adversarial solver corpus green |
| T14 | Reactive channel | T06,T07,T11 | Force fixture + C1 reactive green |
| T15 | Fin channel | T06,T07,T11 | Force fixture + C1 fin green |
| T16 | Drag channel | T06,T11 | Force/work/sign fixtures green |
| T17 | Full frozen-heading step | T12–T16 | B2/B4/B6/B8 and C2 green |
| T18 | Episode and aggregate gate | T17 | B3/B5 green |
| T19 | 100,000-step energy gate | T17 | C3 curves and thresholds green |
| T20 | Churn/graph/repro diagnostics | T03,T17 | A5 and D green |
| T21 | Staged benchmark | T18–T20 | Gate E evidence complete |
| T22 | S0 go/no-go report | T21 | Every gate/falsifier classified from telemetry |

Critical path:

```text
T00 -> T04 -> T05 -> (T06,T07)
T00 -> T01 -> T03
(T03,T06,T07) -> T09 -> T10/T11 -> T12 -> T13-T16 -> T17 -> T18/T19/T20 -> T21 -> T22
```

Do not begin T21 while any correctness gate is red. Do not begin S1 while T22 lacks an accepted decision.

---

## 8. Falsifier register and required responses

| ID | Falsifier | Detector | Required response |
|---|---|---|---|
| F1 | H1/H2 heterogeneity destroys throughput | H1/H0, H2/H0 tax | Profile gather/pose/force contributors; do not authorize from H0 |
| F2 | Pose or segment-axis work dominates | profiler share >50%; optimization trigger >30% | Consider fused/Warp kernel first; arena only in its own measured slice |
| F3 | Launch overhead dominates at realistic B | rung and B curves | Prefer CPU if it wins; otherwise test fused kernel or conditional multi-world batching |
| F4 | Fixed padding exceeds VRAM or useful work | OOM + occupancy telemetry | NO-GO for current capacity or open a separate representation slice |
| F5 | Donor behavior cannot be ported within tolerance | gain0 trace/aggregate failures | Stop; isolate physics/order mismatch; never loosen oracle threshold first |
| F6 | Analytic gain1 and production disagree | gain1 corpus | Stop; audit units, axes, mass, quadrature, and fin convention |
| F7 | f32 mechanical residual exceeds prefix budget | C2/C3 | Audit evaluation order/precision; f64 hot loop is not an automatic fallback |
| F8 | Regularization activates on authorization bodies | intervention counter | Stop and audit `M_eff`, units, condition bounds, or body validity |
| F9 | Lifecycle invalidates graph or is expensive | address and churn timings | Fix static path or record conditional/no-go; no hidden recapture |
| F10 | Deterministic-mode tax is large | diagnostic ratio | Inform later experimental protocol; do not sacrifice physical gates for float identity |
| F11 | All H1 or H2 authorization cells OOM | benchmark manifest | Gate E failure |
| F12 | Kernel passes but projected whole-tick allocation is wrong | later measured `phi_loco` at G-E2E | G-E2E supersedes the S0 assumption; revise there, not retroactively |
| F13 | Untouched donor cannot be extracted or its old fixtures lack verifiable provenance | T06 extraction/provenance report | Try Unity-light console, then untouched Unity batch runner, then verified archived artifacts; analytic gain1 alone cannot authorize unconditional GO |

No risk is retired because a plan contains a mitigation. It becomes resolved only when its named detector is
green in committed evidence.

---

## 9. S0 definition of done

S0 is done only when all are true:

1. The repository has one current S0 authority and no live layout/determinism contradiction.
2. All expected oracle values existed before the production checks that consume them.
3. `corpus.json` satisfies the exact H0/H1/H2 composition in §1.5 and its hashes match every fixture/benchmark
   manifest.
4. Gate A is green, including checked int64 bounds and static lifecycle behavior.
5. Gate B is green on H1 and H2; gain0 donor and gain1 analytic arms remain distinct.
6. Gate C is green with executable mixed tolerances and prefix-budget curves.
7. Gate D hard requirements are green; Tier-3 float identity remains informational.
8. Gate E has non-OOM H1 and H2 evidence at 5,000 and 10,000 live creatures; the 10,000-cell minimum clears
   the revised 1.2e6 real-time floor, while 5×/10× acceleration is reported separately.
9. Regularization incidence is zero on every authorization run.
10. The report leads with H1/H2, identifies the winning device/rung, reports `B*` honestly, and includes profiler
   attribution.
11. Every falsifier is marked clear, tripped, or unresolved with its consequence.
12. The report records GO, CONDITIONAL GO, or NO-GO without moving thresholds after measurement.

The durable S0 output is not merely a kernel. It is a measured decision about whether this particular
representation and faithful locomotion mechanism are a sound foundation for the next slice.
