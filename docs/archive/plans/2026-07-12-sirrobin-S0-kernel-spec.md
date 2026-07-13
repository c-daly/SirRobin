# SirRobin — S0 / SpikeSwim Kernel Spec (standalone, build-ready)

**Status:** Historical; superseded for execution by
`2026-07-12-sirrobin-S0-consolidated-implementation-plan.md`. Retained as design history. · **Date:** 2026-07-12.
**Scope:** ONLY the SpikeSwim go/no-go kernel. All
S1/S3 ecology/energy-economy material is **explicitly deferred** (§9) and does **not** gate S0.

S0 is a **standalone batched-torch port of the donor's one-shot, frozen-heading `SwimEval.Sim.Step`**
(`SwimEval.cs:740`) over `B = W·N_cap` ragged bodies. **No ecology, no genome mutation, no steering, no
metabolic energy ledger.** Frozen-heading path only: `_fThrust`/`_nThrust` constant unit vectors,
`_vCom.y ≡ 0`, no yaw integration. It exists to answer one question: *does faithful per-body swim physics,
vectorized on GPU, hit the conservation, oracle, and throughput gates?*

---

## 1. The four S0 gates (the whole acceptance set)

| Gate | Name | Assertion |
|---|---|---|
| **(a)** | force-law algebraic identities | per step/body, the two donor closures hold to `rel < 1e-6` (§4) |
| **(b)** | discrete KE balance `R_step` | per step/body `|R_step|/max(KE_n,ε) < 1e-6` (f64) / `< 1e-3` (f32); the 1e5-step drift curve is **bounded-oscillating** (§5) |
| **(c)** | independent gain1 oracle | kernel reproduces each committed fixture value to `rel < 1e-4` (f32) (§6) |
| **(d)** | throughput / affordability | measured on ≥1 **non-OOM** authorization-sized cell (all-OOM ⇒ FAIL) (§7) |

Plus the standing **conservation gate** on the scaffold's *fake* reservoir ledger — exact int64, mass
currency only (§3). No energy metabolic ledger is exercised at S0.

---

## 2. Body layout & indexing (from Rev-3 §2–§3, unchanged — RESOLVED in round 3)

Retained verbatim from Rev-3 (Codex round-3 marked these RESOLVED): fixed `[B, S_slot]` padded storage,
`S_slot = 17` = slot-0 **sentinel** + 16 real segments in `[1, S_max]`, root `parent = 0`. (Rev-3's
empty-body `det_safe` masking is **superseded here** by the §4 branch dispatch: an empty/zero-mass body is
`INVALID` and the outer select forces `Δv = 0`.) The lifecycle eager gather uses `free_rank.clamp_min(0)`
before masking. This spec **depends on** that layout; it does not change it.

`body_valid := alive & (masked_segment_sum(seg_mass, seg_mask) > 0)`  → `[B]` bool.

---

## 3. Conservation gate — exact int64 transfer (S0 subset of Rev-3 §1)

S0 exercises **only** the int64 transfer machinery on a **fake mass reservoir** (scaffold gate); the
metabolic *energy* ledger is S1/S3 (§9). Retained from Rev-3 §1, with the round-4 source-cap added:

```python
def transfer_quanta(src_q, dst_q, n, mask):          # all int64; n >= 0 quanta
    req   = torch.where(mask, n, 0)
    n_eff = torch.minimum(req, src_q)                # SOURCE CAP: never overdraw → no negative reservoir
    return src_q - n_eff, dst_q + n_eff, (req - n_eff)   # 3rd return = shortfall for the caller's failure path
```

`close_books()` is an exact integer `==` (order-independent; no float residual, no `R_num`). Gate:
`test_transfer_exact_int`, `test_close_books_order_independent`, `test_transfer_no_negative`
(`(src_q >= 0).all()`). Currency/quanta: mass `q_mass = 1e-9` mol (int64). **No energy currency at S0.**

---

## 4. The solver — one production solve, three branches (fixes round-4 #1)

`M` is the symmetric **2×2** constrained added-mass matrix in the **x/z plane** `[[M00,M02],[M02,M22]]`;
`P = F_stream·Dt` is `[B,2]`; `F_stream = (tReact+tFin)·f̂ + fDrag` (`SwimEval.cs:804`). The vertical (y)
reaction is a **separate** statement (§4.3) — it is **not** a term in this 2-D equation.

### 4.1 One canonical degeneracy predicate — `κ > KAPPA_MAX`, expressed division-free

```python
tr   = M00 + M22
det  = M00*M22 - M02*M02
disc = torch.sqrt((M00 - M22)**2 + 4*M02**2)         # cancellation-free (Rev-3 §7.2)
lam_max = 0.5*(tr + disc)                             # >= 0; == 0 only for M == 0 (an INVALID body)
# κ = lam_max/lam_min = lam_max²/det, so "κ > KAPPA_MAX" ⟺ "det < lam_max²/KAPPA_MAX" — NO division,
# NO lam_min, so no 0/0 and no DET_FLOOR/0 anywhere. Plus a scale guard for a numerically-zero matrix:
degenerate = (lam_max < LAM_FLOOR) | (det < lam_max*lam_max / KAPPA_MAX)
```

`degenerate` **is** the retained `κ > KAPPA_MAX` criterion (`κ = lam_max²/det` for a symmetric 2×2), written
without any division so it is finite on every body — including `M = 0`, where `lam_max = 0 < LAM_FLOOR` marks
it degenerate (and it is also `INVALID`, so discarded). No `det_safe`, no `lam_min`, no `κ` denominator is
formed. **The earlier `det_safe` / `lam_min = det_safe/lam_max` formulation is DELETED** — it gave
`lam_min = DET_FLOOR/0 = ∞` at `M = 0`, and `det_safe` altered an `EXACT` solve for a small-but-well-
conditioned `M = αI` (κ scale-invariant, `DET_FLOOR` absolute). Testing on `det` / `lam_max` directly removes
both.

### 4.2 Three branches, eager denominators masked to `1.0` off-branch, single select

`torch.where` does **not** short-circuit, so **both** branch solves are computed for **all** bodies before
selection. Finiteness is guaranteed by masking each solve's **denominator to `1.0` off its own branch** — a
body only ever divides by its *own* branch's real denominator; every discarded eager value divides by `1.0`:

```python
INVALID     = ~body_valid                              # empty / zero-mass
REGULARIZED = body_valid & degenerate                  # valid but ill-conditioned (κ > KAPPA_MAX)
EXACT       = body_valid & ~degenerate                 # common path

reg     = EPS_SPD * lam_max                            # > 0 on REGULARIZED (lam_max ≥ LAM_FLOOR there)
det_reg = (M00+reg)*(M22+reg) - M02*M02                # = (lam_min+reg)(lam_max+reg) ≥ reg·lam_max (exact arith)

def solve_2x2(a, b, c, Px, Pz, denom):                 # M = [[a,b],[b,c]]; returns [B,2]
    dvx = ( c*Px - b*Pz) / denom
    dvz = (-b*Px + a*Pz) / denom
    return torch.stack([dvx, dvz], dim=-1)

one         = torch.ones_like(det)
denom_exact = torch.where(EXACT,       det,     one)   # EXACT bodies divide by their TRUE det (never floored)
denom_reg   = torch.where(REGULARIZED, det_reg, one)   # REGULARIZED bodies divide by det_reg; all others by 1.0

Δv_exact = solve_2x2(M00,     M02, M22,     Px, Pz, denom_exact)
Δv_reg   = solve_2x2(M00+reg, M02, M22+reg, Px, Pz, denom_reg)

Δv    = torch.where(INVALID[...,None], 0.0,
          torch.where(REGULARIZED[...,None], Δv_reg, Δv_exact))                 # [B,2]
J_reg = torch.where(REGULARIZED[...,None], -reg[...,None] * Δv_reg, 0.0)        # [B,2]  ← the − sign
```

- **Finite on every body.** Off-branch, `denom_exact`/`denom_reg` are `1.0`, so `Δv_exact`/`Δv_reg` are finite
  even at `M = 0`; the outer select discards them. No `1/0`, no `0·∞`, no eager NaN — the whole class of
  round-3/4 eager-evaluation bugs is closed structurally.
- **`EXACT` closes exactly.** An `EXACT` body divides by its *true* `det` (never replaced by a floor), and
  `~degenerate ⇒ det ≥ lam_max²/KAPPA_MAX > 0` with `lam_max ≥ LAM_FLOOR`, so `M Δv = P` holds — the
  scale-invariance defect (`M = αI` selecting `EXACT` yet solving with `DET_FLOOR`) is gone.
- **`REGULARIZED` is finite and f32-safe.** On this branch `lam_max ≥ LAM_FLOOR ⇒ reg = EPS_SPD·lam_max > 0`,
  and `det_reg ≥ reg·lam_max = EPS_SPD·lam_max² ≥ EPS_SPD·LAM_FLOOR²` in exact arithmetic; with `EPS_SPD ≫
  eps_f32` (§4.2a) the f32 cancellation in `det_reg` cannot reach zero. Regularized bodies are damped
  approximations whose energy is accounted by `J_reg` in `R_step` (§5.2); the f64 arm re-checks them.
- **`INVALID` bodies output exactly `0`** (outer select) — the frozen-heading empty-body contract.

### 4.2a Pinned constants (frozen; part of `config_hash`)

```
KAPPA_MAX = 1e6        # condition-number ceiling; above it → REGULARIZED
LAM_FLOOR = 1e-9 kg    # eigenvalue floor = "numerically-zero added-mass matrix" (real bodies are O(0.1–10) kg)
EPS_SPD   = 1e-6       # relative reg strength (reg = EPS_SPD·lam_max); ≫ eps_f32 ≈ 6e-8 ⇒ f32-safe det_reg
# Input bounds, validated at body construction (malformed bodies rejected, NOT silently regularized):
#   M00, M22 ∈ [LAM_FLOOR, M_MAX],  M_MAX = 1e4 kg ;   |M02| ≤ sqrt(M00·M22)  (SPD)
```

### 4.3 The regularization sign and the 2-D vs vertical split

`(M + reg·I)Δv = P` ⇒ `M Δv = P − reg·Δv`, so the **booked** regularization impulse is `J_reg = −reg·Δv`
(NOT `+reg·Δv`). Momentum closes **exactly in the 2-D x/z plane**:

    M Δv = P + J_reg          (EXACT: J_reg = 0;  REGULARIZED: J_reg = −reg·Δv)     # 2×2, x/z only

The **vertical reaction** `J_c` (`SwimEval.cs:807`) is a **separate 3-D statement**: it enforces
`v_y ≡ 0` and is **workless** (`J_c · v_mid = 0` because `v_y ≡ 0`). `J_c` does **not** appear in the 2-D
`M Δv = P + J_reg` equation above, nor in the 2-D work term of §5. It is stated once, here, as the
constraint that the frozen-heading path keeps `v_y = 0`.

### 4.4 Solver gates

- `test_reg_momentum_closes`: force `κ > KAPPA_MAX` on a 10:1 slender body; assert the **2-D**
  `M·Δv == P + J_reg` to f32 rel `< 1e-5`, and that flipping to `+reg·Δv` **fails** it (sign is load-bearing).
- `test_solve_dispatch_selects_reg`: a valid-degenerate body's output equals `Δv_reg`, not `Δv_exact`.
- `test_solve_finite_all_branches`: for empty/zero-mass (`M=0`) and near-singular bodies, `Δv_exact`,
  `Δv_reg`, and `Δv` are all finite (the off-branch denominator masking to `1.0` guarantees it) **before**
  any select — assert `torch.isfinite(Δv_exact).all() and torch.isfinite(Δv_reg).all()`.
- `test_reg_off_on_wellcond`: isotropic/H0 body, `κ ≤ KAPPA_MAX` ⇒ production == exact solve, `J_reg == 0`.
- `test_vertical_workless`: `v_y == 0` maintained; the vertical `J_c` does no work (`J_c·v_mid == 0`).

---

## 5. Energy — the f32 mechanical arm only (fixes round-4 gate-(b) meaning + normalization)

**S0 has no int64 energy ledger.** Energy at S0 is entirely the **f32 mechanical physics**, in two gates
with distinct, non-overlapping meanings:

### 5.1 Gate (a) — force-law algebraic identities (the two donor closures)

```
U_cl = max(0, U)                                                          # donor clamp (SwimEval.cs:395)
(a.reactive):  InputPower           = m_t·U·V_t·W_t ≡ tReact·U   + pWake         # signed U
(a.fin):       CirculatoryInputPower = F_n·V_t       ≡ tFin·U_cl + pFin          # clamped U_cl
```

Each is a **per-step algebraic identity**, tested to `rel < 1e-6` with the safe normalization
`den = max(|p_in|, ε)` (**|·|** — signed reactive power can make `p_in < 0`, so `max(p_in,ε)` was unsafe).
The one authoritative combined input power (used by the diagnostic and, later, S3) is
`p_in = (tReact·U + pWake) + (tFin·U_cl + pFin)` — `tFin·U_cl`, one clamp policy, everywhere.

### 5.2 Gate (b) — the discrete KE balance `R_step` (the ONLY KE equation)

Gate (b) **is** `R_step`, the exact discrete identity for the semi-implicit integrator with pose-varying
`M_eff` (the naive `ΔKE = (p_in − wake − drag)·dt` is deleted everywhere):

```
constrained update (2-D):  M_{n+1}(v_{n+1} − v_n) = P + J_reg,   P = F_stream·Dt
ΔKE   = ½ v_{n+1}ᵀ M_{n+1} v_{n+1} − ½ v_nᵀ M_n v_n
R_step = ΔKE − v_mid·(F_stream·Dt + J_reg) − ½ v_nᵀ ΔM v_n          # v_mid = ½(v_n+v_{n+1});  ≈ 0
```

The added-mass term `½ v_nᵀ ΔM v_n` is **mandatory**. `J_reg = −reg·Δv` (§4.3); on the EXACT branch
`J_reg = 0` ⇒ `R_step ≈ 0` by construction; on the REGULARIZED branch the `v_mid·J_reg` term makes it close
**with** the artificial damping present — the regularization energy lives **here, in the f32 `R_step` term**,
with **no int64 booking** (kinetic/mechanical energy has a single representation — the f32 body velocity —
so there is no int64 mirror; that would be the Law-4 anti-pattern). Threshold and bounded-oscillating drift
per gate (b) in §1.

### 5.3 Kinetic energy is f32 (production); f64 is the validation arm

`E_KE = ½ vᵀ M v` is derived from f32 velocity — its single canonical representation. It is **not** an int64
reservoir and there is no metabolic ledger at S0 to reconcile it against. **Precision (resolving the §1 vs
§5 wording):** **f32 is the production hot-loop representation** and the `< 1e-3` f32 threshold is the
production gate; the tight `< 1e-6` arm of gate (b) is a **separate f64 validation config** running the same
kernel — **never** the throughput-timed path. "Entirely f32" refers to the *production* representation; f64
is a diagnostic reference arm, not a second stored copy. (The honest treatment of energy as a
*metabolic-expenditure* int64 ledger **plus** an f32 total-energy invariant is an **S1/S3** matter, §9 — it
does not exist in the standalone locomotion kernel.)

---

## 6. Gain1 independent oracle (fixes round-4 #5 + the Jacobian)

The kernel's added-mass/`gain1` path is validated against a **standalone, donor-free, torch-free** second
implementation `tools/gain1_oracle.py`, run once, with values committed to the fixture file.

### 6.1 Frozen fixture inputs (literal — H1 canonical body)

The fixture's **inputs** are pure geometry + kinematics — all literal numbers. `m_t`, the Lamb added-mass
coefficients, and the forces are **derived** by *both* the kernel and the oracle, so they are **outputs**,
not inputs (this removes the earlier `m_t = …(…)` ellipsis — `m_t` is never an input):

```
H1 inputs (committed literals):
     a = 0.50 m,  b = 0.05 m,  c = 0.05 m           # 10:1 prolate spheroid
     U = 0.30 m/s,  V_t = 0.10 m/s,  s = 0.20 m     # surge; transverse tail speed; tail-tip lateral offset
     pose = identity quaternion;  f̂ = +x̂,  n̂ = +ẑ   # frozen heading
     ρ_water = 1000 kg/m³,  Dt = 1e-3 s
H1 expected outputs (numbers the kernel must reproduce — committed by tools/gain1_oracle.py):
     α0, β0, γ0  (Lamb added-mass coefficients),  m_t,  the single-step reactive/fin forces
```

The **inputs are frozen literals here and now.** The **expected outputs are produced by running
`tools/gain1_oracle.py` once and committing its numbers** — they cannot be hand-evaluated (a 32-point
Gauss–Legendre Lamb integral), which is exactly why the generator is the **first build artifact** (§9), not
a paper deliverable.

### 6.2 Quadrature — 32-point Gauss–Legendre WITH the change-of-variable Jacobian

The Lamb added-mass integrals `α0, β0, γ0` are `∫₀^∞ g(λ) dλ`. Map to `[0,1]` by `λ = (1−t)/t`, whose
Jacobian is `dλ = −dt / t²`; folding the sign into the limits:

    ∫₀^∞ g(λ) dλ  =  ∫₀¹ g((1−t)/t) · (1/t²) dt        # the 1/t² Jacobian is REQUIRED (was omitted)

evaluated by **32-point Gauss–Legendre** on `[0,1]` (nodes/weights from `numpy.polynomial.legendre.leggauss(32)`
mapped to `[0,1]`, committed and pinned; `test_gain1_quadrature_pinned` matches them bit-for-bit). Endpoint
`t → 0` (`λ → ∞`): the integrand `g` decays faster than `t²`, so `g/t²` is finite; the fixture asserts it.

### 6.3 Tolerances & independence

Fixture-vs-analytic `rel < 1e-6` (f64); kernel-vs-fixture `rel < 1e-4` (f32). **Independence gate**
(`test_gain1_generator_independent`, AST/import audit): `tools/gain1_oracle.py` importing **`torch`, the
production `core`/`physics` packages, OR the donor `SwimEval`** is a **test failure** — not merely
"donor-free." It is the independent second implementation that makes the oracle non-circular.
**Convergence** (`test_gain1_quadrature_converged`): the committed 32-point value must agree with a
higher-order independent evaluation (64-point Gauss–Legendre, or `scipy.integrate.quad`) to `rel < 1e-8`, so
32-point accuracy is *demonstrated*, not asserted. **This is a T11 deliverable and the first build artifact
(§9):** the generator is committed, run once, its outputs (and the 32 GL nodes/weights) frozen into the
fixture file; the kernel test reads the frozen values.

---

## 7. Throughput / affordability gate (d) (fixes round-4 #7 non-vacuous requirement)

The S0 throughput sweep measures the **locomotion kernel** (§4–§5) over the staged cell funnel (Rev-3 §9,
hardware pinned: RTX 5070, 11 GiB cap, Ryzen 7 8700F). **An all-OOM sweep is a FAIL, not a vacuous pass:**
`assert n_valid_cells ≥ 1` before emitting the authorization number `φ_loco`; the affordability floor
`F_loco_S0 = 9.0e7` (Rev-3 §9.2) requires a real non-OOM measurement. Compile-warmup failure falls back to
eager (Rev-3 §9.4). *(The S2 feeding-stub op/tensor/byte model is an S2 concern, not measured here — §9.)*

---

## 8. S0 acceptance checklist (these are S0-DONE gates, not pre-code preconditions)

Green to **complete** S0 (not "before S0 code begins"): `test_transfer_exact_int`,
`test_close_books_order_independent`, `test_transfer_no_negative`; the §4.4 solver gates; gate (a) the two
force-law closures; gate (b) `R_step` 1e5-step bounded-oscillating; gate (c) `test_oracle_gain1_analytic`
+ `test_gain1_quadrature_pinned` + `test_gain1_fixtures_donor_free`; gate (d) non-OOM `φ_loco ≥ F_loco_S0`.

---

## 9. Explicitly DEFERRED to S1/S3 — reviewed as we reach them (NOT S0 blockers)

These round-4 findings are real and belong to later phases; they do **not** gate the standalone locomotion
kernel and are **not** specified here (owner: "review the other parts as we get to them"):

- **Energy economy (S1/S3):** the int64 energy ledger reframed honestly as a **metabolic-expenditure** ledger
  (exact int64 accounting) **plus** an **f32 total-energy** invariant including KE — *not* "exact int64 total-
  energy conservation." Immediate-heat partition `P_basal + (1/η−1)·p_in` (mechanical `p_in` is work, not heat).
- **Feeding/reserve (S3):** the field-biomass(`Bp`)→consumer-reserve energy path (grazing a mass-only producer
  currently has no reserve to debit); reserve inflow partition; predation reserve transfer.
- **Reservoir completeness (S1):** restore **`Bm`** (remineralized biomass, `v1:408,414,443-445`) to the mass
  inventory; `E_sun_cum` is a cumulative **meter** (credit reserve *and* increment the meter — not "transfer
  out of" it); death `reserve → E_sun-return` semantics.
- **Cross-ledger transactions (S1/S3):** atomic growth (mass and energy legs capped together, not
  independently); grouped `[W,N_cap] reserve → [W] meter` reductions; aggregate int64 **sum** overflow bound
  (per-value `< 2^62` does not bound the sum).
- **RNG for mutation (S3):** Philox rejection-continuation packing, Box–Muller (fix the
  `(((w0>>8)&0xFFFFFF)+1)` precedence bug and the `≤ 2^-256` bound when the RNG scaffold lands); Tier-2 =
  reproducible integer draws + discrete decisions.
- **Feeding-stub throughput (S2):** the `[B,K]` gather/mask/segment-sum op/tensor/byte model (include
  `neigh_idx`, `dist2` dtype, pinned `K`).
