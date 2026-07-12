# SirRobin — Implementation Plan Rev. 2: Corrections resolving the Codex review

**Date:** 2026-07-12 · **Base:** the v1 plan (`2026-07-11-sirrobin-implementation-plan.md`).
This document is the **Rev-2 delta** — the concrete resolutions to the 20 findings in the Codex
adversarial review (`docs/superpowers/reviews/2026-07-11-codex-adversarial-review-of-implementation-plan.md`).
**Base plan + these corrections = Rev 2.** Each fix-set is organized by Codex finding number.

## Pre-decided cross-cutting resolutions (adopted throughout)

- **Segment layout → fixed `[B, S_max]` padded + boolean segment-mask.** Static shapes ⇒ CUDA-graph-safe,
  and per-body reductions become **masked axis-sums** (`(vals*mask).sum(dim=segment_axis)`) — no atomics,
  no `scatter_add`/`index_add` over duplicate `body_id`. This simultaneously kills the ragged-reduction
  determinism problem (Codex #3) and the static-shape contradiction (#4), at the cost of ~2.7× masking
  waste (revisitable later). Replaces the flattened/CSR `[S_total]` layout; re-aligns with the design's
  fixed `[P, S_max]` development representation.
- **Reservoir state = float64** + an explicit **numerical-residual reservoir** (fixes #5).
- **RNG = counter-based keyed** `(seed, step, stable_entity_id, gene_iid, event_kind, draw_index)` +
  stable entity IDs (fixes #13; the append-only manifest is removed).
- **S0 energy gate = split** into per-force algebraic identities + a **discrete energy balance** with the
  `½vᵀ(M_{n+1}−M_n)v`, actuator, constraint-impulse, drag, and wake terms; `p_in` includes fin work
  (fixes #1). **Horizontal solve = constrained KKT** with an explicit vertical reaction, ledgered, donor
  deliberately re-baselined (fixes #2).
- **Oracle:** untouched-donor gain0 fixtures; **independent analytic gain1 fixtures** (fixes #8).
- **Throughput target** derived from a required **one-world run-duration**; S0 authorizes only
  **locomotion-kernel feasibility**, not the whole vectorization thesis; a whole-tick benchmark is a
  later gate (fixes #6, #20-scope).

## Fix-sets (by cluster)

The six clusters below resolve, respectively: S0 kernel correctness (#1, #2, #9, #10, #11, #12);
determinism reduction + RNG (#3, #13, #12-compile); representation layout + checkpoint (#4, #14);
conservation ledgers + energy system (#5, #7, #15); throughput/benchmark/risk (#6, #19, #20); and
roadmap sequencing + S4 + import firewall (#16, #17, #18).


---

I have full grounding from SwimEval.cs (donor `Step` at 740–817, `StepLive` at 944–1059, `Quaternion.Inverse` at 778/1000, `SolveSym3` at 1151–1166), the Codex review, plan §2, and the design doc §4.3. Here is the corrected spec.

---

## FIXES: S0 kernel correctness & physics port (Codex #1, #2, #9, #10, #11, #12)

All fixes are written against the **pre-decided canonical layout `[B, S_max]` padded + boolean `seg_mask`** (replacing the flattened `[S_total]`/CSR layout throughout plan §2.1 line 291, §2.3 item, design:471). Per-body reductions become masked axis-sums `(vals*mask).sum(dim=seg_axis)` — no `index_add_`, no `-1` global-parent arithmetic. This is assumed below and is load-bearing for #9.

---

## #1 — The S0 energy gate (replaces plan:314, plan:345–346; design:266)

**Root cause.** The plan's instantaneous test (plan:345) asserts `|p_in − (tReact·U + pWake + pFin)| < 1e-6` — it drops the `tFin·U` term, so any active fin (H1's ~40% fin tails) fails or forces `p_in` to be defined inconsistently. The `ke_budget_1e5` test (plan:346) compares `ΔKE` against `Σ(P_musc−P_diss)·Dt` but `M_eff` is pose-dependent, so `ΔKE = ½vᵀM(t)v` carries a `½vᵀΔM v` term the comparison omits, plus discrete semi-implicit work, the constraint impulse (the `_vCom.y=0` at SwimEval.cs:807), and the tail-tip-`U`-vs-COM-velocity gap. The donor only proves the *algebraic* trailing-edge flux identity (SwimEval.cs:336, `InputPower`), never whole-body KE conservation.

**Resolution — split into (a) per-force algebraic identity tests + (b) a derived discrete energy-balance gate.**

### (a) Per-force ALGEBRAIC identity tests — unit tests, NOT authorization gates

These verify the force-law port reproduces the donor algebra; they hold by construction (SwimEval.cs `InputPower`/`CirculatoryInputPower`).

- **Reactive closure** (`test_energy.py::reactive_algebraic_closure`). With signed `U` and unclamped wake `Wpow_signed = 0.5·mt·U·Wt²`:
  - `ReactiveThrust(mt,Vt,U,s) ≡ ReactiveThrustFlux(mt,Vt,U,s)` (the `0.5·mt·(Vt²−U²s²)` reduction of `mt·(Vt·Wt−0.5·Wt²)`), and
  - `InputPower = mt·U·Vt·Wt ≡ tReact·U + Wpow_signed`.
  Threshold: f32 rel `< 1e-6` on the H0/H1/H2 corpus, all steps.
- **Fin closure** (`::fin_algebraic_closure`). With `Ucl = max(0,U)` (the donor's clamp, SwimEval.cs:395): `CirculatoryInputPower = Fn·Vt ≡ tFin·Ucl + pFin`. Verified derivation: `tFin·Ucl + D·Q = Vt·(L·cosβ + D·sinβ) = Vt·Fn` with `sinβ=Vt/Q, cosβ=Ucl/Q, Q=√(Ucl²+Vt²)`. Threshold f32 rel `< 1e-6`. **Test MUST use `Ucl`, not signed `U`** — the `max(0,U)` clamp is a modeling choice (no lift/wake swimming backward) and the identity only closes with it.
- **Clamp-parity assertion** (`::wake_clamp_parity`): the integrator path applies `max(0,U)` to `pWake` (SwimEval.cs:756) and `Ucl` to the fin channel identically to the donor; asserted bit-exact against the oracle trace.

The corrected *combined* instantaneous identity (documentation only, diagnostic ledger) is `p_in = tReact·U + pWake + tFin·U + pFin` — this is what plan:345 should have written, but it is **demoted from an authorization gate** to a diagnostic.

### (b) DISCRETE energy-balance gate — the authorization gate (replaces plan:346)

Derived for the *actual* semi-implicit integrator with pose-varying `M_eff` and the constrained solve of #2. Let `M_n = M_eff` at pose `n`, `P = F_stream·Dt` (impulse), `J_c` = vertical constraint impulse (#2), `v_mid = ½(v_n + v_{n+1})`, `ΔM = M_{n+1} − M_n`. The constrained update is `M_{n+1}(v_{n+1} − v_n) = P + J_c` with `v_{n+1,y}=v_{n,y}=0`.

Exact discrete identity (proof by symmetry of `M_{n+1}`, add/subtract `½v_nᵀM_{n+1}v_n`):

```
ΔKE  ≡  ½ v_{n+1}ᵀ M_{n+1} v_{n+1}  −  ½ v_nᵀ M_n v_n
     =  v_mid · (P + J_c)  +  ½ v_nᵀ ΔM v_n
     =  v_mid · F_stream · Dt  +  ½ v_nᵀ ΔM v_n            (since J_c·v_mid = J_y·v_mid,y = 0)
```

The constraint impulse is **workless** (`v_y≡0` before and after → `v_mid,y=0`) — this is *why* the ledger closes, and is the physical justification for #2's KKT form. The step residual

```
R_step = ΔKE − v_mid·F_stream·Dt − ½·v_nᵀ·(M_{n+1}−M_n)·v_n
```

must be **≈0 by construction**. `F_stream·v_mid·Dt` decomposes into actuator/thrust work `(tReact+tFin)·(f̂·v_mid)·Dt`, COM-frame quadratic-drag work `Σ F_drag,j·v_mid·Dt`, and (`=0`) constraint work. Note `f̂·v_mid ≠ U`: the thrust work on the COM is *not* `tReact·U` (tail-tip throughflow) — the two differ by exactly the wake/entrainment energy that the (a)-tests account for. Conflating them was the original gate's error.

**Gate (`test_energy.py::discrete_balance_1e5`):** compute `R_step` every step for 1e5 steps; assert `|R_step| / max(KE_n, ε_KE) < 1e-6` (f64 validation config) / `< 1e-3` (f32 hot config), **and gate on the accumulated `Σ R_step` drift curve being bounded-oscillating (slope ≤ 0 over the window), not the endpoint** (design:266, F6). A port carrying the donor's `#2` zero-`v_y` bug produces `J_c·v_mid ≠ 0` (the discarded 3×3 mid-step `v_y ≠ 0`) → `R_step ≠ 0` → **the gate trips on the bug**, which is the intended catch.

`StepLedger` (plan:314) is redefined to carry, per step, the f32 terms `{KE_n, ΔKE, v_mid·F_stream·Dt, ½v_nᵀΔM v_n, J_y}`; the world-level `Σ R_step` accumulator is f64-Kahan (see #11).

---

## #2 — Constrained horizontal solve (replaces plan:309 `vcom[...,1]=0`; design:514, SwimEval.cs:805–807)

**Root cause.** Donor solves the full 3×3 `SolveSym3(m00..m22, fStream·Dt)` then sets `_vCom.y=0` (SwimEval.cs:805–807). When rotated **anisotropic** added mass produces `m01`/`m12 ≠ 0`, the unconstrained `v_y` DOF changes the computed `dvx`/`dvz`; deleting `y` afterward destroys momentum and energy. **The off-diagonals are non-zero exactly when a segment's rest orientation tilts it out of the swim plane** (`p.orient.x`/`.z ≠ 0`): a pure yaw-about-ŷ rotation gives `m01=m12=0`, but any pitch/roll of a segment couples y into x/z through `R·diag(ma)·Rᵀ` (SwimEval.cs:794–801). So the bug is **inert for untilted axial bodies (H0) and active for H1/H2 tilted/mirror-paired bodies.**

**Resolution — solve the KKT-constrained x/z system with an explicit vertical reaction impulse.**

The constraint is `C·v_{n+1}=0`, `C=[0,1,0]` (ŷ). Because the constraint axis is a coordinate axis, KKT elimination reduces to the **2×2 SPD system in (x,z)** with `dv_y ≡ 0`:

```
[ M00  M02 ] [ dvx ]   [ Px ]
[ M02  M22 ] [ dvz ] = [ Pz ]        P = F_stream·Dt
dv_y = 0
```

Solve via a dedicated `solve_constrained_xz` (2×2 closed-form, SPD-robust — see #12), **not** the 3×3. The vertical reaction impulse (for the ledger, workless) is

```
J_y = M01·dvx + M12·dvz − Py         # the impulse the plane constraint supplies
```

recorded in `StepLedger` and asserted `J_y·v_mid,y = 0` (identically, since `v_y≡0`). Kernel step S10 (plan:309) becomes: `dv_xz = solve_constrained_xz(M00,M02,M22, P_xz); vcom[...,{0,2}] += dv_xz; vcom[...,1] = 0; J_y = M01·dvx + M12·dvz − Py; xcom += vcom·Dt`.

**Deliberate donor re-baseline (reconciled with #8's frozen-oracle rule):**
- **Bug-inert fixtures (H0 axial, `coast.npz`, `momentum.npz` all-sphere)**: `m01=m12=0` provably, so 2×2 ≡ 3×3-then-zero. These stay on the **untouched donor** and validate the port byte-for-byte (#8 requirement preserved).
- **Bug-active fixtures (H1/H2 tilted, fin-tail)**: regenerate `forces_H*.npz`/`aggregates_H*.npz` from a **narrowly-reviewed patched donor** (`SwimEval` with `SolveSym3`+`_vCom.y=0` replaced by the 2×2 constrained solve), with retained provenance. We refuse to reproduce the SwimEval.cs:805–807 bug.
- **Regime classifier test** (`test_step_forces.py::offdiag_regime`): for every fixture body, assert `max_t(|m01|,|m12|)` — if `< 1e-6·trace` the untouched-donor fixture is authoritative; else the patched-donor fixture is required. A dedicated regression (`::tilted_solve_divergence`) constructs a 45°-rolled two-segment body, shows `dvx_{3×3-then-zero} ≠ dvx_{2×2}` by a documented margin, and gates the port to the 2×2 result.

---

## #9 — Pose root / empty-body indexing (replaces plan:291, plan:302–303)

**Root cause.** Roots have `seg_parent = -1`; the depth-scan gathers every parent before selecting by depth. In PyTorch `pos[-1]` gathers the **final** row, not identity (SwimEval.cs:669 uses `pPos=0,pRot=identity` for `parentIndex<0`). Empty bodies cannot safely execute `pos[tail_gidx]` (plan:302); `torch.where` does **not** prevent the gather from evaluating.

**Resolution — reserved sentinel slot per body + remapped parents/tails, on the `[B,S_max]` layout.**

- **Reserve slot 0 of every body as an immutable identity sentinel:** `seg_mask[:,0]=False`, `pos[:,0]=0`, `rot[:,0]=(0,0,0,1)`, `mass=ma=areaZ=mt=0`, `depth[:,0]=-1`. Real segments occupy slots `[1, S_max)`.
- **`seg_parent_slot[B,S_max]` ∈ [0, S_max)** (local, never global, never −1): each root's parent is remapped to slot **0**. Gather `pos[b, seg_parent_slot[b,k]]` / `rot[...]` is then always in-bounds, and for a root returns the identity sentinel → `pPos=0, pRot=identity`, bit-matching SwimEval.cs:669.
- **Depth-scan** (plan:303): pass `d∈{0..5}` updates slots with `seg_depth==d ∧ seg_mask`. The sentinel (`depth=-1`, `mask=False`) is never written and stays identity; every real parent (strictly lower depth) is written in an earlier pass → single-valued, static 6 passes.
- **`tail_gidx` for an empty body → slot 0** (sentinel); `mt=0` ⇒ `tReact=pWake=0`; all forces vanish; `dv=0`. Safe with no `torch.where` masking of the gather itself.
- **Root `pRot` is bit-exact identity** because it is a *gather of the identity row*, not a computed near-identity — this matters for the #10 quaternion-inverse chain.

**Test matrix (`test_pose.py`, Codex-mandated):**
| case | body | assertion |
|---|---|---|
| empty | 0 live segs (only sentinel) | all outputs 0, no NaN/Inf; `tReact=0` |
| root-only | 1 seg, depth 0, `hasJoint=False` | `pos=localPos`, `rot=localRot`; root `pRot` bit-exact identity |
| max-depth | chain depth 0..5 (MaxDepth boundary) | 6-pass resolves all; depth-6 child correctly dropped (SwimEval.cs:157) |
| mixed live/dead | interleaved empty + full bodies in one `[B,S_max]` batch | live bodies rel `<1e-4` vs oracle; dead rows exactly 0; no cross-body leakage |
| mirror-paired | fin tail with tie on `restPos.z` | tail tie-break = later DFS slot (SwimEval.cs:142) |

---

## #10 — Quaternion inverse (replaces plan:307, S8 `vloc=quat_rotate(conj(rot),uj)`)

**Root cause.** Donor uses `Quaternion.Inverse(R)` at SwimEval.cs:778 (`Step`) and :1000 (`StepLive`) = `conj(q)/‖q‖²`. The plan uses `conj(rot)` alone (plan:307), which equals the inverse **only for an exactly unit quaternion**. The depth-scan composes `rot = quat_mul(pRot, localRot·angleAxis(θ,ŷ))` in f32 without renormalizing (matching SwimEval.cs:685, which does not renormalize inside `Pose`), so `‖R‖²` drifts from 1 and `conj(R) ≠ R⁻¹`.

**Resolution — exact `q⁻¹ = conj(q)/‖q‖²` with donor degenerate semantics + explicit (non-)normalization policy.**

```python
def quat_conj(q):   # q=(x,y,z,w)
    return stack([-q.x, -q.y, -q.z, q.w])
def quat_inv(q):
    dot = q.x*q.x + q.y*q.y + q.z*q.z + q.w*q.w
    # Unity Quaternion.Inverse: conj·(1/dot); dot==0 → return q unchanged (degenerate)
    return where(dot != 0, quat_conj(q) * (1.0/dot), q)
# S8 body-frame velocity:
vloc = quat_rotate(quat_inv(rot), uj)          # NOT quat_rotate(conj(rot), uj)
```

- **Degenerate branch** (`dot==0 → return q`) mirrors Unity's native guard; it is **provably unreachable in the donor** (every `R` is a product of unit Euler quaternions; `dot ≥ 1−ε`), but replicated so a pathological genome cannot diverge from the oracle.
- **Normalization policy (match donor exactly):** the depth-scan `quat_mul` chain is **NOT renormalized** inside pose (matches SwimEval.cs:685); only `StepLive`'s `_orientation` is renormalized via `Quaternion.Normalize` (SwimEval.cs:1049) — but `_orientation` is a **yaw path (S2)**, out of S0 scope. Therefore S0 uses `quat_inv` (conj/normSq) to absorb the accumulated drift exactly as the donor's `Quaternion.Inverse` does, and does **not** insert extra normalizations that would break byte-match.

**Tests:**
- `test_quat.py::inverse_vs_unity` — grid of deliberately denormalized quaternions (`‖q‖²∈{0.8,1.1,1.5}`); assert `quat_inv` matches Unity `Quaternion.Inverse` to `1e-6`; assert `conj`-only diverges by ~`(‖q‖²−1)` (proves the fix is load-bearing).
- `test_quat.py::long_chain_oracle` — max-depth-5 body, 960 steps; assert `rot`/`vloc` rel `<1e-4` vs `StepTraceForTest`. Euler order (`Quaternion.Euler(x, y·side, z·side)`, SwimEval.cs:161) matched (plan T1 acceptance).

---

## #11 — f32/f64 dtype policy (replaces plan:312–314, plan:308)

**Root cause.** `LambK` emits `seg_ma` in f64 (Simpson quadrature, SwimEval.cs:256–291); the plan calls `StepLedger`/`dv`/`m_eff` "f64-like" (plan:312–314). Mixing f64 `seg_ma` with f32 rotations promotes the entire `M_eff` assembly + `solve_sym3` to f64 via `torch.result_type`, silently invalidating the f32 throughput claim; and per-body f64 ledgers written every step are not the advertised "cheap global compensated ledger."

**Resolution — explicit dtype partition with three f64 islands and a hard f32 hot loop.**

- **Build-time (once/body), f64 → cast to f32:** `LambK` runs the 2048-interval Simpson quadrature in f64; the **stored** `seg_ma[B,S_max,3] = k_i·ρ·V` is immediately **cast to f32**. `assert seg_ma.dtype == float32` at kernel entry.
- **HOT LOOP — all f32, and this is exactly what the throughput gate (§2.7) times:** `pos, rot, prev_pos, vcom, xcom, seg_ma, seg_areaZ, seg_mass, seg_localPos, seg_localRot, m_eff[B,6], dv[B,2]`, and every pose/force/`M_eff`/solve intermediate. A promotion guard (`assert_hot_dtype`) checks `m_eff.dtype == dv.dtype == vcom.dtype == config.dtype_hot` after each step — catches accidental f64 promotion.
- **f64 ISLANDS (excluded from the throughput gate, amortized/O(W)):**
  1. `LambK` quadrature (build-time, once).
  2. The **global energy/conservation ledger**: per-step per-body terms (`R_step`, `ΔKE`, …) are computed in **f32**, reduced by masked axis-sum over bodies, then accumulated into **one f64 Kahan/Neumaier scalar per reservoir per world** — O(W) f64 scalars/step, **not** O(B) f64 tensors. This is the "cheap global compensated ledger."
  3. A **separate validation `Config(dtype_hot="float64")`** that runs the *same* kernel in f64 to produce the `<1e-6` reference arm of gate (b). It is **never** the throughput-timed path (§2.7 bench runs only `dtype∈{f32}` for the authorization number; f64 rows are diagnostic).
- **Plan §2.7** amended: the `dtype∈{f32,f64}` axis is retained but the **authorization floor (gate d) is read only from the f32 rows**; f64 rows are labeled diagnostic. The bench manifest records which tensors were timed (the hot-loop set above) so the number is auditable.

---

## #12 — `SolveSym3` robustness + compile reassociation (replaces plan:215–216, plan:322, plan:350, plan T2)

**Root cause.** Donor `SolveSym3` (SwimEval.cs:1151–1166) uses an **absolute** `|det|<1e-12` fallback (not scale-invariant): `M_eff ~ ρ·V·k` scales with body mass, so `det` can be legitimately small (well-conditioned tiny body) or catastrophically ill-conditioned while `det>1e-12` (slender body → extreme added-mass anisotropy). Cofactor inversion amplifies relative error ~`κ(M)` before the threshold fires. NumPy-random test matrices (plan:377) never sample the ill-conditioned region. And `torch.compile` may fuse/reassociate the cofactor arithmetic, breaking the "exact donor op-order" determinism claim.

**Resolution — production SPD-robust 2×2 solve + retained donor cofactor form for gain0 conformance + compile-parity gate.**

Because #2 reduces the production solve to the **2×2 SPD system** `[[M00,M02],[M02,M22]]` (`M_body·I + Σ SPD added-mass` → SPD by construction), robustness is tractable in closed form:

```python
def solve_constrained_xz(M00, M02, M22, Px, Pz):
    tr  = M00 + M22
    det = M00*M22 - M02*M02
    # scale-RELATIVE guards (τ = trace is the natural scale)
    disc = sqrt(clamp(tr*tr - 4*det, min=0))
    lam_min = 0.5*(tr - disc); lam_max = 0.5*(tr + disc)
    kappa   = lam_max / clamp(lam_min, min=eps_rel*tr)          # closed-form 2×2 condition number
    # Tikhonov ONLY where ill-conditioned; isotropic/H0 bodies never trigger it
    reg = where(kappa > KAPPA_MAX_F32, EPS_SPD*tr, 0.0)          # KAPPA_MAX_F32 = 1e6
    a = M00 + reg; c = M22 + reg; d = a*c - M02*M02
    inv = 1.0 / where(abs(d) > EPS_REL*tr*tr, d, EPS_REL*tr*tr)  # relative det guard, not absolute 1e-12
    dvx = ( c*Px - M02*Pz) * inv
    dvz = (-M02*Px + a*Pz) * inv
    return dvx, dvz
```

- **Scale-relative guards** replace the donor's absolute `1e-12`: det guard `|d| > ε_rel·τ²`, condition gate `κ > 1e6` (f32) triggers diagonal Tikhonov `EPS_SPD·τ`. For isotropic/H0 bodies `κ≈1` → **no regularization, exact**.
- **Retained donor cofactor form** (`solve_sym3_donor`) reproduces SwimEval.cs:1151–1166 op-order and the absolute `|det|<1e-12` fallback **byte-for-byte**, used **only** to match gain0/bug-inert step-trace fixtures. Production uses `solve_constrained_xz`. `test_solve.py::donor_conformance` asserts they agree on well-conditioned bodies and documents the (intended) divergence where the donor is numerically wrong.
- **Extreme-morphology M_eff corpus** (replaces NumPy-random, plan:377) — `M_eff` generated from *legal* extreme donor bodies at poses `t∈{0,Dt,…}` over 960 steps:
  1. 10:1 prolate slender (`kz→0, kx,ky→1`); 2. 10:1 oblate disk; 3. `MinX/Y/Z`-clamped degenerate `0.12×0.12×0.3` (SwimEval.cs:75); 4. `SurfaceThickness=0.1` fin with `finMaPerp` broadside (SwimEval.cs:195); 5. max-tilt 45°/89° roll (max off-diagonal); 6. single-segment; 7. combined worst case (slender + max-tilt + fin tail).
  Per fixture assert: (i) residual `‖M_xz·dv − P_xz‖/‖P_xz‖ < 1e-5` (f32); (ii) `κ(M_xz)` reported, `κ>1e6` flagged; (iii) `dv` all-finite; (iv) SPD (`λ_min>0`).
- **Compile-vs-eager parity** (plan:350 "force/solve-bound" claim depends on this): the "exact donor op-order" determinism claim requires `torch.compile` not reassociate the cofactor arithmetic. Enforce via `test_solve.py::compile_parity` — assert `max_abs(Δ)==0` between r0-eager and r1/r2-compiled `solve_sym3_donor` on the extreme corpus. **If parity fails, `solve_sym3_donor` is emitted as a fixed-lowering custom op (`torch.library`) or run in an eager island outside the compiled graph**, and that cost is reported in the §2.7 bench. `solve_constrained_xz` (production) is SPD-robust and does not require bit-exact donor op-order, so it may compile freely; only the *conformance* solve needs the no-reassociation boundary.

**Plan seam signatures amended** (plan:215–216): `solve_sym3` → two functions — `solve_sym3_donor(m00..m22, rhs)` (donor cofactor op-order EXACTLY, gain0 conformance only) and `solve_constrained_xz(M00,M02,M22, Px,Pz)` (SPD-robust production, scale-relative regularization). Plan:322 degenerate-guard item updated: guards are `torch.where` on **scale-relative** thresholds, never absolute `1e-12`, never dropped rows.

---

### Cross-references for the reviser to patch in the plan
- plan:309 (S10) → constrained 2×2 solve + `J_y` ledger (#2).
- plan:314 → `StepLedger` f32 terms `{KE_n, ΔKE, v_mid·F·Dt, ½vᵀΔM v, J_y}`; f64 world-Kahan only (#1, #11).
- plan:345–346 → split algebraic tests (a) + discrete-balance gate (b) (#1).
- plan:291,302–303 → sentinel-slot `[B,S_max]` indexing, remapped parent/tail (#9).
- plan:307 → `quat_inv = conj/‖q‖²` (#10).
- plan:308,312–314 → dtype partition, f32 hot set enumerated (#11).
- plan:215–216,322,350,377 (T2) → `solve_sym3_donor`/`solve_constrained_xz` split, extreme corpus, compile-parity (#12).

Source anchors: SwimEval.cs:805–807 (the #2 bug), :778/:1000 (`Quaternion.Inverse`, #10), :256–291 (`LambK` f64, #11), :1151–1166 (`SolveSym3`, #12), :669 (root `parentIndex<0` semantics, #9), :336/:426 (`InputPower`/`CirculatoryInputPower` algebraic closures, #1).


---

## FIXES: Determinism core — reduction & RNG

Scope owned here: Codex **#3** (deterministic segmented reduction), **#13** (RNG stream stability), and the **determinism half of #12** (compile/reassociation reproducibility). The shared premise — **canonical segment layout = fixed `[B, S_max]` padded + boolean `seg_mask`** (replacing the flattened/CSR `[S_total]` layout of plan:60, 201, 287–293; design:183–185) — is the pre-decided cross-cutting resolution and is treated as given; the #4 fix owns the layout switch itself, this fix builds the reduction and RNG on top of it and cites the exact plan text each item replaces.

Grounding checked before writing: donor per-body accumulation is a plain sequential loop over `_segs` summing `fDrag`, `wDrag`, and the six `m00..m22` added-mass entries (`SwimEval.cs:773–802`) — i.e. a reduction over the segment axis, nothing ragged-atomic; there is no stable organism id anywhere in the donor or in `ColonyState` (plan:169–181); the innovation registry is a monotone `next_iid` with a within-generation structural-key cache (design:677).

---

### Finding #3 — Deterministic segmented reduction

**Root cause Codex identified:** `segment_index_add(vals, body_id, B)` (plan:60, 216, 293, 307–308, 807) reduces a `[S_total]` tensor into `[B]` bodies keyed by `body_id`, which necessarily has duplicate destination indices (many segments per body). "Precomputed unique slots" cannot make destinations unique without a *prior* deterministic reduction that the plan never named, and the standard `index_add_`/`scatter_add_` implementations are atomic (nondeterministic on CUDA). The grep CI rule (plan:223) is a lint, not an algorithm.

**Resolution — the `[B, S_max]` layout turns the ragged scatter into a dense masked axis-sum, which is deterministic by construction; `body_id`, `segment_index_add`, and the "sanctioned wrapper" are deleted outright.**

Segment tensors are now `[B, S_max, …]` with a companion `seg_mask [B, S_max] bool` (`True` = live segment, `False` = padding). `S_max` is a compile-time constant pinned against the H1 raggedness profile (design:652 suggests `S_max = 32` post-mirror; S0 pins it from the measured H1 tail, open Q #12). There is **no `body_id`, no CSR `body_ptr`, no `S_total`.** A per-body reduction is:

```python
# numerics/reduce.py  — REPLACES segment_index_add entirely
def masked_segment_sum(vals: Tensor,          # [B, S_max, *F]  finite in ALL lanes
                       seg_mask: Tensor,      # [B, S_max]      bool
                       ) -> Tensor:           # [B, *F]
    # segment axis is fixed at dim=1; mask broadcast over trailing feature dims
    m = seg_mask.reshape(seg_mask.shape + (1,) * (vals.dim() - 2)).to(vals.dtype)
    return (vals * m).sum(dim=1)              # plain reduction along a fixed contiguous axis
```

This is the *entire* primitive. It is one line of arithmetic; there is nothing hidden to review.

**Call sites** (replacing plan:307–308 "`seg_sum(...)`"):
- Axial form drag: `fDrag = masked_segment_sum(quat_rotate(rot, floc), seg_mask)` → `[B,3]` (donor `SwimEval.cs:790`).
- Dissipated power: `wDrag = masked_segment_sum(relu(-(fworld*uj).sum(-1)), seg_mask)` → `[B]` (`SwimEval.cs:791`).
- Six added-mass entries: each `m_kl = masked_segment_sum(ma * col_k * col_l, seg_mask)` → `[B]`, then add the structural diagonal `mbody*250` **after** the reduction (`SwimEval.cs:771,796–801`; keeps the ×250 on struct-only, watch-item plan:318).
- `mbody = masked_segment_sum(seg_mass, seg_mask)`.
- Development scatter (genetics/develop.py, plan:488 S2.2) uses the same primitive over `[P, S_max]` — no atomics there either.

**Why this is deterministic (the claim Codex demanded be substantiated):**
1. `torch.sum` along a **fixed dimension** is on PyTorch's list of operations that have a deterministic implementation; the nondeterministic-on-CUDA operations are precisely the atomic scatters (`index_add`, `scatter_add`, `bincount`, `index_put(accumulate=True)`) — none of which appear anywhere in the hot path now. Under `torch.use_deterministic_algorithms(True, warn_only=False)` (plan:222) a residual nondeterministic op would *raise*, not silently pass.
2. The reduction axis has a **static extent `S_max`**, so the reduction tree (block/split geometry of the CUDA kernel) is fixed and data-independent: the number of partial sums and their combination order do not depend on how many lanes are live, only on the constant `S_max`. Two reruns on the same device therefore combine the same floats in the same order → bit-identical.
3. No duplicate-destination hazard exists because there is no scatter: each output row `b` reads exactly its own `[S_max]` slice.

**Finite-padding invariant (must hold or masking silently corrupts):** `vals` must be finite in *padded* lanes too, because `0.0 * NaN = NaN` would leak through the multiply. Every per-segment computation writes a neutral value into padded lanes: padded `seg_mass=0`, `seg_ma=0`, and any lane produced by a division/`quat` op is guarded so padded rows carry finite defaults (this dovetails with the #9 sentinel-row fix for root/empty gathers). Debug builds assert `torch.isfinite(vals).all()` before every `masked_segment_sum`; the mask is applied *before* the sum, never after.

**The grep rule is demoted from "algorithm" to "regression tripwire"** (rewrites plan:223, plan:5.3): the determinism *guarantee* is now structural (the layout admits no scatter). The CI check `test_no_atomic_scatter_in_hotpath` is defense-in-depth with two parts: (a) a source grep forbidding `scatter_add_`, `index_add_`, `index_put_(..., accumulate=True)`, `bincount` under `physics/` and `genetics/develop.py`; (b) a **graph-level audit** that traces the compiled hot step with `torch.compile(backend="aot_eager")` / FX and asserts the captured graph contains **zero** nodes of those aten ops. A grep can be fooled by an alias; the graph audit cannot.

**Replacement plan text (plan:60, 216, 293, 807):**
- plan:60 → `reduce.py  # masked_segment_sum(vals[B,S_max,*],seg_mask[B,S_max]) → axis-sum over segment dim; NO scatter, NO body_id`
- plan:216 signature → `def masked_segment_sum(vals, seg_mask) -> Tensor  # (vals*mask).sum(dim=1); deterministic axis reduction`
- plan:293 → "Every per-body reduction is a masked axis-sum over the `S_max` dimension; there is no `index_select`, no `scatter`, and no `body_id` in the hot loop."
- plan:807 → "`reduce.py::masked_segment_sum` (deterministic masked axis-sum; no atomics, no CSR)."

Acceptance gate for #3 (must be green *before the rest of S0 is built*, per Codex): `test_reduction` corpus below.

---

### Finding #12 (determinism half) — compile / reassociation reproducibility

**Root cause Codex identified:** `torch.compile` (rung r1) and CUDA-graph capture (rung r2, plan:369, 707) may fuse or reassociate floating-point expressions, defeating any "exact op-order" claim, and the plan asserted bit-identity without saying against what.

**Resolution — state the determinism CONTRACT precisely, then guard it with a two-tier test that separates *within-mode reproducibility* (bit-identity, gated) from *cross-mode parity* (tolerance, gated).**

**The contract (rewrites plan:9, 5.3/661–663):**

> **Determinism = reproducible-within-seed on a fixed `(device, dtype, build, execution-mode)` tuple.** For a fixed torch build, fixed device, fixed Inductor config, and a fixed compile mode, the compiled step is a pure deterministic function of `(state, seed)`: two reruns are **bit-identical** (`max_abs(Δ) == 0`). Cross-**mode** identity (eager vs compiled vs CUDA-graph) is **explicitly NOT claimed** — an FP-legal reassociation is permitted between modes; it is bounded by a tolerance parity gate, not by bit-identity. Cross-machine / CPU↔GPU identity remains out of scope (the conservation gate carries cross-device correctness).

**Making the compiled path reproducible run-to-run (the concrete measures):**
- `torch.use_deterministic_algorithms(True, warn_only=False)`; `CUBLAS_WORKSPACE_CONFIG=":4096:8"` set in `seed_everything` **and** in the CI/process env before torch imports (plan:221–222 retained).
- **Static shapes only** → `torch.compile` captures **one** graph with no guard-triggered recompilation; the reduction/pose/solve trees are baked once. This is why the `[B, S_max]` layout matters for #12 as well as #3: a `[S_total]` layout that changes size on birth/death would invalidate the capture (Codex #4) and force recompiles whose kernel choice can vary.
- **Forbid autotuning that selects kernels by measured wall-clock** in the determinism-gated path: `mode` is restricted to `"default"` or `"reduce-overhead"` (cudagraphs, no benchmark-based selection). `mode="max-autotune"` is **banned** on any determinism-gated run because it benchmarks candidate kernels and can pick different ones from timing noise across processes → nondeterministic op selection. (`max-autotune` may be used only for throughput telemetry cells that are *not* determinism-gated, and its result is never a reference-of-record.) Pin `torch._inductor.config.fallback_random = True` and disable `coordinate_descent_tuning`.
- `torch.backends.cuda.matmul.allow_tf32 = False`, `torch.backends.cudnn.allow_tf32 = False` (keep true f32 semantics matching the donor; TF32 also varies the mantissa).

**Reference-of-record:** the **eager f64** path is the oracle reference for gate (c) (LambK/forces/aggregates, plan:347–349). The compiled f32 path is what the throughput gate (d) measures. The compiled path must match the eager reference within the parity tolerance below; it is never compared *bit*-wise to eager.

**Guard tests (added to `test_determinism.py`):**

| test | assertion | threshold |
|---|---|---|
| `test_compile_rerun_bit_identical` | compiled step replayed twice, same process, fixed device | `max_abs(Δ) == 0` |
| `test_cudagraph_replay_bit_identical` | CUDA-graph captured once, replayed N=100× | `max_abs(Δ) == 0` replay-to-replay |
| `test_compile_eager_parity` | eager f32 vs `compile(reduce-overhead)` f32 over 960 steps | rel `< 1e-6` (tolerance, NOT bit-identity — a catastrophic reassociation trips this; a benign one passes) |
| `test_process_restart_reproducible` | run twice in **separate processes**, same build/seed/device/mode | `max_abs(Δ) == 0` (proves no timing-dependent kernel selection leaked in) |

`F4` (determinism tax, plan:358, 388) is measured as gate-(a) timing with/without det mode; it is a *cost* falsifier, not a correctness one — the correctness contract above holds regardless of tax.

---

### Finding #13 — RNG stream stability (counter-based keyed stateless RNG)

**Root cause Codex identified:** the append-only `RngManifest` (plan:56, 224, 807; design I-GENOME-5 line 634) only preserves downstream draws *while a gene stays inert*. Once an inert gene activates and consumes draws, every subsequent **sequential** draw shifts. Slot recycling/compaction changes iteration order, and `ColonyState` has **no stable entity id** to bind a stream to an organism. A sequential-stream RNG cannot be stable under a mutating population.

**Resolution — delete the sequential stream and the manifest entirely; replace with a stateless counter-based keyed PRF where every draw is addressed by content, not by position. Add never-reused stable entity IDs + parent IDs to `ColonyState`.** Because draws are *addressed*, an inert gene occupies no stream position, activating it shifts nothing, and slot recycling is irrelevant.

#### 13.1 The key tuple and its bit budget

Every random draw in the system is produced by evaluating a pseudo-random function `PRF(key, counter)`; the *logical* address is the exact 6-tuple Codex prescribed:

```
(seed, step, stable_entity_id, gene_iid, event_kind, draw_index)
```

packed into Philox-4x32's 64-bit **key** + 128-bit **counter** with fixed, injective bit budgets:

| field | source | bits | placement |
|---|---|---|---|
| `seed` | run seed (i64) | 64 | key `k = (k0=seed&0xFFFFFFFF, k1=seed>>32)` |
| `step` | `SimClock.step` | 40 | counter word budget |
| `stable_entity_id` | ColonyState `stable_id` | 48 | counter |
| `gene_iid` | innovation id (design:677) | 24 | counter |
| `event_kind` | enum (below) | 8 | counter |
| `draw_index` | block index within one draw | 8 | counter |

`40+48+24+8+8 = 128` bits — exactly the four 32-bit counter words. The packing is a fixed bijection into the 128-bit counter; a runtime assert (`_assert_field_widths`) raises if any field exceeds its budget (e.g. `step ≥ 2^40`, `stable_entity_id ≥ 2^48`). Injectivity of the packing ⇒ distinct logical draws never collide on `(key, counter)`.

`event_kind` enum (reserved, extensible to 256): `0 REPRO_ASEXUAL_SELECT`, `1 STRUCTURAL_ADD_NODE`, `2 STRUCTURAL_ADD_EDGE`, `3 STRUCTURAL_DELETE`, `4 PARAM_JITTER`, `5 MUTATION_BERNOULLI`, `6 MATE_CROSSOVER_MASK`, `7 MATE_PARTNER_SELECT`, `8 DEVELOP_STOCHASTIC` (reserved), `9 FIELD_STOCHASTIC` (reserved), … A draw needing more than 128 output bits increments `draw_index` (e.g. a normal via Box–Muller uses `draw_index` 0 for two uniforms; a rejection-sampled draw increments until accept).

#### 13.2 The generator — Philox-4x32-10, torch-implementable, bit-exact

Philox is a stateless counter-based PRF (Salmon et al. 2011; the generator behind JAX/cuRAND). Pure 32-bit integer arithmetic ⇒ bit-identical on CPU and CUDA and across runs (integer ops are deterministic and not affected by reduction order or TF32). Implemented over int64 tensors holding uint32 lanes:

```python
# numerics/rng.py
_M0, _M1 = 0xD2511F53, 0xCD9E8D57          # Philox multipliers
_W0, _W1 = 0x9E3779B9, 0xBB67AE85          # key-bump (golden ratio) constants
_MASK = 0xFFFFFFFF

def _mulhilo(a, b):                          # a,b: int64 tensors of uint32 values
    p  = a * b                               # 32*32 -> fits in int64 exactly
    return (p >> 32) & _MASK, p & _MASK      # hi, lo

def philox_4x32_10(c0,c1,c2,c3, k0,k1):      # all [B] int64, elementwise, batched
    for _ in range(10):
        hi0, lo0 = _mulhilo(c0, _M0)
        hi1, lo1 = _mulhilo(c2, _M1)
        c0, c1, c2, c3 = (hi1 ^ c1 ^ k0) & _MASK, lo1, (hi0 ^ c3 ^ k1) & _MASK, lo0
        k0 = (k0 + _W0) & _MASK; k1 = (k1 + _W1) & _MASK
    return c0, c1, c2, c3                     # 128 random bits

def keyed_bits(seed, step, eid, gene_iid, event_kind, draw_index):  # tensors -> 4x[B] u32
    _assert_field_widths(step, eid, gene_iid, event_kind, draw_index)
    c0 =  step & _MASK
    c1 = ((step >> 32) & 0xFF) | ((eid & 0xFFFFFF) << 8)          # 8 hi step bits + 24 eid bits
    c2 =  (eid >> 24) & 0xFFFFFF | ((gene_iid & 0xFF) << 24)      # remaining 24 eid + 8 gene bits
    c3 =  (gene_iid >> 8) & 0xFFFF | (event_kind << 16) | (draw_index << 24)
    return philox_4x32_10(c0, c1, c2, c3, seed & _MASK, (seed >> 32) & _MASK)
```

(The exact word-packing above is the canonical bijection; the only requirement Codex's concern imposes is injectivity, which the width asserts enforce.) Distribution transforms are pure functions of the 128 output bits: **uniform-f32** `u = (w0 >> 8) * 2**-24 ∈ [0,1)`; **uniform-f64** from `(w0,w1)` 53-bit mantissa; **normal** Box–Muller from two uniforms; **Bernoulli(p)** `u < p`; **categorical** via inversion over a fixed-order CDF. All deterministic, no global state, no `torch.rand`/`Generator` in the mutation/selection path.

`seed_everything(seed)` still seeds torch/numpy/python for any incidental library calls and sets det-mode/CUBLAS env (plan:214), but **the simulation's own stochasticity never reads a torch Generator** — it only calls `keyed_bits`. This is what makes streams position-independent.

#### 13.3 Stable entity IDs — monotone, never reused, with parents

`ColonyState` gains two columns (extends the table at plan:169–181):

| tensor | shape | dtype | notes |
|---|---|---|---|
| `stable_id` | `[W,N_cap]` | i64 | monotone, **never reused**, unique within a world for the whole run; `-1` in dead slots |
| `parent_id` | `[W,N_cap]` | i64 | `stable_id` of the parent at birth; `-1` for seeded founders |

**Per-world monotone allocator** `next_eid [W] i64` (part of the snapshot, §13.5). Birth assignment is deterministic and batched — no atomics, no host loop:

```python
# core births, given newborn_mask [W,N_cap] bool for this step
rank   = torch.cumsum(newborn_mask.to(torch.int64), dim=1) - 1     # deterministic prefix within world
eid    = next_eid.unsqueeze(1) + rank                             # contiguous ids, slot-order
stable_id = torch.where(newborn_mask, eid, stable_id)
parent_id = torch.where(newborn_mask, parent_stable_id_gather, parent_id)
next_eid  = next_eid + newborn_mask.sum(dim=1)                    # advance allocator
```

Assignment order is the fixed `[W,N_cap]` slot order (deterministic `cumsum`), **per world**, so cross-world scheduling never perturbs any world's ids. Because `stable_id` is never reused, recycling a dead slot for a new birth gives the newcomer a fresh id whose RNG stream is disjoint from the deceased's — slot recycling and compaction can no longer shift any stream.

#### 13.4 I-GENOME-5 rewrite (design line 634; and plan:224, 5.3)

Replace the manifest invariant verbatim:

> **I-GENOME-5 (counter-based keyed RNG; §2.7).** The simulation draws no random numbers from a sequential stream. Every draw is produced by `PRF(key, counter)` keyed by `(seed, step, stable_entity_id, gene_iid, event_kind, draw_index)`. Because a draw is *addressed by content* rather than consumed by position, an inert gene occupies no stream position; activating a previously inert gene introduces only that gene's own draws (keyed by its stable `gene_iid`, I-GENOME-4) and shifts no other gene's or organism's draws. Slot recycling, compaction, birth, and death cannot re-baseline determinism because organism streams are keyed by never-reused `stable_id`, not by slot index or draw order. There is no manifest and no sequential draw counter.

Replaces plan:224 (§1.7 bullet) and plan:56/807 (`rng.py` description → "`rng.py  # seed_everything + Philox-4x32-10 keyed PRF (keyed_bits); NO sequential stream, NO manifest`"). The design's mutation-operator note (design:721 "appends to the RNG manifest") is rewritten to "keys its draws by the new gene's `gene_iid`."

#### 13.5 Checkpoint completeness (the determinism state Codex #13 said was missing)

The authoritative snapshot must carry everything the keyed RNG and allocator depend on. `ColonyState` gains `stable_id`, `parent_id`; the versioned `SimulationSnapshot` (the #14-owned object) additionally carries `next_eid[W]`, the innovation registry (`next_iid` + within-generation structural-key cache, design:677), the run `seed`, and `SimClock.step`. With those, `keyed_bits` is fully reconstructible → replay is bit-identical. (Snapshot schema itself is #14's deliverable; this fix only enumerates the RNG/allocator fields it must include.)

#### 13.6 Determinism GATE assertions replacing the manifest claims

These are the tests that *earn* I-GENOME-5' (replacing the unfalsifiable "append zero-count entry" claim):

| test | setup | assertion |
|---|---|---|
| `test_inert_gene_no_shift` | two genomes identical except genome B carries one extra **silenced** gene (fresh `gene_iid`); run N mutation+develop steps | shared genes' realized draws AND developed bodies **bit-identical** between A and B |
| `test_gene_activation_locality` | activate that gene at step k | all *other* genes' draws (steps `<k` and `≥k`) bit-identical to control; only the activated gene's own draws appear |
| `test_slot_recycle_stream_stable` | kill organism X, recycle its slot for new organism Y | every *surviving* organism's stream bit-identical to a control run with no death/birth; Y's stream (new `stable_id`) disjoint from X's |
| `test_key_injective` | synthetic corpus spanning field-budget extremes | no two distinct logical draws map to the same `(key,counter)`; width-overflow asserts fire on out-of-budget inputs |
| `test_prf_reference_vectors` | fixed `(key,counter)` inputs | output matches published Philox-4x32-10 reference vectors (bit-exact), and CPU==CUDA bit-exact |
| `test_rng_process_restart` | mutation soak run twice in separate processes | `max_abs(Δ)==0` (proves no hidden stateful RNG) |

**RK-6 mitigation rewrite (design:1194, plan:770):** "Deterministic-by-construction masked axis-sum reductions (no atomic scatter anywhere in the hot path); counter-based keyed stateless RNG (no sequential stream); det-mode env; CPU-first fallback." The old "precomputed-unique-slot deterministic scatter" phrasing is deleted — there is no scatter to make unique.

---

### Consolidated determinism test set (replaces plan:105 `test_determinism.py`, extends §2.5 gate (a))

Single-worker (`pytest -p no:xdist -m determinism`, plan:225). Gated on CPU; CUDA reported per the contract.

1. **Reduction** — `test_reduction_masked_axis_sum_matches_ref` (masked axis-sum == f64 gather-sum reference, rel `<1e-6`; padded lanes contribute exactly 0); `test_reduction_finite_padding` (NaN in a padded lane is caught by the pre-sum assert); `test_reduction_rerun_bit_identical` (`==0`, fixed device).
2. **No atomics** — `test_no_atomic_scatter_in_hotpath` (source grep tripwire + FX/compile graph audit: zero `scatter_add`/`index_add`/`index_put(accumulate)`/`bincount` nodes under `physics/`, `genetics/develop.py`).
3. **Compile/graph** — `test_compile_rerun_bit_identical`, `test_cudagraph_replay_bit_identical`, `test_compile_eager_parity` (rel `<1e-6`), `test_process_restart_reproducible` (`==0`).
4. **RNG** — the six tests of §13.6.
5. **End-to-end** — `test_within_seed_rerun` (existing gate (a), plan:343): two same-process reruns over 960 steps `max_abs(Δ)==0` on `(vcom,xcom,ledger)`; `test_churn_deterministic` (plan:344) with births/deaths/recycling still `==0`, now *provable* because stable-id RNG + masked-sum reductions have no order dependence on churn.

This set is what authorizes the "determinism" standing gate; it is producible by the specified processes (integer PRF, static-axis reductions, static-shape compile) — none of it rests on an unspecified reduction, a sanctioned wrapper, or a grep-as-algorithm.

---

**Files/sections this fix rewrites:** plan:9 (contract wording), plan:56 & 60 (module comments), plan:105 (test file), plan:165–181 (`ColonyState` + `stable_id`/`parent_id`), plan:213–226 (§1.7 determinism rules), plan:287–293 (§2.1 layout → `[B,S_max]`+`seg_mask`), plan:307–308 (`seg_sum` → `masked_segment_sum`), plan:343–344 (gate (a) tests), plan:661–663 (§5.3 determinism plan), plan:770 (R5), plan:807 (§7 numerics tasks); design:634 (I-GENOME-5), design:721 (mutation-operator manifest note), design:1194 (RK-6). The `numerics/reduce.py::segment_index_add` and `numerics/rng.py::RngManifest` symbols are **deleted**; `masked_segment_sum` and `keyed_bits`/Philox replace them.


---

I have the full context needed for my cluster (#4 representation contradiction, #14 incomplete checkpoint). Here are the fixes.

---

## FIXES: Representation, lifecycle & checkpoint (Codex #4, #14)

These two fixes are coupled: #14's checkpoint cannot be defined until #4 fixes what the authoritative in-memory state *is*. I resolve #4 first (canonical padded layout + static lifecycle), then #14 (`SimulationSnapshot` over that state).

---

## Finding #4 — "Static-shape architecture contradicts the flattened live-segment layout"

**Root cause of the contradiction.** The plan simultaneously asserts (i) static shapes for CUDA-graph capture (plan:99, 264 G-SCAF-4, 277) and (ii) a segment tensor of shape `[S_total] = Σ live_seg_count` (plan:201, 287–293) whose length changes on every birth/death/refill/compaction. Those are irreconcilable: any change to `S_total` reallocates the segment arena, moves base addresses, invalidates a captured CUDA graph, and forces `body_ptr`/CSR `body_base` repair. The design's own development representation is fixed `[P, S_max, …]` (design:638–652), so the flattened layout was never actually load-bearing — it was an unforced contradiction.

**Resolution (adopts the pre-decided cross-cutting decision).** The canonical layout is **fixed `[B, S_max]`-padded segment tensors + a boolean `seg_mask`**, with `B = W·N_cap`. `S_total` is deleted from the design. CSR/flattened storage is demoted to a *deferred, measured* optimization (the exact inverse of the plan's current "padded is only an ablation" stance). This is what kills #3 as well: per-body reductions become plain **masked axis-sums over the `S_max` axis**, `(vals * seg_mask).sum(dim=SEG_AXIS)` — a fixed-shape contiguous-axis tree reduction, deterministic by construction, no `index_add_`, no `scatter_add_`, no atomics, no duplicate-`body_id` destinations. (The detailed determinism argument for the reduction primitive is owned by the #3 cluster; this cluster supplies the layout that makes it a trivial `sum`.)

### 4.1 Canonical shape constants (Config, frozen)

| symbol | value | meaning | source |
|---|---|---|---|
| `W` | Config.w | worlds | — |
| `N_cap` | Config.n_cap (1024 default) | per-world creature capacity | plan:187 |
| `S_max` | **16** | per-creature **physics** segment capacity | donor physics cap, `SwimEval` PropWalk `:442–460` |
| `B` | `W·N_cap` | flattened creature index for the kernel | plan:288 |
| `L` | 6 | development / pose depth passes | design:652, plan:303 |

`S_max = 16` is the *developed-body / physics* capacity (what `swim_step` consumes). It is distinct from the **genotype** capacities `N_max = 24`, `E_max = 48` (design:652), which live at the genetics layer and are never touched by the hot physics loop. Development (genetics→DevelopedBody) is the pure fixed-shape map that fills `[·, S_max, ·]` from `[·, N_max, ·]` (design:630, I-GENOME-1); it can emit at most `S_max` segments and masks the rest. There is **no** `S_max_dev=32` in the physics body — the design's `S_max=32` is genotype post-mirror node capacity, a different tensor at a different layer; this reconciliation removes that ambiguity.

### 4.2 Segment layout — REPLACES plan §2.1 lines 287–293 and §1.6 line 201

Segment tensors are canonically stored `(W, N_cap, S_max, …)` and viewed as `[B, S_max, …]` for the kernel (a reshape, never a copy — `B` and `S_max` are compile-time constants so the view is graph-stable).

| tensor | shape | dtype | notes |
|---|---|---|---|
| `seg_mask` | `[W,N_cap,S_max]` | bool | **sole source of truth for segment existence**; padded slots `False` |
| `seg_localPos` | `[W,N_cap,S_max,3]` | f32 | rest position in parent frame [m] |
| `seg_localRot` | `[W,N_cap,S_max,4]` | f32 | rest orientation quat (parent frame) |
| `seg_abc` | `[W,N_cap,S_max,3]` | f32 | ellipsoid semi-axes [m] |
| `seg_mass` | `[W,N_cap,S_max]` | f32 | box or ellipsoid per gain-stage (§2.0) |
| `seg_areaZ` | `[W,N_cap,S_max]` | f32 | axial cross-section [m²] |
| `seg_ma` | `[W,N_cap,S_max,3]` | f32 | Lamb added-mass per axis (cast f64→f32 after LambK, per #11) |
| `seg_ampDeg,seg_phase,seg_c` | `[W,N_cap,S_max]` | f32 | gait actuation + half-length |
| `seg_isTail,seg_hasJoint` | `[W,N_cap,S_max]` | bool | tail selection / hinge presence |
| `seg_parent_local` | `[W,N_cap,S_max]` | i16 | **parent index within the SAME creature's `S_max` axis**, ∈ `[0,S_max)`; root → own index (self-parent, `seg_hasJoint=False`) |
| `seg_depth` | `[W,N_cap,S_max]` | i8 | DFS depth ∈ `[0,L)`; padded → `-1` (never selected) |
| working: `pos,prev_pos` | `[W,N_cap,S_max,3]` | f32 | pose-pass scratch |
| working: `rot` | `[W,N_cap,S_max,4]` | f32 | pose-pass scratch |

**Key change vs the flattened design:** `seg_parent_local` is now a **creature-local** index into the `S_max` axis, not a global `body_base + local_parent`. All parent gathers in the pose pass (`torch.gather` along `SEG_AXIS`) are therefore in-range `[0,S_max)` and can never read another creature's segment. This eliminates `body_ptr`/`body_base`/`body_id` maintenance entirely — `body_id` no longer exists because segments are indexed by their `(world, creature, slot)` position, not by a flat id. (The root-`-1`/empty-body gather hazard of #9 is handled by the self-parent + `seg_hasJoint=False` convention here and completed in the #9 cluster.)

**Per-body reductions (masked axis-sum, replacing every `segment_index_add`):**
```
SEG_AXIS = 2
m_body   = (seg_mass * seg_mask).sum(SEG_AXIS)                 # [W,N_cap]
# M_eff assembly (donor S9): per-seg outer products, masked, summed over S_max
col_x = quat_rotate(rot, x_hat)                               # [W,N_cap,S_max,3]
m00   = (seg_ma[...,0] * col_x[...,0]*col_x[...,0] * seg_mask).sum(SEG_AXIS)   # etc. for m01..m22
fDrag = (quat_rotate(rot, floc) * seg_mask.unsqueeze(-1)).sum(SEG_AXIS)        # [W,N_cap,3]
```
`torch.sum` over a fixed contiguous axis is a deterministic pairwise-tree reduction under `use_deterministic_algorithms(True)`; no sanctioned-wrapper indirection is needed. `numerics/reduce.py::segment_index_add` (plan:60,216,223) is **deleted**; the Grep-rule scaffold (plan:223) is replaced by an AST/import guard asserting the hot loop contains no `index_add_`/`scatter_add_` at all, which is now enforceable because the code genuinely contains none.

### 4.3 Creature layout — REVISES plan §1.5 (lines 165–181)

The `ColonyState` table gains the identity/lifecycle fields required by #13 (stable RNG keying) and #14 (checkpoint), and **drops the flattened `body_ptr`** (segments are co-indexed with creatures now; only `genome_ptr` into the `[P,…]` genotype pool remains):

| tensor | shape | dtype | notes |
|---|---|---|---|
| `alive` | `[W,N_cap]` | bool | sole source of truth for creature existence |
| `stable_id` | `[W,N_cap]` | i64 | **monotonic, never reused** (per-run); 0 = never-allocated sentinel |
| `parent_id` | `[W,N_cap]` | i64 | `stable_id` of parent (0 for founders) |
| `generation` | `[W,N_cap]` | i32 | slot-reuse counter (ABA guard / lineage debug) |
| `pos,heading,lin_vel,ang_vel` | as plan:172–175 | f32 | physics state |
| `energy` | `[W,N_cap]` | **f64** | metabolic reserve [J] — promoted to f64 per #5 (reservoir) |
| `struct_N` | `[W,N_cap]` | **f64** | structural nutrient [mol] — f64 reservoir per #5 |
| `genome_ptr` | `[W,N_cap]` | i64 | → row in `[P,…]` genotype pool |
| `age` | `[W,N_cap]` | f64 | [s] |
| `species_tag` | `[W,N_cap]` | i64 | observational only (never gates mating) |

(`body_ptr` removed; the developed-body segment tensors of §4.2 are the body, co-indexed `[W,N_cap,S_max,…]`.)

### 4.4 Static-shape lifecycle (birth / death / refill) — REPLACES the "precomputed free-slot list / append" text at plan:167, and `spikeswim/churn.py` semantics at plan:99

**Invariant:** every operation is a `where`/`gather`/`cumsum` over the fixed `[W,N_cap,…]` / `[W,N_cap,S_max,…]` buffers. No tensor is ever resized, reallocated, `index_select`-ed, or compacted. Therefore a single captured CUDA graph replays every tick (§4.5).

**Death** (masked clear, deterministic):
```
alive = alive & ~die_mask          # [W,N_cap]; dead slots' data left stale, masked out everywhere
```
No compaction, ever. Staleness is safe because every reduction multiplies by `alive` (creature axis) and `seg_mask` (segment axis).

**Birth / dead-slot recycling** (deterministic, no atomics, static shape):
```
free       = ~alive                                    # [W,N_cap]
free_rank  = torch.cumsum(free.to(i32), dim=1) - 1     # k-th free slot in world w has rank k
n_free_w   = free.sum(1)                                # [W]
n_birth_w  = torch.minimum(request_w, n_free_w)         # overflow deterministically dropped, logged
claim      = free & (free_rank < n_birth_w[:,None])     # [W,N_cap] slots to fill this step

# child payload is prebuilt indexed by BIRTH ORDINAL (0..n_birth_w) per world, padded to N_cap:
#   payload_*[w, ord, ...]  (parent lookups are gathers → deterministic)
# free_rank is a bijection claimed-slot ↔ ordinal, so scatter-by-gather:
new_field  = torch.gather(payload_field, 1, free_rank.unsqueeze(-1).expand_as(field))
field      = torch.where(claim.unsqueeze(-1), new_field, field)   # per creature/segment tensor
alive      = alive | claim
generation = generation + claim.to(i32)                 # slot reuse bumps generation
```
**Stable-id assignment** (monotonic, never reused; deterministic across worlds by `(world, free_rank)` order):
```
world_off  = torch.cumsum(n_birth_w, 0) - n_birth_w     # [W] global-ordinal base per world
global_ord = world_off[:,None] + free_rank              # valid where claim
stable_id  = torch.where(claim, next_stable_id + global_ord, stable_id)   # next_stable_id: host i64
parent_id  = torch.where(claim, gather(parent_stable_id_by_ord, free_rank), parent_id)
next_stable_id += int(n_birth_w.sum())                  # the ONLY host-side scalar advanced
```
`next_stable_id` is the free-list/allocator state (a single i64 counter) — it, plus the `generation` array, fully captures allocator status; the free-slot *list* itself is never stored because it is recomputed from `alive` each step. This is what the plan's "precomputed free-slot list (never atomic append)" (plan:167) becomes: a per-step `cumsum` over `~alive`, not a persisted mutable list.

Developed-body segment slots of a newly-claimed creature are filled the same way (`gather` the child's developed body along the `S_max` axis, `where` on `claim`), so `seg_mask` for a reused slot is overwritten wholesale — no residue from the previous tenant leaks into a masked sum.

### 4.5 CUDA-graph-capture argument (makes G-SCAF-4 / gate (a) real)

A capture holds iff every replayed op has fixed shapes, fixed addresses, and no host synchronization. Under §4.2–4.4: (1) all tensors are `Config`-constant `[W,N_cap,·]`/`[W,N_cap,S_max,·]`, allocated once at `reset`; (2) birth/death are `where`/`gather`/`cumsum` over those buffers — no resize, no `index_select`, no `.item()` inside the step; (3) `S_max` and `N_cap` are frozen in `Config`, so the `[B,S_max]` view is address-stable. The only host-side scalar (`next_stable_id`) is advanced **outside** the captured region (it feeds the *next* step's payload build), so it does not break capture. `n_birth_w` overflow is resolved by `minimum` on-device, not a Python branch. Gate-(a) `churn_deterministic` (plan:344) and G-SCAF-4 (plan:264) are therefore satisfiable by construction.

### 4.6 Masking-waste vs arena — stated as a MEASURED-LATER decision (replaces plan:293 "ablation" framing)

Padded storage carries `S_max/mean_seg ≈ 16/6 ≈ 2.7×` compute/memory waste on segment-axis ops. This is **accepted for S0 and made canonical.** A flattened/arena layout (segment compaction with generations, allocation/fragmentation policy, pointer repair, graph recapture) is a **deferred optimization**, and the plan's existing padded-vs-flattened ablation (plan:293, T15) is **inverted**: it now measures whether the *flattened* layout is worth its lifecycle complexity, not whether padded is viable. The arena is adopted only if **both** hold, on telemetry: (i) the S0 profiler attributes `> 30%` of step time to segment-axis reductions (falsifier F2, plan:360), **and** (ii) the measured masking tax at H1/H2 exceeds the arena's added lifecycle cost (T15 parquet). Until both are met, `[B,S_max]`-padded is the only representation the build carries. This resolves the design/plan contradiction (design:638–652 fixed `[P,S_max]` now agrees with the plan) and removes the "padded is only an ablation" claim entirely.

**Plan edits for #4:** rewrite §2.1 lines 287–293 (segment layout → `[W,N_cap,S_max]`+`seg_mask`, delete `[S_total]`/CSR/`body_id`, invert the padded/flattened framing); update §1.5 lines 165–181 (add `stable_id`/`parent_id`/`generation`, drop `body_ptr`, promote `energy`/`struct_N` to f64); update §1.6 line 201 (`DevelopedBody` → padded `[B,S_max]`); delete `numerics/reduce.py::segment_index_add` (lines 60,216,223) and rewrite plan:223 as an AST guard; rewrite `spikeswim/churn.py` note (line 99) to the §4.4 recycle-by-`cumsum` scheme.

---

## Finding #14 — "Replay from ColonyState alone is false"

**Root cause.** S7 (plan:602, 604) and design S7 (design:602–604) require replay "from serialized `ColonyState`", but `ColonyState` is the creature-state subset only — it omits field reservoirs, the genotype pool, RNG/counter state, the id allocator, the innovation registry, `Config`, the clock schedule, and external forcing. A run resumed from `ColonyState` alone diverges immediately.

**Resolution.** Define a **versioned `SimulationSnapshot`** as the *complete* checkpoint. `ColonyState` is explicitly re-scoped to "the creature-state subset, not a checkpoint." Counter-based keyed RNG (#13) makes this dramatically simpler: there is **no PRNG internal buffer to serialize** — the entire RNG state is `(master_seed, step, next_stable_id)`, all of which live in Config/clock/allocator. This is called out as the payoff of the #13 decision.

### 14.1 `SimulationSnapshot` — the complete authoritative state (new `core/snapshot.py`)

```python
@dataclass(frozen=True)
class SnapshotHeader:
    schema_version: int          # bump on any field change; load refuses major mismatch
    git_sha: str
    torch_version: str; torch_cuda: str | None
    device: str; dtype_hot: str; dtype_ledger: str   # "float32"/"float64"
    config_hash: str             # sha256 of serialized Config (determinism depends on it)
    created_utc: str

@dataclass(frozen=True)
class SimulationSnapshot:
    header:      SnapshotHeader
    config:      dict            # full Config + WorldConfig (pydantic .model_dump) — §1.6
    clock:       dict            # {now:f64, step:i64, dt:f32, scale, forcing_phase...} — SimClock schedule
    colony:      dict            # ALL ColonyState tensors incl developed-body [·,S_max] segs + seg_mask (§4.2/4.3)
    genotype:    dict            # [P,N_max/E_max] pool: node_f/type/iid/mask, edge_*, body_g (design:640-650)
    reservoirs:  dict            # f64: Nd,Bp,Bd,Bm [W,G,G,B]; struct_N,energy [W,N_cap]; Sed [W,G,G];
                                 #      + numerical_residual reservoir (#5); + double-buffer parity flag
    allocator:   dict            # {next_stable_id:i64}  (free-list is recomputed from alive, §4.4)
    innovation:  dict            # {next_iid:i64, structural_key_cache: {(op,key)->iid}} (design:677)
    forcing:     dict            # stateful external forcing (S6 advected field buffers; empty at S0/S1 — analytic)
    rng:         dict            # {master_seed:i64, salt:i64}  — NO PRNG buffer (counter-based, #13)
```

Authoritative-state coverage checklist (each maps to a Codex-#14 gap):

| #14 gap | Snapshot field |
|---|---|
| field reservoirs | `reservoirs` (f64, incl. residual) |
| genomes / development inputs | `genotype` + `colony.genome_ptr` + `colony` developed-body segs |
| RNG states | `rng` + `clock.step` + `allocator.next_stable_id` (counter-based ⇒ no buffer) |
| stable IDs / free-list | `colony.stable_id/parent_id/generation` + `allocator.next_stable_id` |
| innovation registry | `innovation` |
| config | `config` + `header.config_hash` |
| clock schedule | `clock` |
| external forcing | `forcing` |

### 14.2 Serialize / load

- **Format:** tensors → a single `tensors.safetensors` (typed, deterministic, no pickle, CPU-resident); scalars/registries/config → `meta.json`; `header.json` alongside. Directory or zip. `torch.save`/pickle is forbidden (nondeterministic dict ordering, arbitrary-code hazard).
- **Device policy:** always serialize CPU tensors; `load(path, device)` moves to the configured device. Snapshot is device-agnostic; determinism is only defended within a fixed `(device,dtype,op-order)` (plan:146, §5.3) after load.
- **API** (`core/snapshot.py`):
  ```python
  def save(snap: SimulationSnapshot, path: Path) -> None
  def load(path: Path) -> SimulationSnapshot        # raises on header.schema_version major mismatch
  # Colony gains:
  def Colony.snapshot(self) -> SimulationSnapshot    # gather complete state
  def Colony.restore(self, snap: SimulationSnapshot) -> None   # rehydrate ALL of the above
  ```
  `Colony.load(ColonyState)` (plan:198) is **removed** and replaced by `Colony.restore(SimulationSnapshot)`. `ColonyState.serialize/load` (plan:86) survives only as the creature-subset (de)serializer used *inside* `snapshot()/restore()`, and its docstring is changed to "creature-state subset — NOT a checkpoint; see `SimulationSnapshot`."

### 14.3 Round-trip determinism gate (new `tests/test_snapshot.py`) — makes S7 acceptance real

```python
@pytest.mark.determinism
def test_snapshot_roundtrip_bit_identical():
    c = Colony(cfg); c.reset(seed=S)
    for _ in range(N): c.step(dt, act)          # advance to a nontrivial state (post-birth/death)
    snap = c.snapshot(); save(snap, p); snap2 = load(p)
    # (1) serialize/load is lossless:
    assert_tensors_bit_identical(snap, snap2)   # max_abs(Δ)==0 on every tensor; scalars/registries ==
    # (2) a restored run is indistinguishable from a never-serialized one:
    c2 = Colony(cfg); c2.restore(snap2)
    for _ in range(M): c.step(dt, act); c2.step(dt, act)
    assert_state_bit_identical(c.state(), c2.state())          # ColonyState
    assert_reservoirs_bit_identical(c.reservoirs, c2.reservoirs)  # f64 == exactly
    assert c._ledger.closure_residual == c2._ledger.closure_residual
```

**Two-directional (catches an *incomplete* checkpoint, mirroring the leak-test discipline of plan:229):**
```python
@pytest.mark.determinism
def test_colonystate_alone_is_insufficient():
    # a partial snapshot omitting reservoirs+genotype+allocator+innovation+clock MUST diverge
    partial = strip_to_colonystate(c.snapshot())
    c3 = Colony(cfg); c3.restore_partial(partial)
    for _ in range(M): c.step(dt, act); c3.step(dt, act)
    assert not states_bit_identical(c.state(), c3.state())     # proves ColonyState alone ≠ checkpoint
```
This negative gate is what makes "replay from a complete snapshot" falsifiable rather than asserted — a snapshot that silently drops a field fails `test_snapshot_roundtrip_bit_identical`, and a checkpoint claim narrowed to `ColonyState` fails `test_colonystate_alone_is_insufficient`.

**Plan edits for #14:** add `core/snapshot.py` to the tree (plan:82–95) and `tests/test_snapshot.py`; change plan:198 `Colony.load(s)` → `Colony.restore(snap)` + `Colony.snapshot()`; re-scope `core/state.py` docstring (plan:86); rewrite S7 components (plan:602) and acceptance (plan:604) to read "replay-from-checkpoint from a serialized **`SimulationSnapshot`**" and "reproduces a checkpointed run from a **`SimulationSnapshot`** (not `ColonyState`) alone"; mirror the same wording in design:602–604; add the round-trip + insufficiency gates to §5.2 (plan:646) and the S7 row.

---

### Consistency notes for the other clusters
- The f64 promotion of `energy`/`struct_N` (§4.3) and the `numerical_residual` reservoir in `SimulationSnapshot.reservoirs` are the #5 decision surfacing in this cluster's tables; keep dtypes aligned with the #5 fix.
- `stable_id`/`parent_id`/`generation` (§4.3) are the ColonyState fields that #13's counter-based RNG keys on `(seed, step, stable_id, gene_iid, event_kind, draw_index)`; this cluster provides the fields and the allocator (`next_stable_id`), #13 owns the keyed-draw derivation.
- The deletion of `segment_index_add` / `body_id` (§4.2) is the layout half of #3's resolution; the reduction-determinism proof is #3's.


---

I have everything I need. I've read the plan fully (both pages), the Codex review, and the design doc's energy/conservation sections (§3.9 light, §4.8 energy closure, §6.1–6.9 economy, §7.1.1), plus the donor `SwimEval.Step` energy accounting (lines 740–817, 354–426). Here are the fixes.

---

## FIXES: Conservation ledgers & the energy system (#5, #7, #15)

Voice: engineering spec. Every replacement below is written to drop into the plan verbatim; I cite the exact plan lines/sections it supersedes. Constants are anchored to the design doc (§6.1, §6.2, §1.2 anchored-measurement corollary).

---

### FINDING #5 — The f64 reservoir ledger + explicit numerical-residual reservoir + scale-dependent tolerances

**What Codex is right about.** `transfer(src,dst,a)` written as `src -= a; dst += a` in **f32** does *not* preserve the total: subtracting `0.5` from `1.0` is exact, but adding `0.5` to `1e8` rounds away (f32 ULP at 1e8 is ~8). The debit is realized, the credit is lost → a structural mint/leak. No compensated f64 sum *afterward* recovers information already destroyed in the f32 write. So plan:229, plan:414–425, plan:459–471 ("`residual==0`", "`<1e-9`", "million-step `<1e-6`") are unsupported **for f32 reservoir state**.

**Resolution — three concrete mechanisms, all specified.**

#### 5.1 Reservoir state is f64 (REPLACES plan:414 "Named f32 tensors" and the f32 rows of plan:176–177)

Every *registered reservoir* — the quantity `close_books()` sums — is stored **float64**:

| Reservoir tensor | shape | dtype (was → now) |
|---|---|---|
| `Nd, Bp, Bd, Bm` | `[W,G,G,B]` | f32 → **f64** |
| `struct_N`, `energy` (reserve `E`), `mass` | `[W,N_cap]` | f32 → **f64** |
| `Sed` | `[W,G,G]` | f32 → **f64** |
| `R_num` (numerical-residual reservoir, §5.3) | scalar per `(W, currency)` | **f64** |

The **hot physics stays f32** (swim kernel `pos/rot/vcom/M_eff/dv` per plan:174–178, §4.8). The seam: the f32 kernel *emits fluxes* (a locomotion-power draw, a drag-dissipation credit); those f32 flux amounts are cast up to f64 (an **exact** widening — f32 ⊂ f64, zero rounding) before touching any reservoir. Rationale for f64 *state* is **dynamic range, not conservation** (§5.2 already gives exact conservation): f64 keeps the residual reservoir at true machine-noise level (a small flux onto a large pool lands in the pool, not in the residual). At f64, `1e8 + 0.5` is exact (ULP ≈ 1.5e-8), so a physically-realizable flux never vanishes below the reservoir ULP unless the magnitude ratio exceeds ~1e15 — which no physical flux does.

#### 5.2 The transfer primitive captures its own rounding via TwoSum (REPLACES plan:421 `transfer` and design §6.6's "index_add_ … equal credit")

Even in f64, `src−a` and `dst+a` round. We make the pair **exact** with Knuth's error-free TwoSum (6 flops, branchless, correct for any magnitudes), booking the exact rounding into `R_num`:

```
def two_sum(x, y):            # returns (s, e) with x + y == s + e EXACTLY in f64
    s = x + y
    z = s - x
    e = (x - (s - z)) + (y - z)
    return s, e

def transfer(src, dst, a, currency):     # a: f64 (f32 flux widened losslessly)
    src_new, e_s = two_sum(src,  -a)      # a is masked/elementwise; deterministic, no scatter
    dst_new, e_d = two_sum(dst,  +a)
    src[...] = src_new
    dst[...] = dst_new
    R_num[currency] -= (e_s + e_d)         # the ONLY place matter/energy enters the residual pool
```

After the write, `src_exact + dst_exact + R_num` equals the pre-write `src + dst + R_num` **to the precision at which `R_num` itself accumulates** (O(eps²)). `R_num` is registered as a real reservoir and is included in `close_books()`. This is not a "sanctioned wrapper hiding a nondeterministic op": TwoSum is pure elementwise f64 arithmetic; the debit/credit still use the masked-axis-sum layout (canonical `[B,S_max]`+mask / grid stencils), never `scatter_add`.

- **Cross-magnitude case, now correct:** `a=0.5`, `dst=1e8` (f64): `dst_new=100000000.5` exact (`e_d=0`); `src=1.0→0.5` exact (`e_s=0`); `R_num` unchanged; the 0.5 actually arrives. The f32 mint is gone.
- **Sub-ULP case (only if ratio >~1e15):** `e_d≠0`, the un-representable fraction is booked to `R_num` so **nothing is minted** — and §5.4's leak guard fires if `R_num` ever grows past machine noise, surfacing the dynamic-range failure as a bug to root-cause (not a silently absorbed leak).

#### 5.3 `close_books()` — exact assertion + tolerance derivation (REPLACES plan:229, plan:416/422, and the plan:459–466 thresholds)

```
def close_books(currency) -> LedgerResidual:
    total   = pairwise_sum_f64(registered_reservoirs[currency]) + R_num[currency]   # compensated
    ext     = X_in_cum[currency] - X_out_cum[currency]        # tracked external in/out (0 for S1 mass box)
    res_abs = total - I0[currency] - ext
    res_rel = abs(res_abs) / max(I0[currency], I_FLOOR[currency])
    assert res_rel        < TAU_STEP[currency]                # bookkeeping closure
    assert abs(R_num[currency]) / max(I0,I_FLOOR) < TAU_NUM_LEAK   # residual stays machine-noise
    return LedgerResidual(res_abs, res_rel, R_num[currency], S_max, N_elem)
```

**Tolerance derivation (scale-dependent, defensible — this is the derivation Codex demanded).** The reservoirs are **all-positive** (mass and stored energy ≥ 0), so the summation condition number `κ = Σ|R_i|/|ΣR_i| = 1` — no catastrophic cancellation. A pairwise-compensated f64 reduction over `N_elem` cells then has relative error bounded by:

```
err_reduce ≤ C · ⌈log2 N_elem⌉ · eps_64          (C≈8 safety; eps_64 = 2.22e-16)
```

For the S1 box `N_elem = W·G²·B·N_res + W·N_cap = 1·256²·16·4 + 1024 ≈ 4.19e6` → `⌈log2⌉=22` → `err_reduce ≤ 8·22·2.22e-16 ≈ 3.9e-14`. Per-transfer bookkeeping error is O(eps²) (TwoSum) and does not accumulate coherently (the ±rounding random-walks, so drift is √T·eps², not T·eps). Therefore:

| Tolerance | Value | Derivation |
|---|---|---|
| `TAU_STEP[mass]` | **1e-12** rel (f64) | `err_reduce ≈ 3.9e-14`, ×25 headroom |
| `TAU_STEP[energy]` | **1e-6** rel-to-throughput (f32 fluxes) | see #15.4 — energy fluxes are f32-kernel-sourced |
| drift over 1e6 steps | **< 1e-9** rel, bounded-oscillating (slope≈0 in noise) | TwoSum → O(T·eps²)≈5e-24 systematic; observed drift is reduction-noise random walk, never monotone |
| `TAU_NUM_LEAK` | **1e-10** rel | `R_num` must stay machine-noise; larger ⇒ dynamic-range bug, root-cause per §1.2 |
| `I_FLOOR[mass]` | 1e-6 mol | avoids divide-by-zero on an emptied box |

These **supersede** plan:229 (`< 1e-9` f32), plan:459–466 (the `<1e-9 f64` rows keep their value but are now *derived*, not asserted), and design §6.6 (`τ_mass ~ 1e-9` tightens to `1e-12` on f64+TwoSum; state that the design's 1e-9 was a floor, now provably beatable).

**New tests** (extend plan:459–467):

```
test_transfer_exact_twosum   :  after N random transfers, Σreservoirs + R_num == I0 to 1e-14 rel   [f64]
test_cross_magnitude_transfer:  transfer 0.5 between a 1.0 pool and a 1e8 pool → 0.5 realized at BOTH; res_rel<1e-12  (the exact case Codex named)
test_residual_stays_noise    :  1e6-step soak → |R_num|/I0 < 1e-10 throughout (dynamic-range guard)
test_leak_is_caught          :  an UNPAIRED write (bypassing transfer) → close_books raises  (kept from plan:236; now also catches R_num overflow)
```

---

### FINDING #7 — S1's bloom-and-crash is impossible with S1's processes

**What Codex is right about.** As written (plan:443–444), S1 has *only* production (`Nd→Bp` until `Nd` exhausts) and remineralization/sinking of `Bd`. There is **no `Bp→Bd/Nd` producer loss**, so when `Nd` empties, production merely *stops* and `Bp` sits flat forever. "Bloom must … crash" (plan:467, plan:473) and "drifters plateau" cannot occur. `Bp` never declines; there are no consumers to plateau.

**Resolution — I pick: ADD explicit producer respiration + mortality to S1** (the design already wants an in-S1 crash, §6.8: "drawdown empties `Nd` → production collapses → biomass declines"; and the remin loop that receives `Bd` already exists at S1.8). The **consumer-plateau claim moves to S3** (it genuinely needs grazing). Both moves are specified below.

#### 7.1 New S1 producer-loss processes (INSERT into §S1.C, after S1.7)

All units mol N; all transfers go through the §5.2 primitive; energy side per #15.

```
S1.7b  Producer respiration (maintenance):     Bp -> Nd,  releases heat
   P_resp = m_resp · Bp                         [mol N · s⁻¹],  m_resp ≈ 0.10 /day  (anchor: phyto respiration ≈10–30% of gross production, §6.2 NPP band)
   transfer(Bp, Nd, P_resp·dt, 'mass')
   E_heat += e_N · P_resp·dt   ;  E_chem(Bp) tracks -e_N·P_resp·dt   (Nd carries NO energy — inorganic)

S1.8b  Producer mortality (senescence + crowding):  Bp -> Bd,  energy carried
   M_p = (d0 + d_dd·Bp) · Bp                     [mol N · s⁻¹],  d0 ≈ 0.05/day, d_dd density-dependent
   transfer(Bp, Bd, M_p·dt, 'mass')
   E_chem moves Bp->Bd with the biomass (no heat)
```

`d_dd·Bp` (crowding) is what gives the crash its bite: at the bloom peak `Bp` is large, mortality is super-linear, biomass collapses even as production has already stalled from `Nd` exhaustion. `M_p` feeds `Bd` → S1.8 remin returns `(1−BGE)` of it to `Nd` (deep) → S1.10 mixing lifts it → **damped re-bloom → steady standing stock**. This is genuine bloom-and-crash, fully conserved, **no cap knob** — the crash is a real loss flux, not a ceiling. `m_resp, d0, d_dd` are frozen *rate anchors* (§1.2), never retuned to hit a target biomass; a failure to stabilize is a diagnostic, not a tuning target (§6.8, §6.9).

#### 7.2 S1 acceptance rewrite (REPLACES plan:473 item 3)

> **3.** Blooms/deserts emerge from the loop. A seeded nutrient pulse must: (i) spike production, (ii) draw `Nd` down toward exhaustion, (iii) **crash — `Bp` rises to a peak then declines** as producer respiration+mortality (S1.7b/S1.8b) debit it faster than the stalled production replaces it, (iv) settle to a **damped oscillation / steady standing stock** as mortality→`Bd`→remin→`Nd`→mixing closes the return. **No hand-coded ceiling.** Zonation falls out of the pump. *(The consumer-plateau claim "drifters plateau at production÷intake" MOVES to S3 — it requires grazers, which S1 does not implement.)*

#### 7.3 S1 test rewrites (REPLACE plan:467, ADD)

```
test_bloom_self_terminates (rewrite): seeded pulse → Bp(t) is unimodal-then-declining (argmax not at t_end,
                                       and Bp(t_end) < Bp(peak)·0.7); Nd draws down then partially recovers;
                                       close_books < 1e-12 throughout; no baseCap/carryingCap symbol reachable.
test_producer_loss_closes (new):       respiration Bp->Nd and mortality Bp->Bd each pass INV-TRANSFER (|Δsrc+Δdst|<1e-14 f64);
                                       energy: E_chem drop == E_heat rise (respiration) OR E_chem(Bd) rise (mortality).
test_damped_steady_state (new):        1e6-step soak reaches a bounded standing stock with no cap knob; oscillation amplitude decays (not monotone growth, not zero-flat).
```

The plan-wide claim (plan:9, plan:473, design §6.8) "drifters plateau at production÷intake" is retagged **[S3]** everywhere it appears in S1 contexts.

---

### FINDING #15 — Define the full energy system as a reservoir/flux graph

**What Codex is right about.** S3 debits basal metabolism + locomotion (plan:570) with **no heat/waste reservoir**; production introduces chemical energy from light while S1 models light as an *untracked analytic value* (plan:449); feeding mixes nutrient mass and energy with **no biomass energy-density conversion** (plan:578). "Energy books closed end-to-end" (plan:9, plan:558, plan:572) is rhetoric until every reservoir, flux, unit, and conversion is written down. Below is the implementable graph.

#### 15.1 Conversion constants (frozen anchors, §1.2 — REPLACES the missing energy-density in plan:578)

```
Redfield C:N (molar)      = 106 : 16  = 6.625                          (§6.2)
caloric anchor  c_calC    = 45 kJ·gC⁻¹                                 (Platt & Irwin 1973, §6.2)
molar mass C    M_C       = 12.011 g·mol⁻¹
⇒ energy per mol C  e_C   = 45000 · 12.011      = 5.405e5  J·(mol C)⁻¹
⇒ energy per mol N  e_N   = e_C · 6.625          = 3.581e6  J·(mol N)⁻¹   ← the biomass mass↔energy conversion
sim-energy anchor N_sim   = 300 J per sim-energy unit                  (§4.8, SimUnits)
muscle efficiency η       = 0.20                                       (§4.8)
```

`e_N` is the single conversion that lets **derived** chemical energy of any N-pool be read off its mass: `E_chem(pool) = e_N · N_pool` [J]. This is a *readout*, **not a second stored-and-synced scalar** (design §6.1 — storing it independently is the very sync-glue anti-pattern P1 forbids).

#### 15.2 Reservoirs (all f64, per #5) — the energy currency graph

| # | Reservoir | Symbol | Unit | Stored or derived | Kind |
|---|---|---|---|---|---|
| E1 | Producer/detritus/microbial chemical energy | `E_chem = e_N·(Bp+Bd+Bm)` | J | **derived** from mass pools | internal |
| E2 | Creature metabolic reserve (lipid/glycogen) | `E_reserve = energy[W,N_cap]` | J | **stored** (§6.1: genuinely distinct) | internal |
| E3 | Creature bulk kinetic energy | `E_KE = ½vᵀM_eff v` | J | derived from kernel state (f32) | internal |
| E4 | Absorbed-light input meter | `E_sun_cum` | J | stored, monotone↑ (Kahan) | **INPUT** |
| E5 | Heat / respiration / dissipation sink meter | `E_heat_cum` | J | stored, monotone↑ (Kahan) | **OUTPUT** |
| E6 | Exported (buried) energy meter | `E_export_cum` | J | stored, monotone↑ (off at S1) | **OUTPUT** |

This is an **open system**: E4 is the one external input (sun), E5+E6 the outputs. `Nd` (dissolved inorganic nutrient) and `Sed` carry **zero energy density** — re-fixing `Nd` re-injects sun energy; this decoupling is what makes the loop open, not a closed constant.

#### 15.3 Fluxes (transaction equations + units; each is a §5.2 paired transfer or a metered credit)

Powers in W = J·s⁻¹. `p_in` (muscle→body input power) uses the **cluster-#1-corrected** identity `p_in = tReact·U + pWake + tFin·U + pFin` (consistent with the S0 fix; the plan:345 form omitting `tFin·U` is superseded there).

```
F-L  Photosynthesis (INPUT):    ΔE_chem = +e_N·ΔBp ;  E_sun_cum += e_N·ΔBp
        (ΔBp from S1.7 production; light energy = the energy needed to fix ΔBp from Nd. Incident I(x,z),
         §3.9 Beer–Lambert, remains an analytic DRIVER, not a reservoir; only ABSORBED-and-fixed light is metered.)

F-G  Grazing (chem → reserve + egesta):        [S3]
        E_chem(field) -= e_N·I_bio ; E_reserve += AE·e_N·I_bio ; E_chem(Bd) += (1-AE)·e_N·I_bio
        (mass side: field-=I_bio, struct_N+=AE·I_bio, Bd+=(1-AE)·I_bio ; §6.4. No heat: assimilation loss is egesta, mass-conserved.)

F-M  Metabolism (reserve → heat + mech):        [S3]
        P_basal = B0·M^α  (Kleiber, α≈0.79, §6.5.1)
        E_reserve -= (P_basal + p_in/η)·dt
        E_heat_cum += (P_basal + (1/η - 1)·p_in)·dt        # basal + muscle-inefficiency heat
        E_mech_in  += p_in·dt                               # mechanical work delivered to body
        (identity check: draw = basal + p_in/η = heat + mech ✓  → S3-ENERGY-2)

F-K  Locomotion (mech → KE + hydro dissipation): [S0/S2 kernel]
        ΔE_KE = ½vᵀM_{n+1}v_{n+1} − ½vᵀM_n v_n
        E_heat_cum += (pWake + pFin + wDrag)·dt             # shed wake + fin wake + axial drag → heat
        (the exact discrete balance incl. the ½vᵀ(M_{n+1}−M_n)v added-mass term is the S0 energy-gate cluster;
         THIS ledger consumes the S0 StepLedger's {p_in, pWake, pFin, wDrag, ΔE_KE} as authoritative flux values.)

F-R  Bacterial remineralization (chem → microbial + heat):  [S1]
        R = k_remin(z)·Bd
        E_chem(Bd) -= e_N·R·dt ; E_chem(Bm) += e_N·BGE·R·dt ; E_heat_cum += e_N·(1-BGE)·R·dt
        (mass: BGE→Bm, (1-BGE)→Nd, §6.3. The (1-BGE) N returns to Nd but its ENERGY is respired to heat — the key decoupling.)

F-P  Producer respiration + mortality:          [S1, new §7.1]
        respiration: E_chem(Bp) -= e_N·P_resp·dt ; E_heat_cum += e_N·P_resp·dt
        mortality:   E_chem(Bp) -= e_N·M_p·dt   ; E_chem(Bd) += e_N·M_p·dt   (no heat, biomass carries energy)

F-B  Burial (OUTPUT, off at S1, on at S9):
        E_export_cum += e_N·w_bury·Bd·dt          (mass: Bd→Sed; energy leaves the system)
```

#### 15.4 The master balance — "books close" = every flux accounted (NOT total constant)

```
Σ_stored(t) := E_chem(t) + E_reserve(t) + E_KE(t)
INV-ENERGY:   | Σ_stored(t) − Σ_stored(0) − (E_sun_cum − E_heat_cum − E_export_cum) |  <  τ_E · (E_sun_cum + E_heat_cum + ε)
```

Denominator is the **cumulative throughput**, not the stored total — the correct reference for an open system whose meters grow unboundedly (this is why a raw "energy is constant" check is meaningless here). Two distinct precisions, stated honestly:

- **Bookkeeping closure (each flux paired via §5.2 TwoSum):** closes to O(eps²). Catches any unpaired energy write (a mint). `τ_E_book = 1e-12`.
- **Physics/work consistency (`E_chem` derived from mass; `E_KE`/`pWake`/`wDrag` from the f32 kernel):** limited to f32, `τ_E_phys = 1e-6` per step, `1e-4` drift over 1e6 steps, **gated on bounded-oscillating, not endpoint** (semi-implicit integrators oscillate — F6, §4.8). The f32-ness is a *fidelity* limit checked by the S0 oracle gate, not a conservation leak (the application is exact via TwoSum).

#### 15.5 S1 / S3 energy-conservation test predicates (the deliverable Codex asked for)

**S1** (abiotic + producer loop; reservoirs E1/E4/E5, no E2/E3):
```
S1-ENERGY-1 (bookkeeping):  | E_chem(t) − E_chem(0) − (E_sun_cum − E_heat_cum) | / (E_sun_cum+E_heat_cum+ε) < 1e-12   [f64, all fluxes TwoSum-paired]
S1-ENERGY-2 (derived-not-stored):  AST/audit — E_chem is ONLY ever computed as e_N·(Bp+Bd+Bm); no independent E write to a producer pool exists (kills the sync-glue P1 anti-pattern). ⇒ mass closure (#5, 1e-12) IMPLIES energy closure for producer pools.
S1-ENERGY-3 (crash is energetic):  during the §7 bloom-crash, the ΔE_chem<0 of the decline is matched cell-for-cell by E_heat_cum rise (respiration F-P/F-R) + E_chem(Bd) rise (mortality) — energy released/moved, never lost.
```

**S3** (full economy; all six reservoirs):
```
S3-ENERGY-1 (bookkeeping):  INV-ENERGY (§15.4) < 1e-9 rel-to-throughput  [mix of f64 mass-derived + f32 KE terms booked via TwoSum]
S3-ENERGY-2 (metabolic identity):  per creature/step, ΔE_reserve == −(P_basal + p_in/η)·dt to 1e-6 f32; NO path banks mechanical work at implicit η=1 (design §4.8, plan:534/657).
S3-ENERGY-3 (work consistency, inherited from S0):  |ΔE_KE − (p_in − (pWake+pFin+wDrag))·dt| / throughput < 1e-6 f32, bounded-oscillating over 1e5 steps.
S3-ENERGY-4 (feeding closure):  e_N·I_bio == AE·e_N·I_bio (→reserve) + (1−AE)·e_N·I_bio (→Bd); INV-TRANSFER on the mass split < 1e-14 f64.
S3-ENERGY-5 (no mint):  E_sun_cum is the ONLY reservoir permitted to rise without an internal debit; every other energy increase is a §5.2 paired transfer (audit test forbids raw E_i += x). ⇒ the population "sustains without a mint/cap knob" (plan:558/572) becomes a checkable predicate, not a claim.
```

#### 15.6 Plan edits this finding requires

- **plan:449** ("Light: analytic … no storage at S1"): keep analytic `I(x,z)`, but ADD "absorbed light is *metered* into `E_sun_cum` via flux F-L; incident irradiance stays an analytic driver, not a reservoir."
- **plan:566–572** (S3): replace "debits basal metabolism and locomotion" prose with the F-M/F-K/F-G/F-B flux set and the E1–E6 reservoir table; add `E_heat_cum`, `E_export_cum` to the reservoir registry.
- **plan:646–657 / design §6.6** (INV-ENERGY row): replace the bare `|E_tot − ∫(P_sun−P_heat−P_burial)| < τ_energy` with the §15.4 throughput-referenced form and the two-precision (`τ_E_book=1e-12` / `τ_E_phys=1e-6`) split.
- **plan:558** ("energy books don't close end-to-end" as S3 falsifier): now backed by S3-ENERGY-1…5 predicates.

---

### Consistency notes for the re-review

- **Segment layout / determinism:** every `transfer` above uses masked axis-sums on the canonical `[B,S_max]`+mask / grid stencils — no `scatter_add`, no ragged reduction (aligns with the adopted layout resolution; TwoSum is pure elementwise f64).
- **f32↔f64 seam is explicit and lossless:** f32 kernel fluxes widen to f64 exactly; only f64 *reservoir writes* round, and TwoSum captures that into `R_num`. This is the "explicit numerical-residual reservoir that absorbs any f32↔f64 transfer rounding" the cross-cutting resolution mandates.
- **`p_in` uses the cluster-#1-corrected identity** (`+ tFin·U`) everywhere it appears (F-K, F-M, S3-ENERGY-3), so the energy system and the S0 gate cannot disagree.
- **Anchors are frozen, not tuned** (§1.2): `e_N, c_calC, η, N_sim, m_resp, d0, k_remin, BGE` are measurements; a non-closing ledger or an implausible equilibrium is root-caused upstream, never softened by retuning an anchor or re-hiding a mint (§6.8–6.9).


---

## FIXES: Throughput authorization, benchmark plan & risk register (Codex #6, #19, #20)

This cluster rewrites plan §2.0 (line 285), §2.5 gate-(d) row (line 350), §2.7 sweep (lines 367–369), §2.8 tasks T16 (line 391), the design floor claim (design:272 / plan quotes at 750/787), the benchmark param gaps (lines 821–825), and the entire risk register §6.3 (lines 764–780). It does **not** re-open the layout decision — it *consumes* the pre-decided fixed `[B,S_max]`+mask layout to collapse two axes of the sweep and to retire two of Codex's top-ranked risks.

---

## Finding #6 — Throughput gate is arbitrary, contradictory, not end-to-end

### 6.1 Root cause of the original number
Plan:350 and design:272 derive the floor as `256 worlds × 1000 creatures × 120 Hz = 3.07e7` **then demand `≥10× headroom` (→ effectively 3.07e8)**. Both inputs are wrong for near-term scope: (i) 256 worlds is the *deferred S8 many-worlds* config (design:270 defers many-worlds to S8), not the near-term single small-dense world; (ii) the `×10` is an unexplained blanket multiplier; (iii) it is measured on the *frozen-heading locomotion kernel only*, which excludes development, field sampling, spatial hashing, feeding, encounters, mutation, mating, and telemetry — so it cannot authorize "the entire vectorization thesis" (plan:277).

### 6.2 Re-derivation from a required scientific run (REPLACES the 256-world derivation)

Anchor on **one small-dense world** (design:271), sized to worst case at `N_cap`:

| Symbol | Quantity | Value | Justification |
|---|---|---|---|
| `N_pop` | creatures in the near-term world (worst case = `N_cap`) | **1000** | design:271 "dense hundreds-to-low-thousands" |
| `f_phys` | physics rate | **120 Hz** | plan §1.6 `dt=1/120` |
| `T_run` | ticks in one scientific run | **1e9** | design:272/§2.9 "deep-time evolution 10⁸–10⁹ ticks/run", demanding end |
| `W_budget` | wall-clock budget for one run | **12 h** (43 200 s) | overnight run → 1 run/day iteration cadence |

**End-to-end floor (this is the *real* authorization number):**

```
F_sci = N_pop · T_run / W_budget
      = 1000 · 1e9 / 43 200
      = 2.31e7 creature-steps/s   (whole tick, all subsystems)
```

For scale only: single-world **realtime** is `N_pop·f_phys = 1.2e5` c-steps/s. `F_sci` is therefore **≈193× realtime on one world** — the honest statement of the demand (deep-time evolution must run ~200× faster than the biology it simulates, so a run is iterable overnight). Note `F_sci = 2.31e7` is *coincidentally near* the old `3.07e7` but is now single-world and derived from a run duration, not from 256 deferred worlds.

### 6.3 What S0 actually authorizes (SCOPE — replaces plan:277 "go/no-go for the entire vectorization thesis")

> **S0 authorizes exactly one thing: that the frozen-heading locomotion *kernel* is fast enough to be a viable component of the whole tick — a NECESSARY, not sufficient, condition.** S0 does **not** measure or authorize the end-to-end tick. The end-to-end throughput thesis is decided at a later named gate **G-E2E** (§6.5), post-S2/S3, once StepLive + development + at least one field + spatial hash + feeding exist.

**S0 kernel floor, derived (replaces `3.07e7 ×10`):** allocate locomotion a wall-clock share `φ_loco` of the tick and require the kernel to fit its allotment with explicit (not blanket) headroom.

```
φ_loco = 0.5      # locomotion (articulated pose + M_eff assembly + 3×3 solve) is the
                  # single heaviest per-creature subsystem; assumption VALIDATED at G-E2E
F_loco_bare = F_sci / φ_loco = 2.31e7 / 0.5 = 4.63e7 c-steps/s

headroom = 1.5 (StepLive-over-frozen-Step: yaw state + P-controller + torque assembly)
         × 1.5 (pop growth toward N_cap + world densification)
         × 1.3 (bench↔production variance, thermal throttle, measurement)
         ≈ 2.9×          # REPLACES the unexplained ×10

F_loco_S0 = F_loco_bare · headroom ≈ 1.4e8 creature-steps/s   ← S0 gate (d) floor, H1/H2
```

Every factor is now named and falsifiable. The `256-world × 1000 × 120 = 3.07e7` figure is **retained only as the S8-era many-worlds realtime target** (design:270), explicitly not a near-term gate.

**Reality this exposes (stated, not hidden):** at the near-term single-world batch `B = W·N_cap ≈ 1024`, `1.4e8` c-steps/s ⇒ **≤ 7.3 µs/full-articulated-step**. At B≈1024 a GPU is launch-bound and likely *cannot* clear this in eager mode; CPU-compiled may or may not. **This is the make-or-break the benchmark exists to expose, not paper over.** The crossover sweep and rung ladder (§19) exist precisely to find whether *any* `(device, rung)` clears `F_loco_S0` at the near-term B. Outcomes:
- Best device clears at B≈1024 → **GO** (device chosen on the number; CPU-only for one world is a *legitimate GO*, design:270 device knob — the thesis is "affordable when vectorized on the best device," not "GPU wins").
- Neither clears at B≈1024, but GPU clears at larger B → **conditional GO**: fill the GPU batch with independent **seed-replicate worlds** (`W>1`, non-interacting — a throughput device, distinct from the S8 *scientific* many-worlds claim). Record `B*`.
- Neither clears at any rung/device → trips falsifier F3 → escalate one ladder rung (§5.8) or **NO-GO / narrow-scope**, exactly as the ladder intends.

### 6.4 Replacement text for plan §2.5 gate (d) row (line 350)

> **(d) Throughput** — `bench.py`→`test_throughput.py::floor`: at the near-term single-world batch `B=W·N_cap` (H1/H2), the best `(device,rung)` clears **F_loco_S0 = 1.4e8 creature-steps/s** (derived §6.2–6.3; *not* `3.07e7×10`). Report `B*` (CPU↔GPU crossover) and the per-device curve across the B-ladder; report H1/H0 heterogeneity tax and padded-mask tax. Profiler (winning + any falsifier-tripped cell only) must show **force/solve-bound, not pose-scan/reduction-bound**. **S0 authorizes locomotion-kernel feasibility only; end-to-end authorization is deferred to G-E2E (§6.5).**

### 6.5 NEW named later gate — **G-E2E** (add to §4/§6 milestones, gates post-S2, re-confirmed post-S3/S4)

> **G-E2E (end-to-end throughput authorization).** *Dependency: S2 StepLive + genome development + ≥1 field (S1) + spatial hash + S3 feeding stub.* Benchmark **one whole tick** (`Colony.step`: develop-if-dirty → field sample/advect → spatial hash rebuild → StepLive → feeding → metabolism → churn → telemetry) at `N_pop=1000`, one world, best `(device,rung)`. **Floor: `F_sci = 2.31e7` creature-steps/s (≈193× realtime).** Also emits the *measured* `φ_loco` (locomotion share of the tick); if `φ_loco > 0.5`, the S0 gate was optimistic and G-E2E is the binding authority; if `< 0.5`, S0's floor was conservative (safe). **This — not S0 — authorizes the full vectorization thesis.** A NO-GO here escalates the language/kernel ladder (§5.8) on the *dominant* subsystem the profiler names, which may be fields or hashing, not locomotion.

---

## Finding #19 — Benchmark not executable as a one-week task (576 cells → staged program)

The original 576-cell crossed sweep (`8 B × 2 dev × 2 dtype × 3 rung × 3 het × 2 layout`, plan:369) is replaced by a **staged funnel** that measures only what each stage needs, killing dead axes using the pre-decided layout resolution.

### 19.1 Axis reductions (justified removals)
- **dtype axis removed from throughput.** f64 is *oracle-match only* (design §2.8: LambK precompute + ledger + oracle config). The hot loop is f32 (design §2.8). f64 is a **correctness** dtype, never a throughput cell. −2×.
- **rung axis de-crossed.** Rung is not a full axis; it is chosen per-device by a **3-point probe at one B** (Stage 2), then fixed. −(3→1) on most cells.
- **layout axis collapsed.** Pre-decided: canonical layout is fixed `[B,S_max]`+mask. The old "flattened vs padded" cross vanishes; masking tax is measured **once** vs a single flattened prototype (Stage 4), not crossed. −2×.

### 19.2 The staged program

| Stage | Purpose | Cells (timed) | Config |
|---|---|---|---|
| **0 · Correctness corpus** | gates (a)(b)(c) green — *precondition, not timed* | 0 | T3–T13; fixed small B; f64 oracle + f32; must pass before any timing |
| **1 · Rung probe** | pick best rung per device | **5** | single B=`N_cap`≈1024, H1, f32: CUDA{r0,r1,r2}, CPU{r0,r1} → freeze best rung/device |
| **2 · Crossover sweep** | locate `B*`, confirm scaling | **16** | 8-point B-ladder × {CPU,GPU}, H1, f32, frozen best rung |
| **3 · Authorizing cells** | **decide gate (d)** | **~10** | B∈{near-term, 2×, 4×} × het∈{H1,H2} × best device; H0 baseline ×2 (het-tax denominator) |
| **4 · Masking-tax ablation** | quantify padded `[B,S_max]`+mask waste | **2** | near-term B, H1: canonical padded vs one flattened prototype (≈2.7× waste check) |
| **5 · Churn cost** | churn stub on/off (F7) | **2** | near-term B, H1, best rung |
| **6 · Profiler** | attribution — **winning + falsifier-tripped only** | **~4** | `torch.profiler`, 1 rep each, NOT every cell |
| | **Total timed** | **~35–40** | vs 576 |

### 19.3 Fixed benchmark parameters (fills every gap Codex named)

```python
# spikeswim/bench.py — measurement protocol (frozen)
K              = 5           # reps per cell; report median + IQR
S_window       = 600         # steps timed per rep (= episode measure phase)
W_warmup       = 360         # steps before timing (= episode warmup)
compile_warm   = "discard iters until step-time Δ<2% between consecutive, max 10"  # r1 compile / r2 capture
sync           = "cuda.synchronize() around each rep's 600-step window ONLY"  # never per-step (kills graph)
clock          = "time.perf_counter()"
timeout_cell   = 180.0       # s wall; exceed → recorded 'timeout' = gate-(d) fail for that cell (CPU r0 @262k expected)
vram_cap       = "<PIN Q#1>" # e.g. 20 GiB; cell over cap → recorded 'OOM-skip', not fail; log max_memory_allocated
subprocess     = "one FRESH subprocess per cell"          # isolates CUDA-graph pool + compile cache
env_per_cell   = {"TORCHINDUCTOR_CACHE_DIR": "<per-cell temp>",  # compile artifacts don't leak between cells
                  "TORCHINDUCTOR_FX_GRAPH_CACHE": "0"}           # compile cost stays in warmup, not measured
```

**CUDA-graph recapture rule (uses the pre-decided static layout):** with fixed `[B,S_max]`+mask, all shapes are static ⇒ **the r2 graph is captured ONCE at cell entry (post-warmup) and replayed for all `S_window·K` steps; no mid-cell recapture, ever.** Churn (birth/death) mutates tensor *contents* (`alive`-mask, free-slot writes) but never *shapes* ⇒ no recapture; T14 asserts the captured graph pointer is byte-stable across a churn event. Any op that would force recapture (dynamic shape / host-sync / data-dependent control flow) is a **build failure** caught by the static-shape guard, not a silent recapture.

### 19.4 Budget and statistical confidence
- **GPU-hour budget:** ~40 timed cells × ~90 s (incl. warmup/compile) ≈ 1.0 h + profiler (~4 × 300 s ≈ 0.3 h) + GPU correctness corpus (~0.5 h) + 3× rerun/debug factor ⇒ **≈ 6 GPU-hours** total — one dev-box afternoon. CPU cells run concurrently, off the GPU-hour budget. (vs profiling all 576 cells, which "would dominate the schedule," Codex #19.)
- **Confidence (no p-value theatre):** per cell report **median + IQR over K=5**. A cell **clears** the floor only if its **conservative bound** (`min` of the K reps) clears — never the median. `B*` is reported as the smallest B where CUDA-median < CPU-median **and** the two cells' IQRs do **not** overlap; if they overlap, report "`B*` within noise, unresolved" rather than a false crossover.

### 19.5 Replacement for T16 (plan:391)
> **T16** — Run the **staged** sweep (Stages 1→6, §19.2) with the frozen protocol (§19.3). Acceptance: gate (d) `F_loco_S0=1.4e8` cleared by best `(device,rung)` at near-term B for H1/H2 **or** a falsifier (F1–F7) explicitly tripped; `B*` located with non-overlapping-IQR rule; het-tax + masking-tax reported; profiler (winning + tripped cells only) shows force/solve-bound. **Budget ≤ 6 GPU-h.**

---

## Finding #20 — Risk register badly understates where failure occurs (reweight + honest reclassification)

Two of Codex's top-ranked "first failures" are **retired by the pre-decided layout change** and must be labelled as such; the rest are re-ranked to the top and the originally-"solved" items are honestly downgraded.

### 20.1 Retired-by-design (verify-only) — DROP from top ranks, note why
- **Deterministic ragged reduction (Codex #3, old rank #1).** The fixed `[B,S_max]`+mask layout makes every per-body reduction a **masked axis-sum** `(vals*mask).sum(dim=segment_axis)` — deterministic by construction, no atomics, no `scatter_add_`/`index_add_` over duplicate `body_id`. `numerics/reduce.py::segment_index_add` is **deleted**; there is no bespoke ragged reduction to prototype. Residual work = *verify* the masked-sum determinism test. **Was the #1 load-bearing unknown; now resolved.**
- **Flattened-churn breaks static capture (Codex #4, old rank #2).** Fixed `[B,S_max]` ⇒ the `S_max` axis is static; churn mutates `(W,N_cap)` *contents* via alive-mask + free-slot, never shapes; the CUDA graph is captured once (§19.3). **Was the #2 load-bearing unknown; now resolved.** Residual = verify graph-pointer stability across churn (T14).
- **f32 reservoir drift (Codex #5).** Reservoirs move to **f64** + explicit numerical-residual reservoir. Resolved-by-design (owned by the ledger cluster).

### 20.2 Reweighted register (genuinely-unresolved, load-bearing, ranked highest)

| Rank | Risk | L | I | Mitigation | Trigger |
|---|---|---|---|---|---|
| **1** | **Energy-gate correctness** — the *new* discrete semi-implicit balance (incl. `½vᵀ(M_{n+1}−M_n)v` added-mass term, actuator/constraint-impulse work, quadratic drag, wake loss; `p_in` incl. fin work) may itself be mis-derived; this is the S0 authorization gate | **High** | **High** | derive the discrete balance against the *actual* integrator; split per-force algebraic identities (hold by construction) from the discrete energy-balance test; validate on gain0 analytic single-step | (b) long test fails or algebraic identities disagree with discrete balance / Physics |
| **2** | **End-to-end ≠ kernel throughput** — S0's `1.4e8` kernel number may not predict `F_sci=2.31e7` whole-tick; fields/hashing/development could dominate | **High** | **High** | S0 scoped to kernel-feasibility only (§6.3); **G-E2E named gate** (§6.5) benchmarks the whole tick post-S2/S3 and emits measured `φ_loco`; ladder escalates on the profiler-named dominant subsystem | G-E2E miss, or measured `φ_loco>0.5` / Architecture |
| **3** | **gain1 oracle independence** — a modified-donor "re-recording" is not independent validation (Codex #8) | **Med-High** | **High** | freeze **untouched** donor for gain0 (byte-for-byte); derive **independent analytic** gain1 fixtures (closed-form ellipsoid mass, Lamb k-factors, single-step force/momentum); any donor seam narrowly reviewed w/ retained provenance, validated vs gain0 | gain1 fixture derivable only by editing the oracle / Physics |
| **4** | **S1 closure realizability** — S1 as specified has no `Bp→Bd` loss/grazing, so the required bloom **cannot crash** and drifters cannot plateau (Codex #7) | **Med-High** | **Med-High** | add explicit producer respiration/mortality `Bp→Bd/Nd` transfers to S1 (or defer those AC claims to S3); re-test `test_bloom_self_terminates` for an actual crash | S1 bloom fails to crash with no cap knob / Ecology |
| **5** | **S5 deterministic speciation** — research frontier; may be gated by ecology not encoding | **Med** | **High** | encoding solved; magic-trait + Kleiber/prune; root-cause absence to ecology, never retune | S5 split fails all setups / Evolution |
| **6** | **Sophia action interface unknown (RK-4)** | **Med** | **High** | contract continuous-first; symbolic decode Talos-side; gate on live Sophia code | S8 entry |
| 7 | Near-term single-world B underutilizes GPU (7.3 µs/step at B≈1024) | Med | Med | crossover sweep + rung ladder; seed-replicate worlds to fill batch; CPU-for-one-world is a legitimate GO | (d) miss at near-term B on all rungs / Architecture |
| 8 | Torch dispatch overhead at small batch (F3) | Med | Med | ladder r0→r1→r2; narrow GPU / CPU via `device=`; Warp→Rust | (d) miss or `B*`>real pop |
| 9 | Ragged heterogeneity tax on padded layout (~2.7× masking waste, F1) | Med | Med | accept masking tax at S0; flattened/arena is a *later measured* optimization; ablation quantifies it | H1/H2 masking tax > headroom |
| 10 | Speciation genome inflates unrewardable morphospace (RK-10) | Med | Med | Kleiber + prune | morphospace inflates w/o fitness |
| — | Deterministic reduction (Codex #3) | — | — | **RETIRED by masked-axis-sum layout** (§20.1) — verify-only | masked-sum determinism test |
| — | Churn/static-capture (Codex #4) | — | — | **RETIRED by fixed `[B,S_max]`** (§20.1) — verify-only | graph-pointer stable across churn (T14) |
| — | f32 conservation drift (Codex #5) | — | — | **RETIRED by f64 reservoirs** — verify-only | ledger residual scale test |

### 20.3 Honest reclassification of items the ORIGINAL register/milestones called "solved" (plan:749–759, 630–636)
- **M-S1 "solved (tuning-fragile)" → "mechanism-incomplete."** Downgrade: S1 cannot produce its own acceptance (bloom crash, drifter plateau) until producer-loss transfers exist (Risk #4). Not a tuning issue — a missing process.
- **M-S0a "solved (impl risk)" → conditional.** The authorization gate (energy, Risk #1) rests on a balance equation that must first be *derived correctly*; "solved" overstated it.
- **M-S0b "uncertain — load-bearing"** — keep, but note it now measures the *kernel* floor `1.4e8` (necessary condition); the *sufficient* condition is G-E2E (Risk #2), which the original register omitted entirely.
- **S3/S4 "🟩 solved" mechanisms → keep engineering-solved for the *mechanism*, but S4 *emergence* stays 🟦 frontier-tail and S1→S3 energy-closure inherits Risk #1's discrete-balance dependency.**
- **Removed the false comfort of "≥10× headroom":** replaced by the named `2.9×` composite (§6.3).

Net: the register now ranks **energy-gate correctness, end-to-end throughput, gain1 oracle independence, and S1 closure realizability** at the top (the genuinely load-bearing unknowns), explicitly marks the three layout/precision-retired risks as verify-only, and stops labelling S1 and the throughput authorization as "solved."

---

**Files/lines this cluster rewrites:** plan §2.0 (285), §2.5 gate-(d) (350), §2.7 sweep (367–369), §2.8 T16 (391), §6.2 milestones M-S0b/M-S1 (750–751), §6.3 risk register in full (764–780), §7 week-1 sweep (821–825); design §2.9 floor claim (272) reframed (retained only as S8 target). New artifacts added: gate **G-E2E** (§6.5), staged benchmark protocol (§19.2–19.3), derived floors `F_sci=2.31e7` (end-to-end) and `F_loco_S0=1.4e8` (S0 kernel).


---

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
