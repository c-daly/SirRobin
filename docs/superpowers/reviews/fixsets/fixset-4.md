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
