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
