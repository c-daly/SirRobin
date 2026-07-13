# SirRobin — Implementation Plan Rev. 4: Round-3 Corrections

**Status:** correction overlay · **Date:** 2026-07-12 · **Base:** Rev-3 reconciliation (`2026-07-12-sirrobin-plan-rev3-reconciliation.md`, §§0–10) · **Closes:** Codex round-3 review (`2026-07-12-codex-round3-of-rev3.md`).

Rev-4 closes the round-3 review, whose verdict was *NOT APPROVABLE to begin S0* with the residual failures "concentrated in the conservation/energy **math**." It does so by (a) **correcting the regularization sign** — the near-singular solve `(M+rI)Δv=P` implies `MΔv=P−r·Δv`, so the booked impulse is `J_reg = −r·Δv`, not the `+r·Δv` Rev-3 §7.3 wrote; (b) **adopting a two-currency conservation architecture** in which mass (mass-quanta) and energy (energy-quanta) each close in their own int64 ledger and are *never* summed into one total nor converted quantum-to-quantum inside a conservation equality — which **dissolves** the non-commensurate `3.581 energy-quanta-per-mass-quantum` problem at its root, because that number only ever arose from §6.4 summing a mass-derived chemical energy into the energy total; and (c) **closing the remaining mechanism gaps** (reserve inflow, source-availability caps, int64 overflow bound, death-transaction ordering), **RNG details** (a real continuation address with dedicated `attempt` bits, the corrected probability claim, the Box–Muller endpoint, the Tier-2 restatement), **oracle fixtures** (frozen inputs / expected / quadrature / tolerance via a standalone independent generator), and the **doc contradictions** the internal-consistency audit flagged.

**Rev-2 + Rev-3 + Rev-4 together are the final S0 spec.** Where Rev-4 deletes prior text, it says "**deleted**," not layered over — the deleted Rev-3 clause is removed from the codebase and schema, not shadowed.

---

## Reconciliation index

| Decision | Canonical fix | Closes (Codex findings) | Supersedes / deletes (Rev-3 §) |
|---|---|---|---|
| **D1** | `J_reg = −r·Δv`; repair `R_step`; **one** production solver, three per-body branches (`INVALID`/`REGULARIZED`/`EXACT`) dispatched by a single `where`/select; gates read the *actually booked* impulses | #2, #12, NP-3, B4; contradictions §7.1/§7.3, §3-dispatch | §7.3 sign (`= reg·dv_reg`); §7.1 "exact solve governs every gate"; §3 dispatch (masks only on `body_valid`); §6.2 `R_step` sign |
| **D2** | **Two independently-conserved currencies** — mass in mass-quanta, energy in energy-quanta — never summed, never cross-converted inside a conservation equality; `struct_N` is mass-only; `E_chem=e_N·struct_N` is a float diagnostic readout, never a reservoir; reserve inflow, source caps, overflow bound, death-order all specified | #5, #15, B1, NP-7, NP-8 + new (non-commensurate quanta, source caps, overflow, death-order); contradictions §1.3/§§6-7, §6.4 | §6.4 `Σ_stored` graph + `INV-ENERGY` bookkeeping; §1.1 `e_N` "quanta" note; §1.2/§5 `E_chem` energy row; §6.3 `ΔKE` partition; §7.3 "numerical-residual energy ledger" |
| **D3** | Re-budgeted 128-bit Philox counter with **dedicated `attempt` bits**; corrected `(1−p_accept)^N` claim; Box–Muller `u∈[ulp,1]` + `w0=2^32−1` upper-endpoint fixture; Tier-2 restated | #13 PARTIAL, NP-6 + new (Box–Muller endpoint, Tier-2 overstatement) | §4.5 "folded into the freed high bits" + `<2^-256 for any well-formed sampler`; §4.6 `w0∈{0,2^24-1}` test; §0/§4 Tier-2 wording |
| **D4** | Frozen gain1 fixtures: literal H1 input vectors, **32-point Gauss–Legendre** (not "e.g. Gauss–Kronrod"), tol `rel<1e-6` (f64), expected values from a **standalone independent** `tools/gain1_oracle.py` | #8 PARTIAL, B7 | §8.1 "e.g. Gauss–Kronrod in the fixture generator"; §8.2 unfrozen provenance rows |
| **D5** | Name the S3 genome mutation-rate symbols (placeholder defaults, *not load-bearing for S0/S1*); fix "mutation preserves `S_max`" → **`N_max`/`E_max`**; use the §4.3 enum names | #17 PARTIAL | Rev-2 §17 unnamed rates + `S_max` wording (reached via §4.3 enum) |
| **D6** | Feeding-stub **fixed op/tensor/byte spec**; throughput gate requires **≥1 non-OOM authorization-sized cell** (all-OOM ⇒ FAIL, not a vacuous pass) | #6 PARTIAL, #19 residual, #7 | §9.1 "deterministic fixed-cost placeholder with representative memory-traffic"; gate-(d) OOM handling |
| **D7** | §5: serialization is bit-preserving, resumed *execution* is not bit-identical (per §0 Tier-3 relaxed); pre-flight items are **S0 acceptance gates**, not pre-code preconditions | internal-consistency audit: §0/§5 replay, circular pre-flight | §5 "resumed from the snapshot alone is bit-identical"; pre-flight header "must be true before S0 code begins" |

---

## D1 — Regularization sign + one solver branch

**Supersedes / deletes in Rev-3.**

- **§7.3 (line 457) — sign is backwards, DELETED.** Rev-3 wrote
  `J_reg = [ (Mr00-M00)*dvx_reg + 0, (Mr22-M22)*dvz_reg ] = reg * dv_reg` and built *both* the momentum
  gate (§7.3 line 459) and the `R_step` energy gate (§6.2 line 379) on that `+reg·dv`. Deleted.
- **§7.1 (lines 427–428) — "exact governs every gate," DELETED.** Rev-3 wrote *"For every conservation
  gate (INV-CONSERVE, INV-ENERGY, `R_step`, momentum), the solve is the exact unregularized system,"*
  which directly contradicts §7.3 line 459 (*"Production `solve_constrained_xz` regularizes-and-ledgers"*).
  Codex: "Both can't govern the same step." Deleted; replaced by the single dispatch rule below.
- **§3 (lines 202–214) — dispatch bug, CORRECTED.** Rev-3 computes `degenerate` and `det_safe=1` but then
  masks the result **only** on `body_valid` (`dvx = where(body_valid, dvx, 0)`). A *valid-but-degenerate*
  body therefore keeps the raw `numerator/det_safe = numerator/1` result and **`Δv_reg` is never selected**.
  Corrected to an explicit three-way select.

**The algebra (the whole bug).** The near-singular solve is `(M+rI)Δv = P`, hence `M Δv = P − r·Δv`. The
booked regularization impulse is therefore

    J_reg = −r·Δv          [units: kg·m/s]        (NOT +r·Δv)

Momentum then closes **exactly**: `M Δv = P + J_c + J_reg`, since `P + (−r·Δv) = P − r·Δv = M Δv` and
`J_c` (vertical-plane reaction, `SwimEval.cs:807`) is workless (`v_y ≡ 0`). ✔

**One production solver, three per-body branches, single `select` dispatch.** `M` is the symmetric 2×2
constrained added-mass matrix `[[M00,M02],[M02,M22]]`; `P = F_stream·Dt` is `[B,2]`; all tensors are
`[B]`/`[B,2]` with `B = W·N_cap`.

```python
# branch predicates ([B] bool) — mutually exclusive, exhaustive
body_valid  = alive & (masked_segment_sum(seg_mass, seg_mask) > 0)     # §3 (unchanged)
degenerate  = det(M) < det_floor        # equivalently λ_min < λ_floor, i.e. κ > KAPPA_MAX (§7.2)
INVALID     = ~body_valid                                              # empty / zero-mass
REGULARIZED = body_valid & degenerate                                  # valid but near-singular
EXACT       = body_valid & ~degenerate                                 # the common path

# per-branch solves
Δv_exact = solve_2x2(M00, M02, M22, Px, Pz)                            # exact KKT
Δv_reg   = solve_2x2(M00+reg, M02, M22+reg, Px, Pz)                    # reg = EPS_SPD·tr  (§7.3 form kept)

# THE dispatch — Δv_reg is actually selected on the REGULARIZED branch (fixes §3)
Δv    = where(INVALID, 0.0,
          where(REGULARIZED, Δv_reg, Δv_exact))                        # [B,2]
J_reg = where(REGULARIZED, -reg * Δv_reg, 0.0)                         # [B,2]  ← the − sign
```

- `degenerate := det(M) < det_floor` (or `λ_min < λ_floor`) is computed with the **cancellation-free**
  `disc = sqrt((M00−M22)² + 4·M02²)`, `λ_max = ½(tr+disc)`, `λ_min = det/λ_max` form Rev-3 §7.2 already
  fixed — **kept verbatim**. `tr = M00+M22 > 0` for any valid SPD body; `tr=0` occurs only for an
  `INVALID` body, which this dispatch forces to `Δv=0` *before* any division (Rev-3 §7.2's masking claim
  now holds because the branch, not a post-hoc `body_valid` mask, governs selection).
- **The `INVALID` branch contributes nothing to any gate** (`Δv=0`, `J_reg=0`). For an empty body
  `M=0, P=0` ⇒ `Δv_exact=0/det_safe`; the `INVALID` select overrides it to exactly `0` — finite, no
  `1/0`, no `0·∞` (Rev-3 §3's intent, now actually wired).
- **REGULARIZED's work is accounted in the f32 `R_step` arm** per D2.6 (the `v_mid·J_reg` term, gate (b)) —
  **no int64 booking**; on the `EXACT` branch `J_reg=0 ⇒ W_reg=0`.

**Repaired `R_step` (§6.2, sign fixed).** The single KE-balance gate becomes, with `J_reg=−r·Δv`:

    R_step = ΔKE − v_mid·(F_stream·Dt + J_reg) − ½·vₙᵀ·ΔM·vₙ            [J]
           = ΔKE − v_mid·(F_stream·Dt − r·Δv) − ½·vₙᵀ·ΔM·vₙ  ≈ 0

The mandatory added-mass term `½·vₙᵀ·ΔM·vₙ` (Rev-3 §6.2) and workless `J_c` are unchanged; only the sign
carried into the `v_mid·J_reg` term is corrected. On the `EXACT` branch `J_reg=0` and `R_step≈0` by
construction; on the `REGULARIZED` branch the `−r·Δv` term makes it close *with* the artificial damping
present. Threshold unchanged: `|R_step|/max(KE_n,ε) < 1e-6` (f64) / `< 1e-3` (f32 hot), bounded-oscillating.

**Kill the §7.1-vs-§7.3 contradiction (one rule governs the step).** Production is the per-body branch
above — **not** "the exact unregularized system for every gate." The conservation gates (INV-CONSERVE,
momentum, `R_step`) are evaluated on the **actually booked** impulses `P + J_c + J_reg`, so they hold on
**both** the `EXACT` branch (`J_reg=0`) and the `REGULARIZED` branch (`J_reg=−r·Δv`). There is exactly one
production solve per step; there is no second "gate-only exact solve" that could disagree with it. The
gain0 donor-conformance path (`solve_sym3_donor`, `SwimEval.cs:1151-1166`, absolute `1e-12` floor) never
regularizes and is used only against untouched-donor fixtures (Rev-3 §7.3 / §8.2, unchanged).

**Gate.** `test_reg_momentum_closes` (force `κ>KAPPA_MAX` on a 10:1 slender body; assert
`M·Δv == P + J_c + J_reg` to f32 rel `<1e-5`, and that flipping the sign to `+r·Δv` **fails** it — proving
the sign is load-bearing); `test_reg_energy_in_rstep` (assert the f32 `R_step` closes with the `v_mid·J_reg`
term included and **fails if it is dropped** — no int64 booking involved); `test_solve_dispatch_selects_reg`
(**new**: a valid-degenerate body's output equals
`Δv_reg`, *not* `numerator/1` — the §3 select is exercised); `test_reg_off_on_wellcond`
(isotropic/H0: `reg=0`, production == exact solve, `max_abs(Δ)==0`).

**Closes:** #2, #12, NP-3, B4; internal-consistency contradictions §7.1/§7.3 and §3-degenerate-dispatch.

---

## D2 — Two-currency energy/mass conservation architecture

**Root decision — two independently-conserved currencies, never summed into one total.** Mass (mol N,
`q_mass = 1e-9`) and energy (J, `q_energy = 1e-3`) each close in **their own int64 quanta**. There is **no
mass→energy quantum conversion inside any conservation equality**, so the non-commensurate
`e_N/q_energy·q_mass = 3.581e6 · 1e-9 / 1e-3 = 3.581` energy-quanta-per-mass-quantum problem *cannot arise*
— it only ever arose because §6.4 summed a mass-derived chemical energy into the energy total. This
strengthens Law 2 (the books close) and Law 4 (single canonical representation: biomass has exactly **one**
representation — mass).

**Supersedes / deletes in Rev-3.**

- **§6.4 (line 404) — `Σ_stored` energy graph, DELETED.** Rev-3 wrote
  `Σ_stored = E_chem(Bp+Bd+Bm derived) + E_chem(struct_N derived) + E_reserve + E_KE`. This sums a
  mass-derived chemical energy (`e_N·struct_N`, a **float**) **and** `E_KE` (an f32 kernel quantity) into
  the energy total — the exact operation that (i) makes energy non-commensurate with mass and (ii) puts
  non-integer floats inside an "exact int64" equality. Deleted. Replaced by the two independent closures
  D2.2 (energy) and D2.1 (mass).
- **§6.4 (lines 410–411) — `INV-ENERGY (bookkeeping, EXACT int64)` over `Σ_stored_q`, DELETED** and
  replaced by D2.2's `Σ_energy_q` closure over native energy reservoirs only.
- **§1.1 (line 84) note "1 mol biomass = e_N=3.581e6 J = 3.581e9 quanta," DELETED** as a *conserved* claim
  — `e_N·struct_N` is a diagnostic readout (D2.1), not a booked energy reservoir.
- **§1.2 / §5 — the `E_chem` row in the energy reservoir table and the snapshot energy schema, DELETED**
  (D2.1).
- **§6.3 (line 394) `ΔE_KE = v_mid·F_stream·Dt + ½vᵀΔMv` "partition," DELETED** (D2.5): gate (b) is a
  force-law power-balance identity, not a `ΔKE` partition.
- **§7.3 (line 459) / §6.2 (line 379) "numerical-residual energy ledger," DELETED** (D2.6): regularization
  work is accounted in the f32 `R_step` arm (the `v_mid·J_reg` term); no int64 sink, no `R_num` resurrected.

### D2.1 — Structural biomass is mass, full stop

`struct_N` (per-creature, `[W,N_cap]`, ColonyState) is conserved in **mass quanta only** (int64,
`q_mass=1e-9` mol). Its chemical energy `E_chem = e_N·struct_N` (`e_N=3.581e6` J/mol) is a **derived float
diagnostic readout [J]** — NEVER a conserved energy reservoir, NEVER summed into `Σ_energy_q`, NEVER
assigned to a stored pool. `E_chem` is deleted from the energy reservoir table (Rev-3 §1.2), from the
snapshot energy schema (Rev-3 §5), and from `Σ_stored` (Rev-3 §6.4); it survives only as an observe-layer
readout `E_chem_creature = e_N·(struct_N_q · q_mass)`. This is the whole fix for #15's mint and #5/B1's
commensurability: **no float `e_N·struct_N` ever enters an int64 energy equality**, so there is no
`3.581`-quanta rounding to leak.

### D2.2 — Energy currency = reserve + source + sinks, all native energy quanta

Conserved **energy** reservoirs (all int64 energy-quanta, `q_energy=1e-3` J):

| reservoir | shape | dtype | role |
|---|---|---|---|
| `reserve` | `[W,N_cap]` (ColonyState) | int64 | per-creature metabolic reserve |
| `E_sun_cum` | `[W]` | int64 | environmental **source** meter (autotrophic fixation input) |
| `E_heat_cum` | `[W]` | int64 | dissipation **sink** (basal + muscle inefficiency + drag/wake heat) |
| `E_export_cum` | `[W]` | int64 | advective / other export **sink** |

Closure (the §1.5 `close_books` identity specialized to the energy currency, with `E_sun` declared
external via `declare_external`):

    Σ_energy_q  :=  Σ reserve  +  E_heat_cum  +  E_export_cum  −  E_sun_cum
    close_books(energy):  Σ_energy_q == I0_energy_q         # exact int64 ==, per world, every step

`I0_energy_q` is the **total initial reserve captured at `reset()`** (energy quanta) — the retained §1.5
general form `total == I0_q[currency] + ext`, specialized to the energy currency. It is **not** forced to
`0`: initial reserves are real seeded energy, not an external injection, so conservation checks that
current *stored + sinks − sources* equals what was present at `t=0`. (`declare_external(E_sun)` accounts
inflow *after* `t=0`; it does not re-seed the initial state.) Rev-3 §1's `transfer_quanta` (§1.3) and
`commit_flux` sub-quantum carry (§1.4) are **kept** — they are already exact *within a single currency*;
**every** energy transaction uses them. `E_KE` (f32) and `E_chem` (float) are **not** members of this sum
(D2.1, D2.5).

### D2.3 — Reserve inflow (fixes "reserve only drains")

Reserve gains energy from exactly two paths, both **exact energy-quanta** transfers; loss is one path:

- **Producers (autotrophic fixation):** `E_sun_cum → reserve`, rate set by the producer's captured
  environmental power `P_cap` [W]; a paired `declare_external(E_sun)` / `commit_flux(P_cap·dt, q_energy, …)`
  / `transfer_quanta`. This is the previously-missing inflow.
- **Consumers (predation/grazing):** two *separate* exact transactions in two *separate* currencies —
  prey `reserve → predator reserve` (**energy** ledger) **AND** prey `struct_N → predator struct_N` (or
  `→ Bd` detritus) (**mass** ledger). No mass↔energy conversion; structural energy is counted once (as
  mass, via D2.1's `E_chem` readout).
- **Loss:** the total metabolic spend `(P_basal + p_in/η)·dt` drains `reserve → E_heat_cum` (**energy**
  ledger, via `commit_flux`) — basal `P_basal = B0·M^α` plus the chemical cost `p_in/η` of producing
  mechanical output `p_in` at muscle efficiency `η`. At the metabolic timescale the entire spend is booked
  as dissipated (see D2.5: muscle mechanical output is accounted spent-to-environment; the transient
  kinetic term lives in the f32 arm, not this ledger).

### D2.4 — Building structure couples the two ledgers WITHOUT converting units

Growth of `Δstruct_N` mol (>0) fires **two paired transactions together**, one per currency:

```python
# ENERGY ledger: reserve → E_heat_cum  of magnitude  e_build · Δstruct_N   [J]
n_e, carry_reserve_heat = commit_flux(e_build * Δstruct_N, q_energy, carry_reserve_heat, grow_mask)
reserve_q, E_heat_cum_q = transfer_quanta(reserve_q, E_heat_cum_q, n_e, grow_mask)   # exact, energy-quanta

# MASS ledger: Nd → struct_N  of magnitude  Δstruct_N   [mol]
n_m, carry_Nd_struct    = commit_flux(Δstruct_N, q_mass, carry_Nd_struct, grow_mask)
Nd_q, struct_N_q        = transfer_quanta(Nd_q, struct_N_q, n_m, grow_mask)           # exact, mass-quanta
```

`e_build` (f64, J per mol) sets the **energy-side magnitude only**; it never appears in a cross-currency
equality (it multiplies a mol amount to yield a J amount that is then quantized in *energy* quanta). Both
ledgers close **independently**: energy loses `n_e` quanta from `reserve` to `E_heat_cum`; mass moves `n_m`
quanta from `Nd` to `struct_N`. There is no equation asserting `n_e` and `n_m` are related in quanta.

### D2.5 — Kinetic energy is the f32 mechanical arm, never an int64 reservoir (closes NP-8)

Two arms, one canonical representation of KE. D2.5 fixes **which identity gate (b) uses** and **what may
enter the int64 energy ledger**; it does **not** redefine `p_in` — the authoritative input-power identity
is **Rev-3 §6.1 verbatim** (the `tFin·U_cl` form Codex round-3 finding #1 explicitly accepted).

- **Gate (b) is the §6.1 force-law power balance** (f32/f64), **not** a `ΔKE` equation:

      U_cl = max(0, U)
      p_in = ( tReact·U + pWake ) + ( tFin·U_cl + pFin )                    # Rev-3 §6.1 line 365, unchanged
      gate (b):  | p_in − [ (tReact·U + pWake) + (tFin·U_cl + pFin) ] | / max(p_in, ε)  <  1e-6   (f32/f64)

  The abbreviated `tReact·U_cl + pWake + pFin` form in the decision spec is **superseded** by §6.1's
  canonical RHS: one `p_in` symbol, signed `U` on the reactive channel and clamped `U_cl` on the fin
  channel, used identically in the S0 gate, the ecology ledger, and the diagnostic (§6.1's single-symbol
  mandate). This is the one place the locked spec was abbreviated; §6.1 governs.

- **Kinetic energy has a single canonical representation — the f32 body velocity (Law 4) — and is NOT an
  int64 reservoir.** Storing `E_KE` in int64 *and* deriving it from velocity would be two synced copies of
  one quantity, the exact anti-pattern Law 4 forbids. So `E_KE` is computed from f32 state
  (`½vᵀM v`) and checked by the **f32 mechanical arm** (gate (b) / §6.2 `R_step`), never summed into the
  int64 `Σ_energy_q`. **Delete any claim that `E_KE` (f32) enters `Σ_energy_q`** (Rev-3 §6.4 line 404).

- **The int64 energy ledger (D2.2) is a *metabolic* ledger.** It conserves chemical/thermal energy across
  `{reserve, E_sun_cum, E_heat_cum, E_export_cum}`. Muscle mechanical output is booked as an **exact
  `reserve → E_heat_cum` debit at the metabolic timescale** — the full spend `(P_basal + p_in/η)·dt`
  (D2.3), i.e. locomotion chemical energy is accounted *dissipated*. This closes exactly per-step in int64
  (reserve loses `n_e` quanta, `E_heat_cum` gains the same `n_e`).

- **The two arms are consistent at gait-cycle scale, not forced to reconcile per-step.** The transient
  mechanical exchange (muscle work → KE → wake/drag heat within a stroke) lives entirely in the f32 arm;
  its net over a gait cycle is zero (`∮ ΔKE = 0`), so the metabolic "all-spend-to-heat" booking is exact at
  cycle scale. The per-step mechanical `ΔKE` is the f32 arm's **bounded-oscillating** term (gate (b) /
  §6.2 `R_step`, `τ = 1e-6/step`, drift `1e-4`, bounded-oscillating) — **never** an int64 leak, because it
  is never an int64 ledger member in the first place.

- **Delete §6.3 line-394's ledger reading** of `ΔE_KE = v_mid·F_stream·Dt + ½vᵀΔMv`: that identity is
  retained *only* as the f32 physics-consistency arm (`R_step`), not as a term added into `Σ_energy_q`. The
  COM-work vs tail-throughflow distinction is carried by §6.1/§6.2's `U_cl` and the explicit `(pWake+pFin)`
  wake term (§6.3 line 398), not by folding `ΔKE` into the int64 books.

**Scope note (not a defer — a phase fact):** S0/SpikeSwim exercises **only gate (b)** — the f32 mechanical
arm. The int64 metabolic energy ledger (`reserve`, `E_heat_cum`, …) first *exists* at S1 (Kleiber read,
Rev-3 §9/plan line 524) and closes end-to-end at S3 (`test_energy_loop_closes_e2e`). D2.5 fixes the
architecture now so no float ever enters the int64 books; the S3 gate proves the metabolic ledger closes
against a live economy.

### D2.6 — Regularization work is accounted in the f32 mechanical arm, not the int64 ledger

Rev-3 §1.3 (line 105) deletes `R_num`, but §§6–7 (§6.2 line 379, §7.3 line 459) then book regularization
work into a "numerical-residual energy ledger" that no longer exists in the table or schema. **Fix
(consistent with D2.5):** regularization modifies the **f32 mechanical solve**, so its work is accounted
**entirely in the f32 §6.2 `R_step` balance** — the `v_mid·J_reg` term — and is **NOT booked into any int64
reservoir.** The "numerical-residual energy ledger" is **deleted outright**, not redirected: there is no
int64 sink for it, because regularization never touches the metabolic ledger. This is the D2.5-consistent
resolution of the §1.3-vs-§§6-7 contradiction (an int64 `reserve → E_heat_cum` booking would have been a
second, int64 copy of an f32 mechanical quantity — the Law-4 anti-pattern D2.5 forbids).

- **Why no int64 booking.** The int64 energy ledger (D2.5) is metabolic; it sees only the muscle spend
  `(P_basal + p_in/η)·dt`, which is set by muscle activation and is **unchanged** by whether the 2×2 solve
  was regularized. Regularization alters only the body's f32 velocity response (an f32 KE detail), so its
  energy lives in the f32 arm.
- **Where it closes.** On the `REGULARIZED` branch, gate (b) / §6.2 `R_step` closes *because* it carries the
  `v_mid·J_reg` term:

      R_step = ΔKE − v_mid·(F_stream·Dt + J_reg) − ½·vₙᵀ·ΔM·vₙ  ≈ 0        (J_reg = −r·Δv)

  On the `EXACT` branch `J_reg = 0`, the term vanishes and `R_step ≈ 0` identically. Threshold unchanged
  (`|R_step|/max(KE_n, ε) < 1e-6` f64 / `1e-3` f32, bounded-oscillating).

**Work term — midpoint form (locked).** `W_reg = v_mid·J_reg` with `v_mid = ½(vₙ + vₙ₊₁)` — work is impulse
× mean velocity, the physically correct energy and the form §6.2's `R_step` uses (the decision spec's
literal `J_reg·Δv` is superseded). It appears in exactly one place — the f32 `R_step` term — so there is no
second copy and no int64 mirror to drift.

### D2.7 — Source-availability cap (a reservoir may never go negative)

`transfer_quanta` gains a hard precondition — it may never move more than the source holds:

```python
def transfer_quanta(src_q, dst_q, n, mask):          # all int64; n>=0
    n_eff = torch.minimum(torch.where(mask, n, 0), src_q)   # cap at availability; never overdraw
    return src_q - n_eff, dst_q + n_eff, (torch.where(mask, n, 0) - n_eff)  # shortfall returned
```

The shortfall `n − n_eff > 0` triggers the caller's defined failure path: **maintenance/locomotion that
cannot be paid from `reserve` marks the creature for starvation death this tick** (not a negative reserve).
A debug-gated assert `(src_q >= 0).all()` runs in `close_books`. This closes the new "transfer permits
`n > src` → negative reservoirs while 'conserving' the total" hazard.

### D2.8 — Death-transaction ordering (fixed, gated)

Fixed lifecycle order — egress precedes flag-clear so no mass/energy vanishes from `close_books`:

1. **Compute `die_mask`** (`[W,N_cap]` bool).
2. **While dying creatures are still ledger-live**, egress their **mass** (`struct_N → Bd`, exact
   mass-quanta) and **energy** (`reserve → E_heat_cum` / `E_sun_cum`-return, exact energy-quanta) via
   `transfer_quanta`.
3. **Only then** clear `alive_mask` and free the slots.

**Gate.** `test_cohort_death_conserves` (**new**): kill a whole cohort in one tick and assert **both**
ledgers `close_books() == 0` (exact) across the death step — catching the ordering hazard where clearing
`alive` before the `struct_N → Bd` transfer would drop mass from the sum.

### D2.9 — Overflow bound (checked, not asserted)

World-lifetime headroom: with `q_energy=1e-3`, signed int64 holds `2^63−1 ≈ 9.22e18` quanta ⇒ `9.2e15` J
per world. `close_books` asserts **no reservoir and no cumulative meter exceeds `2^62`** (leaves a 2×
headroom margin) for both currencies; the derived bound

    W · N_cap · reserve_max  +  E_sun_cum(t_max)   <   2^62          # energy
    W · N_cap · struct_N_max  +  Σ_abiotic_mass     <   2^62          # mass

is pinned in `config.py` and re-checked at every snapshot. Same construction for mass with `q_mass=1e-9`
(`9.2e9` mol/world). This closes the new "no overflow proof" gap.

**Master balance (open system) — two independent exact-integer closures replacing Rev-3 §6.4's
`INV-ENERGY`:**

    INV-MASS   (EXACT int64):  Σ struct_N + Σ Bp + Σ Nd + Σ Bd + Σ Sed == I0_mass_q
                               # ALL mass reservoirs (Rev-3 §1.2): struct_N (ColonyState) + Bp (derived
                               # producer-biomass masked sum) + Nd, Bd, Sed (abiotic). No float member.
    INV-ENERGY (EXACT int64):  Σ reserve + E_heat_cum + E_export_cum − E_sun_cum == I0_energy_q
                               # metabolic reservoirs only (D2.5). E_KE, E_chem are NOT members.
    (physics-consistency arm, f32 tolerance, unchanged §1.5): the f32 E_KE tracks §6.2's R_step to
       τ_E_phys=1e-6/step, 1e-4 drift, bounded-oscillating — a fidelity check, never a conservation leak.

**Gates.** `test_reserve_inflow_partitions` (grazing/fixation cycle: reserve strictly can *increase*; no
mint; `Σ_energy_q` and `Σ_mass_q` both exact-close); `test_struct_energy_counted_once` (grazing→death
cycle: `E_chem` never assigned to a stored pool; both ledgers exact-close); `test_cohort_death_conserves`
(D2.8); `test_transfer_no_negative` (D2.7 cap; assert `(src_q>=0).all()`); `test_overflow_bound`
(D2.9 `<2^62` assertion fires on a synthetic over-cap world); `test_energy_no_float_in_ledger` (AST:
neither `E_KE` nor `e_N·struct_N` is ever added into `Σ_energy_q`).

**Closes:** #5, #15, B1, NP-7, NP-8; new mechanism gaps (non-commensurate quanta [dissolved by currency
separation], source caps [D2.7], overflow [D2.9], death-order [D2.8]); internal-consistency contradictions
§1.3/§§6-7 [D2.6] and §6.4 non-integral energy [D2.1/D2.5].

---

## D3 — RNG rejection-continuation + Box–Muller + Tier-2 statement

**Supersedes / deletes in Rev-3.**

- **§4.5 (line 305) — "an `attempt` counter is folded into the freed high bits," DELETED.** Rev-3's 7-field
  address (§4.2) already uses all 128 counter bits (`16+36+40+20+8+8 = 128`); there are **no** freed high
  bits, so the rejection-continuation had nowhere to put `attempt`. Re-budgeted below.
- **§4.5 (line 305) — "probability `< 2^-256` for any well-formed sampler," CORRECTED** to the true
  `(1−p_accept)^N` bound with a stated per-sampler `p_accept` floor.
- **§4.6 (line 309) — endpoint test `w0 ∈ {0, 2^24-1}`, CORRECTED** to `{0, 2^32−1}`: the raw word that
  drives `u1 → 1` is `2^32−1`, not `2^24−1`.
- **§0 / §4 — Tier-2 wording, RESTATED** to match Codex's accepted caveat (float Box–Muller is not itself
  bit-exact under relaxed float).

### D3.1 — Real continuation address (dedicated `attempt` bits)

Re-budget the 128-bit Philox-4x32 counter so the rejection `attempt` index has **dedicated bits**.
Canonical layout (sums to exactly 128, includes `attempt`):

| field | bits | note |
|---|---|---|
| `world_id` | 16 | ≤65536 worlds |
| `entity_id` | 32 | per-world stable id |
| `purpose` | 8 | event_kind (§4.3 enum) |
| `element` | 24 | gene/innovation + draw index |
| `attempt` | 8 | **dedicated** rejection-continuation index → ≤256 tries |
| `step_lo` | 40 | tick counter (1.1e12 ticks) |

`16 + 32 + 8 + 24 + 8 + 40 = 128` — exactly the four 32-bit counter words. This **re-budgets** (supersedes)
the §4.2 table, which spent all 128 bits without an `attempt` field. `attempt` gets ≥8 dedicated bits
(widths of `step_lo`/`element` may be traded as long as `attempt ≥ 8`). `_assert_field_widths` raises on
overflow of any field. If `attempt` exhausts (see D3.2), the sampler **clamps to a defined valid value and
logs** — never undefined behaviour, never a re-key into "freed" bits that do not exist.

**Field mapping to §4.2 (locked).** The renamed fields correspond to §4.2's exactly: `purpose ≡
event_kind`; `element` subsumes §4.2's `gene_iid + draw_index` (one 24-bit element index over the
per-purpose draw space); `entity_id ≡ stable_entity_id`; `step_lo ≡ step`. This **re-slices** §4.2's
128-bit partition to carve out the 8-bit `attempt` field (taken from the previously over-wide
`element`/`step` budget); the only invariants that must hold are **(i)** total = 128 bits, **(ii)** the
map from `(world_id, entity_id, purpose, element, attempt, step_lo)` to counter is injective, and
**(iii)** `attempt ≥ 8`. `_assert_field_widths` enforces (i) and (iii); `test_prf_reference_vectors`
(regenerated for the new partition) pins (ii).

### D3.2 — Correct the probability claim

Rejection failure after `N` tries is exactly `(1 − p_accept)^N`, given a **stated per-sampler `p_accept`
floor**. For uniform-in-range acceptance, `p_accept ≥ 0.5`, so the `attempt:8` budget (`N=256`) gives
failure `≤ 0.5^256 < 2^-256`. Replace the false blanket "any well-formed sampler `< 2^-256`" with this
per-sampler statement. S0–S3 samplers (Gaussian via Box–Muller, Bernoulli, categorical-by-inversion) never
reject, so the continuation is defense-in-depth.

### D3.3 — Box–Muller `log(0)` + corrected endpoint fixture

Map the radius uniform to `u ∈ [ulp, 1]` (**exclude 0**) so `log(u)` is finite:
`u1 = ((w0 >> 8) & 0xFFFFFF + 1) · 2^-24 ∈ [2^-24, 1]`, then `r = sqrt(−2·ln(u1))` (finite; `=0` only at
`u1=1`), `z0 = r·cos(2π u2)`, `z1 = r·sin(2π u2)`.

**Endpoint fixture (corrected).** The upper endpoint `u1 → 1` requires the top 24 bits of the raw word to
be all ones, i.e. the raw word `w0 = 2^32−1` (`0xFFFFFFFF`) ⇒ `(w0>>8) & 0xFFFFFF = 0xFFFFFF` ⇒ `u1 = 1`.
Rev-3 §4.6's `w0 = 2^24−1` (`0x00FFFFFF`) gives `(w0>>8) & 0xFFFFFF = 0x00FFFF` ⇒ `u1 ≈ 0.0039` — it never
reaches the endpoint, so the test never exercised `r=0`.

**Gate (corrected).** `test_boxmuller_no_log0`: sweep raw `w0 ∈ {0x00000000, 0xFFFFFFFF}` ⇒
`u1 ∈ {2^-24, 1}`; assert `z0, z1` finite at both endpoints and `r == 0` exactly at `u1 = 1`.

### D3.4 — Restate Tier-2 (determinism)

**Tier-2 = reproducible raw integer draws + reproducible discrete decisions.** The discrete decisions —
mutation-fires-or-not, mate choice, death roll — are **threshold comparisons on integer / uniform draws**
and reproduce bit-exactly for a given `(seed, world_id, …)` key. **Continuous mutation *magnitudes*** via
Box–Muller use float `log/sqrt/sin/cos` and are reproducible **only modulo float** — explicitly **NOT** a
bit-exact gate under §0's relaxed posture. The reference-vector gate (§4.4, `test_prf_reference_vectors`)
covers the **integer Philox core only**. Evolution depends on discrete decisions + statistical
distributions (both reproduced); exact Gaussian magnitudes are not required and not gated. §4's wording is
aligned to §0 accordingly (Codex's accepted caveat).

**Gates.** `test_prf_reference_vectors` (integer core, unchanged); `test_boxmuller_no_log0` (D3.3);
`test_rejection_attempt_bits` (**new/renamed**: force ≥256 rejects, assert the dedicated `attempt` field
advances and the stream stays deterministic across process restart, and that exhaustion clamps+logs rather
than aliasing).

**Closes:** #13 PARTIAL, NP-6; new Box–Muller endpoint (§4.6); Tier-2 overstatement (§0/§4).

---

## D4 — gain1 oracle fixtures frozen

**Supersedes / deletes in Rev-3.** §8.1 (line 479) — "Evaluate the integrals to f64 by an *independent*
high-order quadrature (**e.g. Gauss–Kronrod** in the fixture generator)" — is **replaced** by a fully
specified quadrature; "e.g." is deleted. §8.2's provenance table is **frozen** with literal values.

Freeze, **in the doc**:

- **(a) Explicit input vectors.** A named canonical body **H1** with its literal segment states —
  positions, orientations (quaternions), and velocities — written as numeric constants (semi-axes
  `(a,b,c)`, pose, `(U, V_t, s, m_t)`), committed to the fixture file. No "regenerate from a body."
- **(b) Quadrature fully specified.** **32-point Gauss–Legendre** on the Lamb added-mass integrals
  (`α0, β0, γ0` per Rev-3 §8.1), with the 32 nodes/weights taken from a **cited source** (Abramowitz &
  Stegun Table 25.4 / a pinned NumPy `numpy.polynomial.legendre.leggauss(32)` call). **Not** "e.g.
  Gauss–Kronrod." The infinite `∫₀^∞` is mapped to `[0,1]` by `λ = (1−t)/t` before the 32-point rule.
- **(c) Tolerance.** `rel < 1e-6` (f64) for the fixture-vs-analytic match; the kernel-vs-fixture match
  stays `rel < 1e-4` (f32), Rev-3 §8.2.
- **(d) Expected values from a standalone independent generator.** `tools/gain1_oracle.py` implements the
  analytic Lamb integral via the (b) quadrature and the closed-form Lighthill/Garrick single-step forces
  (Rev-3 §8.1) — importing **nothing** from the donor `SwimEval` and **nothing** from the torch port. It is
  run **once**; its outputs are committed to the fixture file. It is the **independent second
  implementation** that makes the oracle non-circular.

This is a **T11 deliverable**. `tools/gain1_oracle.py` is the independent arm; the kernel is the arm under
test; agreement to `rel<1e-4` is corroboration, not tautology.

**Gates.** `test_oracle_gain1_analytic` (kernel reproduces each committed fixture value to f32 rel
`<1e-4`); `test_gain1_fixtures_donor_free` (AST/import audit: the generator imports nothing from the donor
or torch port); `test_no_patched_donor` (Rev-3 §8.2, unchanged); `test_gain1_quadrature_pinned`
(**new**: the 32 Gauss–Legendre nodes/weights match the cited source bit-for-bit).

**Closes:** #8 PARTIAL, B7.

---

## D5 — #17 residuals

**Supersedes / deletes in Rev-3.** The two untouched round-2 #17 items (mutation rates unnamed; "genotype
mutation preserves `S_max`" wording) reach the plan through Rev-3 §4.3 (which reconciled the `event_kind`
enum but left the rate symbols and the `S_max` wording from Rev-2 §17 unfixed).

**D5.1 — Name the genome mutation-rate symbols** as S3 config symbols with placeholder defaults, each
marked *"tuned at the S3 gate, not load-bearing for S0/S1"*:

    p_add_node, p_add_edge, p_del_edge, p_perturb_weight, σ_weight, p_mut_scalar, σ_scalar   # S3 Config

They are declared in `Config` (S3 block) so the plan names them, but no S0/S1 gate depends on their values.

**D5.2 — Fix the wording.** Genotype mutation preserves **`N_max` / `E_max`** (the genome node/edge
capacity), **NOT `S_max`** (the developed-body segment cap, `S_max=16`, Rev-3 §2). The two are distinct:
`N_max/E_max` bound the *genome* pool `[P, N_max/E_max]`; `S_max` bounds the *developed body*. Every
mutation operator uses the canonical §4.3 event-kind enum names (`PARAM_MUT`, `STRUCT_ADD`,
`STRUCT_ADD_EDGE`, `STRUCT_DELETE`, `STRUCT_TOGGLE`) — no operator uses a name outside that enum.

**Gate.** `test_mutation_preserves_genome_capacity` (**new**: after any mutation op, `N_max`/`E_max` are
invariant and `S_max` is never referenced as the preserved bound); `test_mutation_rates_named` (AST: the
seven symbols exist in `Config` and are flagged S3-tunable).

**Closes:** #17 PARTIAL.

---

## D6 — Feeding stub spec + non-vacuous throughput gate

**Supersedes / deletes in Rev-3.** §9.1 (line 503) — "a deterministic fixed-cost placeholder with
representative memory-traffic" — is **replaced** by a fixed op/tensor/byte spec. The gate-(d) OOM handling
(§9.4) is **strengthened** so an all-OOM sweep FAILs rather than vacuously passes.

**D6.1 — Fixed op/tensor/byte spec for the S2 feeding-throughput stub.** For each body (`B = W·N_cap`):

1. Gather each body's `K` nearest neighbours' `struct_N` via the spatial hash: `neigh_idx` `[B,K]` int64
   → `torch.gather` → `neigh_struct_N` `[B,K]` int64.
2. Mask by encounter radius: `enc_mask` `[B,K]` bool (`dist2 < r_enc²`).
3. Reduce: `segment_sum(neigh_struct_N · enc_mask, dim=1)` → `intake` `[B]` int64.

Per-cell byte traffic (the memory pattern S3 feeding will use):

    bytes = B·K·(8 + 1)  +  B·8
            └ gather (int64) + mask (bool) ┘   └ [B] result ┘

This benchmarks the exact gather/mask/segment-sum memory pattern of real S3 assimilation without computing
assimilation. It replaces the vague "representative memory-traffic."

**D6.2 — Non-vacuous throughput gate.** The throughput gate must **require ≥1 non-OOM
authorization-sized cell** to produce a valid measurement. An **all-OOM sweep is a FAIL**, not a vacuous
pass — a run that OOM-skips every cell has measured nothing and cannot authorize the `F_loco_S0 = 9.0e7`
floor (Rev-3 §9.2). State it in the gate: `assert n_valid_cells ≥ 1` before emitting `φ_loco`.

**Gates.** `test_feeding_stub_shapes` (**new**: the stub emits the `[B,K]` int64 gather, `[B,K]` bool mask,
`[B]` int64 reduction with the stated `bytes` formula); `test_throughput_requires_nonoom` (**new**: an
all-OOM sweep raises/FAILs; a sweep with ≥1 valid cell passes).

**Closes:** #6 PARTIAL, #19 residual, #7.

---

## D7 — Doc contradictions (internal-consistency audit)

**D7.1 — §0 vs §5 replay.** Rev-3 §5 (line 347) wrote *"a run resumed from the snapshot alone is
bit-identical,"* which contradicts §0's Tier-3 relaxation (§0 lines 28–39: *"bit-for-bit float replay of a
resumed long run is not guaranteed and not gated"*). **DELETE** the "forcing-completeness ⇒ resumed run
bit-identical" claim. **Restate §5:**

> Snapshot **serialization** is bit-preserving — a lossless save/load round-trip of the exact int64 + float
> state (`test_snapshot_roundtrip_bit_identical` stays a save/load correctness test). Resumed **execution**
> is **NOT** bit-identical over time (per §0 Tier-3, relaxed) — only **statistical reproducibility** is
> guaranteed (conservation holds; discrete decisions and distributions reproduce). Forcing-completeness
> (`forcing` in the schema; the negative gate `test_colonystate_alone_is_insufficient`) guarantees a
> resumed run is **valid**, not that its float trajectory is bit-identical.

**D7.2 — Circular pre-flight checklist.** Rev-3's "Updated S0 pre-flight checklist (**must be true before
S0 code begins**)" (line 615) lists items whose gates *require* S0 code (e.g. `test_transfer_exact_int`,
`test_prf_reference_vectors`) — Codex: "They are S0 *acceptance* gates, not pre-code gates." **Reframe** the
ten checklist items as **S0 acceptance gates** — green as part of *completing* S0, gating S0 **done**, not
S0 **start**. The header changes from "must be true before S0 code begins" to "must be green to
**complete** S0"; line 630's "S0 code may begin" becomes "the S0 spec is internally consistent and **S0 is
accepted as done**."

**Gate.** `test_docs_no_replay_contradiction` (**doc-lint**: no "resumed … bit-identical" string survives
in §5); the pre-flight section is labeled *acceptance gates* (reviewer checklist item, not an automated
test).

**Closes:** internal-consistency audit — §0/§5 replay [D7.1], circular pre-flight [D7.2].

---

## Coverage check — every round-3 required change → the Rev-4 decision that closes it

| Codex round-3 required change | Rev-4 decision |
|---|---|
| 1. `J_reg = −reg·dv`, repair `R_step`, one solver branch (exact / regularized / invalid-zero) | **D1** |
| 2. Redesign exact energy: commensurate/conversion; explicit `E_KE` + reg-work; overflow; source caps; death order | **D2** (currency separation ⇒ no conversion; `E_KE` D2.5; overflow D2.9; caps D2.7; death order D2.8) |
| 3. Non-minting feeding partition replenishing reserve; structural energy counted once | **D2.1 / D2.3** |
| 4. Real rejection-continuation address + corrected test/probability | **D3.1 / D3.2** |
| 5. Frozen gain1 inputs / expected / quadrature / tolerance | **D4** |
| 6. Two untouched #17 defects (mutation rates; `S_max`→`N_max/E_max`) | **D5** |
| 7. Feeding-stub op/tensor/byte spec + non-OOM measurement | **D6** |

| Internal-consistency contradiction | Rev-4 decision |
|---|---|
| §7.1 vs §7.3 (which solve is production) | **D1** |
| §1.3 vs §§6–7 (reg-work into a deleted "numerical-residual ledger") | **D2.6** (→ f32 `R_step` arm; no int64 sink) |
| §6.4 non-integral energy (`E_chem`/`E_KE` in `Σ_stored`) | **D2.1 / D2.5** |
| §0 vs §5 (resumed run "bit-identical") | **D7.1** |
| Circular pre-flight ("before S0 code begins") | **D7.2** |
| §3 degenerate dispatch (`Δv_reg` never selected) | **D1** |
| Box–Muller endpoint (`2^24−1` vs `2^32−1`) | **D3.3** |

Every one of Codex's 7 required changes and every consistency-audit contradiction is closed above. Rev-4
introduces no new cross-currency conversion, books no float (`E_KE`, `E_chem`) into the int64 energy total,
resurrects no `R_num`, and carries no surviving §7.1/§7.3, §1.3/§§6-7, §6.4, or §0/§5 contradiction. The
three textual tensions the drafting pass flagged between the locked spec and retained Rev-3 text are now
**resolved in-doc**, each toward the retained Rev-3 canon or the physically correct form: **(1)** gate-(b)
RHS uses §6.1's canonical `p_in = (tReact·U + pWake) + (tFin·U_cl + pFin)` (the spec's abbreviated form
superseded, D2.5); **(2)** regularization work uses the midpoint `W_reg = v_mid·J_reg` matching §6.2's
`R_step` (D2.6); **(3)** the energy closure uses the general `Σ_energy_q == I0_energy_q` with `I0` the
initial reserve captured at `reset()` (D2.2). A fourth (D3.1 counter field-names) is resolved by an
explicit injective re-slice of §4.2's 128 bits. No `AUTHOR-NOTE` remains open.
