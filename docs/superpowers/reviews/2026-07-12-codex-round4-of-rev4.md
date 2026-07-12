# Codex — Round 4 Review of Rev-4

**Date:** 2026-07-12 · Codex `gpt-5.6-sol`, high reasoning, read-only. (Full trace: `codex-rev4-raw.txt`.)
**Overall verdict: NOT APPROVABLE to begin S0 yet — but "close enough to justify continuing preparatory
scaffold work."** Crucially, round 4 **separates real S0 blockers (5, small, mostly mechanical/code) from
S1/S3-phase findings (the whole energy-economy pile).**

## The one deep, correct critique (D2)

Codex's central finding is **right and worth internalizing**: the int64 energy ledger I built is *"an
exactly balanced expenditure ledger, not conservation of the simulator's total physical energy."* The
"metabolic ledger closes exactly, KE is f32-only, consistent at gait-cycle scale" argument is **conditional**
— it holds only if the gait is truly periodic (`∮ΔKE=0`), all mechanical work dissipates locally, and there's
no net acceleration/export/transient. Locomotion doesn't guarantee those, so there's a real per-step physical
gap; excluding KE from the equality and calling it "never an int64 leak" is *category-switching* (the quantity
is omitted, not conserved). Also: booking the **full** `p_in/η` as heat overstates transient heat — the
correct immediate-heat partition is `P_basal + (1/η−1)·p_in`, with the remaining `p_in` being mechanical work
(Rev-2's partition was right).

**The honest reframe Codex hands us (rec #4):** *"Reframe D2 as a metabolic-expenditure ledger, or restore a
physically complete hybrid energy balance. Do not call the current reduced sum total energy conservation."*
→ **Mass** gets exact-int64 *conservation*. **Energy** gets (a) an exact-int64 metabolic-**expenditure**
ledger (accounting; closes tautologically) **plus** (b) a **total-energy** invariant at f32 tolerance that
includes KE. Same hybrid the plan already uses for mass-exact vs physics-tolerance — applied honestly to
energy. The two-currency *direction survived*; only the "energy is exact-int64 conservation" overclaim dies.

## The seven required changes (round-4 scores)

1. **PARTIAL — solver.** Sign now correct (`J_reg=−rΔv`), but: dispatch predicate must be exactly
   `κ>KAPPA_MAX` (not `det<det_floor`, "not generally equivalent"); both eager branch solves need safe
   denominators (`where` doesn't short-circuit → the un-selected singular branch NaNs first); the 2-D x/z
   momentum eq wrongly adds the **vertical** `J_c` (dimensionally wrong — `J_c` is a separate 3-D statement).
2. **STILL-BROKEN — energy.** The D2 expenditure-vs-conservation issue above; plus retained S1/S3 flux/tests
   still use float `E_chem`/`E_KE` (Rev-2 `:899-923`) that Rev-4 didn't cleanly replace.
3. **STILL-BROKEN — feeding/reserve.** Grazing a producer `Bp` (mass-only, no reserve pool) gives the grazer
   no reserve inflow — only creature-on-creature predation transfers reserve. The field-biomass→consumer-
   reserve path is still absent (an S3 energy-economy design task).
4. **PARTIAL — RNG.** Dedicated `attempt` bits exist, but `gene_iid+draw_index→element` packing unspecified;
   Box–Muller has a **precedence bug** (`((w0>>8)&0xFFFFFF)+1` needs the inner parens); probability should be
   `≤2^-256` not `<` (at `p_accept=0.5` it's exactly `2^-256`).
5. **STILL-BROKEN — gain1 fixtures.** Method specified but literal inputs + expected values not committed
   (still a T11 deliverable). **Real math bug:** the `λ=(1−t)/t` map omits the `1/t²` Jacobian.
6. **RESOLVED — #17.** Seven mutation symbols named; `S_max`→`N_max/E_max` fixed.
7. **PARTIAL — feeding stub.** All-OOM fails correctly, but byte model omits `neigh_idx`/`dist2` reads,
   `dist2` dtype unspecified, `K` unpinned.

## Internal-consistency audit: still FAIL (but the surviving items are nameable)

- Gate (b) has two meanings — D2.5 calls the **force-law algebraic identity** "gate (b)," but Rev-3's gate (b)
  is `R_step`. (Fix: gate (b) = `R_step`; the §6.1 force-law identities are the separate **gate (a)** subtests.
  My relabeling introduced this.)
- 2-D `M`/`P` vs vertical `J_c` (finding #1).
- Degeneracy predicate `det<det_floor` vs retained `κ>KAPPA_MAX`.
- `where` doesn't short-circuit → INVALID branch isn't NaN-safe as claimed.
- Retained S1 biomass-energy processes (`E_chem(Bp/Bd/Bm)`, Rev-2 `:862-893`) have no D2 representation.
- Mass inventory omits **`Bm`** (remineralized biomass, `v1:408,414,443-445`) — I added `Bp`, missed `Bm`.

## New Rev-4 problems
- Box–Muller precedence (above). · `p_accept=0.5` gives exactly `2^-256`. · `max(p_in,ε)` unsafe when signed
  reactive power makes `p_in<0` → use `max(|p_in|,ε)`. · gain1 Jacobian omission (above). · feeding-stub byte
  count incomplete.
- Transaction defects (all S1/S3): `E_sun_cum` is a cumulative **meter**, not a source reservoir — can't
  "transfer out of" it (external input credits reserve *and* increments the meter); death `reserve→E_sun-
  return` ambiguous; `transfer_quanta` is elementwise but `[W,N_cap] reserve → [W] heat` needs a grouped
  reduction; growth's two ledgers can cap **independently** (not atomic); per-value `<2^62` check doesn't
  bound the multi-term **sum** (need aggregate bound / wider accumulator).

## Phase split (Codex's own, verbatim in substance)

**Real S0 / SpikeSwim blockers (5):**
1. Unify gate (b) = `R_step`; keep force-law identities as separate algebraic (a) subtests.
2. Executable solver: one `κ>KAPPA_MAX` predicate, safe denominators for the eager branches, 2-D `MΔv=P+J_reg`
   separated from the 3-D vertical `J_c`.
3. Commit literal H1 gain1 fixture inputs + expected values.
4. Correct the gain1 quadrature mapping/Jacobian.
5. Preserve the non-OOM S0 authorization requirement (already done — keep).

**Later-phase (S1/S3) blockers — do NOT gate the standalone locomotion kernel:** D2's biomass/reserve/heat
architecture, missing `Bm`, grazing reserve source, atomic growth caps, death semantics, grouped energy
transfers, overflow-sum proof; RNG mutation details (S3, though fix the Box–Muller/parser bug when the RNG
scaffold lands); feeding stub (gates provisional G-E2E at S2→S3). *"S0 has no ecology, genome mutation, or
steering (`v1:277`), and scaffold conservation runs on a fake reservoir (`v1:25`). D2 therefore should not
block writing unrelated S0 code—but the live S0 solver, gate, and oracle contradictions do."*

## Closing line (verbatim)
> "Rev-4 is close enough to justify continuing preparatory scaffold work, but not yet internally coherent
> enough to authorize the SpikeSwim implementation plan as written."
