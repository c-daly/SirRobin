# Codex — Round 3 Review of Rev-3 (reconciled + determinism-relaxed)

**Date:** 2026-07-12 · Codex `gpt-5.6-sol`, high reasoning, read-only. (Full trace: `codex-rev3-raw.txt`.)
**Overall verdict: NOT APPROVABLE to begin S0** — but the reconciliation and the determinism relaxation both succeeded; the remaining failures are concentrated in the conservation/energy **math**.

## What round 3 confirms worked

- **The reconciliation succeeded.** The cross-fix-set contradictions from round 2 are gone: **#4 layout RESOLVED, #9 empty-indexing RESOLVED, #18 firewall RESOLVED, #19 benchmark RESOLVED, #20 risk register RESOLVED**; new-problems **NP-1, NP-2, NP-4, NP-5, NP-9, NP-10, NP-11 RESOLVED**; blockers **B2, B3, B6, B8 RESOLVED.**
- **The determinism relaxation is accepted.** Codex, verbatim: *"The relaxation itself does not break either retained gate… atomic or reassociated float physics cannot alter an already paired int64 debit/credit… Philox's integer core… remains independent of float-kernel scheduling."* One caveat: Tier-2 must be stated as *reproducible raw draws / discrete decisions* — the float Box–Muller transform (`log/sqrt/sin/cos`) is not itself bit-exact under relaxed float.

Round-2-residual score: **5 RESOLVED · 6 PARTIAL · 4 STILL-BROKEN.**

## What's still broken — all in the two retained hard gates (conservation + mechanics)

1. **Regularization sign error (STILL-BROKEN; #2, #12, NP-3, B4).** From `(M+rI)Δv=P` ⇒ `MΔv=P−r·Δv`, so the booked impulse must be `J_reg = −r·Δv`; Rev-3 wrote `+r·Δv` and built both the momentum and the `R_step` energy gate on it. Elementary, but it makes both gates false whenever regularization fires.
2. **Energy quanta are non-commensurate (STILL-BROKEN; #5, #15, B1).** With mass quantum `1e-9` and energy quantum `1e-3` and energy density `e_N`, one mass quantum ≈ **3.581** energy quanta — not an integer — so structural chemical energy cannot enter an *exact* int64 energy equality. Exact-integer conservation works cleanly for a single currency but breaks at the mass↔energy conversion.
3. **Reserve economy has no inflow (STILL-BROKEN; #15).** Rev-3 removed the grazing reserve-credit to stop the double-mint but supplied no replacement partition, so reserve only decreases — no valid feeding path.
4. **Conservation-mechanism gaps (B1, new problems).** `transfer_quanta` permits `n > src` → negative reservoirs while "conserving" the total (needs an availability cap); no int64 overflow bound; death-order hazard (if `alive` is cleared before `struct_N→Bd`, mass vanishes from `close_books` — transaction ordering unspecified).
5. **RNG detail bugs (PARTIAL; #13, NP-6).** The rejection-continuation has no free counter bits for the `attempt` index; the Box–Muller endpoint test uses `2^24−1` where the upper endpoint needs `2^32−1`.
6. **Two untouched round-2 items (#17):** mutation rates still unnamed; the "genotype mutation preserves `S_max`" wording (should be `N_max/E_max`) unfixed.
7. **gain1 fixtures (PARTIAL; #8, B7):** analytic method given, but no frozen input vectors / expected values / specified quadrature.

## Internal-consistency audit: FAIL (4 real contradictions)

- **§7.1 vs §7.3:** §7.1 says production uses the exact *unregularized* solve for every conservation gate; §7.3 says production regularizes. Both can't govern the same step.
- **§1.3 vs §§6–7:** §1.3 deletes `R_num` and its residual mechanism; §§6–7 then book regularization work into a "numerical-residual energy ledger" that no longer exists in the table/schema.
- **§0 vs §5:** §0 disclaims long-run float replay; §5 still says forcing-completeness makes resumed execution *bit-identical*. (A leftover from before the §0 relaxation — needs "serialization is bit-preserving; execution replay is not.")
- **Pre-flight checklist is circular:** it says implementation tests must be green "before S0 code begins," but those tests require S0 code. They are S0 *acceptance* gates, not pre-code gates.

## Codex's required changes before approval

1. `J_reg = −r·Δv`; repair `R_step`; publish one solver branch selecting exact / regularized / invalid-zero.
2. Redesign exact energy representation: commensurate quanta or an integer conversion scheme; explicit `E_KE` + regularization-work treatment; checked overflow; source caps; death-transaction ordering.
3. Specify a non-minting feeding partition that replenishes reserve while counting structural energy exactly once.
4. Real rejection-continuation address; correct its test/probability claim.
5. Freeze concrete gain1 fixture inputs/expected/quadrature/tolerances.
6. Finish the two untouched #17 defects.
7. Replace the vague feeding stub with a fixed op/tensor/byte spec; require a non-OOM authorization measurement.

> "Until those are fixed, the plan's top two claimed hard gates — exact conservation and correct mechanics — are not real."
