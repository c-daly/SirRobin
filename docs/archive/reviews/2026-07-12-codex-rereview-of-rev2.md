# Codex — Re-review of the SirRobin Plan Rev-2 Corrections

**Date:** 2026-07-12 · **Reviewer:** Codex `gpt-5.6-sol`, high reasoning, read-only (WSL).
**Subject:** `2026-07-11-sirrobin-plan-rev2-corrections.md` (the Rev-2 delta) against the 20 findings of the v1 review.
(Full trace: `codex-rev2-raw.txt`.)

## Overall verdict: **Not approvable to begin S0** — but substantially more concrete than v1.

Score: **5 RESOLVED · 13 PARTIAL · 2 STILL-BROKEN.** The remaining failures are concentrated in the exact
gates meant to authorize S0 (layout, finite mechanics, energy accounting, determinism under churn, oracle
independence), and — critically — many are **contradictions between the six independently-produced fix-sets**
that the (stalled) synthesis pass would have reconciled.

## Per-finding verdicts

- **#3 RESOLVED** — fixed `[B,S_max]` storage makes the reduction an explicit masked axis-sum; concrete, not a renamed atomic wrapper.
- **#7 RESOLVED** — producer respiration/mortality give real `Bp→Nd`/`Bp→Bd` loss; the originally-impossible crash is now mechanically possible.
- **#10 RESOLVED** — `quat_inv=conj(q)/‖q‖²` + normalization policy + denormalized/long-chain tests.
- **#11 RESOLVED** — explicit f32 hot-loop tensor set; Lamb coeffs cast to f32 after f64 construction; throughput read only from f32 rows.
- **#16 RESOLVED** — one consistent artifact-level dependency table; the three conflicting narratives removed.
- **#1 PARTIAL** — real progress (separate force-law identities + discrete residual), but the combined identity reverts to signed `tFin·U`, and the S3 gate re-asserts the invalid `ΔKE=(p_in−wake−drag)dt` (omits `½vᵀΔMv`).
- **#2 PARTIAL** — the 2×2 constrained solve + explicit `J_y` is preferable, but the production solver adds **unledgered Tikhonov mass**, so once regularization fires, `MΔv=P+J_c` is false.
- **#4 PARTIAL** — fixed padded storage eliminates the `[S_total]` contradiction, but defines **two incompatible layouts**: `S_max=16` real segments w/ root self-parent (here) vs slot-0 sentinel + real in `[1,S_max)` under #9 (capacity 15, two parent conventions).
- **#6 PARTIAL** — targets now from a 12-hour run + S0 scoped to locomotion + a whole-tick gate, but G-E2E is "post-S2" yet needs S3 feeding, and S0 headroom double-counts population growth.
- **#8 PARTIAL** — commits to untouched-gain0 + independent-analytic-gain1, but never specifies the gain1 construction; H1/H2 fixtures still regenerated from a patched donor (residual circularity).
- **#9 PARTIAL** — sentinel-row scheme fixes root/empty gathers + test matrix, but conflicts with #4's root-self-parent layout, reduces capacity to 15, and the zero-mass empty body still enters a solver that yields `1/0` then `0·∞`.
- **#12 PARTIAL** — corpus + compile-parity + donor/production split are concrete, but the "robust" solver still uses cancellation-prone `λ_min=(tr−disc)/2`, has no `tr=0` branch, floors bad determinants silently, and changes dynamics via unaccounted regularization.
- **#13 PARTIAL** — Philox + stable IDs + checkpoint fields genuinely replace sequential fragility, but IDs come from per-world `next_eid[W]` while the key omits `world_id` (identical RNG across worlds); #4/#14 instead specify one global host allocator — no authoritative implementation.
- **#14 PARTIAL** — `SimulationSnapshot` is materially complete, but duplicates `energy`/`struct_N` in both `colony` and `reservoirs`, disagrees with #13 on allocator shape, omits external actions for "snapshot-alone" replay, and requires `safetensors` without pinning it.
- **#17 PARTIAL** — mutation/inheritance/lineage moved to S3, S4 sensing trimmed, seeded predation as the gate; but mutation rates unnamed, text says genotype mutation preserves `S_max` (should be `N_max/E_max`), and uses RNG event-kinds absent from the canonical enum.
- **#18 PARTIAL** — exhaustive contracts + drift test much stronger, but #14 adds `core/snapshot.py` absent from #18's forbidden list, so the drift test fails its own config; the S7 viewer's snapshot access isn't exposed via `core.contracts`.
- **#19 PARTIAL** — staged 35–40-cell funnel + policies replace the 576-cell explosion, but leaves `vram_cap="<PIN Q#1>"` unresolved, requires a flattened prototype despite deferring that representation, and leaves compile-warmup-failure behavior unspecified.
- **#20 PARTIAL** — register elevates the right risks, but marks churn/static-capture and conservation "RETIRED" while the birth gather is unsafe, layouts conflict, and the residual correction has the wrong sign.
- **#5 STILL-BROKEN** — moving reservoirs to f64 is sound, but the TwoSum correction has the **wrong sign** (`x+y=s+e` ⇒ closure needs `R_num += e_s+e_d`, not `-=`), so the mechanism *doubles* the discrepancy it was meant to close.
- **#15 STILL-BROKEN** — the creature's structural chemical energy is absent from `Σ_stored`: grazing credits `struct_N` and reserve, then death converts unaccounted `struct_N` into energetic `Bd`, **minting `e_N·struct_N`**; and S3-ENERGY-3 reinstates the very whole-body equation #1 rejected.

## New problems introduced by Rev-2

1. **Lifecycle gather repeats the eager-gather bug** — `free_rank=-1` for occupied slots is gathered before `where` masks it (indices must be clamped/remapped first).
2. **Empty-body solver is non-finite** — `M00=M02=M22=0` ⇒ `tr=0` ⇒ `inv=∞` ⇒ NaN, falsifying #9's promised empty-body behavior.
3. **Regularization invalidates the KKT mechanics + energy gate** — production solves `(M+λI)Δv=P`; ledger claims `MΔv=P+J_c`. Ledger the regularization impulse/work, or stop calling the balance exact.
4. **Stable-ID / RNG specs are mutually exclusive** — per-world IDs without a world key collide; the global-host alternative breaks the all-device allocator. Include `world_id` in the counter and keep `next_eid[W]` on-device.
5. **Philox makes a false arithmetic claim** — a 32×32 product needs 64 bits and does not "fit in int64 exactly"; prove against reference vectors under eager+compiled CPU/CUDA.
6. **Distribution transforms underspecified** — Box–Muller permits `log(0)` at `u=0`; rejection sampling has an 8-bit `draw_index` with no exhaustion rule.
7. **Energy graph omits structural-body chemical energy** — an actual mint on assimilation/death; decide where assimilated biomass energy resides and count it once.
8. **Mechanics vs ecology energy definitions contradict** — #1 (COM thrust work ≠ `p_in`, needs `ΔM`) vs #15 (asserts the opposite). No single authoritative energy equation.
9. **f64 policy internally inconsistent** — #11 advertises O(W) f64 islands; #5 makes every field reservoir f64 (a real whole-tick cost G-E2E must measure).
10. **Snapshot has duplicate authoritative state** — `energy`/`struct_N` in both `colony` and `reservoirs` (violates single-source-of-truth).
11. **Import firewall cannot pass its own drift guard** — `core/snapshot.py` (added by #14) is absent from #18's enumerated core internals.

## The 8 named blockers (Codex)

1. Fix the TwoSum residual sign + specify deterministic reduction of the error terms.
2. Publish ONE canonical segment layout (16 real + separate sentinel, OR 15 real) + one root-parent convention.
3. Add explicit finite handling for empty/zero-mass bodies.
4. Repair the constrained solver so regularization is reflected in momentum + energy accounting.
5. Reconcile `U` vs `Ucl` and delete the contradictory S3 energy equation.
6. Choose one stable-ID allocator; make RNG keys unique across worlds.
7. Supply an actual independent gain1 fixture specification.
8. Fix the unsafe lifecycle gathers; pin benchmark hardware/VRAM.
