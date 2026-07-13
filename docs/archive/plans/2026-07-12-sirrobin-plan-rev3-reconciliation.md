# SirRobin — Implementation Plan Rev. 3: Reconciliation & Final Fixes

**Status:** correction overlay · **Date:** 2026-07-12 · **Base:** Rev-2 delta (`2026-07-11-sirrobin-plan-rev2-corrections.md`) resolving the Codex re-review (`2026-07-12-codex-rereview-of-rev2.md`: 5 RESOLVED / 13 PARTIAL / 2 STILL-BROKEN, 11 new problems, 8 blockers).

This document reconciles the **six independently-produced Rev-2 fix-sets into one internally-consistent spec** and closes every residual gap Codex named. It does **not** re-open any decision Rev-2 got right; where Rev-2 fix-sets disagreed, Rev-3 states the single canonical decision, specifies it concretely (equations with units, tensor shapes, pseudocode, and the gate that enforces it), and lists exactly which Codex findings and new-problems it closes. **Rev-2 + Rev-3 together are the final S0 spec.** Where Rev-3 supersedes a Rev-2 mechanism (e.g. the f64/TwoSum ledger), the Rev-2 text is deleted, not layered over.

Physics ground truth is cited by `SwimEval.cs:line`. The nine non-negotiable laws (fidelity, conservation, form-is-function, single-canonical-representation, clean-boundaries, continuity, grow-on-demand, depth-first, implicit-selection) are the acceptance frame; every reconciliation below is chosen to *strengthen* one of them, and this is called out.

---

## §0. Determinism posture — RELAXED (owner decision 2026-07-12; governs this document)

The determinism target is **three tiers**; SirRobin adopts the first two as hard requirements and
**downgrades the third to an optional same-device diagnostic — not a CI gate.** This realigns the plan
with the design doc's actual stance (conservation invariants are the primary gate; byte-identity is an
optional same-device check) and removes the float-replay machinery that was the disproportionate source
of complexity.

**KEPT — hard requirements (both are cheap by construction):**
- **Tier 1 — exact conservation.** The int64-quanta reservoir ledger (§1) is exact *regardless* of float
  summation order or GPU atomics; integer transfers are order-independent and `close_books` is an exact
  `==`. Float nondeterminism cannot break it.
- **Tier 2 — statistical reproducibility.** The integer counter-based Philox RNG + stable entity IDs (§4)
  reproduce the stochastic decisions (mutation, mating, death rolls) bit-exactly for a given
  `(seed, world_id, …)` key, so ablations are valid and experiments compare across seeds. Its
  bit-exactness is *integer* arithmetic (§4.4) and stays.

**RELAXED — Tier 3 (bit-identical *float* replay) is NOT a gate.** Consequences:
- `torch.use_deterministic_algorithms(True)` is **not required** at runtime; nondeterministic atomic
  float kernels (`scatter_add_`, etc.) are **permitted** where convenient. *(The `[B, S_max]`
  masked-axis-sum of §2 is retained for its **static-shape / CUDA-graph** benefit — a layout choice, not
  a determinism mandate.)*
- `torch.compile` / CUDA-graphs may fuse and reassociate float ops freely — **no compile-parity gate on
  the float physics.**
- **Snapshot completeness is kept** (a run resumes and continues *validly* — conservation holds,
  statistics reproduce), but **bit-for-bit float replay of a resumed long run is not guaranteed and not
  gated.** §5's `SimulationSnapshot` completeness and its lossless *serialization* round-trip
  (`test_snapshot_roundtrip_bit_identical`, a save/load correctness test on int64/exact state) are
  unaffected; only the "resumed run stays bit-identical over time" claim is downgraded.

**Unaffected** (correctness, not float-replay determinism): the conservation-invariant gate (int quanta),
the oracle-match gates (tolerance-based, §8), and the RNG reference-vector / cross-world tests (integer,
§4).

**Optional diagnostic (retained, not a CI gate):** a same-device "determinism smoke test"
(`use_deterministic_algorithms(True)` + no-compile + fixed seed → two reruns compared) may be run *on
demand* to debug a specific issue; it never constrains the production hot loop.

Rationale: the science needs *exact conservation* and *statistically reproducible dynamics* — not a
bit-identical replay of one billion-tick trajectory. Dropping Tier 3 as a gate deletes the hardest
determinism machinery (deterministic float reductions, compile-parity, `use_deterministic` enforcement)
while losing nothing the science requires.

---

## Reconciliation index (what each section fixes)

| § | Canonical decision | Closes (Codex findings) | Closes (new problems) | Blocker |
|---|---|---|---|---|
| 1 | int64 fixed-point quanta reservoirs | #5 STILL-BROKEN | NP-9 (f64 policy) | B1 |
| 2 | one segment layout: slot-0 sentinel, real `[1..S_max]` | #4, #9 (indexing half) | NP-1 (eager-gather) | B2, B8(gathers) |
| 3 | empty/zero-mass finiteness (det_safe + mask) | #9 (finiteness half) | NP-2 (non-finite solver) | B3 |
| 4 | one RNG allocator + world-keyed Philox | #13, #14 (allocator) | NP-4, NP-5, NP-6 | B6 |
| 5 | single source of truth: `energy`/`struct_N` in ColonyState only | #14 | NP-10 (dup state) | — |
| 6 | one energy equation (`U_cl` consistent; exact discrete balance) | #1, #15 STILL-BROKEN | NP-7, NP-8 | B5 |
| 7 | regularization ledgered (exact solve gates; stable κ) | #2, #12 | NP-3 (reg invalidates KKT) | B4 |
| 8 | independent closed-form gain1 fixtures | #8 | — | B7 |
| 9 | throughput/benchmark reconciliation | #6, #19, #20 | — | B8(hardware) |
| 10 | import firewall: `core/snapshot.py` + `core.contracts` snapshot view | #18 | NP-11 (drift guard) | — |

---

## 1. Conservation ledger — int64 fixed-point quanta (supersedes Rev-2 §5.1–5.3)

**Closes:** Codex **#5 STILL-BROKEN** (the f64 TwoSum residual had the wrong sign, `R_num -= (e_s+e_d)` where closure needs `+=`), blocker **B1** (fix the residual sign + specify a deterministic reduction of the error terms), and new-problem **NP-9** (f64-reservoir vs "O(W) f64 islands" inconsistency). Strengthens Law 2 (the books close) and Law 4 (single canonical representation).

**Root decision.** The wrong-sign TwoSum is not patched — it is **deleted**. Conserved *reservoir state* is stored as **int64 fixed-point quanta**, not f64. Integer transfers are exact by construction: there is no rounding, therefore no residual, therefore no sign to get wrong. Physics and rates stay f32 (hot loop) / f64 (Lamb quadrature, oracle arm) exactly as #11 specifies; **only the conserved reservoir totals are int64.** This makes `close_books()` an *exact integer equality* rather than a tolerance, and makes the reduction of totals order-independent (integer addition is associative and exact — deterministic on any device regardless of reduction tree), which is what B1's "deterministic reduction of the error terms" asks for: there are no error terms.

### 1.1 Currencies and quanta (frozen)

| Currency | Unit | Quantum `q` | int64 capacity | Rationale |
|---|---|---|---|---|
| mass | mol N | `q_mass = 1e-9` mol (nano-mol) | `9.22e18·q = 9.2e9` mol/world | resolution 1e-15 relative to a ~1e6 mol world — far below f64 eps |
| energy | J | `q_energy = 1e-3` J (milli-J) | `9.2e15` J/world | reserve/heat/sun meters; 1 mol biomass = `e_N=3.581e6` J = `3.581e9` quanta |

Quanta are declared once in `core/config.py` (`QUANTUM = {'mass': 1e-9, 'energy': 1e-3}`) and are part of `config_hash` (determinism depends on them). A reservoir tensor's dtype is **int64**; its physical value is `quanta * q`.

### 1.2 Reservoir tensors (replaces the f64 rows of Rev-2 §5.1)

| Reservoir | shape | dtype | currency |
|---|---|---|---|
| abiotic: `Nd, Bd, Sed` | `[W,G,G,B]` / `[W,G,G]` | **int64** | mass |
| meters: `E_sun_cum, E_heat_cum, E_export_cum` | `[W]` | **int64** | energy |
| `carry[src,dst,currency]` (§1.4) | per directed channel `[W,...]` | **f64** | — (sub-quantum accumulator) |

**Biomass mass/energy reservoir totals (`Bp`, `E_chem`, and the biomass part of any mass sum) are NOT stored here** — they are *derived masked sums* over ColonyState per resolution 5. `energy` (reserve) and `struct_N` live in ColonyState as int64 (resolution 5), and are summed into the ledger as derived totals. The int64 dtype for `energy`/`struct_N` **replaces** the f64 promotion Rev-2 §4.3 gave them (that f64 choice is deleted; NP-9's "every field reservoir is f64" contradiction dissolves — reservoirs are int64, an 8-byte whole-tick cost G-E2E measures, not an "O(W) f64 island").

### 1.3 Exact integer transfer (replaces Rev-2 §5.2 `two_sum`/`transfer`)

```python
def transfer_quanta(src_q, dst_q, n, mask):     # all int64; n>=0 quanta to move where mask
    moved = torch.where(mask, n, 0)             # elementwise, deterministic, no scatter
    return src_q - moved, dst_q + moved         # EXACT: one integer leaves src, the same enters dst
```
No quantum is minted or destroyed: the identical integer `moved` is subtracted from the source and added to the destination. `two_sum`, `R_num`, and the `-=`/`+=` sign question are **removed from the codebase** (Rev-2 §5.2 deleted).

### 1.4 Quantizing a real flux with a deterministic remainder carry

A physical flux is `f = rate·dt` (f32/f64). To move it into int64 quanta without loss, each directed channel `(src→dst, currency)` keeps a **f64 carry** ∈ `[0, q)` — sub-quantum flux not yet committed. It is *owned by the source* (it has not left) and is snapshot state (so replay is exact):

```python
def commit_flux(f, q, carry, mask):             # f: f64 physical amount >=0; returns (n_quanta:int64, carry:f64)
    acc      = torch.where(mask, f + carry, carry)          # accumulate intent in the source's currency
    n        = torch.floor(acc / q).to(torch.int64)         # whole quanta to commit (deterministic floor)
    carry    = acc - n.to(torch.float64) * q                # bounded in [0,q); stays in source
    return n, carry
# usage: n,carry = commit_flux(rate*dt, q_mass, carry_SD, mask); src_q,dst_q = transfer_quanta(src_q,dst_q,n,mask)
```

The carry is a rate-integrator, **not stored mass**: the reservoir's physical content is exactly `quanta·q`; the carry only decides *when* a whole quantum commits. Lumpiness is at the `1e-9` mol / `1e-3` J scale — orders below any biological relevance. Signed fluxes are split into two non-negative directed channels (src→dst and dst→src) so `f≥0` always and `floor` is unambiguous.

### 1.5 close_books — exact integer equality (replaces Rev-2 §5.3 tolerances for the *bookkeeping* arm)

```python
def close_books(currency) -> bool:
    total = int64_masked_sum(all_reservoir_quanta[currency])      # exact, order-independent
    ext   = X_in_cum_q[currency] - X_out_cum_q[currency]          # metered external source/sink quanta
    return total == I0_q[currency] + ext                          # EXACT ==, per world
```

**INV-CONSERVE (the top CI gate, exact):** for every currency and every world, at every step,
`Σ(all reservoir quanta) == I0_quanta + Σ(external-source quanta) − Σ(external-sink quanta)`, as an **`int64` equality with zero tolerance.** For a closed box (S1 mass, no external in/out) this is `Σ quanta == I0_quanta` exactly. The physics/work-consistency arm (E_KE from the f32 kernel, `p_in`, `wDrag`) remains a *tolerance* gate at `τ_E_phys = 1e-6` per step / `1e-4` drift, bounded-oscillating — that arm is a **fidelity** check on the f32 kernel, never a conservation leak (resolution 6). The two arms are now cleanly separated: bookkeeping is exact-integer, physics-consistency is f32-tolerance.

### 1.6 Tests (replace Rev-2 §5.3 test block)

| test | asserts | threshold |
|---|---|---|
| `test_transfer_exact_int` | after N random `transfer_quanta`, `Σ quanta == I0_q` | exact `==` |
| `test_cross_magnitude_int` | move 1 quantum between a 1-quantum pool and a `1e15`-quantum pool; both change by exactly 1 | exact `==` (the case that minted under f32/f64) |
| `test_carry_bounded_and_conservative` | 1e6-step soak: every `carry ∈ [0,q)`; `Σ quanta == I0_q` throughout; carries snapshot/restore identically | exact |
| `test_close_books_order_independent` | `int64_masked_sum` gives identical total under reversed/shuffled reduction order and on CPU vs CUDA | exact `==` |
| `test_unpaired_write_caught` | a raw `+=` to a reservoir bypassing `transfer_quanta`/`commit_flux` (AST audit) | CI fails |

**Design contradictions retired:** the design's `τ_mass ~ 1e-9` (§6.6) is now provably beaten — mass closes *exactly*; the `1e-9`/`1e-12` tolerances Rev-2 derived apply only to the *derived-energy physics arm*, not to mass. The whole "reservoir precision" question collapses: mass is exact.

---

## 2. One canonical segment layout — slot-0 identity sentinel (reconciles Rev-2 §4.2 vs §9)

**Closes:** Codex **#4** (Rev-2 §4.2 defined `S_max` real segments with root *self-parent*; #9 defined a slot-0 sentinel with reals in `[1,S_max)` — two incompatible layouts, two parent conventions, capacity 16 vs 15), Codex **#9** indexing half, blocker **B2** (publish ONE layout + one root-parent convention), the gather half of blocker **B8**, and new-problem **NP-1** (lifecycle eager-gather at `free_rank=-1`). Strengthens Law 4 (single canonical representation).

### 2.1 The one layout (replaces both Rev-2 §4.2 and the §9 sentinel scheme)

- **Segment-axis extent `S_slot = S_max + 1 = 17`**, with `S_max = 16` (donor `MaxParts`, `SwimEval.cs:72`). Slot **0 is a frozen identity sentinel**; **real segments occupy slots `[1..S_max]` — full `S_max=16` real capacity** (no capacity loss; Codex's "capacity 15" objection is void because the sentinel is an *extra* slot, not a stolen one).
- Segment tensors are `[W, N_cap, S_slot, …]`, viewed `[B, S_slot, …]` (`B=W·N_cap`) for the kernel. `SEG_AXIS = 2`.
- **Sentinel content (slot 0, immutable):** `pos=(0,0,0)`, `rot=(0,0,0,1)` (bit-exact identity), `mass=0`, `ma=0`, `areaZ=0`, `depth=-1`, `seg_mask=False`. It is never a real segment and never written by development or the depth-scan.
- **Parent convention (single):** `seg_parent[b,k] ∈ [0, S_slot)` is a **creature-local** slot index. A **root's parent = slot 0** (the sentinel), **never −1, never self.** Self-parent (Rev-2 §4.2) is deleted; global `body_base+local` is deleted; `−1` (Rev-2 §9 root sentinel value) is deleted. Every parent gather `pos[b, seg_parent[b,k]]` is in-bounds and, for a root, returns the identity sentinel → `pPos=0, pRot=identity`, bit-matching `SwimEval.cs:669` (`parentIndex<0 → pPos=0, pRot=identity`).
- **Empty body:** all real slots `seg_mask=False`; `tail_slot = 0` (sentinel). No real segment ⇒ every gather targets a valid slot; the tail gather returns the sentinel.

`root self-parent + hasJoint=False` (Rev-2 §4.2) and `parent=-1` (Rev-2 §9) are **both replaced** by `parent=0`. `seg_hasJoint` still encodes "carries an actuator" (`False` for roots, matching `SwimEval.cs:210` `parentIndex>=0`), but it is no longer load-bearing for the gather — the sentinel makes the gather safe unconditionally.

### 2.2 Depth-scan and reductions on the one layout

- Depth-scan: `L=6` passes over `depth ∈ {0..5}` (`MaxDepth=5`, `SwimEval.cs:73`), writing slots where `seg_depth==d & seg_mask`. The sentinel (`depth=-1, mask=False`) is never selected and stays identity; every real parent (strictly lower depth) is resolved in an earlier pass. Static 6 passes, single-valued.
- Reductions: `masked_segment_sum(vals, seg_mask)` = `(vals*seg_mask).sum(SEG_AXIS)` (Rev-2 §3, unchanged). Sentinel contributes exactly 0 (mask False + neutral content). **Finite-padding invariant** (Rev-2 §3) still required: padded/sentinel lanes carry finite neutral values so `0*NaN` cannot leak.

### 2.3 Lifecycle free-slot gather — no index is ever −1 (closes NP-1)

Rev-2 §4.4 computed `free_rank = cumsum(free)-1`, which is `−1` for the first slot when it is occupied, and then gathered `payload[free_rank]` *before* `where(claim,…)` masked it — a `-1` gather returns the last row (the eager-gather bug). Fix: the gather index is guaranteed valid by construction.

```python
free       = ~alive                                   # [W,N_cap]
free_rank  = torch.cumsum(free.to(torch.int64), 1) - 1
n_birth_w  = torch.minimum(request_w, free.sum(1))    # overflow dropped deterministically, logged
claim      = free & (free_rank < n_birth_w[:,None])
gather_idx = free_rank.clamp_min(0)                   # ALWAYS in [0,N_cap): occupied slots (free_rank=-1) map to 0,
                                                      #   their payload is discarded by `claim` below
new_field  = torch.gather(payload_field, 1, gather_idx.unsqueeze(-1).expand_as(field))
field      = torch.where(claim.unsqueeze(-1), new_field, field)   # only genuine births write
```

`gather_idx` is provably ≥ 0, so **no gather sees −1** anywhere in the lifecycle (mirrors the segment sentinel rule). Occupied slots gather harmless payload at index 0 and are discarded by `claim`. This is the layout-half of blocker B8; the empty/zero-mass *solver* finiteness is resolution 3.

### 2.4 Tests (extend Rev-2 test_pose / test_lifecycle)

| test | asserts |
|---|---|
| `test_layout_capacity` | real capacity == `S_max` = 16; slot 0 is sentinel in every body |
| `test_root_parent_is_sentinel` | every root has `seg_parent==0`; gather returns bit-exact identity `rot`; never −1, never self |
| `test_empty_body_indexing` | empty body: `tail_slot==0`, all gathers in-bounds, all forces 0, no NaN |
| `test_no_negative_gather_index` | FX/AST audit: no gather in `physics/` or lifecycle can receive a negative index (all sources are `clamp_min(0)`'d or sentinel-remapped) |
| `test_mixed_batch_no_leak` | interleaved empty/full bodies in one `[B,S_slot]` batch: live bodies rel `<1e-4` vs oracle, dead rows exactly 0 |

---

## 3. Empty / zero-mass body finiteness (completes Codex #9; closes NP-2)

**Closes:** Codex **#9** finiteness half (a zero-mass empty body still entered the solver and produced `1/0` then `0·∞`), blocker **B3** (explicit finite handling for empty/zero-mass bodies), and new-problem **NP-2** (`M00=M02=M22=0 ⇒ tr=0 ⇒ inv=∞ ⇒ NaN`). Strengthens Law 1 (fidelity — a bodyless creature has zero capability, not a NaN).

**Decision.** The solve guards the determinant with a **scale-relative** degeneracy test (never the donor's absolute `1e-12`, `SwimEval.cs:1157`) and **masks the result to zero** for invalid/empty bodies, so no division by zero and no `0·∞` can occur:

```python
body_valid = alive & (masked_segment_sum(seg_mass, seg_mask) > 0)   # [W,N_cap] bool
# 2x2 constrained system (resolution 7): M = [[M00,M02],[M02,M22]], P=(Px,Pz)
tr    = M00 + M22                                    # natural scale
det   = M00*M22 - M02*M02
scale = tr*tr                                        # scale of det
degenerate = (~body_valid) | (det.abs() <= EPS_REL*scale)     # EPS_REL ~ 1e-10 (f32-relative)
det_safe   = torch.where(degenerate, torch.ones_like(det), det)
dvx = ( M22*Px - M02*Pz) / det_safe
dvz = (-M02*Px + M00*Pz) / det_safe
dvx = torch.where(body_valid, dvx, torch.zeros_like(dvx))     # empty/invalid -> zero accel, finite
dvz = torch.where(body_valid, dvz, torch.zeros_like(dvz))
```

For an empty body `M00=M02=M22=0` and `P=0` (no segments ⇒ `F_stream=0`): `det=0 → det_safe=1 → dv=0/1=0`, then masked to 0. **Output is finite (zero accel).** For a *valid but near-singular* body (slender/anisotropic), `degenerate` is triggered only by the scale-relative `det` test, and the regularized path of resolution 7 handles it *with the reaction impulse ledgered* — it is **not** silently floored here. The absolute `1e-12` floor (`SwimEval.cs:1157`) is retained **only** inside `solve_sym3_donor` for gain0 byte-conformance (resolution 7), never in production.

**Test (`test_solve.py::empty_zero_mass_finite`):** a batch containing (i) an empty body, (ii) a single zero-mass sentinel-only body, (iii) a `MinX/Y/Z`-clamped `0.12×0.12×0.3` degenerate body (`SwimEval.cs:75`), (iv) a valid body — assert all outputs finite; empty/zero-mass rows exactly `0`; valid row matches oracle rel `<1e-4`; no `inf`/`nan` anywhere (`torch.isfinite(dv).all()`).

---

## 4. RNG — one allocator, world-keyed Philox, correct 64-bit product (supersedes Rev-2 §13.1–13.3)

**Closes:** Codex **#13** (RNG stream stability), Codex **#14** (allocator disagreement: §13 used per-world `next_eid[W]`, §4/§14 used a global host `next_stable_id`), blocker **B6** (one allocator + world-unique keys), and new-problems **NP-4** (per-world IDs without a world key collide), **NP-5** (the false "32×32 product fits in int64 exactly" claim), **NP-6** (Box–Muller `log(0)`, rejection exhaustion undefined). Strengthens Law 4 (single canonical representation) and the determinism contract.

### 4.1 One allocator (resolves the §13-vs-§4/§14 contradiction)

**Canonical: a per-world, on-device, monotonic, never-reused-within-a-world counter `next_eid[W]` (int64), stored in the snapshot.** The global host-side `next_stable_id` scalar of Rev-2 §4.4/§14 is **deleted**. Birth assignment is the batched, atomics-free `cumsum` scheme of Rev-2 §13.3, keyed per world:

```python
rank      = torch.cumsum(newborn_mask.to(torch.int64), 1) - 1   # deterministic within-world prefix
eid       = next_eid.unsqueeze(1) + rank.clamp_min(0)           # contiguous per-world ids
stable_id = torch.where(newborn_mask, eid, stable_id)
next_eid  = next_eid + newborn_mask.sum(1)                      # advance per-world allocator (on-device)
```

Assignment order is fixed `[W,N_cap]` slot order per world, so cross-world scheduling never perturbs a world's ids, and the allocator lives entirely on-device (no host sync — CUDA-graph safe, resolution 9). `ColonyState.stable_id/parent_id` are `[W,N_cap] i64`, never reused *within a world*.

### 4.2 The 7-field world-keyed address (adds `world_id`; closes NP-4)

Every draw is `PRF(key, counter)` addressed by the **7-tuple** (Rev-2's 6-tuple + `world_id`):

```
(seed, world_id, step, stable_entity_id, gene_iid, event_kind, draw_index)
```

Because per-world `next_eid` reuses id *values* across worlds, the key **must** include `world_id` or two worlds' organism-0 share a stream. Bit budget (Philox-4x32: 64-bit key + 128-bit counter):

| field | bits | placement |
|---|---|---|
| `seed` | 64 | key `(k0,k1)` |
| `world_id` | 16 | counter (≤65536 worlds; near-term W≤256) |
| `step` | 36 | counter (6.8e10 ticks ≫ 1e9 demanding-end) |
| `stable_entity_id` | 40 | counter (1.1e12 births/world) |
| `gene_iid` | 20 | counter (1M innovations) |
| `event_kind` | 8 | counter |
| `draw_index` | 8 | counter |

`16+36+40+20+8+8 = 128` — exactly the four 32-bit counter words. The packing is a fixed injective bijection; `_assert_field_widths` raises on overflow (`world_id≥2^16`, `step≥2^36`, `eid≥2^40`, …). Injectivity ⇒ distinct logical draws never collide on `(key,counter)` — across worlds included.

### 4.3 Canonical `event_kind` enum (adds PARAM_MUT, STRUCT_TOGGLE — reconciles §13 vs §17)

Rev-2 §13.1 listed `PARAM_JITTER/STRUCTURAL_ADD_NODE/…`; §17.1 used `PARAM_MUT/STRUCT_ADD/STRUCT_TOGGLE` — names absent from the enum (Codex #17). One canonical enum (extensible to 256; high bit `0x80` reserved as a re-key continuation flag, §4.5):

```
0 REPRO_ASEXUAL_SELECT   4 STRUCT_DELETE       8 MATE_PARTNER_SELECT
1 PARAM_MUT              5 STRUCT_TOGGLE       9 DEVELOP_STOCHASTIC (reserved)
2 STRUCT_ADD             6 MUTATION_BERNOULLI  10 FIELD_STOCHASTIC  (reserved)
3 STRUCT_ADD_EDGE        7 MATE_CROSSOVER_MASK  (0x80 bit) REKEY_CONT (continuation, §4.5)
```

The §17 mutation operators map exactly: parametric→`PARAM_MUT`, structural-add→`STRUCT_ADD`, structural-toggle→`STRUCT_TOGGLE`. §17.1's event-kind names are updated to these; no operator uses a name outside this enum.

### 4.4 Philox-4x32-10 with a correct 64-bit product (closes NP-5)

Codex NP-5 is right: for uint32 operands, `a*b` can reach `(2^32−1)^2 ≈ 1.845e19`, which **overflows signed int64** (max `9.22e18`). The claim "`p=a*b` fits in int64 exactly" (Rev-2 §13.2) is false. Compute mul-hi/mul-lo via 16-bit decomposition so every intermediate is `< 2^33`:

```python
def _mul32(a, b):                      # a,b: int64 tensors holding uint32 in [0,2^32)
    a_lo,a_hi = a & 0xFFFF, a >> 16    # each < 2^16
    b_lo,b_hi = b & 0xFFFF, b >> 16
    ll,lh,hl,hh = a_lo*b_lo, a_lo*b_hi, a_hi*b_lo, a_hi*b_hi   # each < 2^32
    mid = (ll >> 16) + (lh & 0xFFFF) + (hl & 0xFFFF)           # < 2^18
    lo  = (ll & 0xFFFF) | ((mid & 0xFFFF) << 16)               # low 32 bits
    hi  = (hh + (lh >> 16) + (hl >> 16) + (mid >> 16)) & 0xFFFFFFFF   # sum < 2^33 -> safe
    return hi, lo                       # (mulhi, mullo), both exact uint32

_M0,_M1 = 0xD2511F53, 0xCD9E8D57
_W0,_W1 = 0x9E3779B9, 0xBB67AE85
def philox_4x32_10(c0,c1,c2,c3, k0,k1):
    for _ in range(10):
        hi0,lo0 = _mul32(c0,_M0); hi1,lo1 = _mul32(c2,_M1)
        c0,c1,c2,c3 = (hi1^c1^k0)&0xFFFFFFFF, lo1, (hi0^c3^k1)&0xFFFFFFFF, lo0
        k0,k1 = (k0+_W0)&0xFFFFFFFF, (k1+_W1)&0xFFFFFFFF
    return c0,c1,c2,c3
```

All intermediates `< 2^33 ≪ 2^63`; no overflow. Integer ops are TF32/reduction-order-independent ⇒ bit-identical CPU↔CUDA and eager↔compiled. **Gate `test_prf_reference_vectors`:** the implementation must reproduce **published Philox-4x32-10 reference vectors bit-exactly**, and CPU == CUDA == eager == `torch.compile` bit-exactly. (The false-arithmetic claim is the exact kind of unverified assertion this gate exists to catch.)

### 4.5 Distribution transforms + exhaustion rule (closes NP-6)

- **uniform-f32:** `u = ((w0 >> 8) & 0xFFFFFF) * 2**-24 ∈ [0,1)`.
- **Box–Muller (no `log(0)`):** use `u1 ∈ (0,1]` by offsetting to exclude 0: `u1 = ((w0>>8) + 1) * 2**-24 ∈ [2^-24, 1]`; `u2 = (w1>>8)*2**-24 ∈ [0,1)`. Then `r = sqrt(-2*ln(u1))` (finite: `ln(u1)≤0`, `=0` only at `u1=1` giving `r=0`), `z0=r·cos(2π u2)`, `z1=r·sin(2π u2)`. `u1` can never be 0 ⇒ `ln(0)` is unreachable.
- **Bernoulli(p):** `u < p`. **Categorical:** inversion over a fixed-order CDF (no rejection).
- **Rejection sampling exhaustion (defined):** a rejection sampler consumes `draw_index = 0..255`. If all 256 sub-draws are exhausted (probability `< 2^-256` for any well-formed sampler), it **re-keys**: `event_kind |= 0x80` (the reserved `REKEY_CONT` bit), `draw_index` resets to 0, and an `attempt` counter is folded into the freed high bits. This yields effectively unbounded draws while remaining a pure deterministic function of the logical address + attempt number. In practice S0–S3 samplers (Gaussian, Bernoulli, categorical-by-inversion) never reject, so `REKEY_CONT` is defense-in-depth.

### 4.6 Tests (extend Rev-2 §13.6)

Add `world_id` to every key-locality test; add `test_prf_reference_vectors` (§4.4); add `test_boxmuller_no_log0` (sweep `w0∈{0, 2^24-1}` ⇒ `u1∈{2^-24,1}`, assert `z` finite); add `test_rejection_exhaustion` (force 256 rejects, assert `REKEY_CONT` engages and the stream stays deterministic across process restart); add `test_cross_world_disjoint` (two worlds, same per-world `stable_id` value, assert draws differ — the NP-4 collision cannot occur).

---

## 5. Single source of truth for `energy` / `struct_N` (closes Codex #14 duplication)

**Closes:** Codex **#14** (`SimulationSnapshot` duplicated `energy`/`struct_N` in both `colony` and `reservoirs`; omitted external-forcing state for snapshot-alone replay; required `safetensors` without pinning) and new-problem **NP-10** (duplicate authoritative state). Strengthens Law 4 — *"if you are writing code to keep two copies of one thing in sync, you have already lost."*

**Decision.** Per-creature `energy` (reserve, J) and `struct_N` (structural nutrient, mol N) are **canonical in ColonyState and stored nowhere else** (int64 quanta per resolution 1). The biomass mass/energy reservoir *totals* are **derived masked sums** over creatures, computed on demand for `close_books` and never stored:

```python
# derived, never stored — resolution 1 provides exact int64 sums
biomass_struct_total_q = int64_masked_sum(colony.struct_N_q * alive_mask)   # mass currency
biomass_reserve_total_q = int64_masked_sum(colony.energy_q  * alive_mask)   # energy currency
E_chem_creature = e_N * (colony.struct_N_q * q_mass)          # DERIVED readout [J], not stored
```

The **abiotic** reservoirs (`Nd` dissolved nutrient, `Bd` detritus, `Sed` sediment) are field-level `[W,G,G,B]`/`[W,G,G]` int64 and remain in `reservoirs` — they are genuinely distinct quantities, not duplicates. `energy`/`struct_N` are **removed from `reservoirs`** entirely; `close_books` reads the derived creature totals plus the abiotic reservoirs.

**Canonical `SimulationSnapshot` (deduplicated; replaces Rev-2 §14.1):**
```
SimulationSnapshot = {
  header,                 # schema_version, git_sha, torch/cuda, device, dtypes, config_hash, quanta
  config,                 # full Config + WorldConfig (+ QUANTUM table)
  clock,                  # {now, step, dt, scale, forcing_phase}
  colony,                 # ColonyState: alive, stable_id, parent_id, generation, pos/heading/vel,
                          #   energy_q (int64 J-quanta), struct_N_q (int64 mol-quanta),   <-- SOLE COPY
                          #   genome_ptr, age, species_tag, developed-body segs [·,S_slot,·]+seg_mask
  genotype,               # [P,N_max/E_max] pool
  reservoirs,             # abiotic int64 ONLY: Nd,Bd,Sed; energy meters E_sun/E_heat/E_export;
                          #   flux carries (f64, §1.4).  NO energy/struct_N here (single source of truth)
  allocator,              # {next_eid[W] int64}   (resolution 4; no host scalar)
  innovation,             # {next_iid[W or global], structural_key_cache}
  forcing,                # stateful external forcing (S6 advected buffers; empty pre-S6) -- REPLAY-COMPLETE
  rng                     # {master_seed}   (counter-based -> no PRNG buffer)
}
```

**External-forcing completeness (closes the #14 "snapshot-alone omits external actions" gap):** `forcing` carries every stateful external driver (S6 current/advection buffers; any scripted forcing schedule) so a run resumed from the snapshot alone is bit-identical — the negative gate `test_colonystate_alone_is_insufficient` (Rev-2 §14.3) is retained and now also fails if `forcing` is stripped.

**Serializer pinned (closes the #14 "requires safetensors without pinning"):** add `safetensors >= 0.4.3` to the pinned dependency table (the one artifact-level table of Codex #16). Tensors → `tensors.safetensors` (int64 reservoirs/quanta serialize losslessly and deterministically — no float ordering hazard); scalars/registries/config → `meta.json`. `torch.save`/pickle forbidden.

**Gate `test_single_source_of_truth`:** AST/audit — `energy`/`struct_N` appear as a stored field **only** in `ColonyState`; no `reservoirs['energy']`/`['struct_N']` key exists; `E_chem` is only ever computed as `e_N * struct_N` (never assigned to a stored pool). Plus `test_snapshot_roundtrip_bit_identical` (Rev-2 §14.3) on the deduplicated schema.

---

## 6. One energy equation (reconciles Rev-2 #1 vs #15; `U_cl` consistent; deletes the naive gate)

**Closes:** Codex **#1** (the combined identity reverted to signed `tFin·U`; the S3 gate re-asserted the invalid `ΔKE=(p_in−wake−drag)dt` that omits `½vᵀΔMv`), Codex **#15 STILL-BROKEN** (structural chemical energy absent from `Σ_stored` ⇒ mint on death; `S3-ENERGY-3` reinstated the very whole-body equation #1 rejected), blocker **B5** (reconcile `U` vs `U_cl`; delete the contradictory S3 equation), and new-problems **NP-7** (energy graph omits structural-body chemical energy — a mint) and **NP-8** (mechanics-vs-ecology energy definitions contradict). Strengthens Law 1 (fidelity), Law 2 (books close), Law 3 (form is function).

### 6.1 The authoritative locomotion power input `p_in` — `U_cl` consistent (fixes the sign contradiction)

Ground truth: the donor's two closure identities are **`InputPower = m_t·U·V_t·W_t ≡ tReact·U + pWake`** (reactive, *signed* `U`, `SwimEval.cs:336,358`) and **`CirculatoryInputPower = F_n·V_t ≡ tFin·U_cl + pFin`** (fin, *clamped* `U_cl = max(0,U)`, `SwimEval.cs:395,426`). The fin channel *cannot* close with signed `U` — the clamp `max(0,U)` is the modeling choice "no lift/wake swimming backward" (`SwimEval.cs:395,756,764`). Therefore the **one authoritative input power** is:

```
U_cl = max(0, U)
p_in = ( tReact·U + pWake )  +  ( tFin·U_cl + pFin )        # reactive uses signed U; fin uses U_cl
```

**`tFin·U_cl` (not `tFin·U`) everywhere.** This replaces Rev-2 line 65 (the "documentation-only" combined identity that wrote `+ tFin·U`) **and** Rev-2 §15.3 line 859 / `F-K` / `F-M` / `S3-ENERGY-3` (all of which wrote `tFin·U`). The reactive `(a)`-test still uses signed `U`; the fin `(a)`-test still uses `U_cl` (Rev-2 §1(a) was already correct — only the *combined* statements reverted, and those are now fixed). One symbol, one clamp policy, used identically in the S0 gate, the ecology ledger, and the diagnostic.

### 6.2 The discrete energy-balance gate — the ONLY KE equation (deletes the naive form)

The naive `ΔKE = (p_in − wake − drag)·dt` is **deleted everywhere it appears** (Rev-2 §1's demoted diagnostic already flagged it; Rev-2 `S3-ENERGY-3` line 921 re-introduced it — that line is removed). The **single** authoritative KE balance is the exact discrete identity for the actual semi-implicit integrator with pose-varying `M_eff` (Rev-2 §1(b)), now written with the constraint impulse `J_c` **and** the regularization impulse `J_reg` (resolution 7):

```
constrained update:  M_{n+1}(v_{n+1} - v_n) = P + J_c + J_reg,   P = F_stream·Dt,  v_y ≡ 0
ΔKE ≡ ½v_{n+1}ᵀM_{n+1}v_{n+1} − ½v_nᵀM_n v_n
    = v_mid·(P + J_c + J_reg) + ½ v_nᵀ(M_{n+1}−M_n) v_n
    = v_mid·(F_stream·Dt + J_reg) + ½ v_nᵀ ΔM v_n           # J_c·v_mid = 0 (workless: v_y≡0)
R_step = ΔKE − v_mid·(F_stream·Dt + J_reg) − ½ v_nᵀ ΔM v_n   # ≈ 0 by construction
```

`F_stream = (tReact+tFin)·f̂ + fDrag` (`SwimEval.cs:804`). The `½vᵀΔMv` added-mass term is **mandatory** (its omission was Codex #1's and #15's shared error). `J_c` (vertical plane reaction, `SwimEval.cs:807`) is workless. `v_mid·J_reg` (regularization work) is booked into the numerical-residual energy ledger (resolution 7) so the books close even when regularization fires; on the normal path `J_reg=0` and `R_step≈0`. **This single `R_step` gate is used by S0 (`discrete_balance_1e5`) and inherited unchanged by S3 (`S3-ENERGY-3` now *references* it rather than restating a naive form).** Threshold: `|R_step|/max(KE_n,ε) < 1e-6` (f64 arm) / `< 1e-3` (f32 hot), gated on the drift curve being bounded-oscillating (Rev-2 §1(b)).

### 6.3 The one reserve→outflow chain (mechanics = ecology; closes NP-7, NP-8, #15)

There is **one** energy accounting, consumed identically by the kernel (S0/S2) and the economy (S3). Per creature per step:

```
reserve draw (F-M):   ΔE_reserve = −(P_basal + p_in/η)·dt          P_basal = B0·M^α (Kleiber, α≈0.79)
of which p_in/η:      p_in·dt          -> mechanical work delivered to the body–fluid system
                      (1/η − 1)·p_in·dt -> muscle-inefficiency heat  -> E_heat
                      P_basal·dt        -> basal heat                -> E_heat
mechanical p_in·dt then partitions (the §6.2 physics balance):
                      ΔE_KE            = v_mid·F_stream·Dt + ½vᵀΔMv  (the R_step≈0 identity)
                      (pWake+pFin+wDrag)·dt -> shed wake + fin wake + axial drag -> E_heat
```

`p_in`, `pWake`, `pFin`, `wDrag`, `ΔE_KE`, `½vᵀΔMv`, `v_mid·J_reg` are the **StepLedger fluxes** the S0 kernel emits (`SwimEval.cs:814` accumulates `(pWake+pFin+wDrag)·dt` as MechWork; S0 additionally emits `ΔE_KE` and the added-mass term). S3 consumes these *authoritative* values — it never redefines the KE relation. This is the single equation NP-8 demanded: `p_in` (muscle→fluid) ≠ COM thrust work `f̂·v_mid·(tReact+tFin)` — they differ by exactly the wake/entrainment energy `(pWake+pFin)` — and the ledger accounts for the difference explicitly, so mechanics (#1) and ecology (#15) cannot disagree.

### 6.4 Structural chemical energy — counted once, no mint (closes #15, NP-7)

Codex #15/NP-7: grazing credited `struct_N` and reserve, then death converted *unaccounted* `struct_N` into energetic `Bd`, minting `e_N·struct_N`. Fixed by resolutions 5 + 1 together:

- **`struct_N` energy is DERIVED, never independently stored:** `E_chem(struct) = e_N · struct_N` (a readout, resolution 5). It **IS part of `Σ_stored`**: `Σ_stored = E_chem(Bp+Bd+Bm derived) + E_chem(struct_N derived) + E_reserve + E_KE`.
- **Assimilation (grazing, F-G):** mass `field −= I_bio; struct_N += AE·I_bio; Bd += (1−AE)·I_bio` — all exact int64 mass moves (resolution 1). The associated energy is *automatically* moved because `E_chem` is `e_N ×` those exact masses. No independent energy write ⇒ no mint.
- **Death:** `struct_N → Bd` is one **exact int64 mass transfer** (resolution 1). Because `E_chem` is derived from mass, the structural energy transfers to `Bd` **exactly and once** — the mint is impossible by construction (there is no second stored copy to double-count).

**Master balance (open system; replaces Rev-2 §15.4 with the exact-integer bookkeeping arm):**
```
INV-ENERGY (bookkeeping, EXACT int64):
  Σ_stored_q(t) == Σ_stored_q(0) + E_sun_cum_q − E_heat_cum_q − E_export_cum_q
  where the stored/meter energy quanta close by resolution 1's exact integer equality
INV-ENERGY (physics consistency, f32 tolerance):
  the E_KE term (f32 kernel) tracks the §6.2 R_step balance to τ_E_phys=1e-6/step, 1e-4 drift, bounded-oscillating
```

### 6.5 Tests

`test_energy_uses_Ucl_everywhere` (AST: every `tFin·…` term uses `U_cl=max(0,U)`; no signed-`U` fin term exists). `test_no_naive_ke_gate` (AST: the string/expression `ΔKE=(p_in−wake−drag)dt` exists nowhere; the only KE gate is `R_step`). `test_struct_energy_counted_once` (grazing then death cycle: `Σ_stored_q` exact-closes; `E_chem` never assigned to a stored pool). `S3-ENERGY-3` (rewritten): "inherits the S0 `R_step` gate verbatim" — not a restated equation.

---

## 7. Regularization ledgered (repairs Codex #2 + #12; closes NP-3)

**Closes:** Codex **#2** (the production 2×2 solver added *unledgered* Tikhonov mass, so once regularization fires `MΔv=P+J_c` is false), Codex **#12** (the "robust" solver used cancellation-prone `λ_min=(tr−disc)/2`, had no `tr=0` branch, floored bad determinants silently, changed dynamics via unaccounted regularization), blocker **B4** (regularization reflected in momentum + energy), and new-problem **NP-3** (regularization invalidates the KKT mechanics + energy gate). Strengthens Law 2 (books close) and Law 1 (fidelity — the gate uses the *true* physics, not a regularized surrogate).

### 7.1 Conservation gates use the EXACT (unregularized) constrained solve

The 2×2 constrained KKT solve (Rev-2 §2, unchanged in form) is the production dynamics:
```
[ M00 M02 ][dvx]   [Px]
[ M02 M22 ][dvz] = [Pz]      P = F_stream·Dt,   dv_y = 0
J_y = M01·dvx + M12·dvz − Py   (vertical reaction impulse, workless; ledger only)
```
**For every conservation gate (INV-CONSERVE, INV-ENERGY, `R_step`, momentum), the solve is the exact unregularized system.** Regularization is a numerical fallback that is *permitted only* for a valid-but-near-singular `M` — and empty/degenerate bodies are already excluded by the resolution-3 sentinel/`body_valid` mask, so the fallback fires only for genuinely ill-conditioned real bodies (10:1 slender, extreme anisotropy), never for empties.

### 7.2 Numerically stable condition estimate (replaces `λ_min=(tr−disc)/2`; handles `tr=0`)

The cancellation-prone `λ_min = (tr − disc)/2` (Rev-2 §12) is replaced by a cancellation-free form. For the symmetric 2×2 `[[M00,M02],[M02,M22]]` (SPD by construction: `M_body·I + Σ SPD added-mass`):
```
tr    = M00 + M22
det   = M00*M22 − M02*M02
disc  = sqrt( (M00 − M22)^2 + 4*M02^2 )     # NO cancellation: sum of squares, always >= 0
lam_max = 0.5*(tr + disc)                    # well-conditioned (sum of positives)
lam_min = det / max(lam_max, TINY)           # EXACT (lam_min*lam_max = det); avoids (tr−disc) cancellation
kappa   = lam_max / max(lam_min, eps_rel*tr) # condition number
```
`disc = sqrt((M00−M22)² + 4M02²)` never subtracts large near-equal quantities (the donor/Rev-2 `tr²−4det` form does). `lam_min = det/lam_max` is exact and stable. **`tr=0` is not a special case here** — it can only occur for an invalid/empty body, which resolution 3 masks out *before* this code (`body_valid=False ⇒ result forced to 0`); a valid SPD body has `tr>0`. The `TINY`/`eps_rel·tr` guards prevent `0/0` only on the already-masked degenerate path.

### 7.3 When regularization fires, its impulse and work are ledgered (closes #2, NP-3, B4)

```
reg    = where(body_valid & (kappa > KAPPA_MAX), EPS_SPD*tr, 0.0)   # KAPPA_MAX = 1e6 (f32)
Mr00,Mr22 = M00+reg, M22+reg
dv_reg = solve_2x2(Mr00, M02, Mr22, Px, Pz)                          # regularized dynamics (numerical fallback)
# the regularization changed the momentum balance by:
J_reg  = [ (Mr00-M00)*dvx_reg + 0,  (Mr22-M22)*dvz_reg ]  = reg * dv_reg   # reaction impulse from Tikhonov mass
```
`J_reg` is recorded so **momentum still closes**: `M·dv = P + J_c + J_reg` is now *true* (the added `reg·dv` is booked, not hidden). Its **work** `v_mid·J_reg` enters the numerical-residual **energy** ledger (resolution 1, int64 quanta), so `R_step` (resolution 6.2) closes with `J_reg` present. Regularization is thus a *ledgered numerical event*, not an invisible dynamics change. On the overwhelmingly-common well-conditioned path `reg=0`, `J_reg=0`, and production == exact solve bit-for-bit.

**Regularization is never permitted in the gain0 donor-conformance path:** `solve_sym3_donor` (Rev-2 §12, reproducing `SwimEval.cs:1151-1166` byte-for-byte incl. the absolute `1e-12` floor) is used *only* to match untouched-donor gain0 fixtures; it never regularizes. Production `solve_constrained_xz` regularizes-and-ledgers. `test_solve.py::donor_conformance` asserts they agree on well-conditioned bodies.

### 7.4 Tests

`test_reg_momentum_closes` (force `κ>KAPPA_MAX` with a 10:1 slender body; assert `M·dv == P + J_c + J_reg` to f32 rel `<1e-5`, and that dropping `J_reg` makes it fail — proving the ledger is load-bearing). `test_reg_energy_ledgered` (assert `R_step` closes with `v_mid·J_reg` booked, fails without it). `test_stable_kappa_no_cancellation` (extreme-anisotropy corpus, Rev-2 §12: assert `disc` and `lam_min=det/lam_max` match a f64 eigen-reference to f32 rel `<1e-5`, whereas `(tr−disc)/2` diverges — proving the stable form is load-bearing). `test_reg_off_on_wellcond` (isotropic/H0: `reg==0`, production == exact solve, `max_abs(Δ)==0`).

---

## 8. Independent, closed-form gain1 oracle fixtures (closes Codex #8)

**Closes:** Codex **#8** (committed to untouched-gain0 + independent-analytic-gain1 but never specified the gain1 construction; H1/H2 fixtures were still regenerated from a *patched* donor — residual circularity) and blocker **B7** (supply an actual independent gain1 fixture spec). Strengthens Law 1 (fidelity validated against theory, not a self-referential oracle).

**Decision.** gain1 fixtures are **closed-form analytic values derived from first principles, independent of any donor build**. The patched-donor regeneration of Rev-2 §2 (for the bug-active tilted/fin fixtures) is **replaced** by the analytic single-step construction below — this removes the circularity Codex flagged.

### 8.1 Analytic primitives (with sources)

- **Ellipsoid inertial mass (gain1):** `m = ρ·(4/3)·π·a·b·c` for semi-axes `(a,b,c)`. (Matches the donor gain1 path `massBox·(π/6)` since `(π/6)(2a)(2b)(2c) = (4/3)πabc`, `SwimEval.cs:88,173-174` — but derived, not read from the donor.)
- **Lamb added-mass coefficients:** the standard ellipsoid integrals (Lamb, *Hydrodynamics* 1932, §114):
  `α0 = a·b·c ∫₀^∞ dλ / ((a²+λ)·Δ(λ))`, similarly `β0` (`b²`), `γ0` (`c²`), `Δ=√((a²+λ)(b²+λ)(c²+λ))`, with `α0+β0+γ0=2`. Added-mass factor per axis `k_i = α0/(2−α0)`, etc. Added mass `m_a,i = k_i·ρ·V`, `V=(4/3)πabc`. Evaluate the integrals to f64 by an *independent* high-order quadrature (e.g. Gauss–Kronrod in the fixture generator — deliberately a *different* scheme from the donor's fixed 2048-interval Simpson at `SwimEval.cs:263`, so agreement is corroboration, not tautology). Sphere check: `a=b=c ⇒ k=0.5` each.
- **Single-step reactive force (Lighthill EBT):** at a stated pose + velocity `(U, V_t, s, m_t)`, `T_react = ½ m_t (V_t² − U² s²)` and `P_wake = ½ m_t U_cl W_t²`, `W_t = V_t + U s` (Lighthill 1960/71; `SwimEval.cs:326,341,355`) — evaluated by the closed-form expression, not the integrator.
- **Single-step fin force (Garrick):** `L = q·C_L`, `D = q·C_Di`, `q = ½ρ U_cl² S`, `C_L = a_L·α`, `C_Di = C_d0 + C_L²/(π e AR)`, `T_fin = L·sinβ − D·cosβ`, `P_fin = D·Q` (Garrick 1936; `SwimEval.cs:392-408`).
- **Single-step momentum:** `momentum = (m + m_added)·v`; for an all-sphere body `M_eff = (m_body + Σ m_a)·I` is constant and isotropic, so `(v_final·f̂)·(m_body + Σ m_a) = Σ T·Dt` exactly (the L6 identity, `SwimEval.cs:1205-1210`), computed analytically.

### 8.2 Fixture provenance rules

| fixture class | source | validation |
|---|---|---|
| gain0 (byte-conformance) | **untouched donor** build, byte-for-byte | `solve_sym3_donor`, quat, LambK ported bit-exact vs recorded traces |
| gain1 axial / isotropic | **closed-form** (§8.1) | analytic mass/added-mass/force/momentum; sphere `k=0.5` |
| gain1 tilted / anisotropic (was patched-donor) | **closed-form single-step** (§8.1) | analytic `M_eff = Σ R·diag(m_a)·Rᵀ` at the stated pose, analytic `F_stream`, analytic 2×2 KKT `dv` — **no donor, patched or otherwise** |
| gain1 fin-tail | **closed-form** Garrick (§8.1) | analytic `T_fin`, `P_fin`, `p_in = tReact·U + pWake + tFin·U_cl + pFin` |

**Gate `test_oracle_gain1_analytic`:** each gain1 fixture value is reproduced by the kernel to f32 rel `<1e-4`; the fixture generator imports **nothing** from the donor (AST/import audit: `test_gain1_fixtures_donor_free`). **Gate `test_no_patched_donor`:** grep/AST — no fixture is produced by a donor with `SolveSym3`/`_vCom.y=0`/`ellipMassGain` modified. The Rev-2 §2 "narrowly-reviewed patched donor" clause is **deleted**; the tilted-solve regression (`tilted_solve_divergence`) now compares the kernel's 2×2 result to the **analytic** 2×2 KKT solution, not a patched-donor recording.

---

## 9. Throughput / benchmark reconciliation (Codex #6, #19, #20)

**Closes:** Codex **#6** (G-E2E ordering: "post-S2" yet needs S3 feeding; S0 headroom double-counts population growth), Codex **#19** (`vram_cap` unresolved; Stage-4 required a *flattened prototype* despite that representation being deferred; compile-warmup-failure unspecified), Codex **#20** (marked churn/static-capture + conservation "RETIRED" while the birth gather was unsafe, layouts conflicted, and the residual had the wrong sign), and the hardware half of blocker **B8**. Strengthens Law "measure, don't assert."

### 9.1 G-E2E ordering resolved — stubbed-feeding whole tick (fixes the #6 contradiction)

Rev-2 §6.5 placed G-E2E "post-S2" but listed an S3-feeding dependency — contradictory. **Decision: G-E2E runs at the S2→S3 boundary on a *whole tick with a feeding STUB*** (a deterministic fixed-cost placeholder with representative memory-traffic, not real assimilation), then is **re-confirmed post-S3 with real feeding.** This removes the dependency contradiction: G-E2E needs S2 (StepLive + development + ≥1 field + spatial hash) plus a feeding stub — all available at the S2→S3 boundary. The floor `F_sci=2.31e7` c-steps/s and the emitted measured `φ_loco` are unchanged; the stub run is labeled provisional, the post-S3 run authoritative.

### 9.2 S0 headroom double-count corrected (fixes #6)

`F_sci = N_pop·T_run/W_budget = 1000·1e9/43200 = 2.31e7` c-steps/s uses `N_pop = N_cap = 1000` (**worst case, already at capacity**). Rev-2 §6.3 then multiplied by a `1.5×` "pop growth toward N_cap + world densification" factor — which **double-counts**, because `F_sci` is *already* evaluated at `N_cap`. That factor is removed:

```
F_loco_bare = F_sci / φ_loco = 2.31e7 / 0.5 = 4.63e7 c-steps/s
headroom    = 1.5 (StepLive over frozen Step: yaw + P-controller + torque)
            × 1.3 (bench↔production variance, thermal throttle, measurement)
            ≈ 1.95×                         # was 2.9× (the 1.5 pop-growth factor removed)
F_loco_S0   = 4.63e7 × 1.95 ≈ 9.0e7 creature-steps/s     <-- corrected S0 gate (d) floor
```
At near-term batch `B = W·N_cap ≈ 1024`, `9.0e7` c-steps/s ⇒ **≤ 11.4 µs/full-articulated-step** — a looser, honest target than the double-counted `1.4e8` (7.3 µs). The `256-world×1000×120 = 3.07e7` figure remains the S8-era many-worlds realtime target only. Gate-(d) text (Rev-2 §6.4) updated: floor = `F_loco_S0 = 9.0e7`.

### 9.3 No flattened prototype required (fixes #19)

Rev-2 §19.2 Stage 4 required a *flattened prototype* to measure masking tax — but the flattened/arena layout is a **deferred** optimization (Rev-2 §4.6). **Decision: Stage 4 is replaced by an *analytic + same-layout* masking-tax estimate.** The `S_slot/mean_seg ≈ 17/6 ≈ 2.8×` waste is reported analytically, and cross-checked empirically by timing the canonical `[B,S_slot]` kernel at the H1 mean-segment count vs at full `S_slot` occupancy (both on the *same* padded layout — no separate flattened implementation built). This keeps the deferred representation deferred (Law 7, grow-on-demand) while still quantifying the tax. Stage 4 cell count: 2 (same-layout occupancy sweep), 0 prototype builds.

### 9.4 Hardware pins parameterized (fixes #19; hardware half of B8)

The following are explicit owner-supplied parameters, **not** resolved here:
```
vram_cap  = 11 GiB              # RTX 5070 = 12 GiB GDDR7; ~11 usable; cell over cap -> 'OOM-skip', not fail
gpu_model = "RTX 5070"          # Blackwell; ~30 TFLOP FP32, FP64 ~1:64 -> keep f64 minimal (LambK + validation)
cpu_model = "Ryzen 7 8700F"     # Zen 4, 8 cores / 16 threads; CPU-rung baseline = 16 threads
```
`bench.py`'s `vram_cap` (Rev-2 §19.3 `"<PIN Q#1>"`) reads this pin; a cell exceeding it is recorded `OOM-skip` (logs `max_memory_allocated`), never a gate-(d) fail. **Compile-warmup-failure behavior (Rev-2 gap):** if r1/r2 warmup fails to reach the `<2%` step-time convergence within `max 10` iters, the cell is recorded `warmup-fail` and **falls back to the r0 (eager) number for that (device,B)** with a logged flag — it is not silently dropped and does not fail the gate; the ladder (§5.8) then decides on the r0 number.

### 9.5 Risks stay ACTIVE until verified in code (fixes #20)

Codex #20: Rev-2 §20.1 marked "deterministic ragged reduction," "flattened-churn static capture," and "f32 conservation drift" **RETIRED** — but the birth gather was unsafe (NP-1), the layouts conflicted (#4), and the residual had the wrong sign (#5). **Decision: reclassify all three from "RETIRED" to "resolved-in-spec / verify-in-code (ACTIVE)".** They remain in the *active* register with mitigation "Rev-3 §2/§1/§4 fix specified; gate green required before close" until the Rev-3 tests (`test_no_negative_gather_index`, `test_close_books_order_independent`, `test_transfer_exact_int`, `test_layout_capacity`, graph-pointer stability T14) pass in code. No risk is retired on the strength of a spec alone. The register's top ranks (energy-gate correctness, end-to-end throughput, gain1 independence, S1 closure realizability) are unchanged.

### 9.6 Tests / gates

`test_headroom_no_double_count` (assert the S0 floor derivation contains no factor evaluated twice at `N_cap`). `test_ge2e_dependencies` (G-E2E's dependency set is exactly {S2, feeding-stub}, provisional; re-confirm gate post-S3). `test_bench_no_flattened_build` (AST: the benchmark builds no flattened/arena layout). `test_risk_register_no_premature_retire` (the three §9.5 risks are `ACTIVE` until their named Rev-3 gates are green).

---

## 10. Import firewall — `core/snapshot.py` + `core.contracts` snapshot view (closes Codex #18, NP-11)

**Closes:** Codex **#18** (`core/snapshot.py`, added by #14, was absent from #18's enumerated `core` internals, so the drift guard failed its own config; the S7 viewer's snapshot access was not exposed via `core.contracts`) and new-problem **NP-11** (import firewall cannot pass its own drift guard). Strengthens Law 5 (clean abstraction boundaries).

### 10.1 `core/snapshot.py` is an enumerated core internal

Add `sirrobin.core.snapshot` to the `core-internals-private` forbidden list (Rev-2 §18.2 contract 5), so the exhaustive-enumeration drift guard `test_every_private_module_is_firewalled` (Rev-2 §18.3) passes on its own config:

```ini
[importlinter:contract:core-internals-private]
...
forbidden_modules =
    sirrobin.core.config
    sirrobin.core.clock
    sirrobin.core.state
    sirrobin.core.colony
    sirrobin.core.spatialhash
    sirrobin.core.ledger
    sirrobin.core.economy
    sirrobin.core.parcels
    sirrobin.core.snapshot          # <-- ADDED (resolution 5 / #14); private, reachable only via core.contracts
```

### 10.2 S7 reads snapshots through `core.contracts`, not core internals

The S7 viewer (`observe` layer) must **not** import `core.snapshot` directly (it would trip the contract in §10.1). Instead, `core/contracts.py` (Rev-2 §18.1) exposes a **read-only snapshot view**:

```python
# core/contracts.py  (public surface)
class ReadOnlySnapshotView(Protocol):
    """Immutable, read-only projection of SimulationSnapshot for observers.
       Exposes colony state, abiotic reservoirs, clock, header — NO mutation, NO internal module handles."""
    def colony_view(self) -> ColonyStateView: ...
    def reservoir_view(self) -> ReservoirSnapshot: ...     # abiotic totals (derived biomass on request)
    def clock(self) -> ClockView: ...
    def header(self) -> SnapshotHeaderView: ...

def open_snapshot(path: Path) -> ReadOnlySnapshotView: ...  # the ONLY way observe/S7 touches a snapshot
```

`observe` imports only `core.contracts.open_snapshot`/`ReadOnlySnapshotView`; the concrete `core.snapshot.load`/`SimulationSnapshot` stay private to `core`. This satisfies Law 5's "query *what*, never *how*" — the viewer sees a read-only projection, never the serializer or the mutable dataclass.

### 10.3 Tests / gates

`test_every_private_module_is_firewalled` (Rev-2 §18.3) now passes because `core.snapshot` is enumerated. Add to the injected-violation probe (Rev-2 §18.3 / G-SCAF-2): `observe → core.snapshot` (direct snapshot-internal reach) **must fail** `lint-imports`; `observe → core.contracts.open_snapshot` **must pass**. `test_s7_snapshot_via_contracts_only` (AST: `observe`/S7 imports no `core.snapshot` symbol; snapshot access is exclusively `core.contracts.open_snapshot`).

---

## Cross-section consistency ledger (what changed, so no two sections disagree)

- **Reservoir dtype is int64 quanta (§1)**, superseding Rev-2 §4.3/§5.1's f64 promotion of `energy`/`struct_N`. NP-9 dissolves: there is no f64 reservoir; the whole-tick int64 cost is measured by G-E2E (§9.1).
- **`energy`/`struct_N` are int64 in ColonyState only (§5)**, summed as *derived* totals for the ledger (§1). No `reservoirs['energy']`. `E_chem` is always the derived readout `e_N·struct_N` (§6.4).
- **One segment layout: `S_slot=17`, slot-0 sentinel, real `[1..S_max=16]`, parent=0 for roots (§2).** All of Rev-2 §4.2 (self-parent), §9 (`-1`), and the "capacity 15" concern are void. Reductions and `masked_segment_sum` unchanged.
- **The solve (§3 finiteness + §7 regularization) is one function:** `body_valid` mask → exact 2×2 for conservation gates → ledgered `J_reg` only for valid near-singular bodies → `det_safe` + zero-mask for invalid/empty. `solve_sym3_donor` is gain0-only and never regularizes (§7.3, §8.2).
- **`p_in = tReact·U + pWake + tFin·U_cl + pFin` with `U_cl=max(0,U)` (§6.1)** is the single input-power symbol used by the S0 gate, the ecology ledger, and the diagnostic. The naive `ΔKE=(p_in−wake−drag)dt` is deleted (§6.2). `R_step` (with `½vᵀΔMv`, `J_c`, `J_reg`) is the one KE gate, inherited by S3.
- **RNG key includes `world_id` (§4.2); allocator is per-world on-device `next_eid[W]` (§4.1).** The Rev-2 §4.4/§14 host `next_stable_id` is deleted; §13's per-world counter is kept but now world-keyed. event_kind enum includes `PARAM_MUT`/`STRUCT_TOGGLE` (§4.3), reconciling §13 and §17.
- **`safetensors>=0.4.3` pinned (§5); `core.snapshot` enumerated in the firewall, S7 reads via `core.contracts` (§10).**

---

## Residual open items requiring owner input

These were explicit owner-input pins (Law: measure, don't assert — don't fabricate hardware numbers). **RESOLVED 2026-07-12** with the owner's dev box (Ryzen 7 8700F + RTX 5070):

1. `vram_cap` = **11 GiB** — RTX 5070 (12 GiB GDDR7, ~11 usable). Feeds `bench.py`'s OOM-skip policy (§9.4). ✔ pinned 2026-07-12.
2. `gpu_model` = **RTX 5070** (Blackwell; strong FP32, FP64 ~1:64 → f64 kept to LambK + validation only). Determines the `(device)` axis winner. ✔ pinned.
3. `cpu_model` = **AMD Ryzen 7 8700F** (Zen 4, 8 cores / 16 threads; CPU-rung baseline = 16 threads). ✔ pinned.
4. `W_budget` = **12 h / overnight cadence** — the owner-accepted default behind `F_sci=2.31e7` (§9.2); revisit only if the run-iteration cadence changes. ✔ pinned.

All four are `bench.py`/`Config` parameters; none blocks writing S0 code, and each is logged in the bench manifest so the authorization number is auditable.

---

## Updated S0 pre-flight checklist (must be true before S0 code begins)

Concrete decisions Rev-2 + Rev-3 have now settled; each has a named gate that must be green:

1. **Reservoirs are int64 quanta** (`q_mass=1e-9` mol, `q_energy=1e-3` J); `transfer_quanta` is exact; `close_books` is an exact int64 `==`; flux carries are f64, bounded `[0,q)`, snapshotted. Gates: `test_transfer_exact_int`, `test_close_books_order_independent`, `test_carry_bounded_and_conservative`. (§1)
2. **One segment layout:** `S_slot=17`, slot-0 identity sentinel, real `[1..16]`, root parent = 0, no `-1` anywhere (segments or lifecycle gathers). Gates: `test_layout_capacity`, `test_root_parent_is_sentinel`, `test_no_negative_gather_index`. (§2)
3. **Empty/zero-mass bodies are finite:** `det_safe` + `body_valid` mask ⇒ zero accel, no `1/0`, no `0·∞`. Gate: `test_solve.py::empty_zero_mass_finite`. (§3)
4. **RNG:** one per-world on-device `next_eid[W]`; key includes `world_id`; Philox mul-hi via 16-bit decomposition (no int64 overflow); Box–Muller uniforms in `(0,1]`; rejection exhaustion via `REKEY_CONT`; event_kind enum includes `PARAM_MUT`/`STRUCT_TOGGLE`. Gates: `test_prf_reference_vectors`, `test_cross_world_disjoint`, `test_boxmuller_no_log0`, `test_rejection_exhaustion`. (§4)
5. **Single source of truth:** `energy`/`struct_N` stored only in ColonyState; biomass totals derived; `SimulationSnapshot` deduplicated and forcing-complete; `safetensors>=0.4.3` pinned. Gates: `test_single_source_of_truth`, `test_snapshot_roundtrip_bit_identical`, `test_colonystate_alone_is_insufficient`. (§5)
6. **One energy equation:** `p_in = tReact·U + pWake + tFin·U_cl + pFin`; the only KE gate is `R_step` (with `½vᵀΔMv`, `J_c`, `J_reg`); structural energy is derived and counted once. Gates: `test_energy_uses_Ucl_everywhere`, `test_no_naive_ke_gate`, `test_struct_energy_counted_once`, `discrete_balance_1e5`. (§6)
7. **Regularization ledgered:** conservation gates use the exact solve; stable `disc=√((M00−M22)²+4M02²)`, `lam_min=det/lam_max` (no cancellation, `tr=0` masked upstream); `J_reg` in the momentum + energy ledgers; never on the gain0 path. Gates: `test_reg_momentum_closes`, `test_reg_energy_ledgered`, `test_stable_kappa_no_cancellation`, `donor_conformance`. (§7)
8. **Independent gain1 fixtures:** closed-form ellipsoid mass `ρ(4/3)πabc`, Lamb `k=α0/(2−α0)`, analytic single-step Lighthill/Garrick forces and 2×2 KKT `dv`; gain0 from an untouched donor; no patched donor. Gates: `test_oracle_gain1_analytic`, `test_gain1_fixtures_donor_free`, `test_no_patched_donor`. (§8)
9. **Throughput:** S0 floor `F_loco_S0 ≈ 9.0e7` (double-count removed); G-E2E on a stubbed-feeding whole tick at S2→S3, re-confirmed post-S3; no flattened prototype; hardware pins parameterized; churn/static-capture/conservation risks ACTIVE until their Rev-3 gates pass. Gates: `test_headroom_no_double_count`, `test_bench_no_flattened_build`, `test_risk_register_no_premature_retire`. (§9)
10. **Import firewall:** `core.snapshot` enumerated as a core internal; S7 reads snapshots only via `core.contracts.open_snapshot`. Gates: `test_every_private_module_is_firewalled`, `test_s7_snapshot_via_contracts_only`, strengthened G-SCAF-2 probe. (§10)

When these ten gates are green (and the Rev-2 gates they extend remain green), the S0 spec is internally consistent and S0 code may begin. Rev-2 + Rev-3 = the final S0 spec.
