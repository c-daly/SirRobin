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
