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
