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
