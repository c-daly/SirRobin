# SirRobin — Restart Architecture Design (grounded)

**Date:** 2026-07-11
**Status:** Design draft for user review. Supersedes the substrate/skeleton parts (Sections 4–5) of
`docs/archive/2026-07-11-restart-brief.md`. The north star, fatal-flaw diagnosis, and salvage *intent* in
that brief still stand; this doc replaces the *how* (C#/Unity → Python/PyTorch) with an
evidence-grounded architecture.
**Grounding:** verified against real prior art (Isaac Gym/Lab, Brax, MuJoCo-MJX, NVIDIA Warp) and
against the actual donor source `…/game prototype/Assets/ProceduralWorld/Scripts/Life/SwimEval.cs`.

---

## 1. North star & principles (recap — see brief for full)

A faithful, consequence-bearing, non-text **grounding substrate** for an embodied agent (Sophia,
via a Talos HAL over ROS2, transferable to a physical TurtleBot3). Non-negotiables: **first-law
closure** (conservation is an invariant, not a toggle); **form-is-function** (behavior/energetics
emerge from morphology through real physics); **open-ended evolution, implicit selection only**;
**anchored measurement discipline**; **research-grade reproducibility**; **depth-first — close the
books before breadth.** Emblematic endpoint: an unscripted sea-to-land crossing.

## 2. The organizing thesis — faithful vectorization

The prior build's root architectural wound was a **data-far/physics-near LOD proxy** that existed
*only* because faithful full-population physics seemed unaffordable — and that one assumption spawned
every laundered scalar and arbitrary knob. **Vectorizing the true per-body physics for all N
creatures across many worlds as batched tensor ops dissolves the reason the proxy exists** →
faithful *and* affordable, and it yields cluster-scale throughput for deep-time evolution + RL.

**Discipline (the same trap, inverted):** the old build let an *unverified* perf premise drive the
architecture. This thesis is the *optimistic* perf premise, so **S0 must verify it before the build
is bet on it** (Section 7).

**Grounding verdict: SOUND / GO.** Production precedent already surfaces thousands of parallel bodies
as tensors (Isaac Gym/Lab in torch; Brax/MJX in JAX). The donor's hot loop (`Sim.Step`, SwimEval.cs
740–817) is already elementwise and branch-light — a per-segment accumulation of 6 unique added-mass
`M_eff` entries + axial quadratic drag, a closed-form cofactor 3×3 solve (`SolveSym3` 1151–1166), an
algebraic energy identity `P_in = T_react·U + P_wake` (338–358), fixed `Dt=1/120`, RNG-free — so it
maps to batched tensor ops without re-deriving physics. What is *unproven* is SirRobin-specific and
is exactly what S0 must falsify: ragged heterogeneous bodies, per-op launch overhead at realistic N,
the GPU determinism tax, and float32 long-run energy drift.

## 3. Substrate & architecture

- **PyTorch is the one unified array engine** — physics math now, any neural minds later. numpy =
  optional oracle cross-check. **JAX = escape hatch only** (triggers in §10). Keep **NVIDIA Warp**
  (zero-copy torch interop) as the *lighter* escape hatch for a single branchy per-body kernel before
  ever reaching for a full JAX rewrite.
- **GPU from day one (user decision), scope = ONE large world** batched over its creature population.
  Many-parallel-worlds/vectorized-envs batching is **deferred to S8 (RL)** — not a near-term
  requirement. GPU's payoff over multicore-CPU rises with population size: a large population fills the
  GPU; a few-hundred-creature world is where CPU can match it. The identical batched-torch code runs on
  either via a `device=` knob on the immutable Config, and **S0/SpikeSwim measures the CPU↔GPU crossover
  at the real population size** so the device is chosen on numbers, not vibes. GPU determinism becomes
  day-one work: **avoid nondeterministic atomic scatter (use precomputed unique slot indices)** so
  within-seed reproducibility holds on GPU; the conservation-invariant tolerance gate stays primary,
  bitwise goldens are a same-device-CPU regression check only.
  *(Honest note: single-thread compiled throughput is ~5× realtime at ~1000 creatures — the earlier
  "100–1000×" claim was wrong by ~2 orders of magnitude; performance is first-order for deep-time
  evolution, which is exactly why GPU-vectorization is justified for a large single world.)*
- **Layering as Python packages with an enforced import-boundary firewall** (replaces the compile
  firewall): `numerics → physics → fields → genetics → core → observe`, with Unity as a downstream
  viewer. A lint rule (e.g. `import-linter`) makes "physics imports ecology" a CI failure. `physics`
  defines mechanical contracts (`DevelopedBody`, `FluidSample`); `genetics` produces bodies, `fields`
  produce fluid samples; neither reaches into `physics` internals.
- **Unity = remote/replay viewer** over the same state contract Talos uses; it no longer compiles or
  hosts the sim. (This box's Unity-first CLAUDE.md assumption is now void for the Core.)
- **Distributed/cluster is first-class:** Core is pure Python/torch (runs on any CPU/GPU node);
  `ColonyState` is fully serializable (snapshot/checkpoint/resume/migrate); `Reset(seed)`/`Step(action)`
  is the unit an orchestrator drives; batching folds a "worlds" dimension; no global mutable state.
- **Determinism posture:** **conservation invariants are the primary correctness gate** (robust to
  sub-tolerance FP noise). Bit-identity is a *same-machine-same-device* regression check only;
  cross-machine bit-identity is out of scope. torch determinism requires deliberate setup (seed
  torch/numpy/python, `use_deterministic_algorithms(True)`, `CUBLAS_WORKSPACE_CONFIG=:4096:8`,
  deterministic segment reductions).
- **Precision — deliberate HYBRID (never blanket float64; consumer RTX FP64 is a 32–64× cliff):**
  hot loop in **float32** with *relative* conservation tolerances; **float64 in exactly three
  places** — (1) the one-time `LambK` added-mass precompute (the donor computes it in double; matching
  the oracle requires it), (2) the **global conservation ledger** via compensated/pairwise sum, (3)
  the S0 oracle-match config (to isolate "is the math right" from float32 transcendental divergence).

## 4. The canonical representation (one layout for physics AND the RL seam)

Struct-of-arrays torch tensors with fixed creature and segment capacity:

- **Creature level:** state shaped `(W, N_cap, k)` = worlds × fixed-max-population, carrying a
  **boolean alive-mask**. Births/deaths mutate the mask and **recycle dead slots** → tensor shapes
  stay static (required for `torch.compile`/CUDA-graph stability; mirrors what JAX would force).
- **Segment level:** fixed `[W,N_cap,17,...]` tensors, where slot 0 is a finite identity sentinel and
  slots 1..16 are real capacity. `seg_mask` is authoritative; parents are creature-local nonnegative slot
  indices; roots point to sentinel 0. Per-body reductions are masked sums over the fixed slot axis. Pose is a
  bounded six-pass depth scan (MaxDepth=5). There is no `S_total`, `body_id`, atomic scatter, compaction, or
  pointer repair in the canonical path.
- Flattened/CSR storage is deferred to a dedicated optimization slice and requires profiler evidence plus a
  lifecycle-inclusive win over the fixed-slot baseline.
- **This same `(W, N_cap)` + alive-mask structure is the batched observation tensor the Talos
  contract exposes.** Do not design two population representations.

Key torch APIs: `torch.inference_mode()` around the whole step; hand-rolled elementwise quaternion
compose and cofactor `SolveSym3` (control op order for the oracle; avoid `linalg.solve` dispatch);
masked sums over the fixed segment-slot axis for reductions;
`torch.where` for **every** data-dependent branch (never a Python `if` over the batch);
`torch.compile(mode='reduce-overhead')` as the *last-mile* dispatch killer after batching is correct.

## 5. The sim ↔ Talos state contract

**Dict-of-tensors** with a leading `(W, N_cap)` batch dim + alive-mask; nested **CORE/EXT** structure
mirroring `gymnasium.spaces.Dict({'core':…, 'ext':…})` for both obs and action. SI units; ROS REP-103
body frame FLU (x-forward, y-left, z-up); radians; seconds. Every exchange carries a `Header`
{contract_version SemVer, tick, sim_time_s, world_id, agent_id, embodiment∈{SIM_FISH,TURTLEBOT3},
ext_present}.

- **CORE (the differential-drive-executable intersection of fish and TurtleBot3):**
  - Action OUT: `surge_effort∈[-1,1]` (→ `Twist.linear.x`), `yaw_rate∈[-1,1]` (→ `Twist.angular.z`).
    **This 2-vector is the entire CORE action.**
  - Obs IN: `lin_vel(3)`, `ang_vel(3)`, `orientation(4)` (or heading(1) in a 2-D profile),
    `range_egocentric(K)` (shared exteroception: robot LaserScan / fish nearest-neighbour+terrain on
    the same K beams), `flow_rel(3)`, `energy(1)` (fish metabolic reserve / robot battery SoC),
    `contact`.
- **EXT (non-load-bearing richness):** fish — chemical gradients {food, predator kairomone, …} on the
  same K beams (the fish's *primary* nav sense, deliberately EXT because it has no robot analogue),
  light/depth/temp/marine-snow, per-segment proprioception, gape, heave/pitch/gait actions, feeding
  strike. robot — camera, IMU, wheel ticks, joint states.
- **Validated by the crown jewel, not merely compatible:** SwimEval's cruise path already zeroes
  vertical COM velocity (`Sim.Step` line 807) and `StepLive` integrates yaw only (line 956) — so the
  2-DOF {surge, heading} CORE **is the fish's actual realized locomotion DOF today**, not a lossy
  down-projection. Heave/pitch/roll belong in EXT precisely because the kernel doesn't integrate them.
  → fish↔robot **action-side** transfer risk ≈ 0; the residual risk is entirely **observation-side**
  (can a CORE-only, chem-gradient-free policy forage — see open Q).
- **Serialization:** in-process (Sophia in same Python) → pass torch tensor dicts directly (zero
  serialization), pydantic shape/dtype check in debug only. At a process/network boundary → canonical
  IDL is **Protobuf proto3** (field-number evolution, additive-only within a major; `reserved` for
  removals; SemVer-major bump for breaking reshape). ROS2 face = generated `.msg` + `geometry_msgs/Twist`
  matching CORE 1:1 (pin a distro). msgpack/JSON for telemetry/checkpoints. FlatBuffers only if proto
  (de)serialization ever profiles as a hot-loop cost. A conformance test freezes a CORE fixture and
  asserts pydantic + Gymnasium space + `.msg` agree field-for-field.

## 6. S0 = "SpikeSwim" — the verification spike (redefines the old S0)

**Goal:** a standalone PyTorch program (no Unity, no ecology, no feeding) that ports `Sim.Step`
(one-shot, frozen-heading — *not* `StepLive`; steering is out of S0 scope) for a batch of
`B = W·N` ragged bodies, and answers ONE question with real measurements: *can we step faithful
Lighthill per-body physics for a batched, heterogeneous, masked, churning population deterministically,
with energy closing, at throughput that makes full-population faithful sim affordable — and does GPU
help — before the architecture is bet on it?*

**Simulates:** the horizontal-cruise capability episode exactly as the donor's `RunEpisode`
(1076–1120): 3 s warmup + 5 s measure = **960 steps at dt=1/120**, gait kinematic
`θ_j = ampDeg_j·sin(2π·swimFreq·t + phase_j)`, controller = frozen-heading reactive drive (no mind).
Plus a **10^5-step** config for energy-ledger drift. Plus a **population-churn stub** (fixed-capacity
  + alive-mask, kill ~2%/1000 steps, refill from a committed schedule in place) to measure
  masking and lifecycle cost. Two dtype configs: **float64** (oracle-match, isolates math
correctness) and **float32** (throughput/invariant).

**Scale sweep:** staged batches on CPU and CUDA. The original experiment used `N_cap=1024`, `N_live=1000`; heterogeneity
**H0** (homogeneous 6-seg — isolates vectorization), **H1** (realistic ragged 2..16, mean ~6, mirror
pairs, ~40% fin tails), **H2** (skewed worst-case: mostly 2–3 seg + rare 16). Acceleration ladder:
r0 eager → r1 `torch.compile(reduce-overhead)` → r2 explicit CUDA-graph capture. The original **9.0e7
creature-steps/s** floor at 1,000 live remains a recorded NO-GO. The later pre-registered population gate used
5,000/10,000 live creatures with 1× real-time floors of 600k/1.2M; all H1/H2 cells passed. Whole-tick
affordability remains later authority.

**Acceptance authority:** the A–E matrix, equations, mixed tolerances, frozen H0/H1/H2 corpus, executable
100,000-step prefix budget, 11 GiB VRAM cap, and decision history are normative in the archived consolidated S0
plan, the population-grounded Gate-E revision, and both S0 reports. Exact int64 bookkeeping
and discrete schedules are hard determinism gates; bit-identical floating replay is an informational
same-device diagnostic. H0 never authorizes.

**Falsifiers (any trips → cheaply kill/revise the thesis before pouring commits on it):** F1 ragged
heterogeneity defeats batching (H1 flattened < per-body loop at single-world N; H2 padding craters);
F2 depth-scan/reduction >50% of step time; F3 per-op launch overhead dominates at realistic N even at
r2 (→ narrow GPU to many-worlds, keep S0–S2 CPU); F4 determinism tax >2× on GPU; F5 oracle
un-portable without a de-vectorized loop (aggregates diverge >1e-3); F6 float32 semi-implicit energy
drift breaches the gate (monotonic not oscillating); F7 churn compaction swamps the step.
**META-FALSIFIER: a green H0 number does NOT authorize the architecture — H0 hides the raggedness
(F1) and churn (F7) that are the real risk. The gate that matters is H1/H2.**

Telemetry-first: dump a parquet/jsonl of every gate metric + profiler attribution; no assertion from
code inspection.

## 7. Slice roadmap (revised)

- **S0 — SpikeSwim** (above): verify the vectorization thesis. Go/no-go with measured gates.
- **S1 — Conserved single-nutrient economy** (keystone): closed loop (Liebig×Monod drawdown,
  bacterial remineralization BGE split, Martin profile, mixing). *Books must close* to tolerance;
  blooms/deserts emerge from the loop. Nothing proceeds until green.
- **S2 — One canonical body + live locomotion for every creature:** `BodyGraph → DevelopedBody →`
  **`Sim.StepLive`** (yaw-integrating P-controller — the *actual* live loop, heavier than the `Step`
  S0 ports; **re-measure throughput against StepLive before committing**). Feeding/metabolism/defense
  derived from morphology (kill the `eff[]` vector).
- **S3 — Feeding/metabolism/reproduction on conserved energy** (Holling-II + assimilation loss →
  detritus; Kleiber; real construction cost).
- **S4 — Predation as a staged contest between bodies** (find→close→seize→consume; conserved; no
  seeded predator).
- **S5 — Speciation/mating/taxonomy** (recombination + genetic-distance gating; observational taxonomy).
- **S6 — Transport/currents/upwelling** (advect nutrients/plankton/detritus; Ekman).
- **S7 — Unity viewer + observation surface** (render never feeds fitness).
- **S8 — RL/embodiment loop** (Talos state contract → ROS2 → TurtleBot3 → CWM-G). Gated on books
  closed (S1–S4) **and** Sophia's action interface verified against real code.
- **S9 — Plants + the sea-to-land crossing** (the emblematic endpoint; hardness ramps up).

## 8. Salvage as oracle (revised)

The C# crown jewels are **re-ported to torch**, not "called live." The C# donor is demoted to an
**offline oracle / fixture generator** (a tiny headless console harness, or the existing
`ReconstructForTest`/`LambKForTest`/`CoastTest`/`MomentumLedger` seams — SwimEval is Unity-light:
only `Vector3`/`Quaternion`/`Mathf`). Conservation-invariant tests move from C# BitConverter/FNV
goldens to **pytest + tolerance invariants**. Frozen conformance fixtures (LambK grid, single-step
forces, 8 s aggregates across H1/H2 genomes) guard the port. The **validated equations + recorded
oracle values** are the durable salvage; the C# text does not run in SirRobin.

## 9. Risks & mitigations

See the grounding brief; the load-bearing ones: **F1 ragged batching** (measure H0/H1/H2 taxes on fixed slots,
then profile); **F3 launch overhead**
(acceleration ladder; narrow GPU to many-worlds if r2 fails); **F4 determinism tax** (CPU-first;
deterministic segment reductions); **F6 float32 mechanical drift** (dimensioned residual/prefix gates;
conserved mass reservoirs use exact int64 closure, not a float ledger); **F7 churn cost** (fixed-cap in-place recycling, measured); **C#→torch port bugs the
algebraic identity won't catch** (conformance fixtures are the real guard); **Sophia interface unknown**
(keep contract continuous-first; push any symbolic decode into a Talos-side adapter, never the sim
schema; don't build S8 until verified).

## 10. Open decisions (flagged for user review — recommendations given)

1. **Port target for S0:** `Sim.Step` (one-shot, frozen-heading — *recommended*, simpler oracle,
   de-risks vectorization) vs `Sim.StepLive` (the eventual live loop). Rec: **Step for S0, StepLive
   re-measured before S2.**
2. **Differentiate the world, ever?** The single biggest torch-vs-JAX fork. "Never differentiate the
   world" keeps torch+`inference_mode` ideal; "differentiate the world" (analytic policy gradients /
   system-ID through the sim) is the strongest JAX pull. Rec: **assume no; flag, don't decide now.**
3. **Target hardware** (dev CPU core-count; GPU model/VRAM/FP64 tier; eventual cluster shape) — pins
   the throughput floor and whether the float32+float64-ledger hybrid is mandatory or merely prudent.
   **Needs your input.**
4. **Determinism numeric definition:** resolved as exact bookkeeping, reproducible discrete decisions, and
   informational floating replay; cross-device physical correctness uses mixed tolerances rather than
   "within 1e-5 relative" (would let nondeterministic GPU scatter pass, dodging F4).
5. **Batch semantics:** flat `B=W·N` (simplest for field-less S0) vs keep `(W,N)` 2-D now because S1+
   per-world fields need it. Rec: **(W,N) 2-D from the start** to avoid an S1 re-layout.
6. **Realistic `N_cap` and max-segments-per-body** — set padded-batch size / masking overhead.
7. **CORE-only foraging sufficiency** — chemical gradient (fish's primary sense) has no robot analogue;
   if a CORE-only policy can't forage, the shared fish↔robot task becomes navigation-to-goal, not
   foraging. (Research question for S8-ish; noted now because it shapes the contract's EXT boundary.)
8. **ROS2 distro** (pins `Twist` vs `TwistStamped`, sensor_msgs versions) and whether the physical
   robot needs a chemical-sense analogue.
9. **Sophia in-process** (share torch tensors, skip proto) **vs out-of-process** (proto wire) — decides
   whether proto serialization is ever in the hot path.
