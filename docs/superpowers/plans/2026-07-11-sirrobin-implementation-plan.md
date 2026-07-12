# SirRobin — Implementation Plan

**Status:** Historical for S0; superseded by `2026-07-12-sirrobin-S0-consolidated-implementation-plan.md`.
Later-phase material remains planning input, not S0 execution authority. · **Date:** 2026-07-11 · **Repo root:**
`C:\Users\cddal\SirRobin` · **Top-level package:** `sirrobin`

## Scope

This plan is the actionable build sequence for SirRobin: a batched-torch, full-population marine evolution simulator whose architectural bet — *faithful full-population physics is affordable when vectorized* — is verified by measurement before commits are poured on it. It plans **repo scaffolding → the S0/SpikeSwim go/no-go spike → the S1/S2 keystone slices → the S3–S9 milestone contracts → the verification/salvage/language protocol → sequencing, risks, and definition-of-done.** It does not re-open settled design decisions (§2.7–2.9, §4.8, §6, §7); where a "why" is needed it cross-references the design doc rather than restating it.

The build is **depth-first**: nothing in a later slice begins until the prior slice's gate is green **on a telemetry artifact** (P7, §7.1.4 — no claim from code inspection). Three hard phase gates structure it: **(A) Prove the substrate** (scaffold + S0 SpikeSwim clears four gates at H1/H2), **(B) Close the books** (S1→S2→S3→S4, energy and mass ledgers close end-to-end), **(C) Emergence & embodiment** (S5→S9, each with a falsifiable acceptance criterion). "Done" is judged the same way everywhere: conservation ledger green, reproducible-within-seed on a fixed device, no capability read from a stat vector, no green-keeping `gain=0` dial, and the deciding number recorded in a telemetry artifact — never asserted from source.

## Table of Contents

1. [Repo Scaffolding & Module Map](#1-repo-scaffolding--module-map)
2. [S0 — SpikeSwim (the go/no-go spike)](#2-s0--spikeswim-the-gono-go-spike)
3. [S1–S2 — Keystone economy + canonical body](#3-s1s2--keystone-economy--canonical-body)
4. [S3–S9 — Milestone Contracts](#4-s3s9--milestone-contracts)
5. [Verification, Salvage & Language Protocol](#5-verification-salvage--language-protocol)
6. [Sequencing, Risks & Definition-of-Done](#6-sequencing-risks--definition-of-done)
7. [Immediate Next Actions (first week)](#7-immediate-next-actions-first-week)

---

## 1. Repo Scaffolding & Module Map

The scaffold's exit condition: a green repo where a trivial empty `Colony` steps deterministically from a console entry point and a **conservation-invariant CI job runs green on a fake reservoir ledger**. Everything is laid out so S0 and S1+ drop in with zero re-layout. All paths Windows-absolute under `C:\Users\cddal\SirRobin`.

### 1.1 Enforced layer order (one-way, CI-gated)

```
numerics → physics → fields → genetics → core → observe
```

Any upward import is a build failure (`import-linter`, the Python replacement for the C# asmdef firewall, §2.2). Cross-layer access is via `*/contracts.py` Protocols only (INV-W4, "query WHAT not HOW", §3.4).

### 1.2 Directory tree

```
SirRobin/
├─ pyproject.toml                 # build + all tool config (§1.4)
├─ setup.cfg                      # import-linter layered contract (§1.3)
├─ .gitattributes                 # * text=auto eol=lf ; *.py text ; oracle fixtures binary
├─ .gitignore                     # __pycache__, .venv, *.parquet, runs/, .pytest_cache, .mypy_cache
├─ README.md · CLAUDE.md · docs/
├─ oracle/                        # C#-donor fixture generator + FROZEN fixtures (§5.6)
│  ├─ SirRobinOracle.csproj        # net8.0 Unity-light console (Vector3/Quaternion/Mathf shim)
│  ├─ Program.cs                   # drives Reconstruct/LambK/Coast/MomentumLedger/StepTrace seams
│  └─ fixtures/                    # lambk_grid.npz, forces_H*.npz, aggregates_H*.parquet, coast/momentum
│                                  #   two configs each: gain0 (box mass), gain1 (ellipsoid) — §2.0
├─ scripts/
│  ├─ smoke_step.py                # console: empty Colony steps N ticks, prints ledger (accept artifact)
│  └─ run_gpu_checks.ps1           # dev-box CUDA determinism/throughput wrapper (CI skips gpu marker)
├─ src/sirrobin/
│  ├─ __init__.py                  # __version__ only; imports NOTHING from subpackages
│  ├─ numerics/                    # LAYER 0
│  │  ├─ dtype.py                  # HOT=float32, LEDGER=float64, device policy, Config-driven
│  │  ├─ rng.py                    # seed_everything(seed) + RngManifest (append-only draw stream)
│  │  ├─ ledger.py                 # ConservedLedger: f64 compensated (Kahan/pairwise) reservoir book
│  │  ├─ quat.py                   # quat_mul/rotate/from_euler_zyx/angle_axis/conj/normalize (batched)
│  │  ├─ solve_sym3.py             # batched closed-form cofactor solve (DONOR op-order; not linalg.solve)
│  │  └─ reduce.py                 # segment_index_add(values, body_id, n_bodies) — the ONLY sanctioned reduction
│  ├─ physics/                     # LAYER 1
│  │  ├─ contracts.py              # DevelopedBody, Pose, MediumSample, ForceTorquePower (dataclasses)
│  │  ├─ force.py                  # ForceContributor Protocol + NullContributor stub
│  │  ├─ lamb.py                   # LambK: f64 Simpson-2048 quadrature, α-normalize, k=α/(2−α)
│  │  ├─ reconstruct.py            # genome→DevelopedBody DFS build → flattened/CSR (mirrors BuildPart)
│  │  ├─ pose.py                   # bounded 6-pass depth-scan (gather-compose-scatter)
│  │  ├─ swim_step.py              # S0 kernel: batched FROZEN-HEADING Step → state + StepLedger
│  │  ├─ step_live.py              # S2 kernel: yaw-integrating StepLive + P-controller
│  │  └─ capabilities.py           # S2: morphology-derived feeding/metabolism/defense (kills eff[])
│  ├─ fields/                      # LAYER 2
│  │  ├─ contracts.py              # Field.sample(x)->(value,grad), Geology, SourceSet Protocols
│  │  ├─ geology.py                # FlatGeologyStub (const elevation, empty sources)
│  │  ├─ light.py                  # AnalyticLight (Beer–Lambert, no storage at S1)
│  │  ├─ scalar_field.py           # S1: co-grid (W,G,G,B) + interp sample + conservative flux ops
│  │  ├─ nutrient.py               # S1: Nd dissolved field (deplete/deposit/mix, double-buffered)
│  │  ├─ chem.py                   # S1: Monod/Liebig/Redfield/Martin pure fns
│  │  └─ detritus.py               # S1: Bd marine-snow reservoir + Martin sinking
│  ├─ genetics/                    # LAYER 3
│  │  ├─ genotype.py               # SoA node/edge graph tensors (§5.2)
│  │  ├─ develop.py                # batched fixed-depth frontier unroll → DevelopedBody (§5.3)
│  │  └─ innovation.py             # monotone iid registry
│  ├─ core/                        # LAYER 4
│  │  ├─ config.py                 # Config (frozen), WorldConfig
│  │  ├─ clock.py                  # SimClock (Now f64, Dt f32, Step i64, Scale)
│  │  ├─ state.py                  # ColonyState SoA (§1.5) + serialize/load
│  │  ├─ colony.py                 # Colony.reset/step/state/load (S0 no-op step)
│  │  ├─ spatialhash.py            # shared point-entity spatial hash (stub at S0)
│  │  ├─ ledger.py                 # S1: reservoir registry + paired transfer + close_books
│  │  ├─ economy.py                # S1: conserved-loop step orchestration
│  │  └─ parcels.py                # S1 [MEASURE-1] Lagrangian escape hatch (stubbed, wired only if fork trips)
│  ├─ observe/                     # LAYER 5
│  │  ├─ telemetry.py              # TelemetryWriter: parquet/jsonl of gate metrics
│  │  ├─ contract.py              # CORE/EXT dict-of-tensors builders (Talos seam stub)
│  │  ├─ telemetry_cons.py         # S1: [CONS] line, inventory parquet, bloom/desert heatmaps
│  │  └─ telemetry_loco.py         # S2: cruise/COT/reactive-ratio, morphospace, impulse budget
│  └─ spikeswim/                   # S0 driver (NOT a runtime layer; imports physics/numerics only)
│     ├─ genomes.py                # H0/H1/H2 genome-batch generators + padded ablation builder
│     ├─ episode.py                # RunEpisode: 360 warmup + 600 measure = 960 steps; aggregates
│     ├─ churn.py                  # fixed-cap (W,N_cap)+alive-mask; kill 2%/1000; refill; compact/K
│     ├─ bench.py                  # sweep harness (B×device×dtype×rung×het×layout) → parquet/jsonl
│     └─ cli.py                    # python -m sirrobin.spikeswim.bench …
└─ tests/
   ├─ conftest.py                  # device param, seeded_rng, fake_ledger, tiny_config, fixture loaders
   ├─ test_conservation.py         # PRIMARY GATE: closed box closes; leak is caught
   ├─ test_determinism.py          # within-seed bit-identity on fixed device
   ├─ test_import_boundary.py      # import-linter contract as a pytest (belt+braces)
   ├─ test_smoke_step.py           # empty Colony reset/step; static shapes; ledger closes
   ├─ numerics/  physics/  fields/ # package-local unit tests
   └─ fakes/                       # fake_geology.py, fake_body.py, fake_field.py (contracts, not mocks)
```

`src/` layout is deliberate: it forces the installed-package import path so the linter contract and CI reflect the shipped topology (§2.2).

### 1.3 Import boundary (`setup.cfg`)

```ini
[importlinter]
root_package = sirrobin

[importlinter:contract:layers]
name = SirRobin one-way layering
type = layers
layers =
    sirrobin.observe
    sirrobin.core
    sirrobin.genetics
    sirrobin.fields
    sirrobin.physics
    sirrobin.numerics

[importlinter:contract:contracts-only-crossings]
name = Cross-layer access via contracts modules only
type = forbidden
source_modules = sirrobin.physics
                 sirrobin.genetics
forbidden_modules = sirrobin.fields.geology
                    sirrobin.fields.light
                    sirrobin.core
                    sirrobin.observe
```

CI: `lint-imports --config setup.cfg`. `tests/test_import_boundary.py` also runs the contract programmatically so a regression fails a normal `pytest`. Interface opacity (INV-W4): no `plate`/`seed`/`hotspot`/`octave` symbol is reachable from a consumer.

### 1.4 Pinned deps (`pyproject.toml`, key excerpts)

`requires-python = ">=3.11,<3.12"`. Exact `==` pins on everything — determinism is only defined against a fixed `(device, dtype, op-order)` tuple (§2.7).

```toml
dependencies = [ "torch==2.5.1", "numpy==2.1.3", "pyarrow==18.1.0", "pydantic==2.10.4" ]
[project.optional-dependencies]
dev = [ "pytest==8.3.4", "pytest-xdist==3.6.1", "import-linter==2.1",
        "ruff==0.8.4", "mypy==1.13.0", "hypothesis==6.122.3" ]
[tool.pytest.ini_options]
markers = ["conservation", "determinism", "oracle", "gpu"]
```

CUDA install (the one med-risk item, verified against the real GPU in T2 acceptance):
```powershell
python -m venv .venv ; .\.venv\Scripts\Activate.ps1
python -m pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cu124
python -m pip install -e ".[dev]"
```
**T2 acceptance:** `torch.zeros(1, device="cuda")` and `"cpu"` both succeed; `pip check` clean; `ruff check` + `mypy src` pass on stubs. Record `torch.__version__, torch.version.cuda` in the run manifest.

### 1.5 Canonical state layout (`core/state.py`)

`ColonyState` is struct-of-arrays, **shapes STATIC**, `alive`-mask is the sole source of "exists". Births recycle dead slots via a **precomputed free-slot list** (never atomic append); dead rows are **masked, never `index_select`-ed** (every reduction multiplies by `alive`).

| tensor | shape | dtype | notes |
|---|---|---|---|
| `alive` | `[W,N_cap]` | bool | source of truth for existence |
| `pos` | `[W,N_cap,2]` | f32 | periodic world position [m] |
| `heading` | `[W,N_cap]` | f32 | yaw [rad] |
| `lin_vel` | `[W,N_cap,2]` | f32 | [m/s] |
| `ang_vel` | `[W,N_cap]` | f32 | [rad/s] |
| `energy` | `[W,N_cap]` | f32 | metabolic reserve [J] |
| `nutrient` | `[W,N_cap]` | f32 | structural nutrient [mol] |
| `body_ptr` `genome_ptr` | `[W,N_cap]` | i64 | → flattened segment / genome tensors |
| `age` | `[W,N_cap]` | f64 | [s] |
| `species_tag` | `[W,N_cap]` | i64 | observational read-out ONLY (never gates mating) |

### 1.6 Frozen seam signatures (created as importable stubs at scaffold time)

```python
# core/config.py
@dataclass(frozen=True)
class Config: device="cpu"; dtype_hot="float32"; n_cap=1024; w=1; dt=1/120; world=WorldConfig()

# core/clock.py — advance() is the ONLY writer of time
@dataclass
class SimClock: now=0.0; dt=1/120; step=0; scale=1.0
    def advance(self, dt): self.now += dt; self.step += 1; self.dt = dt

# core/colony.py — Gymnasium-shaped; step is a pure fn of (prior state, dt, action); torch.inference_mode()
class Colony:
    def reset(self, seed:int)->dict          # seed_everything; SimClock→0; ledger.reset(reservoirs); obs
    def step(self, dt:float, action:dict)->dict   # S0: NO-OP integrator; ledger.assert_closed(tol=1e-9)
    def state(self)->ColonyState ; def load(self, s)->None

# physics/contracts.py
@dataclass class DevelopedBody:   # flattened/CSR [S_total] seg tensors + per-body scalars (§4.2)
@dataclass class MediumSample:    # density, flow, submersion∈[0,1] continuous waterline (§3.11)
@dataclass class ForceTorquePower:# force[B,3], torque_yaw[B], power_dissipated[B] — SUMMED never switched

# physics/force.py
class ForceContributor(Protocol):
    def accumulate(self, body:DevelopedBody, pose, vel, medium:MediumSample)->ForceTorquePower: ...

# fields/contracts.py
class Field(Protocol):   def sample(self, x)->tuple[Tensor,Tensor]: ...   # (value, grad); INV-W3
class Geology(Protocol): def elevation(x_hz); mineral(x_hz,el); heat_sources(); active_sources(t); river_sources(t)

# numerics
def seed_everything(seed:int)->None            # seeds py/np/torch; use_deterministic_algorithms(True); CUBLAS env
def solve_sym3(m00,m01,m02,m11,m12,m22, rhs)->Tensor["B,3"]   # donor cofactor op-order EXACTLY
def segment_index_add(vals, body_id, B)->Tensor  # deterministic; precomputed unique slots; NEVER scatter_add_
```

### 1.7 Determinism rules baked in day one (§2.7)

- `CUBLAS_WORKSPACE_CONFIG=":4096:8"` set in `seed_everything` **and** exported in CI env before Python starts.
- `torch.use_deterministic_algorithms(True, warn_only=False)`; no cudnn benchmark.
- **No nondeterministic atomic scatter.** `numerics/reduce.py::segment_index_add` is the only sanctioned segment reduction; a Grep-based CI check forbids raw `scatter_add_`/non-unique `index_add_` in the hot loop.
- `RngManifest` is append-only: an inert gene appends a zero-count entry, so latent genome capacity never shifts the stream (I-GENOME-5).
- Determinism tests run single-worker (`pytest -p no:xdist -m determinism`); xdist for the rest.

### 1.8 Conservation gate scaffold (PRIMARY, `tests/test_conservation.py`)

`ConservedLedger` (f64 compensated): `closure_residual = |ΣR(t) − ΣR(0) − ∫X_ext| / max(ΣR(0),ε)`. Both directions asserted — a gate that only ever passes is the byte-identity trap reborn (§7.1.1):

```python
@pytest.mark.conservation
def test_closed_box_books_close_over_soak():        # closed box < 1e-9/step over 10^4 steps, slope ≤ 0
@pytest.mark.conservation
def test_leak_is_caught():                          # minting matter with no declared flux MUST raise
```

### 1.9 CI pipeline (job order encodes priority)

`.github/workflows/ci.yml`, CPU-only runner (CUDA determinism validated on dev box, reported not gated cross-machine, §7.1.2). Order: **`boundary` (import-linter + ruff + mypy, fail-fast) → `conservation` (hard `needs:` for all downstream) → `determinism` → `unit`.** `gpu`-marked tests skipped in CI, run via `scripts/run_gpu_checks.ps1`.

### 1.10 Scaffold task list & acceptance gates

| # | Task | Effort | Risk | Blocks |
|---|---|---|---|---|
| T1 | Repo init, `.gitignore/.gitattributes`, dir tree, `__init__.py` per pkg | S | low | all |
| T2 | `pyproject.toml`, pinned deps, editable install, CUDA verify | M | med | T3–T10 |
| T3 | `numerics`: dtype/device, rng+manifest, `ConservedLedger`, `quat`, `solve_sym3`, `reduce` | M | med | T5,T6,T8 |
| T4 | Contract stubs: `physics/contracts+force`, `fields/contracts+geology+light` | M | low | T5,T7 |
| T5 | `core`: Config, SimClock, ColonyState, Colony.reset/step (no-op) | L | med | T7–T10 |
| T6 | `observe`: telemetry writer + reservoir snapshot surface | M | low | T9 |
| T7 | import-linter contract wired to CI | S | low | T10 |
| T8 | determinism harness + `smoke_step.py` | M | med | T9 |
| T9 | pytest harness, fakes, **conservation scaffold test** + determinism + boundary | M | high | T10 |
| T10 | CI pipeline (conservation-first) + console smoke app | M | low | — |

Sequencing: **T1 → T2 → (T3 ∥ T4) → T5 → (T6 ∥ T7 ∥ T8) → T9 → T10.**

| Gate | Assertion |
|---|---|
| G-SCAF-1 | `pip install -e ".[dev]"` clean; cuda+cpu tensors both allocate |
| G-SCAF-2 | `lint-imports` passes; an injected `physics→core` import **fails** it |
| G-SCAF-3 | Closed box `closure_residual < 1e-9` over 10⁴ steps, bounded-oscillating; leak fake caught |
| G-SCAF-4 | `reset(seed).step(...)` runs; all `ColonyState` shapes static; no `index_select` in step path |
| G-SCAF-5 | Within-seed bit-identical on fixed device (CPU gated; CUDA reported) |
| G-SCAF-6 | `smoke_step.py` prints `OK` + `bit-identical: True` on CPU and CUDA |
| G-SCAF-7 | `FlatGeologyStub` substitution test: swap for a second fake `Geology` impl → no consumer line changes |

Only when G-SCAF-1…7 are green does S0 begin.

---

## 2. S0 — SpikeSwim (the go/no-go spike)

### 2.0 Purpose & the one decision S0 exists to make

S0 is a **standalone batched-torch port of the donor's one-shot, frozen-heading `SwimEval.Sim.Step`** (`SwimEval.cs:740`) over `B = W·N_cap` ragged bodies, measured against four gates. It is the **go/no-go for the entire vectorization thesis** (§7.2). No ecology, no genome mutation, no steering. The frozen-heading path only: `_fThrust`/`_nThrust` constant unit vectors, `_vCom.y≡0`, no yaw integration (`StepLive` is S2, re-measured then — open Q #6).

**Authorization = H1/H2 clear gate (d) while (a)(b)(c) hold** (meta-falsifier). A green **H0** number is non-conforming as authorization; any report leading with H0 is rejected.

**Mass-model migration handled in two stages** (open item #3, §4.2) so "is my vectorization correct" is never entangled with "did I change the mass model":
- **Stage A (oracle-clean):** generate fixtures from the donor at `ellipMassGain = 0` (box mass `max(0.1, sc.x·sc.y·sc.z·density)`), port to bit-clean match.
- **Stage B (port target):** flip port segment mass to ellipsoid `(π/6)`, re-record `gain1` fixtures, re-pass gate (c). Both fixture sets retained.

**Prerequisites to reading the result as authoritative:** open Q #1 (pin throughput floor to the real dev CPU/GPU — 3.07e7 is a placeholder), open Q #6 (StepLive re-measure deferred to S2), open Q #12 (`N_cap`, max-seg pinned against the H1 raggedness profile). Recorded in T17.

### 2.1 Tensor layouts

**Creature level `(W,N_cap)`**, flattened to `[B]` for the kernel: `alive` bool; `vcom[B,3]` (`vcom[...,1]≡0` each step); `xcom[B,3]`; `fthrust,nthrust[B,3]`; `swim_freq[B]`; `mt[B]` (tail transverse added-mass/length); `fin_active[B]` bool + `fin_S,fin_AR,fin_AL,fin_align[B]`; `mbody[B]`; `tail_gidx[B]` i64; `body_base[B]` i64.

**Segment level — flattened/CSR `[S_total]`** (NOT padded-to-16, §2.4.2): `seg_localPos[S,3]`, `seg_localRot[S,4]`, `seg_abc[S,3]`, `seg_mass[S]` (box or ellipsoid per stage), `seg_areaZ[S]`, `seg_ma[S,3]`, `seg_ampDeg[S]`, `seg_phase[S]`, `seg_c[S]`, `seg_isTail[S]` bool, `seg_parent_g[S]` i64 (**global** parent = `body_base+local_parent`, or −1), `seg_depth[S]` i64, `body_id[S]` i64. Working: `pos[S,3]`, `rot[S,4]`, `prev_pos[S,3]`.

**Padded ablation** `(B,16,·)+seg_mask[B,16]` — built only to *measure* the masking tax (F1), never on the main path. Every per-body reduction is `segment_index_add` over `body_id`; no boolean `index_select` in the hot loop.

### 2.2 Kernel port — `Sim.Step` → batched torch (`physics/swim_step.py`)

Each substep maps 1:1 to `SwimEval.cs:740–817`; all `torch.where`/mask, no Python `if` over the batch, wrapped in `torch.inference_mode()`.

| # | Donor | Batched operation |
|---|---|---|
| S2 | `_prevPos=_pos` | `prev_pos = pos.clone()` |
| S3 | `prevTip=TailTip()` | gather tail: `pos[tail_gidx] + quat_rotate(rot[tail_gidx], ẑ)*c` |
| S4 | `Pose(_t)` | **6-pass depth-scan**: pass k updates `depth==k` segs by gathering parent `(pos,rot)` at `seg_parent_g`; `rot=quat_mul(pRot, localRot·angleAxis(θ,ŷ))`, `pos=pPos+quat_rotate(pRot,localPos)`; `θ_j=ampDeg_j·sin(2π·swimFreq·t+phase_j)` masked by `hasJoint`. Fixed 6 passes ⇒ static shape |
| S5 | tail kinematics | `uTail=vcom+(tipNow−prevTip)/Dt`; `U=⟨uTail,fthrust⟩`, `Vt=⟨uTail,nthrust⟩`, `s=⟨quat_rotate(rot[tail],ẑ),nthrust⟩` |
| S6 | reactive | `tReact=0.5·mt·(Vt²−U²s²)`; `Wt=Vt+U·s`; `pWake=0.5·mt·relu(U)·Wt²` |
| S7 | fin channel | `where(fin_active, CirculatoryLiftThrust(...), 0)`→`tFin,pFin`; AoA clamp ±0.35, `q=0.5ρ·relu(U)²·S`, `CDi=0.02+CL²/(π·0.9·AR)` |
| S8 | segment drag | `vloc=quat_rotate(conj(rot),uj)`; `floc=(0,0,−0.5ρ·cd·areaZ·|vloc.z|·vloc.z)`; `fDrag=seg_sum(quat_rotate(rot,floc))`, `wDrag=seg_sum(relu(−⟨fworld,uj⟩))` |
| S9 | M_eff | cols `cx=quat_rotate(rot,x̂)`…; 6 entries `m00..m22 += Σ ma·(col⊗col)`; add diagonal `mbody·250` **(250 on struct only, not added-mass)** |
| S10 | integrate | `fStream=(tReact+tFin)·fthrust+fDrag`; `dv=solve_sym3(m_eff, fStream·Dt)`; `vcom+=dv`; `vcom[...,1]=0`; `xcom+=vcom·Dt` |
| S11 | measuring | f64 ledger accum: `ReactiveImpulse, DragImpulse, MechWork, FwdThrustImpulse` |

**Build-time precompute (once/body):** reconstruct segments; LambK→`seg_ma` (f64); `mt = maX_tail/max(2c_tail,1e-4)` OR fin-rebuilt from `finMaPerp·FinReactiveScaleOf(align,gain)/(2c)` (`SwimEval.cs:575–576`); `mbody=Σ seg_mass`; `tail_gidx` by max `restPos.z`, ties → later DFS index (`SwimEval.cs:142`).

`StepLedger` (f64): `p_in, t_react, p_wake, t_fin, p_fin, w_drag, dv[B,3], m_eff[B,6]`.

### 2.3 Watch-items (flag, don't bulldoze)

1. **KgPerSimMass=250 is asymmetric** (`SwimEval.cs:771,1023`) — scales *structural* inertia diagonal only, applied **before** adding `Σ ma·(col⊗col)`. Getting it wrong silently fails gate (c) on `dv` while `M_eff` off-diagonals match; the momentum-ledger test (T9) is the independent catch.
2. **Fin-tail `mt` rebuilt from `finMaPerp`, not `maX`** — H1's ~40% fin tails exercise this in the authorizing config.
3. **`_pos[j]` excludes own joint flex; `_rot[j]` includes it** (`SwimEval.cs:685–687`); `TailTip` adds `_rot[tail]·ẑ·c`. Depth-scan must preserve this asymmetry or `Vt`/`s` drift.
4. **Tail tie-break = later DFS index** on equal `restPos.z` — or H1 mirror pairs pick the wrong tail.
5. **Degenerate guards** (`SolveSym3` det<1e-12 fallback; `mBody≤1e-6`; empty body) must be `torch.where`, never dropped rows (breaks static shape + determinism).

### 2.4 Oracle fixtures (disposition B, §7.5)

`oracle/SirRobinOracle.csproj` — a Unity-light headless console (Vector3/Quaternion/Mathf shim preferred over linking `UnityEngine.dll`) driving the confirmed seams `ReconstructForTest`/`LambKForTest`/`CoastTest`/`MomentumLedger` plus a new `StepTraceForTest`. The donor is **never called live**; recorded values are checked to tolerance (§7.1.4).

| Fixture | Seam | Contents | Gate |
|---|---|---|---|
| `lambk_grid.npz` | LambKForTest | (a,b,c) grid: sphere/2:1/4:1/10:1 prolate/oblate/degenerate → (kx,ky,kz) f64 | (c)-1 |
| `reconstruct.jsonl` | ReconstructForTest | per-seg restPos/restRot/abc/mass/areaZ/ma/phase/ampDeg/isTail/parent/depth | T6/T7 |
| `forces_H*.npz` | StepTraceForTest | first 32 steps of tReact,pWake,pFin,m00..m22,dv | (c)-2 |
| `aggregates_H*.parquet` | Evaluate | 8s (960-step) cruiseSpeed,costOfTransport,reactiveRatio,mechWork | (c)-3 |
| `coast.npz` | CoastTest | vEnd after 600 zero-force steps | invariant |
| `momentum.npz` | MomentumLedger | vFinal,fHat,forwardThrustImpulse (all-sphere) | invariant |

**Genome corpus** (one canonical JSON source feeds both C# and torch): **H0** homogeneous 6-seg axial; **H1** ragged 2–16, mean~6, mirror-paired, ~40% Surface fin tails; **H2** skewed (mostly 2–3 seg + rare 16). ~64 seeded genomes/class. Two configs recorded: `gain0`, `gain1`.

### 2.5 The four gates (concrete tests)

| Gate | Test | Assertion |
|---|---|---|
| **(a) Determinism** | `test_determinism.py::rerun_bit_identity` | Two same-process reruns, det-mode, CPU **and** CUDA-det: `max_abs(Δ)==0` over `(vcom,xcom,ledger)` after 960 steps. GPU-nondet: report only. |
| | `::churn_deterministic` | With churn stub, rerun bit-identity still `==0` (RK-6). |
| **(b) Energy inst.** | `test_energy.py::algebraic_closure` | Every step/body `|p_in−(tReact·U+pWake+pFin)|/max(|p_in|,ε) < 1e-6` (f32). |
| **(b) Energy long** | `::ke_budget_1e5` | 1e5 steps `|ΔKE−Σ(P_musc−P_diss)·Dt|/Σ|work| < 1e-3` f32 / `<1e-6` f64; **gate on drift curve bounded-oscillating, not endpoint** (F6). |
| **(c) LambK** | `test_lamb.py::vs_fixture` | `<1e-6` abs (f64); sphere-exact `k==0.5`; slender limits. |
| **(c) Forces** | `test_step_forces.py::vs_trace` | tReact,pWake,pFin,dv,6×M_eff rel `<1e-4` (f64), first 32 steps, H1/H2. |
| **(c) Aggregates** | `test_episode_oracle.py::vs_agg` | cruise/CoT/reactiveRatio rel `<1e-3`. Never gate a long-horizon bit-trace (chaotic gait, F5). |
| **(d) Throughput** | `bench.py`→`test_throughput.py::floor` | creature-steps/s clears **3.07e7 with ≥10× headroom** at H1/H2 *(placeholder until Q#1)*; locate CPU↔GPU crossover `B*`; report H1/H0 + padded/flattened tax; profiler shows **force/solve-bound, not pose-scan/reduction-bound** (F2). |

Ported invariants (must stay green): `test_coast_momentum.py` — zero-force coast `|vEnd−v0|<1e-4`; all-sphere `(vFinal·fHat)·M_eff == ΣtReact·Dt` within `1e-3·|impulse|+1e-6` (independent catch for the ×250 folding).

### 2.6 Falsifiers

| ID | Falsifier | Detector |
|---|---|---|
| F1 | Ragged heterogeneity defeats batching | H1 flattened slower than per-body loop, or H2 padded craters |
| F2 | Depth-scan/reductions > 50% of step | profiler attribution row |
| F3 | Per-op launch overhead dominates at realistic N even at r2 | rung sweep r0→r1→r2 |
| F4 | Determinism tax > 2× on GPU | gate-(a) timing with/without det mode |
| F5 | Oracle un-portable without de-vectorized loop | (c) aggregates diverge >1e-3 irreducibly |
| F6 | f32 semi-implicit energy drift monotone | (b) long-horizon curve shape |
| F7 | Churn/compaction swamps step | `churn.py` cost % of step |
| **META** | **H0-green ≠ authorization** | report must lead with H1/H2 |

### 2.7 Benchmark sweep (`spikeswim/bench.py`)

Axes: `B ∈ {1,64,256,1024,4096,16384,65536,262144}` × `device∈{cpu,cuda}` × `dtype∈{f32,f64}` × `rung∈{r0 eager, r1 torch.compile(reduce-overhead), r2 CUDA-graph}` × `het∈{H0,H1,H2}` × `layout∈{flattened,padded}`. Protocol: warm/JIT excluded, `cuda.synchronize()` around timed region, median of K reps, `torch.profiler` on one rep/cell. One parquet row/cell + jsonl manifest (seed, config, git SHA, HW). No claim from anything but this artifact.

### 2.8 Task list

| # | Task | Dep | Effort/Risk | Acceptance |
|---|---|---|---|---|
| T0 | Repo scaffold (§1) done, import-linter, det conftest | — | S/L | linter passes; `pytest` collects |
| T1 | `numerics/quat.py` | T0 | M/M | matches reference 1e-6; **Euler order matches UnityEngine.Quaternion.Euler** |
| T2 | `numerics/solve_sym3.py` | T0 | S/L | vs `numpy.linalg.solve` rel<1e-6; degenerate branch matches donor |
| T3 | `oracle/` console + shim + StepTrace seam; emit **gain0** | T0 | M/M | 6 fixture files; re-run bit-stable; corpus shared |
| T4 | `numerics/ledger.py` compensated f64 | T0 | S/L | Kahan of 1e7 terms beats f32 by ≥6 digits |
| T5 | `physics/lamb.py` (f64 Simpson-2048) | T1 | M/M | (c)-1 <1e-6 abs; sphere-exact 0.5 |
| T6 | `physics/reconstruct.py`+`developed_body` (DFS→CSR, global parent, tail select) | T1,T3 | L/M | pose part-1 rel<1e-4 (gain0) |
| T7 | `physics/pose.py` 6-pass depth-scan | T1,T6 | L/H | world pos/rot rel<1e-4 at t∈{0,Dt,17Dt}; static 6 passes |
| T8 | `physics/swim_step.py` S3–S11 | T2,T5,T7 | L/H | (c)-2 all terms rel<1e-4 vs trace (gain0) |
| T9 | ported CoastTest + MomentumLedger | T8 | S/M | coast `<1e-4`; sphere ledger identity `<1e-3·|imp|` |
| T10 | `spikeswim/episode.py` + (c)-3 | T8 | M/M | cruise/CoT/Γ rel<1e-3 (gain0) |
| T11 | **Stage B ellipsoid flip**; re-record **gain1**; re-pass (c) | T10 | S/M | (c)-1/2/3 green vs gain1; both sets retained |
| T12 | gate (b) energy inst + 1e5 budget + drift curve | T8,T4 | M/H | inst<1e-6; budget<1e-3 f32/1e-6 f64; bounded-oscillating |
| T13 | gate (a) determinism CPU+CUDA | T8 | S/M | `max_abs(Δ)==0`; F4 tax measured |
| T14 | `spikeswim/churn.py` + determinism + cost | T13 | M/M | churn bit-identical; compaction cost % emitted (F7) |
| T15 | padded `(B,16)+mask` ablation + `test_ablations.py` | T8 | M/L | padded≡flattened within f32 tol; masking tax (F1) |
| T16 | `spikeswim/bench.py` full sweep r0→r2 + profiler + parquet | T8,T14,T15 | L/H | gate (d): floor+≥10× headroom at H1/H2 OR falsifier tripped; `B*` located; force/solve-bound |
| T17 | **Go/No-Go report** (H1/H2-led; record Q#1, Q#6, Q#12 caveats) | all | S/L | all four gates' status at H1/H2, falsifier ledger, explicit GO/NO-GO |

**Critical path:** T0→T1→T6→T7→T8→(T10,T12,T13)→T16→T17. Do not start T16 until T8/T12/T13 green (a fast wrong or nondeterministic kernel is worthless). Write T17 from bench telemetry only. **Only on green at H1/H2 does S1 begin.**

---

## 3. S1–S2 — Keystone economy + canonical body

Two measure-first, reversible decisions flagged inline:
- **[MEASURE-1] S1 field representation** — Eulerian co-grid (default) vs Lagrangian parcels for `Bp`/`Bd`, decided by the healing-vs-consumption ratio under dense grazing (§6.2, open Q #7). This slice builds the Eulerian path + the measurement rig; parcels are the escape hatch, queued as its own slice if the fork trips.
- **[MEASURE-2] `StepLive` throughput** — the yaw-integrating kernel must re-clear the S0 affordability floor **before S2 is committed** (§4.8, §7.3, open Q #6).

### S1 — Conserved single-nutrient economy (the keystone)

**Design inversion from the donor:** the donor grew biomass toward a static Perlin cap (`PlanktonField.cs:114`, the #1 mint bug) and discarded remineralized snow (`MarineSnowField.cs:56–58`). S1 makes both structurally impossible: every gram of N moves between tracked reservoirs, `Σ reservoirs = const` (§6.1, §6.6). The donor's *equations* (`NutrientChem.cs`) and its *conservation-test pattern* port forward; its non-conserving *architecture* does not.

Closed inventory: `I_N = ΣNd + ΣBp + ΣBd + ΣBm + Σ_alive struct_N + ΣSed`. Box closed at S1 (no geology source, no burial sink) → `I_N(t)==I_N(0)` to tolerance.

#### S1.A — Reservoir model + ledger (build first)

| Task | Detail | Effort/Risk |
|---|---|---|
| **S1.1** Reservoir registry | Named f32 tensors: `Nd,Bp,Bd,Bm (W,G,G,B)`, `struct_N (W,N_cap)`, `Sed (W,G,G)` (inert at S1). Each registered with name+shape. | S/Low |
| **S1.2** Paired-transfer primitive | `core/ledger.py::transfer(src,dst,amount)` debits/credits by the **identical tensor** via deterministic masked add — the ONLY sanctioned way to move N. | M/Med |
| **S1.3** `close_books()` | f64 compensated sum over all registered reservoirs every N ticks; returns `closure_residual`, emits `[CONS]` line. | S/Low |

```python
# core/ledger.py
def register_reservoir(name:str, t:Tensor)->None
def transfer(src:Tensor, dst:Tensor, amount:Tensor)->None   # |Δsrc+Δdst|==0 by construction
def close_books()->LedgerResidual                            # f64; asserts INV-MASS/INV-TRANSFER
```

**Acceptance S1.A:** random `transfer` sequence leaves `close_books().residual==0` (f64); any direct write to a registered reservoir outside `transfer` is caught by an audit test (INV-TRANSFER `<1e-6` f32/op).

#### S1.B — [MEASURE-1] field representation

| Task | Detail |
|---|---|
| **S1.4** `fields/scalar_field.py` port | Co-grid `(W,G,G,B)` with `sample(x)->(value,grad)` **trilinear interpolated** (port wrap-torus bilinear of `PlanktonField.cs:274–287`, generalized to 3D + analytic grad); transactional `deplete` returning amount-removed; f64 `total_inventory`. All biological reads via `sample` (INV-W3); graze stays on continuous position, never a cell index (P5). |
| **S1.5** `core/parcels.py` stub | `Bp`/`Bd` as point-entities in the shared spatial hash. Built but **not wired** unless S1.13 trips. |
| **S1.6** quantization rig | Log healing rate (diffusion+regrowth) vs consumption at graze hotspots; a forage-probe smaller than one cell must read nonzero slope from `sample.grad`. |

**Decision rule (§6.2):** switch producer/detritus biomass to parcels only if dense grazing quantizes or the co-grid needs ~1 param/feature; `Nd`/temp/light stay on the grid regardless. Report the measured ratio in the S1 artifact; do not assert the default is fine.

#### S1.C — The closed nutrient loop

`fields/chem.py` ports `NutrientChem.cs` verbatim (batched): `monod`, `liebig_min` (S1 is `min`, not the ramp Lerp), `redfield_uptake`, `martin_weights` (sum==1).

| Task | Detail |
|---|---|
| **S1.7** Primary production | `primary_production(Nd,Bp,light,cfg)->(dBp,dNd)`; `Gp=μ_max·min(γ_L,f_N)·Bp`; **`dNd=−dBp` exactly, `dBp←min(dBp,Nd)` strict guard, via `transfer(Nd,Bp,dBp)`. NO `baseCap`** — bloom self-terminates by emptying its own `Nd`. |
| **S1.8** Remineralization + BGE split | `R=k_remin(z)·Bd`; `dBm=+BGE·R·dt` (BGE≈0.2), `dNd=+(1−BGE)·R·dt`, `dBd=−R·dt`; INVARIANT `dBm+dNd==−dBd`; two paired transfers. Fixes `MarineSnowField.cs:56` discard. |
| **S1.9** Martin sinking | `Bd` sinks with `w_sink`; depth-dependent `k_remin(z)` via `martin_weights`; deposit split across bands (port `NutrientField.DepositColumn`). Detritus **accumulates** (days–weeks turnover, not donor's ~9s) so deposit-feeders get standing stock. |
| **S1.10** Vertical mixing | `vertical_mix(Nd,Kz,dz,dt)` flux-form, **double-buffered** (port `NutrientField.TickTransport:46–61`); positivity clamp (INV-W5). |
| **S1.11** Pump wiring | `core/economy.step()`: production→`Bd` sink→deep remin (Martin)→deep `Nd` enrichment→mixing returns it. Zonation must **emerge**; each sub-step calls only `transfer`. |

Light: analytic Beer–Lambert `I(z)=I0·exp(−k_att·(−z))` (§3.9), no storage at S1.

#### S1.D — Telemetry

**S1.12** `[CONS]` line + inventory curve (gate on bounded-oscillating drift, not endpoint). **S1.13** Bloom/desert heatmaps of `Bp`/`Nd` + healing/consumption ratio → this is where [MEASURE-1] is read off.

#### S1 Tests

| Test | Asserts | Threshold |
|---|---|---|
| `test_transfer_paired` | `|Δsrc+Δdst|` | `<1e-6` f32 |
| `test_closed_box_inventory` | draw+remin+deposit+mix leaves `I_N` invariant | `<1e-9` f64 |
| `test_production_drawdown_1to1` | `dNd==−dBp`; `dBp≤Nd` | exact/guard |
| `test_bge_split_closes` | `dBm+dNd==−dBd` | `<1e-9` f64 |
| `test_mixing_conserves` | column Σ conserved, `≥0` | `<1e-9` f64, positivity exact |
| `test_no_static_cap` | AST guard: no `baseCap`/`carryingCap`/`nicheCapFrac` drives production | symbol absent |
| `test_amortized_equals_whole` | row-sliced tick == whole-grid tick | `<1e-9` f64 |
| `test_soak_drift` | 1e6-step run bounded-oscillating, not monotone | drift `<1e-6` |
| `test_bloom_self_terminates` | seeded pulse → bloom → drawdown → crash, no cap knob | qualitative + closed |

#### S1 Acceptance Gate

1. Books close `<1e-9`/step (f64), drift `<1e-6` over ≥1e6 steps, bounded-oscillating.
2. Every transfer passes INV-TRANSFER `<1e-6`.
3. Blooms/deserts emerge from the loop — bloom draws its own `Nd` down and crashes with **no hand-coded ceiling**; drifters plateau at production÷intake; zonation falls out of the pump.
4. **[MEASURE-1] recorded** — Eulerian-vs-parcel fork decided from telemetry, number in the artifact.
5. No-go honored: if the ledger cannot close `<1e-9` without an ad-hoc correction, root-cause upstream — **never re-soften depletion to re-hide a mint** (§1.2, §6.9).

**S1 sequencing:** S1.1→S1.2→S1.3 (ledger first) → S1.4 → S1.7/S1.8/S1.9/S1.10 (each with its paired-transfer test green before the next) → S1.11 → S1.12/S1.13 → soak + gate. S1.5/S1.6 parallel with S1.4.

### S2 — One canonical body + live locomotion

**Design stance:** replace S0's hand-authored segment tensors with **genome→develop→DevelopedBody**, swap the frozen-heading kernel for the yaw-integrating live one, and **delete the scalar proxy** — morphology-through-physics is the only capability driver (§4.5, §5, §7.3). The donor laundered capability through `Measure().swimProxy` (`BodyGraph.cs:349–358`) and the `eff[]` stat vector; S2's headline gate is "kill `eff[]`."

#### S2.A — Batched development scan (genome P0; dovetails S0)

| Task | Detail | Effort/Risk |
|---|---|---|
| **S2.1** DevelopedBody contract freeze | Lock `[S_total]` SoA (§4.2): center, rest_rot, abc, mass, area, m_add, parent_idx, depth, amp_deg, phase, is_surface, is_tail, has_joint, body_id + per-body tail_index/swim_freq/swim_wave. Frozen `physics`↔`genetics` boundary. | S/Low |
| **S2.2** Batched frontier unroll | `genetics/develop.py::develop(genotype)->DevelopedBody`: fixed `L=6` layers, gather→compose→scatter, precomputed unique slots (no atomics), mask-multiply never `index_select`. Reuses the S0 6-pass pose kernel shape. Port `BodyGraph.MeasureWalk:377–437` DFS **including mirror double-count (:433–435)** and the `parts≥28/depth>6` data-cap vs `16/5` physics-cap `PropWalk (:442–460)`. | L/High |
| **S2.3** Aggregate hard-diff | Emit donor `Measure()` aggregates from torch: `{mass,area,intake,bulk,avgDensity,parts}` + `{length,girth,fineness,propArea,asymmetry,compactness}` (`BodyGraph.cs:314–375`). | M/Med |

**P0 acceptance:** torch develop reproduces C# `Measure()` aggregates within **rel 1e-4 (f64)** on an H1 sample, preserving traversal order + mirror double-count, *before* any new capability is layered on.

#### S2.B — Tree→graph + ellipsoid readout (genome P1)

| Task | Detail | Effort/Risk |
|---|---|---|
| **S2.4** Genotype SoA + innovation ids | `genetics/genotype.py`: `PartGene` tree (`BodyGraph.cs:23`)→node/edge tables; children→edges; `mirror`/`recursion_r` as edge fields; monotone `next_iid` (`genetics/innovation.py`). Still asexual (crossover is S5). | L/Med |
| **S2.5** Box→ellipsoid `(a,b,c)` | Replace box `size` with ellipsoid semi-axes; **ellipsoid volume `(π/6)·2a·2b·2c` for BOTH inertia and displacement** (§4.2 migration note). Log-scale `log_a/log_b/log_c` to kill the additive ratchet (§5.2). | Med-High |

**P1 acceptance:** develop still matches P0; **ellipsoid readout re-validated vs box baseline, and SwimEval oracle fixtures (LambK grid, single-step forces) re-recorded** if the a/b/c reinterpretation breaches `<1e-4` rel (§4.8, RK-11). This is the `gain1` re-baseline flagged in S0/T11.

#### S2.C — [MEASURE-2] StepLive + throughput re-measurement

| Task | Detail | Effort/Risk |
|---|---|---|
| **S2.6** Additive-contributor core | Stand up `ForceContributor` (§4.5) with `F_hydro` (ported SwimEval) as the **sole** contributor. Core owns articulated state, `M_eff` assembly, `solve_sym3` semi-implicit integrator, energy ledger. Gravity/buoyancy/contact NOT built now — but the summation seam is (avoids the donor's swimming-only dead-end). | M/Med |
| **S2.7** `step_live.py` (yaw-integrating) | Extend S0 `Step` (§4.4): latched DC bias `θ_j += turnCmd·depth_j`; real yaw torque `τ=Σ r×F` integrated as **angular momentum** `L_yaw+=τ·Dt`, `ω=L_yaw/I_yaw`, quadratic yaw drag `τ_drag=−Cyaw·ω·|ω|`. `turnCmd` = P-controller on heading error (the one "placeholder brain" seam), bounded within `AmpMax=58°`. Realized DOF exactly `{surge,yaw}` (validates CORE contract, §7.4.2). | L/Med |
| **S2.8** Throughput re-measure | Re-run the S0 sweep on `StepLive` (heavier: yaw state + P-controller + steering torque). | M/High |

**[MEASURE-2] rule:** `StepLive` at H1/H2 must re-clear the floor (3.07e7 ×10 headroom, or the Q#1-pinned number) with the profiler force/solve-bound. **If it fails, stop and re-scope** (narrow GPU to many-worlds / keep S2 on CPU via `device=`, RK-2) — do not bulldoze a slower kernel.

#### S2.D — Kill `eff[]`; derive capability from morphology (P2)

The `eff[]` stat vector and `swimProxy` scalar are **deleted, not migrated** (§2.5, §5.4, §7.3).

| Task | Capability | Derived from | Effort |
|---|---|---|---|
| **S2.9** Locomotion | realized surge+yaw | DevelopedBody→`F_hydro`→`StepLive` (no `swimProxy`) | falls out of S2.6/2.7 |
| **S2.10** Feeding | `C_form` gape/uptake | intake-port surface area+size as a *live physics query* (§6.4) | M |
| **S2.11** Mass/buoyancy | sinking, COT, neutral-buoyancy niche | `Σ density·volume` + displaced ellipsoid volume | S |
| **S2.12** Defense | armored vs fast | bulk/density vs agility(compactness+streamlining) — derived, not a slot | M |
| **S2.13** `eff[]`/`swimProxy` deletion + guard | remove the proxy path; AST guard test | grep: no `eff`/`swimProxy`/`speed`-scalar drives feeding/loco/defense | S |

Metabolism (Kleiber `P_basal=B0·M^α`, §6.5.1) wired here as a read on derived mass, feeding S1 energy accounting; full feeding economy is S3.

#### S2 Tests

| Test | Asserts | Threshold |
|---|---|---|
| `test_develop_matches_measure` (P0) | torch aggregates == C# `Measure()` | rel `<1e-4` f64 |
| `test_develop_fixed_shape` | develop never changes `S_max`; growth only in `Mutate` | shape static |
| `test_ellipsoid_oracle` (P1) | re-recorded LambK + forces vs re-baselined fixtures | LambK `<1e-6`; forces `<1e-4` |
| `test_steplive_energy_closure` | per-step `<1e-6` f32; 1e5 KE budget bounded-oscillating | `<1e-3` f32 / `<1e-6` f64 |
| `test_steplive_metabolic_debit` | `ΔE=ΔW_mech/(η·N)`; no work banked at η=1 | identity holds |
| `test_yaw_is_drag_set` | turn rate set by physical `τ_drag`; `{surge,yaw}` only DOF | qualitative + DOF check |
| `test_no_stat_vector` | no capability reads `eff[]`/`swimProxy`/`speed` scalar | symbol absent |
| `test_two_bodies_differ` | streamlined vs stubby → different cruise AND feeding | measurable divergence |

#### S2 Acceptance Gate (§7.3)

1. Every live creature swims via the one canonical body (no hand-authored segment tensor remains).
2. `eff[]` gone; `swimProxy` gone — `test_no_stat_vector` green (P2/P3).
3. Two differently-shaped bodies swim AND feed differently from morphology-through-physics alone.
4. StepLive energy closure green, drift bounded-oscillating.
5. **[MEASURE-2] re-cleared** at H1/H2 (or scope explicitly narrowed on the recorded number).
6. P0 develop-match green (`<1e-4`); P1 ellipsoid oracle re-validated.

**S2 sequencing:** S2.1 → S2.2/S2.3 (P0) → S2.4/S2.5 (P1 oracle re-record) → S2.6 → S2.7 → **S2.8 [MEASURE-2] gate** → S2.9–S2.13 → acceptance. Development (A/B) and kernel (C) proceed in parallel until they meet at S2.6.

---

## 4. S3–S9 — Milestone Contracts

Lighter by intent: each slice is defined by its *contract* (scope, components, falsifiable done-gate, dependency, top risk). Per-task decomposition is authored when the prior slice's gate goes green (P7). Every slice inherits the standing gates (books close, reproducible-within-seed, no stat vector, no green-keeping dial, telemetry-before-claim). Risk classes: 🟩 engineering · 🟦 engineering-with-frontier-tail · 🟥 research frontier.

| Slice | Scope | Class | Gating dep | "Done" is falsified if… |
|---|---|---|---|---|
| **S3** | Feeding+metabolism+reproduction on conserved energy | 🟩 | S2 | energy books don't close end-to-end, or population needs a mint/cap knob |
| **S4** | Predation as staged contest between real bodies | 🟦 | S3 | a predatory role never arises unseeded, or prey mass isn't fully accounted |
| **S5** | Speciation / mating / taxonomy | 🟥 | S4 | no disruptive-selection setup produces a durable deterministic split |
| **S6** | Currents / weather / transport | 🟩 | S1 fields | advected fields stop closing books, or no patchy productivity emerges |
| **S7** | Render / observation viewer | 🟩 | S1–S6 telemetry + state contract | render-on vs render-off changes any sim number |
| **S8** | RL / embodiment (Talos) | 🟩 seam / 🟥 gate | S1–S4 green AND Sophia iface verified | a CORE-only policy cannot drive a fish through Step/Reset over the contract |
| **S9** | Plants + bidirectional water↔land | 🟥 | S2 additive core + S3/S4 benthic gradient; full crossing needs all of S1–S6 | sea-robin walk never emerges; then no two-way crossing |

### S3 — Feeding / Metabolism / Reproduction 🟩

Close the *energy* loop the way S1 closed the *nutrient* loop.

**Components:** `graze(pos,C_form,field,cfg,dt)→(I_assim,egesta,field_debit)` Holling-II, `I_bio←min(I_bio,D_local)`, `C_form` from morphology (§6.4); `metabolism(mass,temp,cfg)→P_basal` Kleiber α≈0.79, `P_active=P_basal+P_loco` (SwimEval muscle power); `reproduce(...)` mass-scaled eligibility, paired debit==credit `(E,struct_N)` parent→child, juvenile body from genome (real `M_off`, never a flat tank, §6.5.2); egesta `(1−AE)·I_bio` and death-routed `struct_N`→`Bd`; birth/death into `(W,N_cap)` via free-slot claim / alive-mask clear (deterministic); telemetry: per-cause mortality, lifespan, reserve histograms, `[CONS]` energy line.

**Acceptance (falsifiable):** in one deterministic run a cohort **feeds→grows→reproduces→dies with energy books closed end-to-end** (`INV-ENERGY < τ_energy`/close, drift bounded-oscillating over ≥1e6 steps); every transfer passes `INV-TRANSFER<1e-6`; population **sustains (plateaus at production÷real-per-capita-intake) without any `carryingCap`/`energyCap` knob** (§6.8–6.9). **Top risk RK-13:** loop oscillates/tuning-fragile → single-dial activation with ledger watched; a collapse is a diagnostic that fidelity was dropped upstream, never fixed by retuning the Joule/SMR anchor.

### S4 — Predation as a Staged Contest 🟦

A second trophic level that is *earned*, not seeded: `find→close→seize→consume` in continuous space over the shared spatial hash, reading only form+physics-derived capabilities — never a `carnivory` flag (P8).

**Components:** `find` two-sided multi-modal detection `Detect(modality)` vs `Signature` across {vision↔light, smell↔chemical, lateral-line↔flow, electro↔electric}, no dominant modality; `close` pursuit from real drag/yaw-torque + burst gear, not a speed stat; `seize` two-sided `GripRate` vs `Evade`, overpower = mass ratio, `F_crack` from jaw geometry; `consume` gated `CanCapture∧CanDigest`, one paired transaction prey `(E,struct_N)`→predator(AE)+detritus(1−AE); spatial-hash neighbor reuse (no cell-bucketed prey mass); telemetry: trophic occupancy, kill/attempt, per-modality stats, mass-flow ledger.

**Acceptance (falsifiable):** in a run seeded with **no predator**, a predatory strategy **arises implicitly** and transferred mass is fully accounted (prey debit == predator credit + egesta). Report the trophic pyramid + arms-race signature. **Top risk (frontier tail):** emergence depends on encounter economics (world is *dense not large*, §2.9); if no predator emerges, root-cause to encounter economics before any incentive knob (P8 forbids a hunt-reward term).

### S5 — Speciation / Mating / Taxonomy 🟥

Genome Phase P2 (§5.12) — the payoff phase, the first hard research-frontier gate.

**Components:** NEAT compatibility distance `δ` over innovation-aligned genes (batched **within spatial buckets**, not global N²); innovation-aligned crossover (align by iid, matching/disjoint/excess); spatial assortative mating gated `δ<δ_t` within `r_mate`, sweep `(w,r_mate,δ_t)` (bimodal-fragile, RK-9); biological species = connected components of the interbreeding graph; observational taxonomy (phenotypic-cosine Linnaean ladder, one-way latch, procedural naming) — **separate axis, never feeds selection or gates mating**; a disruptive-selection test harness (parameters = open Q #11).

**Acceptance (falsifiable):** **one deterministic run where a panmictic population splits into two non-interbreeding clusters under disruptive selection**, stable and reproductively isolated for a defined run length. **Do not proceed until demonstrated — or its absence root-caused to the ecology (insufficient niche diversity), not the encoding** (RK-9). Mitigations: magic-trait coupling option; Kleiber + prune pressure so new DOF must earn fitness (RK-10).

### S6 — Currents / Weather / Transport 🟩

Coupled velocity/temperature/light fields climbing the circulation ladder behind the unchanged `sample(x)` interface (P4).

**Components:** **L0** prescribed analytic gyres from stream function `ψ` (divergence-free `u=−∂ψ/∂y, v=∂ψ/∂x`), diel/seasonal solar forcing, hand-placed upwelling — no PDE solve; **flux-form advection** with positivity/flux limiter (INV-W5); **Ekman upwelling** `w_e ∝ curl(τ/ρf)` (a divergence-free horizontal current gives zero upwelling by construction); **L1 (optional)** barotropic QG vorticity on the periodic box with FFT Poisson solve, deterministic on a fixed device; light promoted to a stored cloud/current-coupled field; currents deflect around bathymetry and carry vent plumes.

**Acceptance (falsifiable):** advected fields **still close the books** (INV-W1/W5, concentrations `≥0`), patchy productivity emerges causally (nutrients upwell, blooms advect downstream, dispersal patterns form), **and every downstream consumer runs unchanged through the field query** (CI check that fields/ecology imports didn't change). **Top risk:** L1 solver stability/determinism → ship L0 first; add L1 only if biology demonstrably needs eddies (P6).

### S7 — Render / Observation Viewer 🟩

A thin remote/replay viewer (Unity or web) over the **read-only** `observe` surface; render is a consumer, never a producer.

**Components:** read-only feed over canonical `(W,N_cap)` tensors + fields (the observation tensor *is* the state, §2.4.3); replay-from-checkpoint from serialized `ColonyState`; viewer client renders morphospace/field heatmaps/trophic structure; CI equivalence harness (same seed headless vs viewer-attached).

**Acceptance (falsifiable):** **render never feeds fitness** — a deterministic run produces byte-identical (same-device) sim numbers with viewer attached and detached, and the viewer reproduces a checkpointed run from `ColonyState` alone. Any headless-vs-viewer divergence fails the slice. **Top risk:** accidental coupling → the one-way `observe` import boundary + render-off equivalence gate make any leak a CI failure.

### S8 — RL / Embodiment (Talos) 🟩 seam · 🟥 gate

Stand up the sim↔Talos state contract: versioned CORE/EXT dict-of-tensors over a gym-shaped Step/Reset env. The seam is day-one architecture; S8 adds schema, serialization, gating.

**Components:** `Header` + CORE/EXT dict-of-tensors, `(W,N_cap)` batch + alive-mask, SI/REP-103/radians/seconds; **CORE action** = 2-vector `{surge_effort∈[−1,1]→Twist.linear.x, yaw_rate∈[−1,1]→Twist.angular.z}` (validated by StepLive integrating yaw only, §7.4.2); **CORE obs** `lin_vel, ang_vel, heading, range_egocentric(K), flow_rel, energy, contact`; **EXT** (sim-only) chemical gradients (the fish's *primary* sense, deliberately EXT — no robot analogue), light/depth/temp, proprioception, feeding strike; serialization ladder in-process torch-dict → Protobuf proto3 → ROS2 `.msg`/`geometry_msgs/Twist`; conformance test freezing a CORE fixture so **pydantic + Gymnasium space + `.msg` agree field-for-field**; Talos-side symbolic→continuous adapter kept **out of** the sim schema.

**Acceptance (falsifiable):** a headless gym-style `Step`/`Reset` env drives a sim fish end-to-end — a **CORE-only** policy produces coherent locomotion through the versioned schema, conformance fixture passes across pydantic/Gymnasium/`.msg`. **Dual hard gate (§7.4.4):** (1) books closed — S1–S4 invariants green; (2) **the Sophia symbolic-vs-continuous action interface verified against live Sophia code**, not the `ExecutorShim` placeholder. **Top risk RK-4** (highest strategic): contract continuous-first; symbolic decode lives Talos-side; if Sophia is continuous the adapter collapses to pass-through. Secondary open-Qs #8 (ROS2 distro), #9 (in-process vs out-of-process).

### S9 — Plants + Bidirectional Water↔Land 🟥

The emblematic endpoint, staged: **(9a) sea-robin milestone** — a benthic fish walking on re-purposed fin-rays *inside the water* (de-risks the frontier); **(9b) full bidirectional crossing** — sea→land limb and land→sea re-streamlining on the additive medium-dependent physics with **no mode switch**. Also lands plants (L-system/CPPN, genome P3) and the land/rivers/sediment cascade (§3.7).

**Components:** additive `Gravity`, `Buoyancy` (ellipsoid displaced volume), `Contact/Friction` contributors on the already-additive core (S2, no rewrite); continuous `medium(x)` submersion-fraction query (§3.11, a smooth gradient, not a boolean); land cascade `is_land = elevation > sea_level`, flow-accumulation rivers→`river_sources`, sediment/burial as a **transfer** to a tracked reservoir (never deletion), intertidal two-way corridor with a survivable ramp both directions; plant kingdom rooted in substrate; genome reversibility guard (`Segment↔Surface` flip + neutral drift lets a walking limb drift back to swimming — no irreversible ratchet); benthic-foraging gradient (buried prey reachable by ventral fin-ray probing).

**Acceptance (falsifiable) — staged.**
- **9a (milestone):** ≥1 established lineage where (1) ventral Surface segments spend a measured gait fraction in seafloor contact producing **net upward + forward ground-reaction impulse**; (2) the same appendages **retain nonzero swimming thrust**; (3) the walking capability is **heritable and improves under the benthic gradient**. Report the impulse budget + gait phase from telemetry (§5.10, §4.6).
- **9b (frontier):** an **emergent, unscripted crossing of the water/land interface in both directions** on the additive physics with no mode switch; convergent re-derivation of streamlined aquatic form by independent lineages as the signature that physics drives form (§1.6).

**Top risk (research frontier):** the crossing may not occur — capability is necessary but not sufficient; the *gradient* (survivable ramp, unexploited niche) is the open variable. Secondary: an irreversible ratchet in a mutation operator would break the bidirectional requirement — guarded by the reversibility test.

### Engineering vs research-frontier at a glance

| Slice | Mechanism (build) | Emergent outcome (bet) |
|---|---|---|
| S3 | 🟩 solved | population persistence — 🟩 |
| S4 | 🟩 solved | unseeded predator + arms race — 🟦 |
| S5 | 🟩 solved | **deterministic species split — 🟥 (RK-9)** |
| S6 | 🟩 solved | patchy productivity/dispersal — 🟩 |
| S7 | 🟩 solved | — |
| S8 | 🟩 solved seam | **gated on external Sophia verification — 🟥 (RK-4)** |
| S9 | 🟩 solved | **sea-robin walk — 🟥 near-term; two-way crossing — 🟥 long-horizon** |

---

## 5. Verification, Salvage & Language Protocol

### 5.1 Why conservation replaces byte-identity (the load-bearing inversion)

The donor's gate was a seeded-run FNV/BitConverter hash vs a stored golden — satisfiable by code that does nothing, which is *precisely* how faithful mechanics shipped disabled behind `gain=0`/`couple=0` dials to keep goldens green (§1.2, §7.1.1). **Byte-identity answers "did the numbers change?"; conservation answers "do the books close?"** and exerts zero pressure to gate mechanics off. Byte-identity survives only as a same-device regression tripwire (§5.4), never the pass/fail gate.

### 5.2 Conservation gate as concrete assertions (PRIMARY, P1)

Every tracked `Q ∈ {nutrient_mass [mol], energy [J]}` lives in a fixed named-reservoir registry; the ledger is f64 compensated. Every mutation goes through `transfer`/`declare_external` — no raw `R_i += x`.

| Assertion | Threshold | Dtype | Source |
|---|---|---|---|
| Per-step inventory residual | `< 1e-9` | f64 ledger | §7.1.1, S1 AC |
| 1e6-step drift, bounded-oscillating (slope≈0 in noise, `max<1e-6`) | `< 1e-6` | f64 | §7.1.1 |
| `test_no_free_channel`: `|ΣΔR_i − X_ext_step| < τ_cons` | catches free-channel | — | CI-CONS-1 |
| Instantaneous power identity | `< 1e-6` | f32 | §4.8 |
| 1e5-step KE budget | `< 1e-3` f32 / `< 1e-6` f64 | both | §4.8 |
| Metabolic: `ΔE=ΔW_mech/(η·N)`, η=0.20, N=300 J — no path banks at η=1 | identity | — | §4.8 |

Gate on the drift *curve*, never the endpoint — a slow monotone leak is the failure mode (RK-13).

### 5.3 Determinism plan

Target = bit-identical across two same-machine, same-`(device,dtype,op-order)` reruns with a fixed seed. Cross-machine / CPU↔GPU bit-identity is explicitly out of scope; the conservation gate carries cross-device *correctness*. `seed_everything(seed)`: seeds torch/numpy/python, `use_deterministic_algorithms(True)`, `CUBLAS_WORKSPACE_CONFIG=:4096:8`, no cudnn benchmark. **No nondeterministic atomic scatter** — all births (unique free-slot), segment reductions (`body_id` via `segment_index_add`), development scatter use precomputed unique indices; a Grep CI check forbids raw `scatter_add_`/non-unique `index_add_` in the hot loop. `test_within_seed_rerun`: `max_abs(Δ)==0` on the fixed device, gated only in deterministic mode; GPU-nondeterministic drift reported informationally. RNG manifest: adding an inert gene leaves the seeded stream byte-identical. **Open Q #4: recommend "bit-identical per-device"** (free on CPU, the cheaper contract to defend) over "within 1e-5 rel" (which would let nondeterministic GPU scatter dodge RK-6).

### 5.4 Bit-golden as a demoted tripwire

A same-device-CPU golden hash regression, run **nightly, reported not gated**. Answers "did anything change unexpectedly?" as a review aid; can never block a merge; no mechanic may ship dark to keep it green. The only surviving role of the donor's byte-identity harness.

### 5.5 Test culture

Tolerance/invariant tests, never golden-value tests (`closure_residual < τ`, `|ported−oracle|/|oracle| < τ_rel`, drift bounded-oscillating — never `state_hash == 0xDEADBEEF`). Long-horizon trajectory divergence under the chaotic gait is *expected* and must not be gated (F5). Fakes over mocks; boundaries over internals — every subsystem tested through its published contract (`FakeGeology`, `FakeField`, `FakeBody`). `test_import_boundary.py` runs the import-linter contract as a pytest and asserts interface opacity (INV-W4). Telemetry-first: every foundation slice ships a headless observation surface before any render; **no claim of correctness or performance is ever made from code inspection — only from a telemetry artifact.**

### 5.6 Oracle-fixture regression vs the C# donor

The donor is **never called live** — compiled once into `oracle/SirRobinOracle.csproj` (Unity-light, driven by `Reconstruct`/`LambK`/`Coast`/`MomentumLedger` + `StepTrace`) emitting frozen fixtures committed to the repo (§2.4 table). **Gate on aggregates + short-horizon forces, NOT a long-horizon bit-trace** (chaotic gait, F5) — this is what lets the port be *correct* without being *bit-identical to C#* across an episode.

> **Migration hazard (RK-11, open Q #3):** the donor stored box mass with ellipsoid `π/6` displacement. The port uses **ellipsoid volume for both**, shifting the added-mass baseline — validate against the box baseline first (Stage A/gain0), then re-record fixtures against the ellipsoid reinterpretation (Stage B/gain1, S0/T11 and S2/P1). Do **not** silently re-baseline. Porting validation rule (uniform): no port is done until it passes both its donor tolerance golden **and** its conservation invariant — a balanced sign error passes energy closure but fails the force-term fixture (RK-7).

### 5.7 Salvage disposition of every donor module

Four dispositions: **PORT** · **CONSTANTS** · **OBSERVATIONAL** · **ORACLE**. Donor tree: `C:\Users\cddal\game prototype\Assets\ProceduralWorld\Scripts\`.

| Donor module | Disposition | Torch destination | Validated by |
|---|---|---|---|
| `Life/SwimEval.cs` + `SwimmerSim` | **PORT + ORACLE** | `physics/swim_step.py`/`step_live.py` (first additive contributor) | forces/lambk/aggregates/momentum goldens |
| `Life/BodyGraph.cs` + `Measure()` | **PORT** | `genetics/` graph + `develop.py` 6-pass scan → DevelopedBody | hard-diff vs `Measure()` (P0/P1) |
| `PlanktonField/NutrientField/MarineSnowField/WorldSampler/OceanField`, vents | **PORT** | `fields/` behind `sample(x)→(value,grad)` | `NutrientConservationTests` etc → pytest conservation invariants |
| `SpatialHash.cs` | **PORT** | `core/spatialhash.py` (point-entities) | neighbor-query equivalence |
| `SimClock.cs` | **PORT** | `core/clock.py` (Now f64, Dt, Step, Scale) | pytest; assert no wall-clock read |
| `SimUnits.cs` | **CONSTANTS** | `numerics/units.py`: `N=300 J`, `η≈0.20`, `KgPerSimMass=250`, `ρ_w=1000` | test asserts never mutated at runtime |
| `Taxonomy*.cs` | **OBSERVATIONAL** | `observe/` — `species_tag` read-out, never gates mating | assert tag derived, not authoritative (INV-W2) |
| `Ent.cs` (POD) | **PORT (dissolved)** | `(W,N_cap,k)` SoA columns | representation change |
| `NutrientChem/LifeEconomyConfig/NutrientConfig` (design) | **PORT (design)** | `core/economy.py` conserved reservoirs (S1) | `NutrientChemTests`/`FeedingConservationTests` → pytest closure |
| Talos EntitySpec CORE/EXT seam | **PORT** | `observe/contract.py` (§7.4) | pydantic+Gymnasium+`.msg` conformance |
| `GoldenMasterDeterminismTests` etc | **DEMOTE** | same-device-CPU tripwire only | invariant flips byte-identity→conservation |
| `SwimEvalTests`/`SwimFin*Tests` etc | **ORACLE source** | drive the fixture emitter | become the recorded goldens |

**Explicitly superseded — do NOT salvage as architecture:** `OceanColony.cs` (3,067-line god-class); the dual `eff[]` genome (`TraitSchema.cs`/`Allocation.cs` — **deleted, not migrated**, the S2 gate); LOD proxy / `SpeciesSwimCache.cs`; static non-conserving Perlin fields; byte-identity-as-gate; the four-builder `SwimmerBuilder`/`Measure`/`MakeBody` divergence; asexual clone-and-mutate + post-hoc cosine species. The durable salvage is **validated equations + recorded oracle values**, not C# text. (The donor's 74-commit small-verifiable-commit sequence is retained only as *process* discipline.)

### 5.8 Language / kernel decision protocol

**Torch-first, measurement-gated, per-kernel reversible.** The sealed `physics`-module interface (`DevelopedBody` in, forces + ledger out) makes the hot kernel **swappable: swap the kernel, not the system.** Climb the ladder only when a measured gate cannot be cleared in the current rung.

| Rung | Kernel | Trigger to escalate | Reversible? |
|---|---|---|---|
| **0 — torch eager** | plain batched ops | default | — |
| **1 — `torch.compile`/CUDA-graphs (r1→r2)** | fuse ops, kill dispatch | **F3**: per-op launch overhead dominates at realistic N after r1 → try r2. If r2 clears (d), **stop.** | trivial (mode flag) |
| **2 — one custom kernel (Triton/NVIDIA-Warp)** | hand-write only the branchy per-body force/solve loop | **F1** or **F2** surviving r2. Warp is the lighter hatch, tried first. | moderate (swap one fn behind the interface) |
| **3 — Rust core via PyO3** | whole hot loop in Rust | rung-2 still can't clear (d) at H1/H2 AND bottleneck is host-side/control-flow, not arithmetic | expensive, still local to `physics` |
| **fallback — narrow GPU scope** | keep S0–S2 on CPU via `device=`; GPU for S8 many-worlds | F3/F4 persist at realistic single-world N | free (`device=` knob) |

**Decision rule:** climb exactly one rung per surviving falsifier, re-measure, stop at the first rung clearing (d) at H1/H2 with ≥10× headroom while (a)(b)(c) hold. Record every rung's number; the go/no-go is read from H1/H2, never H0. **Stays torch regardless** (never on the ladder): RL/orchestration, `core` economy, `fields`, `genetics` develop scan, `observe`, the Gymnasium env API. JAX is a last-resort full rewrite, not on the near-term path.

---

## 6. Sequencing, Risks & Definition-of-Done

### 6.1 Dependency graph

```
SCAFFOLD (S-1) ──> S0 SpikeSwim ──(GO: H1/H2 clear + a/b/c/oracle hold)──> S1 (KEYSTONE)
   │ numerics · import-linter · SolveSym3/quat · segment reduce ·          │ books close <1e-9
   │ f64 ledger · RNG manifest · SimClock · Config · CI · oracle harness    │
   │                                                                        ▼
   └──> genome P0 (develop-scan, shares pose kernel) ──┐         S2 (canonical body + StepLive)
                                                       └────────> · [MEASURE-2] re-clear before commit
   fields pkg (Geology/Field protocols + stubs) ─────────────────> · kill eff[] · P1 ellipsoid re-record
                                                                            │
                                                          ┌─────────────────┤
                                                          ▼                 ▼
                                                   S3 feeding ──────> S4 predation
                                                          │ genome P2      │
                                                          ▼                ▼
                                                   S5 speciation (falsifiable split)
                                                          ▼
                                          S6 currents ──> S7 viewer ──> S8 embodiment (dual-gated)
                                                                                │
                                                                                ▼
                                          S9 plants + water↔land (9a sea-robin de-risk → 9b crossing)
```

**Critical path:** Scaffold → S0 → S1 → S2 → S3 → S5. S4 branches off S3; S6/S7 parallelizable once S5 lands; S8 dual-gated, slips independently; S9's 9a is reachable as soon as S2+S3+S4 are green — pull it forward as the frontier de-risk before committing to 9b. **Cross-cutting:** genome P0→P4 shadows physics/ecology (P0 with S0, P1 with S2, P2 with S5, P3/P4 deferred); fields protocol layer exists before S1 consumes it, rich generator defers to S6/S9 (zero downstream change); Talos schema authored (not wired) once `(W,N_cap)` freezes in S0.

### 6.2 Milestones

| ID | Milestone | Observable = done | Risk class |
|---|---|---|---|
| M-0 | Scaffold green | numerics + import-linter + pytest/CI + oracle harness pass; `Colony.reset/step` no-op tick deterministic | solved |
| M-S0a | SpikeSwim correct | gates (a)`==0`, (b)`<1e-6`/f32, (c)`<1e-4`/`<1e-3` at H1/H2 | solved (impl risk) |
| M-S0b | SpikeSwim affordable | gate (d) 3.07e7×10 headroom at H1/H2; `B*` located; force/solve-bound | **uncertain — load-bearing** |
| M-S1 | Books close | `<1e-9`/step, drift `<1e-6` over 1e6 steps; bloom self-terminates, no cap knob | solved (tuning-fragile) |
| M-S2 | One body, one physics | all swim via canonical body; `eff[]` deleted; StepLive re-cleared; P0/P1 met | solved |
| M-S3 | Energy loop closes | cohort feeds→reproduces→dies, books closed; sustains without minting | solved (tuning) |
| M-S4 | Predation emerges | unseeded predator; prey mass fully accounted | solved / 🟦 tail |
| M-S5 | Speciation proven | one run, panmictic pop splits into two isolated clusters | **research frontier (RK-9)** |
| M-SR | Sea-robin (de-risk) | ventral Surfaces produce net ground-reaction impulse AND retain swim thrust, heritable, no mode-flag | frontier (physics half) |
| M-S6 | Transport | advected fields close books; patchy productivity through unchanged query | solved |
| M-S7 | Viewer | reproduces a checkpointed run; render never feeds fitness | solved |
| M-S8 | Embodiment | CORE-only policy drives a fish; dual gate (books + Sophia verified) | **highest strategic (RK-4)** |
| M-S9 | Crossing | emergent unscripted water↔land crossing both directions, no mode switch | frontier |

### 6.3 Risk register

| # | Risk | L | I | Mitigation | Trigger / Owner |
|---|---|---|---|---|---|
| R1 | Torch dispatch overhead at small batch (F3/RK-2) | Med | High | ladder r0→r1→r2; else narrow GPU to many-worlds / CPU via `device=`; Warp→Rust | S0 (d) miss or `B*`>real pop / Architecture |
| R2 | Ragged heterogeneity defeats batching (F1/RK-1) | Med | High | flattened/CSR + `segment_index_add`; measure H0/H1/H2 tax; padded ablation | S0: H1<per-body loop / Architecture |
| R3 | f32 conservation drift monotone (F6/RK-5) | Med | Med | separate f64 ledger; gate on bounded-oscillating not endpoint | S0 1e5 drift monotone / Physics |
| R4 | C#→torch port bug energy identity won't catch (RK-7) | Med | Med | conformance fixtures (LambK+forces+aggregates); pin `solve_sym3` op-order | S0 aggregates >1e-3 (F5) / Physics |
| R5 | GPU determinism tax / atomic scatter (F4/RK-6) | Med | Med | precomputed-unique-slot scatter everywhere; det-mode env; CPU-first | S0 (a)≠bit-identical or tax>2× / Architecture |
| R6 | Speciation gated by ecology not encoding (RK-9) | Med | High | encoding solved; magic-trait + Kleiber/prune; root-cause absence to ecology | S5 split fails all setups / Evolution |
| **R7** | **Scope-creep / recursive-scope-explosion** (§1.5) | **High** | **High** | P3/P4 bound each sub-project behind an interface; P6/P7 forbid opening a new one until books close; a hack → queue as its own slice, never cascade | any PR adding a knob/sync-glue/second representation, or work spanning >1 slice's AC / Tech lead |
| R8 | Sophia action interface unknown (RK-4) | Med | High | contract continuous-first; symbolic decode Talos-side; gate on live Sophia code | S8 entry / RL+Sophia |
| R9 | Wrong-invariant regression (RK-3) | Med | High | conservation the only pass/fail gate; CI rejects any mechanic behind a green-keeping `gain=0` dial | any new term defaults to "today's behavior" / Tech lead |
| R10 | Churn/compaction swamps step (F7/RK-8) | Low-Med | Med | fixed-cap+mask, free-list recycle, compact every K; cost measured in S0 | S0 churn>step budget / Architecture |
| R11 | Ellipsoid a/b/c shifts Lamb added-mass (RK-11) | Low-Med | Low-Med | validate vs box baseline in P1; re-record fixtures | S2/P1 oracle tol breached / Physics |
| R12 | Closed loop oscillates / tuning-fragile (RK-13) | Med | Med | source/burial damping; one-dial activation; never re-soften depletion to hide a mint | S1 stock won't stabilize without cap knob / Ecology |
| R13 | Expressive genome manufactures unrewardable morphospace (RK-10) | Med | Med | Kleiber + prune; new DOF must earn fitness | morphospace inflates without fitness gain / Evolution |
| R14 | Pose depth-scan/reductions dominate (>50%, F2/RK-12) | Med | Med | shared gather→compose→scatter kernel; fixed 6-pass; profiler | S0 scan/reduce>50% / Physics |
| R15 | Target hardware unpinned (open Q #1) | High | Med | pin CPU/GPU/VRAM/FP64-tier before reading (d) as authoritative | S0 read as go/no-go / Owner input |

### 6.4 Per-slice Definition-of-Done

A slice is DONE only when all four are checked on a telemetry artifact. "Books close" and "telemetry in place" are non-negotiable everywhere.

- **Scaffold (S-1):** ledger+f64 reduction exist and run on the no-op tick · import-linter passes (violation=build fail), numerics unit tests green, oracle harness emits a LambK/force fixture a pytest reads · `Colony.reset/step` deterministic no-op, `SimClock` advances, state round-trips · CI emits `[BUILD]` artifact.
- **S0:** energy closure (b) per-step `<1e-6` f32, 1e5 KE bounded-oscillating · determinism (a) `==0` CPU+CUDA, oracle (c) LambK`<1e-6`/forces`<1e-4`/aggregates`<1e-3` across H0/H1/H2 · throughput (d) 3.07e7×10 at **H1/H2** (H0 doesn't authorize), `B*` located, taxes reported, no F1–F7 tripped · go/no-go artifact with profiler attribution (force/solve-bound).
- **S1:** `<1e-9`/step, drift `<1e-6` over 1e6 bounded-oscillating, every transfer `<1e-6`, amortized==whole · INV-MASS/ENERGY/positivity/continuity green · bloom self-terminates with no cap knob, deserts emerge, [MEASURE-1] decided · `[CONS]` line + reservoir CSV/heatmap + bloom/desert series.
- **S2:** energy closure under StepLive, `ΔE=ΔW_mech/(η·N)` no η=1 banking · P0 `<1e-4`, P1 ellipsoid validated + momentum guard · **`eff[]` deleted** (grep-clean), all swim via canonical body, StepLive re-cleared, additive core live w/ F_hydro sole · cruise/COT/reactive-ratio + morphospace.
- **S3:** books close feed→metabolize→reproduce→die, `I_bio=I_assim+E_eg`, repro debit==credit · Holling-II clamped, Kleiber, mass-scaled eligibility, egesta→`Bd` · cohort lives+dies books-closed, sustains without minting · per-cause mortality, lifespan, energy budget.
- **S4:** prey `(E,struct_N)`==predator credit(AE)+detritus(1−AE) one transaction · stages read only form-derived capabilities+intent, two-sided seize · predatory strategy arises implicitly, mass accounted · predation log, trophic accounting.
- **S5:** repro/crossover conserve E+N, no birth leak · P2 crossover+distance+mating, deterministic repair, append-only manifest, taxonomy never gates mating · **falsifiable split** demonstrated OR absence root-caused to ecology (Q#11 pinned first) · interbreeding-graph components over time, morphospace clusters.
- **S6:** advected fields close books (flux-form, positivity limiter) · L0→L1 (FFT Poisson deterministic), Ekman upwelling, double-buffered mixing · patchy productivity+dispersal emerge through unchanged query · current/upwelling heatmaps, dispersal metric.
- **S7:** viewer read-only, render never feeds fitness · reproduces checkpointed run byte-for-byte (same-device), contract slice matches canonical tensors · remote/replay viewer renders live/replayed run · headless↔viewer parity.
- **S8:** S1–S4 invariants green · conformance (pydantic+Gymnasium+`.msg` agree), CORE action `{surge_effort,yaw_rate}`→`Twist` · **dual gate** (books + Sophia verified vs live code), CORE-only policy drives a fish · episode returns, round-trip timing.
- **S9:** burial=transfer to tracked sediment (never deletion), closed total = biotic+dissolved+snow+microbial+sediment+geological · additive contact/gravity/buoyancy on already-additive core (no rewrite), continuous `medium(x)`, reversibility (walking→swimming drift, no ratchet) · emergent crossing both directions no mode switch; 9a sea-robin reached first · impulse budget + gait phase, convergence clusters.

---

## 7. Immediate Next Actions (first week)

Goal: stand up the shared substrate and get the *first* SpikeSwim numbers against oracle fixtures. **Prerequisite before reading any S0 gate-(d) number as authoritative: pin open-Q #1 (target hardware) and plan for open-Q #6 (StepLive re-measure).**

**Day 1–2 — Repo + `numerics` + CI firewall**
1. Create the six-package skeleton `numerics→physics→fields→genetics→core→observe` with a stub module each (§1.2).
2. Wire the **`import-linter` contract** as a CI job (`physics→core` import must fail the build) — cheap, must exist before real code lands (R7 mitigation).
3. Implement `numerics`: dtype/device policy + `Config.device`; `quat.py`; `solve_sym3.py` (donor cofactor op-order, **not** `linalg.solve`); `reduce.py::segment_index_add` (deterministic precomputed-unique-slot path); `ledger.py` f64 compensated `close_books()`; `rng.py` `seed_everything` + append-only manifest.
4. Stand up pytest + CI + the conservation-scaffold + determinism harness (`use_deterministic_algorithms(True)`, `CUBLAS_WORKSPACE_CONFIG=:4096:8`). Green G-SCAF-1…7.

**Day 2–3 — C# donor → offline fixture generator (§5.6)**
5. Reduce C# `SwimEval` to a Unity-light headless console (`Vector3`/`Quaternion`/`Mathf` shim; drive `Reconstruct`/`LambK`/`Coast`/`MomentumLedger` + new `StepTraceForTest`).
6. Freeze **gain0** fixtures across H0/H1/H2: LambK grid, single-step forces (`tReact,pWake,pFin,6×M_eff,dv`), 8s aggregates (`cruiseSpeed,costOfTransport,reactiveRatio`). Store as recorded values (no live C# dependency).

**Day 3–5 — Port `Sim.Step` (frozen-heading) to the batched kernel**
7. Implement the flattened/CSR segment layout (`[S_total]`+`body_id`) and `(W,N_cap)`+alive-mask with dead-slot recycling.
8. Port the frozen-heading Lighthill step (§2.2 S3–S11) as batched masked `torch.where` under `torch.inference_mode()`: reactive thrust, Garrick fin lift (Surface caudal only), f64 LambK precompute, axial quadratic drag, `M_eff` assembly (×250 on struct diagonal only), semi-implicit `solve_sym3` COM integration.
9. Implement pose as the bounded **6-pass depth-scan** (MaxDepth=5) — the shared gather→compose→scatter kernel genome development (P0) reuses.
10. Build the **padded `(B,16)+mask` ablation** alongside so S0 *measures* the masking tax it avoids (F1).
11. Add the **population-churn stub** (kill ≈2%/1000 steps, refill, compact every K) to measure churn/compaction cost (F7).

**Day 5–7 — Run the sweep, produce the go/no-go artifact**
12. Run gate (c) first — port matches oracle fixtures (LambK`<1e-6`, forces`<1e-4`, aggregates`<1e-3`) at H0 → then H1/H2. (Correctness before throughput.)
13. Run gate (a) determinism (`==0`, CPU+CUDA-det) and gate (b) energy closure (per-step`<1e-6`; 1e5 budget bounded-oscillating).
14. Run the gate (d) throughput sweep: `B∈{1,64,256,1024,4096,16384,65536,262144}`×{CPU,CUDA}×{H0,H1,H2}×{r0,r1,r2}×{f32,f64}.
15. Emit **one telemetry artifact**: all four gates, H1/H0 heterogeneity tax, padded/flattened masking tax, CPU↔GPU crossover `B*`, and **profiler attribution proving force/solve-bound, not scan/reduce-bound** (F2).
16. **Record the go/no-go from the H1/H2 numbers.** The meta-falsifier is binding — a green H0 does not authorize. Only on green does S1 begin.

**Parallel (does not block week-1 critical path)**
- Author (schema-only, not wired) the Talos CORE/EXT state-contract dict once the `(W,N_cap)` layout freezes — it *is* the canonical layout (§2.4.3), so this is a naming/versioning pass.
- Author the `Geology`/`Field` protocol interfaces with trivial S0/S1 stubs (flat `h(x)`, uniform minerals, empty `SourceSet`) so S1 has its field query ready; defer the rich generator (R7 — interface now, generator later, zero downstream change).
