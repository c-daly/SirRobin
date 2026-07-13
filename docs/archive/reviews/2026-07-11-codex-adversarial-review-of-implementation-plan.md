# Codex — Adversarial Review of the SirRobin Implementation Plan

**Date:** 2026-07-11 · **Reviewer:** Codex CLI `gpt-5.6-sol`, high reasoning effort, read-only, run in WSL.
**Subject:** `docs/superpowers/plans/2026-07-11-sirrobin-implementation-plan.md`
**Brief:** find what is wrong; be harsh, specific, concrete; do not soften.
(Full agentic trace preserved at `codex-raw.txt`.)

## Verdict

> The plan should not be approved. Its S0 authorization gate is built on invalid physics accounting,
> an unrealizable deterministic reduction primitive, and contradictory tensor-shape requirements. The
> build will fail in S0 before throughput numbers mean anything.

## Findings (Codex, verbatim, ranked)

**1. The S0 energy gate is mathematically wrong.** The instantaneous assertion omits circulatory
thrust work: `p_in = tReact·U + pWake + tFin·U + pFin`, but the plan checks only
`tReact·U + pWake + pFin` (plan:345). Any active fin either fails the gate or forces `p_in` to be
defined inconsistently. The 100k-step KE test is worse: `M_eff` changes with articulated pose, so
`KE = ½vᵀM(t)v` and its change contains a `½vᵀΔM v` term; the proposed comparison against
`Σ(P_musc−P_diss)dt` omits that term, discrete semi-implicit work, constraint impulse, and the
tail-tip-`U` vs COM-velocity difference. The donor only proves an algebraic trailing-edge flux
identity, not whole-body KE conservation (SwimEval.cs:320). **Fix:** separate reactive and fin
algebraic tests; derive a discrete balance for the actual integrator including `M_n→M_{n+1}`, actuator
work, constraint work, drag, and wake losses. Do not use the current KE gate for authorization.

**2. The horizontal dynamics solve is physically invalid.** The kernel solves the full 3×3 system
then sets `v_y=0` (plan:308-309). When rotated anisotropic added mass creates `m01`/`m12`, the
unconstrained vertical DOF changes the computed x/z acceleration; deleting y afterward destroys
momentum and energy. The donor has the same bug (SwimEval.cs:804); matching the oracle merely
reproduces it. **Fix:** solve the constrained x/z system (or a KKT system with explicit vertical
reaction impulse), account for that impulse in the ledger, and deliberately re-baseline the donor.

**3. `segment_index_add` cannot satisfy the determinism claim as specified.** `body_id` necessarily
contains duplicate indices (several segments reduce into each body). "Precomputed unique slots" cannot
make destination indices unique without first performing a separate deterministic reduction; the plan
names no algorithm for that reduction while forbidding the standard atomic implementations
(plan:213-223, plan:663). A grep rule is not an algorithm, and hiding `index_add_` inside a sanctioned
wrapper does not make it deterministic. **Fix:** specify and prototype the actual reduction now (fixed
padded-tree reduction, sorted CSR + deterministic prefix reduction, or a custom fixed-order kernel).
Gate bit-identity on that exact implementation before building the rest of S0.

**4. The "static-shape" architecture contradicts the flattened live-segment layout.** Creature tensors
are fixed-capacity, but segment tensors have shape `[S_total]` = sum of live segment counts
(plan:165-178, 287-293). Birth/death/refill/mutation/compaction changes `S_total`, which invalidates
CUDA-graph capture, changes allocation addresses, recompiles dynamic graphs, and makes `body_ptr`
maintenance nontrivial. The design's later development representation is actually fixed `[P,S_max,...]`
(design:638-652), contradicting the plan's insistence that padded storage is only an ablation.
**Fix:** choose one real representation — fixed `[B,S_max]`, or a fixed-capacity segment arena with
allocation/fragmentation/generations/compaction/pointer-repair/graph-recapture policies. Benchmark the
real lifecycle.

**5. The f32 transfer ledger cannot close exactly "by construction."** Debiting and crediting the same
f32 amount does not preserve the f64-observed total when source and destination differ in magnitude
(subtracting `0.5` from `1.0` is representable; adding it to `1e8` rounds away). A compensated f64 sum
afterward cannot recover information already lost in the f32 writes, so `residual==0`, `<1e-9`, and
million-step `<1e-6` are unsupported for f32 reservoirs (plan:414-425, 459-471). **Fix:** use f64
reservoir state, fixed-point/integer quanta, or track every realized post-rounding discrepancy in an
explicit numerical-residual reservoir; set tolerances from measured scale-dependent error bounds.

**6. The throughput gate is arbitrary, internally contradictory, and not end-to-end.** Near-term scope
is one world with hundreds–low-thousands of creatures, but the floor is derived from the deferred S8
target of 256 worlds then multiplied by an unexplained 10× (design:268-272, plan:350) — effectively
demanding ~`3.07e8` creature-steps/s without deriving it from a required experiment duration. Worse,
S0 measures only frozen-heading locomotion; it excludes development, fields, spatial hashing, feeding,
encounters, mutation, mating, telemetry, and the real StepLive kernel, so passing it cannot authorize
"the entire vectorization thesis." **Fix:** define required wall-clock for a scientifically useful run,
convert to end-to-end simulated-years/hour, and benchmark a representative whole tick. Keep kernel
throughput as diagnostic telemetry, not architectural authorization.

**7. S1's acceptance behavior is impossible with the processes S1 implements.** S1 claims a bloom must
grow, draw down nutrient, and crash, and that drifters plateau (plan:469-475). But S1 has no `Bp→Bd`
mortality/respiration and no grazing (feeding is S3); production transfers `Nd→Bp` until nutrient
exhaustion, after which `Bp` simply remains. No crash, no consumer plateau. **Fix:** add explicit
producer loss/respiration/mortality/`Bp→Bd/Nd` transfers to S1, or defer those acceptance claims to S3.

**8. The gain1 oracle is circular and does not exist in the live donor.** The live donor hard-codes
`ellipMassGain = 0` (SwimEval.cs:83); `ReconstructForTest` has no gain parameter; `StepTraceForTest`
does not exist. The plan proposes modifying/reimplementing the oracle then "re-recording" gain1
fixtures (plan:324-337, 386). Once both port and oracle are changed to the new model, agreement is not
independent validation — the same misunderstanding can live in both. **Fix:** freeze an untouched donor
executable for gain0; for gain1 derive independent analytic fixtures (mass, added mass, force, momentum)
or add a narrowly-reviewed parameter seam with retained provenance validated against gain0 byte-for-byte.

**9. The pose port has invalid root and empty-body indexing.** Roots have `seg_parent_g=-1`, but the
depth pass gathers every parent before updating depth-selected rows (plan:291-303); in PyTorch, `-1`
gathers the *final* segment, not identity. Empty bodies cannot safely execute `pos[tail_gidx]`, and
`torch.where` does not prevent those gathers from being evaluated. **Fix:** allocate explicit
identity/sentinel rows and remap every root/empty parent and tail index to them before any gather; test
empty, root-only, max-depth, and mixed live/dead batches.

**10. The quaternion inverse is not the donor operation.** The plan uses `quat_rotate(conj(rot), uj)`
(plan:307); the donor uses `Quaternion.Inverse(R)` = `conj/‖q‖²` (SwimEval.cs:1000). Conjugate equals
inverse only for an exactly unit quaternion, and repeated f32 compositions do not guarantee unit norm.
**Fix:** implement `q⁻¹=conj(q)/‖q‖²` with the donor's degenerate semantics, or prove/enforce
normalization at every matching point; oracle-test long pose chains.

**11. The float32/float64 policy will accidentally push the hot loop into float64.** LambK produces
`seg_ma` in f64 and `StepLedger`/`dv`/`m_eff` are "f64-like" (plan:312-314) while the plan claims a f32
hot loop; mixing f64 added mass with f32 rotations promotes the M_eff assembly/solve. Per-body f64
ledgers written every step are also not a cheap "global compensated ledger." **Fix:** compute LambK in
f64 once, explicitly cast stored hot-loop coefficients to f32, keep high-precision validation in a
separate config, and state exactly which tensors participate in the throughput gate.

**12. `SolveSym3` is treated as a trivial port despite being numerically fragile.** The donor's absolute
`|det|<1e-12` branch is not scale-invariant (SwimEval.cs:1151); cofactor inversion can produce
catastrophic relative error for ill-conditioned matrices well before that threshold fires, and
NumPy-random tests do not cover the failure region (plan:375-383). Compilation may also fuse/reassociate
expressions, defeating the claimed "exact donor op-order." **Fix:** generate M_eff from extreme legal
morphologies; test residual, condition number, finiteness, and compile/eager parity; add scale-relative
regularization or a robust SPD solve for production, retaining the cofactor form only for donor
conformance.

**13. The RNG manifest does not guarantee stream stability.** A zero-count entry for an inert gene
preserves downstream draws only *while it remains inert*; once activated and consuming draws, every
subsequent sequential draw shifts. Slot recycling/compaction further change iteration order, and there
is no stable entity ID in `ColonyState` to key RNG to organism identity (plan:167-180, 663). **Fix:**
use counter-based keyed RNG streams such as `(seed, step, stable_entity_id, gene_iid, event,
draw_index)`; add stable entity IDs, parent IDs, allocator state, and innovation-registry state to
checkpoints.

**14. "Replay from ColonyState alone" is false.** The listed `ColonyState` omits field reservoirs,
genomes, developed bodies (or their regeneration inputs), RNG states, stable IDs, free-list state,
innovation registry, config, clock schedule, and external forcing — yet S7 requires replay from that
object alone (plan:602-604). **Fix:** define a versioned `SimulationSnapshot` containing all
authoritative state; `ColonyState` remains the creature-state subset, not a complete checkpoint.

**15. The ecological energy ledger has no actual closed system.** S3 debits basal metabolism and
locomotion but identifies no heat/waste reservoir; production introduces chemical energy from light,
yet S1 models light as an untracked analytic value; predation/feeding mix nutrient mass and energy
without a biomass energy-density conversion. "Energy books closed end-to-end" is rhetoric, not an
implementable equation (plan:566-572, 646-657). **Fix:** define every energy reservoir and external
flux (incident/absorbed light, chemical biomass, reserve, mechanical work, wake/drag dissipation,
metabolic heat, exported energy) and the unit conversions/transaction equations before S1/S3.

**16. The milestone dependency graph contradicts itself.** The table says S5 depends on S4
(plan:556-564); the critical-path text says `S3→S5` with S4 branching separately (plan:742); S6
nominally depends only on S1 yet the same paragraph postpones it until S5. **Fix:** publish one DAG with
explicit artifact-level dependencies and remove the prose alternatives.

**17. S4 demands evolution before an evolutionary engine is specified.** S4 requires an unseeded
predator to arise, but the plan specifies no mutation/selection loop, rates, inheritance, or sufficient
sensory fields first, while asking for vision/smell/lateral-line/electroreception despite the associated
fields not existing (plan:574-580). **Fix:** land and validate asexual mutation, inheritance, lineage
tracking, and the minimum sensing environment first; seed a functional predator for mechanism
validation; reserve "unseeded emergence" for a separate research experiment.

**18. The import-linter contract does not enforce "contracts-only crossings."** The forbidden contract
covers only two source modules and four destinations (plan:114-140); it does not prevent `core` from
importing concrete internals, `observe` from reaching everything, or consumers from importing
non-contract modules. **Fix:** define exhaustive forbidden/independence contracts for every layer;
permit only public contract modules.

**19. The benchmark program is not executable as a one-week task.** The advertised sweep is 576 cells
before repetitions/compilation/graph-capture/profiler/correctness/100k-step energy/fixtures/two-layouts,
with `K`, timing duration, memory limits, timeout policy, compile-cache isolation, and graph-recapture
rules unspecified (plan:367-369, 821-825). Profiling every cell will dominate the schedule and compiled
graphs obscure operator attribution. **Fix:** stage the benchmark — correctness corpus, small crossover
sweep, selected representative H1/H2 cells, profiler only on winning/failing configs; budget GPU-hours;
define statistical confidence.

**20. The risk register badly understates where failure will occur.** It labels gain1 mass migration
low/medium, deterministic reduction medium, S1 closure "solved," and S3/S4 mechanisms solved — the
load-bearing unresolved problems. Likely first-failure sequence: (1) deterministic ragged reduction has
no implementation; (2) flattened churn breaks static capture; (3) the energy gate fails because its
equation is wrong; (4) gain1 oracle generation requires altering the oracle; (5) S1 cannot produce its
required bloom crash; (6) end-to-end throughput bears no relation to the S0 number. Until corrected,
"S0 is not a go/no-go spike. It is an expensive benchmark of a partially ported donor bug against
self-authored fixtures."
