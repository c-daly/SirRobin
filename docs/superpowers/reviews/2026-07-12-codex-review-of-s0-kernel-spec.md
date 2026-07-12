# Codex Review — S0 / SpikeSwim Kernel Spec

**Date:** 2026-07-12  
**Reviewed document:** `docs/superpowers/plans/2026-07-12-sirrobin-S0-kernel-spec.md`  
**Review frame:** scientific fidelity, numerical validity, and fidelity to the documented purpose of S0  
**Verdict:** **NOT APPROVABLE as the standalone, build-ready S0 authorization spec**

## Executive position

The kernel spec contains several real improvements over Rev-4. In particular, it corrects the
regularization-impulse sign, separates the vertical constraint from the two-dimensional solve, masks eager
branch denominators, restores the quadrature Jacobian, caps source transfers, and distinguishes the f32
production path from the f64 validation arm.

Those corrections make parts of the proposed solver more internally coherent. They do not make the document
a faithful S0 specification.

The central problem is that the document silently changes what S0 proves. The authoritative design defines
SpikeSwim as a go/no-go experiment for the full vectorization thesis: faithful hydrodynamics over realistic
ragged bodies must be deterministic, agree with independent and donor evidence, survive long-horizon energy
checks, and clear the affordability target on H1/H2 populations while exposing the CPU/GPU crossover,
heterogeneity tax, masking tax, compaction cost, and profiler bottleneck. The new spec replaces much of that
experiment with algebraic identities, one incompletely specified gain1 fixture, a fake mass-transfer scaffold,
and a throughput requirement that can be satisfied by a single non-OOM cell.

That is not a minor editorial difference. It allows S0 to pass without testing the risks S0 was created to
falsify.

There is a second, independent problem: the document specifies a robust 2x2 numerical solve, but not the full
physical step that supplies its matrix and impulse. The effective inertial matrix, gait and pose advancement,
force construction, state-update ordering, episode protocol, and complete oracle corpus remain missing or
ambiguous. Algebraic closure can prove that internally shared symbols agree with one another; it cannot prove
that the symbols were produced by the correct hydrodynamics.

The appropriate disposition is therefore:

1. retain the solver and quadrature corrections;
2. restore the full S0 experimental acceptance matrix;
3. reconcile the canonical-layout conflict with the authoritative design;
4. specify the complete physical step and commit the oracle values before implementation is authorized.

## The standard this review applies

The repository instructions name the master technical design as authoritative
(`CLAUDE.md:9-16`). They also establish the relevant non-negotiable laws:

- fidelity is the product (`CLAUDE.md:41-42`);
- conserved quantities move between tracked reservoirs (`CLAUDE.md:43-44`);
- capability is derived from morphology through physics (`CLAUDE.md:45`);
- there is one canonical representation per quantity (`CLAUDE.md:46-47`);
- claims must be measured rather than asserted (`CLAUDE.md:65-67`);
- completion requires evidence, not prose (`CLAUDE.md:68-75`).

For S0 specifically, the master design says the spike exists to test the optimistic performance premise before
the architecture is built on it (`docs/2026-07-11-sirrobin-design-document.md:1074-1078`). It defines:

- a 3 s warmup plus 5 s measured episode at `dt = 1/120`, a 100,000-step energy configuration, and a
  population-churn stub (`design-document.md:1078`);
- flattened/CSR segments as canonical, with padded storage as an ablation (`design-document.md:1080`);
- CPU and CUDA sweeps over H0, H1, and H2 populations and the r0/r1/r2 acceleration ladder
  (`design-document.md:1082`);
- determinism, energy, oracle, and throughput gates (`design-document.md:1084-1094`);
- explicit falsifiers for ragged batching, reduction cost, launch overhead, determinism tax, oracle divergence,
  energy drift, and churn (`design-document.md:1096-1103`);
- a meta-falsifier stating that an H0 result cannot authorize the architecture and that H1/H2 must clear all
  gates (`design-document.md:1105`);
- the S0 roadmap criterion: all four gates pass at H1/H2 and a telemetry-backed go/no-go decision is recorded
  (`design-document.md:1113`).

The new document calls itself standalone and says its four gates are "the whole acceptance set"
(`S0-kernel-spec.md:1-5,15-25`). It must therefore contain, or explicitly and safely supersede, all of the
above. It currently does neither.

## Blocking findings

### 1. The acceptance set no longer tests the S0 thesis

**Severity: critical**

The proposed gates are:

- two force-law algebraic identities;
- the discrete `R_step` identity;
- an independent gain1 fixture comparison;
- a throughput measurement on at least one non-OOM authorization-sized cell;
- an additional exact-int64 fake mass-reservoir scaffold test.

These are listed as the entire acceptance set at `S0-kernel-spec.md:15-25`; the throughput condition is expanded
at `:263-269` and the final checklist at `:273-278`.

The set omits or no longer makes load-bearing:

- same-device bit-identical CPU and CUDA determinism;
- H1 and H2 as the mandatory authorization populations;
- the CPU/GPU crossover `B*`;
- H1/H0 heterogeneity tax;
- padded/flattened masking tax;
- population churn and compaction cost;
- the requirement that profiling show the step is force/solve-bound;
- the 8-second episode aggregates;
- the explicit S0 falsifier register;
- the meta-falsifier forbidding H0-only authorization.

#### Concrete failure scenario

An implementation can benchmark one homogeneous padded body population at a favorable batch size on CUDA,
produce a non-OOM value above `F_loco_S0`, and pass gate (d). It need never show that realistic ragged H1 bodies
or skewed H2 bodies fit, remain fast, or avoid pose/reduction domination. It need never measure compaction or the
CPU crossover. That result would pass the new gate while directly violating the authoritative S0
meta-falsifier.

Similarly, an implementation can be nondeterministic on CUDA and still satisfy the stated four gates, because
determinism is no longer in the acceptance table.

#### Required correction

Restore an explicit S0 acceptance matrix covering:

- `B` sweep;
- CPU and CUDA;
- f32 and the f64 validation configuration;
- r0 eager, r1 compile, and r2 graph capture;
- H1 and H2 as mandatory authorization populations;
- determinism and determinism tax;
- throughput, crossover, heterogeneity tax, masking tax, and profiler share;
- churn/compaction;
- single-step oracle terms and episode aggregates.

The fake-reservoir transfer tests can remain preparatory scaffold coverage. They must not substitute for any
part of the physics/vectorization experiment.

### 2. The spec changes the canonical layout without resolving document authority

**Severity: critical**

The new spec adopts fixed `[B, S_slot]` padded segment storage as its required layout
(`S0-kernel-spec.md:29-38`). The master design says the canonical segment representation is flattened/CSR and
that padded `(B,16)+mask` exists only as an ablation (`design-document.md:183-210,447-471,1080`). The master
design also makes the masking comparison part of the S0 throughput gate (`design-document.md:1094`).

Rev-3 contains a deliberate later decision favoring padded storage and an analytic/same-layout estimate instead
of a flattened prototype (`2026-07-12-sirrobin-plan-rev3-reconciliation.md:520-528`). That may be a reasonable
decision. The repository nevertheless contains two incompatible normative answers, while the new spec says it
supersedes only Rev-1 through Rev-4 for S0. It does not state that it supersedes the master design, and the
project instructions continue to name the master design as authoritative.

#### Concrete failure scenario

An engineer faithfully implements padded storage from the S0 spec. A later engineer faithfully implements the
canonical flattened `DevelopedBody` contract from the master design. S2 then requires a representation change
or synchronization/conversion boundary immediately after S0, precisely where the single-representation and
measure-before-commit rules were meant to prevent architectural debt.

#### Required correction

Record one explicit architecture decision and reconcile every normative document:

- **Option A:** padded storage is now canonical. Amend the master design, explain why the measured ablation was
  replaced, define the new representation seam through S2, and retain an executable masking/occupancy risk
  gate.
- **Option B:** flattened/CSR remains canonical. Restore it to S0 and retain padded storage as the measured
  ablation.

Until this is done, the spec cannot accurately call itself standalone or authoritative.

### 3. `M` is physically ambiguous and the complete production step is absent

**Severity: critical**

The new spec defines `M` as the symmetric 2x2 "constrained added-mass matrix"
(`S0-kernel-spec.md:59-63`). The governing dynamics in the master design use the **effective inertial matrix**:

```text
M_eff = M_body I + sum_j R_j diag(maX, maY, maZ)_j R_j^T
```

(`design-document.md:509-515`).

These are not interchangeable. Added mass alone is the inertia of entrained fluid. The body's own inertial mass
must also resist acceleration. Omitting `M_body I` produces incorrect acceleration and makes singular or extreme
condition numbers possible for bodies that should have a positive isotropic inertial floor. If the new spec
intends `M` to mean `M_eff`, calling it an added-mass matrix is a build-breaking ambiguity.

The same section begins at the solve rather than at the physics. The document does not build-readily define:

- how segment pose advances from the frozen-heading gait;
- how tail-tip `U`, `V_t`, and slope `s` are obtained;
- the actual reactive and fin force equations, rather than only their power identities;
- segment drag and its reduction;
- how `M_n` and `M_{n+1}` are constructed;
- whether force sampling uses the old, new, or midpoint pose;
- the complete velocity and position update order;
- the production `dt = 1/120` contract;
- the 960-step warmup/measurement episode.

The force identities at `S0-kernel-spec.md:168-179` cannot fill this gap. They test relationships between force
and power terms after those terms exist; they do not specify how the terms are calculated.

#### Concrete failure scenario

An implementation accidentally samples lateral tail velocity at the segment center instead of the tail tip. It
then computes `tReact`, `pWake`, and `InputPower` consistently from those wrong kinematics. The algebraic identity
passes exactly, `R_step` closes against the resulting impulse, and the implementation remains scientifically
wrong. Only a complete independent force oracle or donor fixture exposes it.

#### Required correction

Add a normative one-step dataflow covering:

1. pose/gait evaluation;
2. tail-tip and per-segment kinematics;
3. Lamb/body mass construction and `M_eff` reduction;
4. reactive, fin, and drag force construction;
5. constrained solve;
6. velocity and position integration;
7. ledger emission;
8. episode accumulation.

Define `M` unambiguously as the x/z principal submatrix of `M_eff`, including `M_body I` and the sim-mass-to-kg
conversion. Pin `dt = 1/120` for production and explain any deliberately different oracle-only timestep.

### 4. The independent oracle is still promised rather than committed

**Severity: critical**

The H1 fixture section supplies geometry and kinematics, followed by symbolic expected-output names
(`S0-kernel-spec.md:217-231`). It then states that the generator will be run later and its numbers committed
(`:233-236`). Round 4 explicitly identified the absence of literal expected values as an unresolved S0 blocker
(`reviews/2026-07-12-codex-round4-of-rev4.md:41-42,70-76`). Repeating the intended generation procedure does
not close that blocker.

The stated inputs also do not uniquely determine all promised outputs. At minimum, they omit or leave ambiguous:

- inertial/tissue density or literal inertial mass;
- the body-mass unit conversion;
- segment topology and segment count;
- surface/tail flags;
- fin area/aspect-ratio convention;
- fin profile drag, efficiency, and stall constants;
- drag coefficient and reference-area convention;
- whether the fixture represents the production `M_eff` or added mass alone.

One 10:1 prolate body also does not cover the required error surfaces: isotropic, tilted anisotropic, fin-tail,
multi-segment ragged reduction, mirror-paired H1, skewed H2, or the untouched-donor gain0 path. The master design
requires frozen H1/H2 fixtures for Lamb factors, force terms, all six `M_eff` entries, `dv`, and 8-second episode
aggregates (`design-document.md:598-602`). Rev-3 separately retained untouched-donor gain0 conformance and
analytic gain1 coverage (`rev3-reconciliation.md:488-493`). The new checklist drops both the broader corpus and
the donor-conformance arm.

There is also an internal checklist defect: section 6 requires a demonstrated 32-vs-64-point or SciPy
convergence result (`S0-kernel-spec.md:249-259`), but the final acceptance checklist does not include
`test_gain1_quadrature_converged` (`:273-278`).

#### Concrete failure scenario

The oracle generator and kernel both use the same mistaken axis convention or omit body inertial mass. The
single axial H1 fixture agrees. The error only becomes visible for a tilted multi-segment body, which the new
acceptance set never requires.

#### Required correction

Commit the oracle artifact before authorization, including:

- literal, complete input records;
- literal expected numeric outputs;
- units, dtype, and configuration hash;
- generator version/provenance;
- isotropic and tilted-anisotropic cases;
- reactive-only and fin-tail cases;
- multi-segment H1 and skewed H2 cases;
- the higher-order convergence result;
- a separate untouched-donor gain0 conformance corpus;
- episode aggregates where the donor remains the intended behavioral reference.

The test suite must read committed values. It must not generate expected results during the test run.

## Scientific and numerical findings

### 5. The energy normalizations are ill-conditioned at scientifically ordinary states

**Severity: high**

Gate (b) normalizes each residual as:

```text
abs(R_step) / max(KE_n, epsilon)
```

(`S0-kernel-spec.md:19-20,181-197`). Neither the value nor units of `epsilon` are pinned. At the first powered
step from rest, `KE_n = 0` while impulse and `KE_{n+1}` are nonzero. A small floating-point residual is then
divided by an arbitrary floor rather than by the energy scale of the event. The test can fail or pass depending
primarily on the chosen epsilon.

The force-law closures use `max(abs(p_in), epsilon)` (`:176-179`). This is better than the earlier signed
denominator, but it remains unstable near a genuine cancellation where signed input power approaches zero while
the individual work and wake terms are finite.

The master S0 budget instead normalizes the long-horizon error by cumulative absolute work
(`design-document.md:1090`). That scale remains meaningful at rest and across signed-power cancellations.

#### Required correction

Use an explicitly dimensioned mixed absolute/relative criterion. For example, the step scale should include the
largest of:

- `abs(KE_n)`;
- `abs(KE_{n+1})`;
- `abs(v_mid dot impulse)`;
- `abs(0.5 v_n^T DeltaM v_n)`;
- a pinned absolute energy floor in joules.

For each power identity, scale by the largest absolute constituent term as well as the two sides, not only by
the potentially cancelling net input power.

### 6. "Bounded-oscillating" is not an executable gate and is not implied by `R_step`

**Severity: high**

The spec requires the 100,000-step drift curve to be bounded-oscillating (`S0-kernel-spec.md:19-20,196-197`)
but supplies no mathematical definition. There is no specified cumulative quantity, normalization, envelope,
window length, trend estimator, or permissible bound.

There is also a conceptual conflation. `R_step` is the algebraic residual of the same discrete update:

```text
R_step = DeltaKE
         - v_mid dot (F_stream Dt + J_reg)
         - 0.5 v_n^T DeltaM v_n
```

When all quantities are calculated from the same update, this is primarily an implementation-consistency
identity. Its remaining f32 error is roundoff and evaluation-order error. There is no general scientific reason
for accumulated floating-point roundoff in an actively driven, dragged, pose-varying system to trace a
symplectic bounded oscillation. Random-walk error may be unbiased without being bounded-periodic; a shared
systematic error can also cancel algebraically and appear perfectly flat.

#### Required correction

Separate the claims:

- **step identity:** verifies the solve/integrator bookkeeping;
- **cumulative numerical bias:** verifies that evaluation order does not cause systematic drift;
- **physical power closure:** connects muscle input, thrust work, wake loss, fin loss, drag loss, and KE change;
- **oracle validation:** verifies that the underlying hydrodynamic terms are correct.

Define the cumulative test numerically, for example as cumulative signed `R_step` divided by cumulative
absolute work, with a pinned maximum magnitude and a pinned monotonic-trend test over fixed windows. Do not use
the word "oscillating" unless the measured quantity and the expected oscillatory mechanism are identified.

### 7. Regularization is accounted algebraically but is not a physical damping mechanism

**Severity: high**

For a regularized solve,

```text
(M + reg I) Delta v = P
M Delta v = P - reg Delta v
J_reg = -reg Delta v
```

The sign correction is mathematically right (`S0-kernel-spec.md:138-143`). Adding `J_reg` to `R_step` also makes
the discrete identity reflect the dynamics actually executed.

The spec goes further and calls the branch a damped approximation and `J_reg` artificial damping
(`:122-125,192-195`). That physical interpretation is not generally valid. Its work is

```text
J_reg dot v_mid = -reg Delta v dot (v_n + 0.5 Delta v)
```

which is not sign-definite. It can remove or add body kinetic energy depending on the prior velocity and
impulse. It is a numerical modification to inertia, not a physical dissipative force with a guaranteed sink.

Recording the signed term explains why the modified discrete equation closes. It does not satisfy the physical
rule that removed mechanical energy reaches a wake, heat, or other tracked sink
(`design-document.md:439-445`). Nor does it prove that a regularized body's motion remains faithful to its
morphology.

#### Required correction

For S0 authorization bodies, require regularization incidence to be zero. Positive-mass `M_eff` should provide
an isotropic inertial floor; an ill-conditioned authorization body should trigger investigation of units,
geometry, or matrix construction rather than silently receive modified dynamics.

If a protective fallback remains:

- call it a numerical intervention, not physical damping;
- report its signed work;
- count and expose every activation;
- reject malformed bodies;
- compare valid ill-conditioned cases with an f64 stable reference solve;
- make any activation in the H1/H2 authorization corpus a failure or an explicit go/no-go finding.

### 8. The f32 determinant-safety claim is stronger than its proof

**Severity: medium**

The spec claims that `EPS_SPD = 1e-6`, because it is greater than f32 machine epsilon, guarantees that
`det_reg` cannot become zero through cancellation (`S0-kernel-spec.md:122-135`). That does not follow merely
from `EPS_SPD > eps_f32`.

`det_reg` is still evaluated as a subtraction:

```text
(M00 + reg)(M22 + reg) - M02^2
```

For a highly rotated, highly anisotropic matrix, both products can be much larger than their difference.
Bounding the exact determinant below by `EPS_SPD lam_max^2` is useful, but a floating-point nonzero guarantee
requires a forward-error bound for this particular evaluation order and the admitted input range. Comparing a
constant with machine epsilon is not that proof.

#### Required correction

Either:

- provide a conservative f32 rounding-error bound over the pinned matrix domain;
- evaluate the regularized determinant through a numerically safer factorization/eigenvalue form;
- or use the f64 validation/reference path to demonstrate safety over an adversarial matrix corpus and phrase
  the result as a tested domain guarantee rather than an algebraic impossibility claim.

### 9. The int64 transfer primitive lacks the preconditions needed for its "exact" claim

**Severity: medium**

The source cap prevents an ordinary nonnegative source from being overdrawn
(`S0-kernel-spec.md:46-55`). It does not prevent:

- `dst_q + n_eff` overflowing signed int64;
- a negative input `src_q` making `n_eff` negative;
- a negative input destination remaining invalid.

The spec defers aggregate overflow analysis to S1/S3 (`:295-297`), but the S0 section calls the primitive exact
and order-independent without stating bounded-domain preconditions. Integer arithmetic is exact only while the
mathematical result remains representable.

#### Required correction

State and enforce:

```text
0 <= src_q
0 <= dst_q
0 <= n
dst_q <= INT64_MAX - n_eff
```

Pin construction-time reservoir bounds and add boundary tests for destination overflow. Aggregate world-lifetime
proofs may remain a later-phase responsibility, but the primitive's own contract cannot be unbounded.

## What should be retained

The following changes are technically worthwhile and should survive revision:

1. **Correct regularization sign.** `J_reg = -reg Delta v` follows directly from the modified solve.
2. **2-D/vertical separation.** The x/z momentum equation and workless vertical constraint are now stated
   separately.
3. **Eager-branch denominator masking.** Masking each denominator before division addresses the real
   `torch.where` non-short-circuit hazard.
4. **Scale-relative degeneracy direction.** Avoiding the old absolute determinant floor removes a genuine
   scale-invariance defect.
5. **Quadrature Jacobian.** The `1/t^2` factor is required by the stated change of variable.
6. **Independent generator boundary.** Prohibiting imports from torch production physics and the donor is the
   right anti-circularity rule, once the literal outputs and coverage are committed.
7. **Source cap and shortfall.** Returning a shortfall is the right transaction shape once range preconditions
   are added.
8. **Production/validation precision split.** f32 hot-loop state plus a separate f64 reference configuration is
   compatible with the single-representation law.

These corrections justify continuing preparatory solver and oracle work. They do not authorize the full S0
implementation or an S0 go decision in the current acceptance frame.

## Minimum revision required for approval

The next revision does not need to reopen S1/S3 ecology. It does need to close the actual S0 boundary:

1. Resolve padded versus flattened canonical storage in the master design and S0 spec.
2. Restore determinism, H1/H2, crossover, heterogeneity, masking, churn, profiler, and episode requirements to
   the S0 acceptance set.
3. Specify the complete physical step, including `M_eff = M_body I + M_added`, force construction, timing, and
   integration order.
4. Commit complete literal gain0/gain1 oracle fixtures and expected outputs before kernel authorization.
5. Replace cancellation-sensitive energy normalizations with dimensioned mixed tolerances.
6. Define the 100,000-step cumulative drift test mathematically.
7. Require zero regularization activations on the authorization corpus, or explicitly fail/escalate them.
8. Add int64 domain and overflow preconditions.

Once those items are present, S0 will again answer the question it was created to answer:

> Can the intended canonical representation run scientifically faithful, reproducible hydrodynamics over
> realistic ragged bodies at an affordable rate, with independent evidence strong enough to justify building
> the rest of SirRobin on it?

The current spec can answer whether a particular numerical solve and several locally defined identities are
self-consistent. That is useful, but it is a smaller question and cannot safely stand in for the S0 go/no-go
decision.
