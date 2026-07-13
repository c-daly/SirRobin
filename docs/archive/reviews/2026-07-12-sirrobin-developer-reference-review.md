# Review of the SirRobin developer reference

**Reviewed:** `docs/2026-07-12-sirrobin-developer-reference.md`  
**Date:** 2026-07-12  
**Disposition:** substantive correction required before the document can serve as S1 developer authority

## Executive position

The developer reference is a strong S0 API orientation, especially around the effective-mass equation, the
two oracle arms, the frozen H0/H1/H2 corpus, mixed mechanical tolerances, and the executable prefix-drift gate.
It was nevertheless stale in several load-bearing places: S0 decision status, Gate-E authority, the S1
reservoir and energy scope, the field/economy/physics dependency graph, and the claimed yaw capability.

This is a focused correction, not a wholesale rewrite. Parts I and most of the concrete S0 kernel reference are
sound. The status, economy, field architecture, authority hierarchy, and a handful of API details need updating.

## Findings, ranked

### 1. S0 status and Gate-E conclusion were stale

The reference described S0 as “substantially implemented,” “in verification,” and “under active development.”
The repository now contains two distinct, provenance-preserving decisions:

- the original 1,000-creature/90M Gate E is `NO-GO / REVISE` under its frozen threshold;
- the later pre-registered population-grounded Gate E is `GO` at both 5,000 and 10,000 creatures.

The reference must state both results and must not relabel the original failure. The revised result authorizes
S1 while leaving the whole future tick budget unresolved.

**Required edit:** update the header, roadmap, S0 section, maintenance note, and authority list to cite:

- `docs/archive/plans/2026-07-12-sirrobin-locomotion-gate-E-revision.md`;
- `docs/superpowers/reports/2026-07-12-sirrobin-S0-decision-report.md`;
- `docs/superpowers/reports/2026-07-12-sirrobin-S0-population-gate-revision-report.md`.

### 2. The throughput default was presented as current authority

`LocomotionConfig` still contains the historical `n_cap=1024`, `n_live=1000`, and
`throughput_floor=9.0e7`. The reference listed these without explaining that they belong to the original failed
experiment.

**Required edit:** label them historical S0 defaults and document the current population-derived floors:

| Live population | Capacity | Hard floor |
|---:|---:|---:|
| 5,000 | 5,120 | 600,000 creature-steps/s |
| 10,000 | 10,240 | 1,200,000 creature-steps/s |

**Follow-up code decision:** eventually remove the single scalar throughput floor or replace it with a derived
population/timestep rule. Preserve old config hashes in historical reports rather than treating the old default
as current policy.

### 3. The economy section conflicted with the S1 authority and overstated energy conservation

The reference described two currencies closing in integer quanta, an exact “metabolic-expenditure ledger,” and
an f32 “total-energy invariant.” That is not the landed S0 mechanism and is not the S1 plan:

- an expenditure counter is not a conserved energy reservoir;
- S0's `R_step` is a discrete mechanical work-consistency residual, not total physical-energy conservation;
- S1 conserves nutrient mass only;
- S1 has exactly `Nd_q`, `Bp_q`, `Bd_q`, and `Bm_q`;
- `struct_N`, `Sed`, creature reserve, grazing, metabolism, reproduction, and predation are absent;
- a native energy ledger waits for real reserve and heat reservoirs.

**Required edit:** split the S1 nutrient economy from later trophic/energy work and remove every implication that
S0 or S1 already possesses a conserved total-energy reservoir system.

### 4. The entity model and package layering could not represent S1 correctly

The reference divided state into point entities and read-only abiotic fields. S1 introduces mutable conserved
Eulerian reservoir fields: they are neither point entities nor read-only exogenous backgrounds.

The total chain `numerics -> physics -> fields -> genetics -> core -> observe` also conflicts with the S1
capability design. Physics and fields/economy are siblings over numerics. Economy may consume public field
contracts; physics must remain independent.

**Required edit:** distinguish three storage roles:

1. continuous-position point entities;
2. mutable conserved Eulerian reservoir fields, changed only through economy transactions;
3. exogenous/abiotic drivers sampled through one-way interfaces.

Ratify these ownership boundaries:

- `fields` owns generic `FieldSample`, geometry, interpolation, and transport;
- `economy` owns reservoir state and transactions;
- `physics` owns mechanical `FluidSample`/future `MediumSample`;
- future `core` samples fields and constructs the physics input without either sibling importing the other.

### 5. The embodiment section claimed a yaw capability S0 does not have

The reference said frozen-heading S0 realizes `{surge, yaw}` and therefore action-transfer risk is low. S0 has
no yaw state or angular integration. It solves x/z translation while holding heading fixed.

**Required edit:** state that StepLive/yaw remains an S2 performance and fidelity risk and that the later Talos
action mapping remains unverified.

### 6. Several S0 scope and repository descriptions were inaccurate

- S0 was described as having a flat seabed, uniform minerals, and empty vents; S0 has no field/geology system.
- The “actual module layout” omitted `validation/drift.py` and most offline tools.
- “No `report.py`” needed qualification because `tools/build_s0_report.py` exists; only a production report
  module is absent.
- Several API line references had drifted, notably `Pose`, `MassProperties`, and `StepLedger` in
  `physics/contracts.py`.

**Required edit:** update the tree and wording, and continue treating line numbers as navigational rather than
stable API identifiers.

### 7. Import-linter was described as running in CI when no CI workflow exists

`setup.cfg` defines import-linter contracts, but the repository does not contain a CI workflow that executes
them.

**Required edit:** say that boundaries are configured and checked with import-linter. Either add CI later or do
not claim CI enforcement.

### 8. Authority and provenance needed refreshing

The reference treated the master design as the unqualified target, although later correction plans supersede
its float-ledger, Gate-E, and S1 package statements. It also contained repeated citations to unspecified
“memory,” which are not durable repository provenance.

**Required edit:** make authority slice-specific, add the consolidated S1 plan once accepted, and replace
“memory” citations with repository documents or remove them.

## Coupled-document corrections

Correcting the reference alone would leave contradictory instructions:

- `CLAUDE.md` said there was no simulation code and described a float64 conservation ledger;
- the master design retained the original 90M gate;
- the master design described f32/f64 nutrient reservoirs and tolerance-based mass closure;
- the master design made fields depend on physics and assigned mechanical medium construction to fields.

These must be reconciled in the same documentation tranche so repository search returns one current answer for
S0 status, S1 reservoirs, energy scope, and dependency ownership.

## Material retained as sound

The following parts should remain substantially intact:

- the project premise and nine laws, with explicit system-boundary wording for conservation;
- the S0 frozen-heading scope;
- `M_eff = M_body*I + sum(R*diag(m_add)*R^T)` and the separation of body and added mass;
- gain0 donor conformance and gain1 independent analytic corroboration;
- the frozen 192-body H0/H1/H2 corpus and anti-gaming constraints;
- mixed dimensioned tolerances and zero authorization regularization;
- the 100,000-step prefix-budget drift detector;
- fixed-slot representation and capability-based runtime module names.

## Adopted disposition

The S1 execution tranche adopts all required corrections above. The developer reference, master design, and
project instructions are amended before S1 fixtures are frozen. The review remains as the durable rationale for
those edits.
