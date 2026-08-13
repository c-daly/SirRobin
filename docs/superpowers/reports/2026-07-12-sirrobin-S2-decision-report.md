# SirRobin S2 decision report

**Date:** 2026-07-12

**Decision:** **NO-GO**

**Blocking falsifier:** F12 — the frozen desired-heading controller does not home and settle

**Execution authority:** `../plans/2026-07-12-sirrobin-S2-canonical-body-live-locomotion-implementation-plan.md`

## Outcome

S2 produced the intended canonical genotype-derived body, ENU/FLU live mechanics, additive force seam, yaw
angular-momentum integration, stateless morphology queries, field/core composition, and lossless restart. The
mechanical kernel is finite, work-consistent, and fast enough on the declared CUDA Graph rung at both 5,000 and
10,000 live creatures. It is not authorized for S3 because the frozen donor-shaped heading policy fails its own
home-and-settle test. This is a terminal scientific gate, not a performance exception.

## Bound evidence

- Live-config SHA-256: `89a3fe8964f7bd493d4a1ee3514b0ab5e974931942786142e088d49da8a4b6f2`.
- Fixture manifest SHA-256: `0c3aeb253329929cf51ef72968a8413869646766ba03a9e3e569f603af7f9c50`.
- Development source: `d813838f788cbad1f1c7e757ae9f75d5214fc03fc5fddcd8504e3c52af557aa6`.
- Hydrodynamic source: `5d6527c0204c3dcde3938707621c402a1dc7dc6a2c3416a72524efd96e16c759`.
- Live-step source: `8305dccd2f5b699c94a56017d6374bc031b3c23341d1a5ecaac8ffd771fe859d`.
- Controller source: `88119e2fc05658a5099d8c6a2e7ae0dee75c0107f5bb760051dd69d4e7ebbd79`.
- Benchmark harness: `7819a4a970932abe9bbfbfb8b8401d3b10ff659b543a388b8ccdfa1060760007`;
  runner: `5115bb93d6edef740749bdb7659ffd0a0b494a43f23584772b4e227da9eeb8fd`.
- CUDA device: NVIDIA GeForce RTX 5070, 12,227 MiB, compute capability 12.0; PyTorch 2.13.0+cu130.

## Gate disposition

| Gate | Result | Evidence |
|---|---|---|
| A — authority/architecture | PASS | ENU/FLU docs reconciled; seven import-linter contracts green; capability names only |
| B — development/representation | PASS | 32 D0/A1 bodies; fixed `[W,N,17]`; GL256/GL512 tolerance; exact repeated development; CPU/CUDA agreement |
| C — live physical fidelity | **FAIL** | open-loop mechanics and work identities pass; F12 closed-loop settlement fails on CPU f64 and CUDA f32 |
| D — form is function | PASS | no stat vector; locomotion is stepped physics; intake and morphology are stateless geometry queries |
| E — affordability | PASS, CUDA Graph scope | all four H1/H2 5k/10k CUDA cells clear the hard floor with zero interventions; CPU is non-viable |
| F — integration/restart | PASS | passive-current transport, ENU wrapping, unchanged depth, genotype+live-state exact continuation |

The 100,000-step drift authorization and profiler retirement were not run after F12 became terminal. F10 and the
full profiler detector therefore remain unretired; this report does not use partial mechanics evidence to claim
them green.

## Frozen population results

Five repetitions follow warmup. The table reports the minimum, which is the authorizing statistic.

| Corpus | Live / capacity | Rung | Minimum creature-steps/s | Hard floor | Peak allocation | Result |
|---|---:|---|---:|---:|---:|---|
| H1 | 5,000 / 5,120 | CUDA Graph | 1,979,326 | 600,000 | 95,562,240 B | PASS |
| H2 | 5,000 / 5,120 | CUDA Graph | 1,971,100 | 600,000 | 95,562,240 B | PASS |
| H1 | 10,000 / 10,240 | CUDA Graph | 2,806,508 | 1,200,000 | 123,399,680 B | PASS |
| H2 | 10,000 / 10,240 | CUDA Graph | 2,814,791 | 1,200,000 | 123,399,680 B | PASS |

Every authorizing cell records zero solve regularizations, inertia-floor hits, omega backstop hits, and nonfinite
events. None reaches the 5x aspiration (3M at 5k; 6M at 10k). A non-authorizing two-repetition compiled H1/5k
diagnostic reached a 3,008,573 minimum after 109.8 seconds of compile time; it does not replace the five-run matrix.

CPU eager minima were 59,401–65,527 at 5k and 80,935–81,088 at 10k, far below the hard floors. CPU is not a
viable S2 live device scope. CUDA eager also missed (251,892–260,628 at 5k; 505,559–528,344 at 10k); graph capture
is the required production rung.

## F12 failure

The f64 swimmer initially turns toward a +90-degree request but does not settle. After 1,200 steps it ends at
`yaw=-1.4623 rad`, `turn_bias=+0.1338 rad/depth`, with a wrapped error of `3.0331 rad`. The matching CUDA Graph
test also fails its 15-degree settlement detector. Open-loop positive/negative phase-reflected commands remain
opposite and symmetric, so the failure is in the feedback policy/reference interaction rather than a yaw clamp or
force-sign fabrication.

Exploratory damping/reference/authority changes were rejected and reverted because they did not settle and were
not part of the frozen policy. The failing tests remain strict expected failures so ordinary regression suites are
usable without disguising the decision: changing them to pass requires a successor authority and a new S2 report.

## Verification summary

- S2 CPU suites: 16 passed, 1 strict expected failure.
- Existing S0/S1/field/numerics suites: 59 passed, 1 CUDA skip across the separately reported groups.
- S2 CUDA suite: 1 passed, 1 strict expected failure.
- Ruff: green.
- Import-linter: 7 contracts green.
- `git diff --check`: green at report construction.

## Required next move

Do not begin S3. Write a narrow successor plan for a stable heading controller whose constants and settlement
corpus are committed before execution. A promising direction is a rate-damped or stopping-distance policy using
derived `omega=L/I`; it must continue to actuate only gait shape and must not write yaw, yaw rate, momentum, or a
kinematic turn clamp. Then re-run Gate C, the 100,000-step drift budget, and the full Gate-E matrix under a new
config hash.
