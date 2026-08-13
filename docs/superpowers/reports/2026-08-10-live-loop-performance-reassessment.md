# Live-loop performance reassessment

**Date:** 2026-08-10

**Branch:** `restart/original-baseline`

**Status:** exploratory performance evidence from the uncommitted lifecycle
candidate. No reduced model is accepted or integrated by this report.

## Decision summary

The current complete loop is too slow for evolutionary use. The best measured
eight-creature canonical path advances only a few simulated seconds per wall
second. Moving the existing runner to the RTX 5070 does not fix it because the
small live batch, Python control flow, fine-grained synchronization, and repeated
energy-funding replays prevent efficient GPU use.

The canonical 120 Hz solver remains useful as an oracle and for bounded transient
bursts. It should not remain the ordinary ecological motion integrator.

The recommended production direction is a second-generation, state-dependent,
phase-averaged force/torque response evaluated as one large GPU batch. It must
retain current velocity, yaw momentum, developed form, effort, steering, fluid,
actuator work, and dissipation. The first prototype is fast enough to justify
continuing, but its yaw and energy errors are still selection-relevant, so it is
not ready to merge.

## Current complete-loop measurements

All new headless measurements used the current 0.1-second ecological cadence,
eight live creatures in 64 fixed slots, float32 mechanics, no rendering, no
Unity, no socket, and exact matter closure.

| Path | Simulated seconds / wall second | Result |
|:---|---:|:---|
| Compiled CPU, short headless sample | 2.47 | Fastest current canonical path |
| Compiled CPU, 50-tick profile | 1.63 | Slower after funding replays become common |
| CUDA compiled substep, original validation | 0.34 | Host synchronization dominates |
| CUDA compiled substep, effort validated once per interval | 0.49 | 44% better, still slower than CPU |
| Eager CUDA Graph over 12 substeps | 0.37 | Launch consolidation without compiled kernels regresses |
| Compiled substep inside 12-step CUDA Graph | 0.63 | Best CUDA result, still sub-real-time |
| CPU server with snapshot/socket work | about 1.17 | Earlier populated live observation |

The CUDA experiments all retained the canonical 1/120-second equations and exact
matter closure. The forced-CUDA server change was reverted after measurement.

## Profile attribution

A 50-tick compiled CPU profile took 3.064 wall seconds for five simulated seconds.

| Component | Approximate share | Important detail |
|:---|---:|:---|
| Funded mechanics | 73% | 141 complete mechanics trials for 50 ticks |
| Feeding | 11% | Per-creature intents and point-deposit planning |
| Maintenance | 10% | Per-creature Python settlement |
| Field economy | 4% | Already batched; not the primary problem |
| Other runner/accounting | 2% | Identity, closure, and reporting |

The energy-funding algorithm averaged 2.82 full mechanics trials per tick. It
first runs requested effort, restores state and retries reduced effort when the
chemical budget is insufficient, and can restore and retry a third time at zero
effort. This is scientifically conservative but computationally disastrous.

## GPU interpretation

The GPU itself is not the limitation. WSL and PyTorch 2.13 detect the NVIDIA
GeForce RTX 5070 and its 12 GB of memory. Existing isolated locomotion evidence
records roughly 1.97 million creature-steps/s at 5,000 bodies and 2.81 million at
10,000 with CUDA Graph replay. The fixed-capacity tensor representation is therefore
sound.

The complete runner defeats that representation by synchronizing scalars to Python,
dispatching small work at an eight-creature live batch, looping over creatures in
feeding and maintenance, and replaying entire trajectories to settle energy. CUDA
support is real; efficient whole-loop CUDA execution is not yet real.

## Lower-cadence canonical probe

The same equations were integrated for one second at lower rates and compared with
the 120 Hz reference swimmer. These are exploratory errors, not acceptance limits.

| Rate | Speedup | Position error | Velocity error | Yaw error | Work ratio |
|---:|---:|---:|---:|---:|---:|
| 60 Hz | 2.32x | 0.0257 m | 0.1286 m/s | 0.0114 rad | 0.952 |
| 30 Hz | 3.90x | 0.0709 m | 0.4058 m/s | 0.0335 rad | 0.877 |
| 15 Hz | 8.69x | 0.2255 m | 0.8923 m/s | 0.0710 rad | 0.640 |
| 10 Hz | 12.87x | 0.3434 m | 0.9827 m/s | 0.0505 rad | 0.529 |

Naively reducing cadence reaches an order-of-magnitude speedup only by changing
mechanical cost by 36-47%. That would materially alter selection and is rejected as
the scientific default. A 30-60 Hz mode could be useful as a temporary interactive
viewer mode if it is explicitly labeled, but it does not meet deep-time needs.

## Phase-averaged response prototype

The prototype sampled eight evenly spaced gait phases in one tensor batch. At every
reduced step it recomputed canonical hydrodynamic force, yaw torque, effective mass,
inertia, actuator power, and dissipation from the organism's current dynamic state.
It then integrated the averaged response over 0.1 second. Unlike the rejected 2026-08-09
scalar reduced model, it does not treat an early cycle mean as a permanent terminal
speed or yaw rate.

One-second comparison against all 120 canonical steps:

| Turn bias | CPU speedup | Position error | Velocity error | Yaw error | Actuator-work ratio | Dissipation ratio |
|---:|---:|---:|---:|---:|---:|---:|
| 0 | 9.97x | 0.0825 m | 0.1835 m/s | 0.0756 rad | 0.879 | 0.897 |
| +0.025 rad/depth | 9.15x | 0.0919 m | 0.3115 m/s | 0.00944 rad | 0.979 | 0.993 |
| -0.025 rad/depth | 7.93x | 0.1147 m | 0.3027 m/s | 0.145 rad | 0.863 | 0.854 |

Using two or four reduced integration steps per 0.1 second improved energy ratios,
but did not repair the important yaw/momentum discrepancy and reduced speedup to
roughly 2.3-5.1x. The missing behavior is phase-state correlation, not merely coarse
time integration.

### Benefit

- Replaces 120 sequential mechanics steps/s, often repeated nearly three times, with
  a small number of large phase batches.
- Naturally uses the GPU across phase, creature, and world dimensions.
- Remains form-derived and state-dependent.
- Can report the same named force, torque, work, and dissipation channels.
- Makes exact chemical budgeting possible without rerunning a full trajectory if the
  response is parameterized by effort inside one batched solve.

### Risk

- Simple phase averaging erases correlations between gait phase and evolving yaw
  momentum/velocity.
- Current errors are asymmetric between paired turn commands.
- A response trained or calibrated only near the founder could bias mutation and
  selection.
- Per-genotype calibration cannot become a hidden large birth cost.

## Recommended architecture

1. **Keep the canonical solver as an oracle and transient lane.** It remains the
   source of held-out evidence and handles states outside the reduced domain.
2. **Build a state-dependent phase response, not a terminal-target surrogate.** Batch
   phase samples for all creatures. Preserve signed inability and backward motion.
3. **Add phase-state memory.** At minimum retain one or more harmonics or a compact
   latent transient state so averaged yaw and surge can depend on where momentum sits
   relative to the gait cycle. Paired turn signs and zero-turn drift are mandatory
   held-out checks.
4. **Solve effort and energy together.** Evaluate force/work over batched effort knots
   or a monotone bounded response, choose the affordable effort on-device, and advance
   once. Do not replay the whole interval from Python.
5. **Preserve fixed addresses through lifecycle churn.** Rebuild developed body and
   response tensors in place so births and deaths do not invalidate compiled/CUDA
   graphs.
6. **Tensorize feeding and ordinary maintenance next.** Keep exceptional death/birth
   event creation on the host, but compute per-creature demands, allocations, carries,
   and masks in batches.
7. **Measure multiple population scales.** Eight creatures are below the GPU crossover.
   The acceptance surface must include interactive 64-slot performance and scientific
   128/5,000-creature complete-loop performance.

## Acceptance and stop conditions

Before the reduced response can drive selection, held-out tests must attack:

- straight, positive-turn, and negative-turn trajectories;
- founder-adjacent amplitude, frequency, and wave mutations;
- structural zeros, weak movers, backward movers, and hard nonrecurrent forms;
- current-velocity and yaw-momentum perturbations;
- actuator work and dissipation over both transient and settled horizons;
- CPU/CUDA finiteness and exact matter/chemical settlement; and
- complete-loop throughput, not isolated response throughput.

If one phase-state correction cannot materially improve the asymmetric yaw and work
errors while retaining a substantial speedup, stop that model. The fallback is a
hybrid per-organism recurrence cache plus canonical transient bursts, not silent
acceptance of the current phase average.

## Other finding: apparent mutation repetition

The deterministic mutation generator is balanced over large identity samples. For
the live seed, the first eight founder births happen to include four swim-frequency
mutations, two of which are `2.0 -> 1.9`. Across a deterministic fill from 8 to 64,
six of 56 births are `2.0 -> 1.9` and seven are `2.0 -> 2.1`. The repeated console
string is a seed-local early cluster plus the common founder baseline, not evidence
of a global negative-frequency bias.

## Repository state

- No approximation was integrated into production.
- The forced-CUDA live default was reverted after it measured slower.
- One small canonical optimization remains in the uncommitted candidate: a constant
  effort tensor is validated once per mechanics interval rather than once per 1/120
  substep. Public single-step validation remains intact.
- Focused lifecycle, energy, periodic-motion, and server verification: 103 passed.
- Complete CPU/default suite: 288 passed, five CUDA-skipped, and one unchanged strict
  F12 heading-controller xfail.
- Direct RTX 5070 CUDA suite: four passed and the matching unchanged F12 strict xfail.
- Ruff and `git diff --check`: clean.
- The larger uncommitted mutation, lineage, mortality, server, and lifecycle work was
  preserved.
