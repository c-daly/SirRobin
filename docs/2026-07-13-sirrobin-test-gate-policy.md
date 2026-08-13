# SirRobin test-gate policy

**Date:** 2026-07-13

**Status:** active project policy

## Position

SirRobin simulates fallible creatures with limited physical abilities. Correctness means that causes are honest,
books close, state remains valid, and declared safety boundaries are respected. It does not mean that every
creature performs every behavior well or that every scientific experiment produces a desirable outcome.

Test burden must be proportional to the claim and its next consumer. A review recommendation is evidence to
consider, not automatically a new project requirement. Before adding a hard gate, name the universal failure it
prevents or the immediate capability that cannot proceed without it.

## Gate classes

| Class | Examples | Disposition |
|---|---|---|
| **Invariant** | exact int64 closure, debit/credit partition, unit/sign identities, no second live authority | Universal hard gate. |
| **Validity and safety** | finite state, declared bounds, no hidden repair, no direct physical-state write | Hard over the declared operating domain. |
| **Minimum mechanism capability** | a capable actuator causes signed motion; a seeded grazer feeds; paid birth and death transfer work | Demonstrate the smallest end-to-end capability the next slice needs, normally with one deliberately viable reference case. |
| **Phenotype quality** | overshoot, turn time, cost, feeding efficiency, lifespan, persistence | Telemetry and selection input. Hard only when a named consumer requires a minimum. |
| **Research outcome** | unseeded predator emergence, speciation, sea-robin evolution, land crossing | Falsifiable experiment; never a generic engineering blocker. |
| **Stress diagnostic** | long soak, adversarial morphology, phase grid, emergency-start corpus | Investigates limits. Hard only when the full stressed domain is part of the actual claim. |
| **Performance** | complete-world throughput, memory, usable experiment time | Hard at a demonstrated consumer need; aspirations and isolated-kernel multiples are reported separately. |
| **Provenance** | frozen external oracle, source hash, literal fixture | Required when a confirmatory comparison depends on identity of the source; not a ceremony for ordinary exploratory work. |

## Fallible-creature rule

An action is an intent applied through physical structure, never a state assignment. A requested heading or
yaw-rate may change gait or appendage effort; it may not snap yaw, angular momentum, velocity, or position.
Inertia, drag, gait phase, morphology, and actuator limits determine the response.

Consequently:

- causal signed response can establish that a steering mechanism exists;
- exact heading settlement, absence of overshoot, fast response, or universal maneuverability are not locomotion
  correctness requirements;
- navigation quality is measured by ecological progress and cost, not servo precision;
- actuator absence produces no intervention and is a legitimate morphology; and
- crossing a real safety boundary, fabricating state, or bypassing physics remains a hard failure.

The same distinction applies elsewhere: a feeding mechanism must not overdraft stock, but a creature may forage
poorly; reproduction must be paid, but a population need not persist; medium physics must permit crossing, but
evolution need not discover it on demand.

## Exploration, confirmation, and corrected mistakes

Exploration is allowed to look at results, change the question, and discover a useful operating domain. It should
record what was tried and stop when additional effort is no longer increasing the chance of answering the
project's next question. A speculative mechanism receives one implementation and at most one bounded correction
before the project uses its fallback or returns to the living loop.

Confirmation is different. When the project intends to make a narrow comparative or authorization claim whose
threshold could be gamed after seeing results, freeze the relevant corpus, metric, and threshold first. Bind an
external fixture or source only when its independence or identity is part of that claim.

If an existing gate is discovered to test the wrong claim:

1. preserve its result as historical evidence;
2. state the category error and the actually needed claim;
3. retain applicable invariants and safety checks;
4. demote phenotype, stress, or research measures to telemetry when that is what they are; and
5. continue under the corrected current policy without manufacturing a ceremonial successor programme.

Historical evidence is not edited into a pass. It also does not retain veto power over a different question.

## Minimum-capability cases

A purpose-built reference case is legitimate when the question is whether an engineered mechanism exists. It may
be deliberately viable, articulated, fed, or positioned to expose the path. Negative controls should rule out
obvious bypasses, such as movement with no actuator or birth without a material debit.

Add holdouts, random corpora, adversarial cases, or statistical thresholds only when the next consumer needs a
generalization or robustness claim. “A larger corpus would be more rigorous” is not by itself sufficient.

## Conservation and numerical evidence

Exact tracked-matter closure is checked on every complete step that is run because it is cheap, universal, and
protects the project's central scientific boundary. Exact closure does not imply exact continuous trajectories,
universal float64 execution, byte-identical replay across devices, or a mandatory million-step soak.

Timestep/refinement comparisons are required when a conclusion may plausibly be a discretization artifact. They
need not be repeated as a universal matrix after a stable operating domain is established. Numerical emergency
bounds protect against invalid state; they are not quality targets creatures must approach or avoid elegantly.

## Performance and scale

Measure the complete world at the population and simulated horizon needed for actual experiments. Small tests
remain valid for local correctness. For the living-loop recovery, 5,000 creatures is the routine scale target and
10,000 is the stretch target on the user's CUDA device.

Report all repetitions, memory, simulated-time advance, and subsystem attribution. Do not let an isolated kernel,
an arbitrary real-time multiple, or one noisy timing sample replace the whole-world result. A 10,000-creature
miss does not invalidate a useful 5,000-creature world; it identifies the next measured scaling question.

## Review checklist

For every proposed hard test, answer:

1. What class is it?
2. What concrete false implementation would pass without it?
3. Which immediate consumer needs this domain, duration, corpus, or threshold?
4. Is there a cheaper detector of the same failure?
5. Would a legitimate fallible phenotype or scientific result fail it?
6. If it fails twice, what fallback lets the living loop continue?

If those answers are missing, the default is telemetry or an exploratory diagnostic, not a project blocker.
