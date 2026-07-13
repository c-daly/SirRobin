# SirRobin — project disposition

SirRobin is a headless, GPU-vectorized ocean-life evolution simulation. Its scientific aim is a world where
form causes function, matter and energy have honest paths, and ecological/evolutionary outcomes are discovered
rather than scripted. The sea robin—fins that can eventually walk—is an emblem and research ambition, not a
target phenotype the code is allowed to grant.

## Disposition

- **Preserve causes, approximate machinery.** Use the cheapest declared model that retains the
  selection-relevant causal relationship. Full fidelity is justified where a cheaper model changes the question.
- **Close the books.** Tracked matter is transferred between explicit reservoirs. Energy has explicit sources,
  transfers, and dissipative outputs. Conservation and unit/sign identities are hard gates.
- **Form is function.** Capability comes from developed morphology and environment, never a parallel speed,
  agility, efficiency, fitness, or target-form vector.
- **Keep one authority.** Each live quantity has one authoritative state. One-way rebuildable caches, response
  models, snapshots, telemetry views, and render assets are allowed.
- **Build a vertical living loop.** Prefer the thinnest runnable path through movement, feeding, metabolism,
  reproduction, death, recycling, and observation over component perfection.
- **Let organisms be fallible.** Intent acts through bounded physical ability. Overshoot, bad foraging, failed
  reproduction, extinction, and bodies that cannot perform are legitimate phenotypes or findings.
- **Selection is implicit.** Survival and paid reproduction are the score. Never add a hidden carrying capacity,
  population target, target morphology, or success branch to make a run look healthy.
- **Headless is non-negotiable.** The complete world resets, runs, checkpoints, benchmarks, and evolves without
  Unity. A viewer is a detachable read client and never owns state or time.
- **Observe early.** A living run and owner-visible telemetry are evidence. Compile success and isolated kernel
  tests cannot establish that the world is useful.

## Working stance

- Read current code before trusting prose or historical verdicts.
- Ask which immediate consumer needs a model or test. Distinguish universal invariants, minimum mechanism
  capability, phenotype quality, research outcomes, stress diagnostics, and performance aspirations.
- Exploration is not preregistration. Freeze fixtures and thresholds only for a confirmatory claim that depends
  on them.
- Give a speculative mechanism one implementation and at most one bounded correction. If prospects are not
  improving, use the simpler fallback and return to the living loop.
- Keep changes small and reversible. Each tranche must leave a runnable headless world.
- Measure the complete world at its intended population and horizon; do not substitute isolated throughput.
- Name modules for durable domains, not project phases.
- Verify before claiming completion. Be candid about approximations, unresolved research, and failed outcomes.

## Current direction

Work on `recovery/living-loop` starts from `0cea74e`, before the later controller/actuator investigations. Those
investigations remain preserved on `main`. S2's articulated mechanics, body development, and work accounting are
reusable; its exact-heading-settlement failure is historical phenotype evidence, not a veto on the living loop.

The current inquiry is a physics-derived, per-genotype mobility response. The full articulated solver remains the
reference instrument; one uniform reduced-order integrator may serve the ecological hot loop if it preserves the
physical distinctions the biology uses. Failure of that inquiry falls back to measured full-physics/multi-rate
operation and does not block feeding, lifecycle, or observation.

Python/PyTorch is the current core because it already contains the exact economy and batched CUDA mechanics.
That is pragmatic, not a ban on C#. Game Prototype is primarily an executable C# donor and behavioral reference.
Its headless mechanics/development code is valuable; Unity scene ownership is not simulation authority, while a
detachable Unity observer remains useful.

## Routing

- `docs/2026-07-13-sirrobin-game-prototype-recovery-synthesis.md` — recovery position and evidence.
- `docs/superpowers/plans/2026-07-13-sirrobin-living-loop-recovery-implementation-plan.md` — active execution
  authority.
- `docs/2026-07-13-sirrobin-test-gate-policy.md` — test classification and proportionality.
- `docs/2026-07-12-sirrobin-developer-reference.md` — detailed live API and architecture reference.
- `docs/2026-07-11-sirrobin-design-document.md` and earlier S0/S1/S2 plans/reports — design intent and historical
  evidence; the recovery documents supersede conflicting roadmap and gate language.

## Environment and Git

- Use the repository's `uv` workflow. CUDA is the intended production device; CPU remains useful for small tests.
- Reproducibility means within seed on a fixed device unless a narrower claim says otherwise. Exact conservation
  does not imply byte-identical floating-point trajectories.
- Branch before committing from a default branch. Commit or push only when the user asks.
- Never add attribution trailers or generated-by credits to commits or pull requests.
