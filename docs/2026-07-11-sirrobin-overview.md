# SirRobin — What I'm Trying to Build

*A plain description of the project's intent at this stage (2026-07-11). A draft to refine, not a spec.*

> **2026-07-13 disposition.** This vision still stands, but current execution is governed by
> `docs/2026-07-13-sirrobin-game-prototype-recovery-synthesis.md` and the living-loop recovery plan. In this
> document, “one representation” means one authoritative live state, not a ban on rebuildable derived caches or
> response models. Full articulated physics is the causal reference rather than necessarily every organism's
> ecological substep. The complete simulation remains headless; Unity may observe it but never own it.

## In one breath

SirRobin is a small, faithful, continuously-simulated ocean world in which life evolves from
scratch — where the *shape* of a creature is the *reason* it can swim, feed, fight, and reproduce,
and where new kinds of creatures, new niches, and eventually a crossing from sea to land all *emerge*
from a handful of conserved physical laws rather than being scripted. It exists to be a world real
enough that, one day, an autonomous mind can be embodied in it and learn the way a body learns —
from consequences, not from text.

## Why it exists

The far horizon is embodiment. My cognitive core (Sophia) is meant to eventually inhabit a creature
in this world through a stable control-and-observation interface, feel real survival stakes, and
learn a grounded model of a world with genuine causes — the same interface later carrying over to a
physical robot. Text alone is an insufficient substrate for that kind of learning. So the near-term
work isn't the agent; it's building a world *faithful enough to be worth learning in* — and not
foreclosing that agent while I build it.

## The thing that matters most: fidelity

Fidelity is the product, not polish. This world has to be *causally* faithful — every effect
genuinely produced by its causes — even where it's mechanically abstract. "Abstract the world" never
meant a shallow world; it means abstracted in *mechanism* (simple fields, simple laws, a few sources)
but faithful in *causality*. You cannot ground learning in a world that fakes its own causes.

Two commitments enforce this:

- **Conservation.** Nothing mints or destroys matter or energy. Every channel — production, feeding,
  predation, death, burial — moves conserved stuff between tracked reservoirs. The books close,
  always. Collapse is a diagnostic that fidelity was dropped upstream, never a knob to soften.
- **Form is function.** What a creature can do is *derived* from its body run through real physics —
  swimming from hydrodynamics, metabolism from size, feeding from mouth geometry — never read off an
  arbitrary stat sheet.

## The world: one substrate, expressing everything

The environment is a thin stage whose whole job is to generate the pressures evolution needs — and
it's built field-first, from a single structural substrate that everything else is a *product* of.
One geological field expresses the terrain's shape; the terrain and field decide where vents,
volcanoes, and slowly-drifting hotspots sit; the geology sets where minerals and elements (like the
iron that decides where the ocean can even bloom) are found; and, later, land, rivers, and the
sediment they carry down to the coast. Nothing is placed by hand — it all falls out of the field.

Over the top runs an atmosphere: sunlight and rotation drive winds, winds drive currents, currents
carry nutrients and heat and larvae — making blooms and deserts, gyres and fronts, and the dispersal
patterns that connect or isolate populations. Clouds shade; storms mix and disturb. It all
interlocks: currents bend around the seafloor, upwell along ridges, and carry the vent plumes.

The world is continuous — not chopped into cells you can feel — and every live quantity has exactly
one authority, never two mutable copies kept in sync. Rebuildable derived caches and observer views are
allowed. It starts *small, dense, and all-ocean* — a
wrap-around patch of sea with no edges — because density, not size, is what makes life interact
richly, and a small world runs fast enough to actually tune. It grows only as the life inside it
asks for more.

## The life: discrete creatures, evolving genuinely

The creatures are discrete individuals carrying an open-ended genome that is a *recipe*, not a
blueprint: a body-graph that *develops* into a body, so evolution can grow new structure, reuse it
(repeated segments, serial limbs), co-opt it (a fin becoming a limb), and — with the right lineage
bookkeeping — recombine and drift apart into genuinely separate species. Behavior starts simple and
reactive; the sophistication is spent on the body and its evolution, which is the part I most care
about getting deep.

Selection is only ever survival: energy earned by a faithful body in a faithful world, spent on
staying alive and reproducing. From that, and from the world's shifting gradients, the hope is real
emergence — adaptive radiation, coevolutionary arms races, and an unscripted crossing from sea to
land where a swimmer becomes a walker. The vehicle is built to *reach* that endpoint; whether it
*arrives* is the hard, honest, long-horizon question.

## How it's built, so it doesn't become soup

The last attempt drowned in complexity, because ideas kept spawning hidden sub-projects that got
bulldozed in place. This one is held together by a few hard rules:

- **One authoritative state per thing** — no duplicated mutable truth or glue keeping copies agreed;
  rebuildable derived responses, snapshots, and views are fine.
- **Clean abstraction boundaries** — each subsystem is asked for *what it provides*, never *how it
  works*, so any piece can start trivial and grow arbitrarily faithful later without touching the rest.
- **Build a vertical living loop** — close the books across movement, feeding, lifecycle, recycling, and
  observation, then deepen the bottleneck that the running world exposes.
- **Start simple, grow on demand** — the world's richness tracks what the biology actually needs and
  never runs ahead of it.
- **One vectorized substrate** (GPU-capable) with the full body physics as its causal reference. A cheaper
  per-genotype response may run the whole ecological population when it preserves the physical differences
  selection actually uses.

## Where it goes

The current recovery joins the existing conserved nutrient cycle and canonical physical body into the thinnest
headless living loop: imperfect movement → local feeding → maintenance → paid reproduction → death → recycling,
with replay/observation alongside it. Once that loop is visible and measured at useful population scale, add
variation, then let its actual bottlenecks justify predation, mating/speciation, richer transport, embodiment,
and, last and hardest, land and the fish that grew legs.

That's what I'm trying to do: a small ocean with honest causes, life that earns its own complexity,
and a world faithful enough to eventually learn in.
