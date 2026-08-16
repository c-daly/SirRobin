# Bare causal living-loop design

**Date:** 2026-08-15

**Status:** Implemented and bounded mechanism proof passed; see
`../reports/2026-08-16-bare-causal-living-loop-proof.md`

**State examined:** `recovery/living-loop` at `59161e6`, including the preserved
uncommitted live-locality work

## Decision

Build the smallest headless ocean world that closes this loop:

```text
local producer food
    -> local food state
    -> bounded intent
    -> developed actuators and aquatic physics
    -> physical movement and paid work
    -> local capture and depletion
    -> reserve, maintenance, and construction
    -> paid asexual birth with mutation
    -> independently released developed offspring
    -> death and material return
```

This is a concrete animal loop, not a framework for imagined future biology. It
stays flexible through explicit state and causal boundaries, not registries,
plugins, generic nervous systems, or empty abstractions.

The current runtime already contains local finite-stock feeding, developed-body
hydrodynamics, metabolism, paid birth, mutation and redevelopment, death returns,
lineage, headless execution, and a detachable Unity viewer. Two defects prevent it
from being the target instance:

1. behavior is divided into seeking/searching/cruising modes and golden-angle
   search legs; the current scientific profile disables food-gradient input; and
2. newborns begin exactly at their parent's position.

## Food state

For the founding population, producer biomass is food. Creatures are born able to
recognize it. We deliberately do not yet model whether the underlying sense is
smell, sight, taste, mechanoreception, or some combination.

Each live organism receives only:

- producer abundance at its physical position; and
- the local horizontal producer gradient in body-relative coordinates.

Both come from the actual depletable finite-volume reservoir through its existing
local interpolation stencil. There is no nearest-food target, world maximum,
remote-patch query, or global food pool. A uniform field provides no direction;
consumption changes the same field that later produces the organism's food state.

This generic food state is an owner-approved approximation of unresolved vertebrate
sensing. If accepted, it supersedes the uncommitted scientific audit's near-term
recommendation to disable gradient input until receptor machinery exists; the
audit's conservation and calibration warnings remain evidence.

The current developed `sense` marker does not gate this first population's food
state and gains no new scientific meaning. It may remain dormant until a concrete
future sensory mechanism justifies removing or replacing it.

Perception never transfers matter. Feeding still requires physical position,
relative motion, developed intake geometry, available local stock, and the exact
shared-stock transaction.

## Intent and physical movement

Remove the named behavior modes and fabricated search headings. The first
population shares one simple inherited response:

- maintain its ordinary autonomous locomotor drive; and
- when a nonzero local gradient exists, request a turn toward recognized food.

With no gradient, it continues its current physical course. It receives no search
direction and no behavior branch intended to create dispersion, persistence, or
attractive paths.

Food state may change only controller state and requested effort. It may never
assign position, velocity, yaw, angular momentum, or a successful turn. Developed
joints, actuator budget, hydrodynamics, drag, inertia, and integration remain the
only path to displacement. A malformed or underpowered body may fail to move or
turn.

The response is shared unchanged by founders and offspring. A child samples its
own local food state and begins requesting its own locomotion on its first active
step; it never follows, remains attached to, or receives steering from its parent.
Controller evolution is deferred; body mutation is sufficient for the first
genetically diverging lineages. Keep the response isolated from field and physics
authority so a later genotype-owned controller can replace it without changing
either domain.

## Reproduction, mutation, and release

Preserve the existing transaction order:

1. mutate the parent's authoritative genotype;
2. develop and structurally price the candidate child;
3. accept birth only when material, slot, and identity are available; and
4. commit the candidate genotype and body only for an accepted birth.

Do not reject unfamiliar, inefficient, immobile, or doomed forms. Only invalid
state, invalid geometry, capacity, or an unpayable transaction blocks birth.

Birth remains a coarse complete-offspring approximation, but it may not create two
material bodies in the same occupied space. An accepted child is placed immediately
outside the parent's developed support extent along the release axis, with only
numerical clearance between them. Horizontal wrapping and vertical bounds still
apply.

For this milestone the paid child is a fully functional mini-adult: its complete
mutated body, initial reserve, food state, controller, feeding, locomotion, and
metabolism are active without a juvenile stage. This deliberately makes prenatal
construction the whole developmental payment and gives the first world no built-in
reason for parental care. It does not freeze that life history into the architecture:
candidate development and the birth transaction remain separate causal boundaries,
so later work may make the amount constructed before release heritable without
changing field, physics, mutation, or lineage authority.

Release is an instantaneous physical event, not a period of parental association.
The child receives clean lifecycle/controller state and its own linear and angular
momentum. It does not inherit or continuously copy the parent's velocity, heading,
controller state, or target. A bounded outward release impulse separates the pair;
the parent receives the equal-and-opposite impulse. Any positive change in pair
kinetic energy is debited by the birth transaction and recorded at the existing
energy boundary. If the transaction cannot pay it, birth waits. The impulse's
magnitude is an explicit birth-approximation parameter until reproductive release
form is represented, not a hidden dispersion or anti-clumping control.

After that instantaneous release, neither body receives positional assistance or
special interaction. The offspring senses food and participates in controller,
actuator, hydrodynamic, and metabolic updates as an independent organism beginning
with its first complete active step.

This slice adds no population repulsion or anti-clumping force. General body-body
contact waits unless validly released bodies demonstrably interpenetrate enough to
block interpretation of the run.

## Environment and time

The first world may remain an explicitly idealized, horizontally periodic,
still-water ocean using the current producer, dissolved, detrital, and microbial
reservoirs. Existing growth, vertical transport, feeding, and returns operate on
those same reservoirs.

Do not add horizontal transport to manufacture dispersion. Currents, advection, or
diffusion become a separate slice when their physical driver and approximation are
stated. Persistent depletion is an acknowledged consequence of this first
still-water world.

One physical simulation clock remains authoritative. Playback rate changes only
how quickly the driver advances and publishes simulated time. Multirate scheduling
waits for a measured stability or complete-world performance need.

## Acceptance evidence

One deliberately viable reference world must run headlessly and then be observed
through a fresh Unity Play Mode connection. It must show:

- paid physics-derived motion, with an actuator-absent negative control;
- local food state changing requested turning without writing motion state;
- exact local producer debit and later depletion-derived food state;
- one fully paid asexual birth with an accepted mutation;
- a child developed from its committed mutated genotype;
- nonoverlapping parent/child initialization, conserved release impulse, paid
  release energy, and immediate independent sensing and motion;
- lineage identity across founder and child generations;
- exact tracked-matter closure and existing named energy boundaries; and
- finite state, bounded telemetry, device, seed, duration, and measured throughput.

Tests protect conservation, authority, locality, paid birth, mutation commitment,
birth initialization, and viewer noninterference. Path quality, dispersal,
population persistence, reproductive success rate, and mutation benefit remain
telemetry or scientific outcomes, never pass conditions.

The accepted claim is a runnable causal mechanism, not biological calibration or
a stable ecosystem. Current material/body scaling and idealized environmental
inputs remain explicit unresolved limitations. CUDA is the production target; CPU
reference tests and a CUDA skip do not establish production runtime behavior.

## Non-goals and change boundary

This design does not add sensory organs, sensory evolution, learning, carnivory,
sexual reproduction, symbiosis, photosynthetic animals, empirical oceanography,
general collision, a phenomenon framework, or a long-horizon evolution gate. It
must not introduce a closed enum of senses, foods, forms, or environmental
phenomena.

Implementation remains three independently runnable corrections:

1. local food state to bounded intent;
2. geometry-derived newborn release; and
3. multi-generation headless and Unity evidence.

Correct only a demonstrated blocker after that run. Preserve the existing dirty
work until it is explicitly retained or superseded; this design authorizes no
reset, cleanup, or broad rewrite.

Acceptance of this design confirms that generic local food state is sufficient for
the first sensory approximation, the controller may remain fixed for this
milestone, coarse geometry-derived release precedes general contact, and horizontal
resource transport is not required to manufacture a successful-looking result.
