# SirRobin — Genome / Genotype→Phenotype Encoding Design (grounded)

**Date:** 2026-07-11
**Status:** Design draft for user review. The central creative bet of the project ("something that
expresses itself like genes"). Grounded against the literature (Sims 1994; Stanley & Miikkulainen
NEAT 2002; Cheney/Kriegman soft robots & xenobots; de Aguiar 2009; Dieckmann & Doebeli 1999;
Servedio 2011; TensorNEAT/EvoX 2024) and against the donor source `…/game prototype/…/Life/BodyGraph.cs`.
**Substrate context:** the sim runs GPU-vectorized (torch, many worlds batched) from day one.

---

## 1. Recommendation (the sweet spot)

**Encode each creature as a bounded recursive directed part-GRAPH (Karl Sims) carrying NEAT
historical-marking "innovation numbers" on every node and edge, developed by a fixed-depth,
level-synchronous transform *scan* into a fixed-capacity, masked segment tensor (the "DevelopedBody")
that feeds the existing Lighthill/Lamb `SwimEval` physics unchanged.** A CPPN parameter-modulator
(HyperNEAT-of-a-tree) is a *deferred, optional* later layer for regularity/gradients and plants.

It is the **minimal in-place evolution of your BodyGraph, not a rewrite** — a part-*tree* becomes a
part-*graph*, and the one load-bearing addition is the innovation-number bookkeeping Sims himself
lacked. Each hard requirement is met by a well-matched piece:

- **Expressive / evolvable** — from the Sims graph: nodes *are* ellipsoid segments; add-node/add-edge
  is genuine complexification; a **self-edge with a recursion count** gives serial repetition and
  module reuse (repeated segments, serial limbs) for free — this is the tensor form of your existing
  `duplicate-diverge`.
- **Vectorizable (GPU-friendly)** — all *growth* stays in the between-generation **mutation** operator,
  so **development is a pure fixed-shape function** of padded genome tensors, unrolled in a fixed
  number of batched gather/compose/scatter passes (the same kernel shape as the S0 pose depth-scan).
  Your donor already enforces this: `Measure()`/`Develop()` never change part count — only `Mutate()`
  does. Preserve that invariant; it is the make-or-break for batching.
- **Speciation-capable** — the NEAT innovation numbers are the *one* addition that turns a growing
  graph into an **alignable** genome, which unlocks aligned crossover, a compatibility-distance
  metric, assortative mate choice, and **emergent reproductive isolation** — all four structurally
  impossible for the prior asexual clone-and-mutate + cosine-cluster overlay.

**Why not the alternatives:** pure **CPPN/HyperNEAT** batches best but is the wrong *backbone* — it
emits a *field* over coordinates, not discrete ellipsoid segments, so its readout is variable-count
(re-introduces dynamic shapes) and its genetic distance lives in weight-space, which doesn't map to
morphology (awkward speciation). **L-system/GRN/NCA** are rejected as backbone: variable,
data-dependent iteration counts are the hardest thing to batch statically and the biggest determinism
hazard, and their crossover/distance are far less principled than NEAT alignment. The **current
direct tree** is rejected because it has no reuse and — decisively — **no gene alignment, so it can
never speciate.**

**Honesty — solved vs frontier:** the backbone (Sims graph, NEAT markings, compatibility distance,
aligned crossover, batched padded-tensor development à la TensorNEAT/EvoX) is **solved, well-cited
technique.** The emergent-speciation *outcome* and any claim of *open-endedness* are **research
frontiers**, gated primarily by the **ecology** (see §5, §8).

## 2. Genotype representation (tensors)

Two padded, fixed-capacity tensors per population of size P — static shape, boolean masks
(heterogeneous population = same shape, different masks/values):

- **Node table** `N:[P, N_max, F_node]` + `node_mask[P, N_max]`. `F_node` = { type one-hot
  (Segment / Surface / extensible plant-organ slots); base semi-axes as **log(a), log(b), log(c)**
  (log-scale → ratio-symmetric mutation, matching the donor's `SizeLogScale` that killed the additive
  mass ratchet); density; port flags (Intake/Sense); joint genes jFreq/jPhase/jAmp + evolvable hinge
  axis; an **"expressed" toggle** for neutral drift; a global **node innovation id** }.
- **Edge table** `E:[P, E_max, F_edge]` + `edge_mask[P, E_max]`. `F_edge` = { src node idx, dst node
  idx, attach position (3), orientation (**quaternion**, renormalized each compose — avoids gimbal in
  the batched matmul), scale factor, **reflection/mirror flag**, **terminal recursion count r**, a
  global **edge innovation id** }.
- The old `PartGene` children become **edges**; a **self-edge (src==dst) with r>1** is the Sims
  repeated-segment/limb.
- Body-level genes (swimFreq, swimWave, sessility, tropism, hungerGain) ride as a per-genome vector.
- **Pre-expand each mirrored part into two side=±1 slots at encode time** so bilateral pairs never
  change part count during development.
- Choose `N_max/E_max/S_max` from the *realistic* ceiling (donor mean ~6 segments; data cap 28;
  physics cap 16), not the theoretical max, to keep pad-to-max overhead acceptable.

## 3. Development step (batched, deterministic)

A **bounded instance-frontier unroll** — genotype → DevelopedBody as a pure, fixed-shape function.
Maintain a frontier of pending traversal *instances*, each carrying (current node idx, accumulated
4×4 world transform, per-edge remaining-recursion counters, bilateral side sign). Iterate a **fixed**
number of layers L (= max developmental depth, e.g. 6, mirroring the donor's depth caps). Each layer,
batched over `[P, W_frontier]`: (1) gather node params; (2) deterministically **scatter-write** one
ellipsoid segment into `DevelopedBody[P, S_max, F_seg]` at a precomputed slot index; (3) for each
outgoing edge with recursion counter > 0, compute the child world transform via batched 4×4 matmul
(applying a reflection matrix on mirror/side flip), decrement the counter, enqueue the child —
**masked off** when frontier or segment budget is exhausted (the analogue of the donor's `parts>=cap`
early-out). Fixed L and fixed frontier width make `S_max` static; no data-dependent loop bounds. This
is the *same* gather/compose/scatter kernel as the S0 pose depth-scan.

`F_seg` = { center xyz, orientation quat, semi-axes a/b/c, mass, is_surface/fin flag, fin area +
normal for Lighthill lift, port flags, per-segment gait phase = −depth·swimWave } + validity mask.
**SwimEval reads masked segments only (multiply dead rows by zero — never boolean index-select, which
changes shape).**

## 4. Mutation operators (heredity)

Parametric jitter (locality: masked Gaussian/Laplace on param slices, log-scale size drift);
**add-node** (split an edge, mint a node innovation id — complexification); **add-edge** (connect two
nodes; a self-edge with r>1 = serial repetition — complexification + reuse); increment/decrement
recursion count r (tune serial-segment count cheaply); toggle mirror/reflection (bilateral symmetry);
**flip Segment↔Surface** (the exaptation lever: drag fin → thrust fin); **enable/disable the
"expressed" toggle** (neutral drift → re-express in a new context = exaptation); toggle Intake port
(grow/lose a mouth, with `EnsureMouth` repair); prune (metabolic pressure against bloat).
**Discipline:** every structural op mints monotone innovation ids; every new heritable-but-inert gene
**appends** to the RNG draw manifest, short-circuited to zero draws while inert, preserving
determinism.

## 5. Crossover, distance, and emergent speciation (the payoff)

- **Crossover (new — the donor has none):** align two genomes **by innovation id**, separately for
  node and edge tables (gather/sort by id; batchable). Matching genes → pick/blend params; disjoint +
  excess → inherit from the fitter/closer parent. Because edges reference nodes by *stable global id*,
  a **graph** (not just a tree) recombines coherently; a deterministic validity-repair pass drops
  dangling edges and re-runs `EnsureMouth` + caps. Innovation-number bookkeeping is **non-optional**:
  naive positional alignment hits the **competing-conventions / permutation problem** and yields
  non-viable monsters — the exact reason the prior build never added crossover.
- **Genetic distance (new):** NEAT compatibility distance `δ = c1·E/N + c2·D/N + c3·W̄` over
  innovation-aligned genes (excess/disjoint/mean-param-diff), a masked reduction, batchable pairwise
  within spatial buckets. **Keep the donor's phenotypic Morpho/GenomeVec cosine as a *separate*
  observational-taxonomy axis** — genotypic δ drives *mating*; phenotypic cosine drives *naming*.
- **Emergent reproductive isolation (not imposed):** assortative mating on genetic distance within a
  spatially structured population —
  `P(mate | a,b) = spatial_neighbour(a,b,radius) · sigmoid((mateThreshold − δ(a,b))/w)`, a masked
  pairwise op within the existing SpatialHash buckets. Under **disruptive/ecological selection**,
  local drift + assortment let a panmictic cloud split into non-interbreeding clusters with **no
  species label ever assigned** (de Aguiar 2009 topopatric; Dieckmann & Doebeli 1999 sympatric).
  Optionally couple the mating cue to a heritable trait already under ecological selection (a **"magic
  trait"**, Servedio 2011) for the fastest sympatric split. "Species" are **read out** for analytics
  as connected components of the who-can-breed-with-whom graph — never imposed. **Deliberately avoid
  NEAT's own fitness-sharing speciation** (it restricts mating within species *by fiat* — re-imposing
  the label meant to emerge).
- **Honesty:** the mechanism is established in population-genetics *models*, but whether it splits in
  *this* substrate is a **research frontier gated by the ecology** — too-strong assortment fragments
  into non-viable singletons, too-weak stays panmictic. The Phase-2 milestone must *prove* a split.

## 6. Evolvability scorecard

Regularity/modularity: **medium** natively (self-edge repetition + duplicate-diverge + mirror),
**high** once the Phase-3 CPPN paints along-body gradients. Locality/smoothness: **high** (jitter on
one node changes one region — the graph's advantage over pure-CPPN pleiotropy). Complexification:
**high & genuine** (NEAT-style topology growth with homology preserved). Exaptation: **high**
(Segment↔Surface flip; express/silence toggle). Neutrality/robustness: **medium-high** (expressed bit
+ inert appended genes = neutral networks). Pleiotropy/epistasis: **low-medium** for the backbone
(rising where the CPPN is added — the regularity-vs-locality trade to tune). Open-endedness:
**unsolved** — expect to need novelty search / MAP-Elites over a behavior descriptor, and even then
it's a frontier.

## 7. Vectorization plan (GPU-first)

Development is a deterministic pure function of the padded genome tensors; **all growth stays
host-side in `Mutate()` between generations** (the make-or-break invariant). Fixed L layers of
`index_select` gather → batched 4×4/quaternion compose → **deterministic scatter-write** into
`DevelopedBody`; zero dead rows everywhere; **multiply-by-mask, never boolean index-select** (keeps
shapes static). GPU determinism: **avoid nondeterministic `scatter_add_`/`index_add_` atomics — use
precomputed unique slot indices for a plain scatter**; `use_deterministic_algorithms(True)`, fixed
dtype/device/op-order. Precedent this tensorizes: **TensorNEAT/EvoX, PyTorch-NEAT** evaluate
variable-topology genomes as padded node/edge tensors with masked fixed-layer scans. The later CPPN
modulator batches trivially (fixed coordinate queries broadcast over P genomes, branchless
activation).

## 8. Migration from BodyGraph

**Keep** (verified in `BodyGraph.cs`): the DFS develop-walk structure and its exact traversal order
(the float-sum-order golden invariant); the mirror/side-flip bilateral mechanism; the fixed-capacity
dual-cap discipline (data 28/6, physics 16/5 → `N_max/E_max/S_max/L` masks; degrade by masking, not
resize); the duplicate-diverge operator (→ recursive self-edge); the Segment/Surface vocabulary and
Intake/Sense ports; swimFreq/swimWave and per-segment gait phase; `EnsureMouth`; `MutationNoise`
(Laplace + SizeLogScale); the append-only RNG manifest; the AeroSurface readout contract and Morpho
aggregates.
**Change:** tree → directed graph (edge table + recursion count r); boxes → **ellipsoid semi-axes
a/b/c** (reinterpret the same three size numbers — but *verify the Lamb added-mass terms don't need
re-tuning*, see §10); **add global innovation ids** to every node and edge (the load-bearing change);
**add crossover + compatibility distance + threshold assortative mating** (asexual → sexual — the
speciation enabler); port the recursive C# walk to the batched torch unroll; make the joint genes
actually *read* by SwimEval (currently selectively dead — pointless to migrate until the actuation
proxy reads them).
**Migration acceptance test:** the torch batched unroll must reproduce the C# `Measure()` aggregates
(mass/area/intake/bulk/avgDensity + shape descriptors) within tolerance on a representative genome
sample **before any new capability is layered on.**

## 9. Phasing

- **Phase 0 (refactor, ZERO behavior change):** port the C# `Measure()`/`Develop()` walk to a batched
  fixed-iteration torch unroll producing the DevelopedBody tensor; acceptance = hard-diff against the
  C# aggregates. **This is the first genome slice to build**, and it dovetails with S0/SpikeSwim
  (which needs developed bodies to swim).
- **Phase 1 (topology upgrade, still asexual):** tree → node+edge tables; children → edges; add
  recursion count r; attach innovation ids; reinterpret box sizes as ellipsoid a/b/c. Genome is now
  *alignable* though nothing recombines yet.
- **Phase 2 (speciation — the payoff):** NEAT compatibility distance + innovation-aligned crossover +
  spatial assortative mating. **Falsifiable milestone:** a single deterministic run in which a
  population splits into two non-interbreeding clusters under disruptive selection — the proof the
  prior build could never produce. Do **not** proceed until demonstrated or its absence root-caused
  to the ecology.
- **Phase 3 (generative modifier, OPTIONAL / plants):** HyperNEAT-lite CPPN indexed by developmental
  coordinates (depth, serial index, bilateral side) → graded parameter offsets. Required for the
  plant kingdom (S9) and the sea-to-land body-plan shift; deferrable for animals if recursion-indexed
  deltas already give enough along-body gradient.
- **Phase 4 (open-endedness pressure, research):** novelty search / MAP-Elites over a behavior
  descriptor, *if* fitness-only search plateaus. A frontier, not a switch.

## 10. Risks & open decisions

**Risks:** competing-conventions without innovation ids (→ ids mandatory + validity-repair);
variable-length development breaking batching (→ all growth in `Mutate()`, fixed-shape development);
nondeterministic CUDA scatter (→ deterministic mode, precomputed unique slots, no atomics); **emergent
speciation may not occur** — the binding constraint is *ecological niche diversity*, not encoding
power (→ magic-trait coupling, gate on the Phase-2 test); **an expressive genome may manufacture
morphospace the ecology can't reward** — complexity without emergence (→ Kleiber + prune pressure; new
structure must earn fitness before more DOF); ellipsoid reinterpretation may shift added-mass terms
(→ validate vs box baseline, re-tune Lamb coefficients if needed).

**Open decisions (for review):**
1. **Determinism contract** — bitwise-identical across GPU/CPU (forbids atomics; may push the physics
   eval out of the byte-golden tier) vs **reproducible-within-seed on a fixed device** + conservation
   tolerance gate. *Given GPU-from-day-one, recommend the latter* (within-seed on fixed device;
   conservation-invariant is the primary gate; avoid atomics via precomputed unique slots so within-
   seed determinism holds even on GPU; bitwise goldens only as a same-device-CPU regression check).
2. Tree-first vs full-graph in Phase 1 (reuse gain vs batching cost).
3. Ellipsoid vs box readout — does a/b/c require re-tuning SwimEval's Lamb terms, and at what tolerance?
4. Compatibility-distance coefficients & threshold: fixed vs auto-tuned to a target species count.
5. Mating cue: pure genetic-distance vs an evolvable preference gene (Fisherian runaway) vs a magic trait.
6. How much CPPN for *animals* vs defer entirely to plants/sea-to-land.
7. Haploid vs diploid (dominance/recessive neutrality vs doubled genome complexity).
8. Do plants (S9) share the innovation/crossover machinery or keep a simpler mean+variance genome.
9. The precise Phase-2 acceptance test: disruptive-selection setup, population size, spatial radius,
   run length that constitute a valid two-species split.
