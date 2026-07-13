# SirRobin — project instructions

SirRobin is a small, faithful, GPU-vectorized **ocean-life evolution simulation**: a world where a
creature's *form* is the *reason* it can swim, feed, fight, and reproduce, and where niches,
speciation, and an eventual **bidirectional sea↔land crossing** (the sea robin — fins that walk —
is the emblem) *emerge* from conserved physical laws rather than being scripted. Its long-horizon
purpose is a grounding substrate for an embodied agent (Sophia), deferred until the world's books close.

**Read these first (authoritative context):**
- `docs/2026-07-11-sirrobin-design-document.md` — the master technical design.
- `docs/2026-07-11-sirrobin-overview.md` — the short vision.
- `docs/superpowers/specs/2026-07-11-sirrobin-restart-architecture-design.md` and `…-genome-encoding-design.md`.
- `docs/superpowers/plans/2026-07-12-sirrobin-S2-canonical-body-live-locomotion-implementation-plan.md`
  — the S2 execution authority; its 2026-07-12 run records NO-GO on controller falsifier F12.
- The file-based memory (`…/.claude/projects/…/memory/`, auto-loaded via `MEMORY.md`) — the running decision record.

**Current state:** **S0 / "SpikeSwim" is implemented and verified.** The original 1,000-creature/90M Gate E
remains a recorded NO-GO; the subsequently pre-registered population-grounded gate records GO at 5,000 and
10,000 creatures. **S1 is implemented and records GO** for the exact four-reservoir nutrient cycle. **S2 is
implemented but records NO-GO**: mechanics and CUDA throughput pass, while the frozen desired-heading controller
fails F12 (home-and-settle). Do not begin S3 until a successor controller authority closes Gate C.

## Environment (this documents decisions that diverge from the global `~/.claude/CLAUDE.md`)

The sim **Core is Python / PyTorch** — headless, GPU-vectorized, cross-platform, deterministic. It is
**not** a Unity project. The global config assumes Unity/Coplay for this machine; that applies here
**only** to a *possible later, optional render viewer* over the Core's state contract — never to where
the simulation lives.

- **Determinism:** reproducible-within-seed on a fixed device. **Conservation invariants are the primary
  test gate — NOT byte-identity.** (Byte-identity was the prior build's wrong invariant; it let fidelity
  ship disabled behind `gain=0` dials.) Conserved reservoir state is exact int64 fixed-point quanta; float64
  is used for independent rate/oracle work and float32 for hot mechanical state. There is no float mirror of a
  conserved reservoir.
- **Coordinates:** public world state is ENU (`x` east, `y` north, `z` up; water `z<0`) and new body contracts
  are FLU (`x` forward, `y` left, `z` up). S0 remains a frozen donor-native XZ/Y-up validation experiment;
  conversions belong in one tested boundary adapter, never inline swaps.
- **PowerShell is 5.1:** when reading/writing files other tools will read, handle UTF-8 explicitly
  (`[System.IO.File]::ReadAllText/WriteAllText` with a UTF-8 encoding) — the default codepage corrupts
  non-ASCII (em-dashes, arrows) into mojibake.
- **The prior build (`C:\Users\cddal\game prototype`) is a donor, not a template.** Salvage its proven
  code (`SwimEval`, `BodyGraph`, field grids, `SimUnits`, taxonomy) and its hard-won science/lessons —
  but **not** its architecture. Superseded there: C#/Unity, the data-far/physics-near LOD proxy, the dual
  `eff[]` genome, static-Perlin non-conserving fields, byte-identity-as-gate.

## Non-negotiable laws (why the last build died for lack of them)

Hold these like invariants:

1. **Fidelity is the product.** Abstract in *mechanism*, faithful in *causality*. Signatures must be
   *derived* from conserved, relational constraints — never imposed by a knob.
2. **Conservation — the books close.** Nothing mints or destroys mass/energy; everything moves between
   tracked reservoirs. This is the top CI gate.
3. **Form is function.** Capability is derived from morphology through real physics — never a stat vector.
4. **Single canonical representation.** One representation per quantity. If you are writing code to keep
   two copies of one thing in sync, you have already lost.
5. **Clean abstraction boundaries.** Query *what* a subsystem provides, never *how*. This is what lets any
   piece start trivial and grow arbitrarily faithful later without touching its consumers.
6. **Continuous, not a discrete grid that leaks into biology.** Continuous positions, interpolated field
   sampling, continuous encounters.
7. **Start simple, grow on demand.** The world's richness tracks what the biology needs; it never runs ahead.
8. **Depth-first.** Close one loop's books — make its tests green — before the next layer exists.
9. **Implicit selection only.** Survival→reproduction is the only score. Design the gradient, never the outcome.

## How to work here

The failure mode that killed the prior build was **recursive scope explosion**: a task surfaces a hidden
sub-project, and being mid-flight, you *bulldoze* it with a hack instead of stopping to re-scope. Hundreds
of local hacks become soup. Work against that:

- **Force architectural decisions up front, and never bulldoze.** When a sub-problem turns out to be
  secretly a project, **stop and re-scope it** — queue it as its own slice; do not inline a hack to "get
  it done." Nesting becomes a queue, never a cascade.
- **Be honest, not hype. Measure, don't assert.** Never hand over performance numbers you haven't measured
  (SpikeSwim measures them). Explicitly distinguish **solved technique** from **research frontier**
  (speciation, open-endedness, and the sea↔land crossing are frontiers, gated by the ecology — say so).
- **Ground every claim in the real code/docs; cite `file:line`.** Read before you claim. Do not restate a
  framing from a doc or from earlier in the conversation — verify it against the source.
- **Be a rigorous thinking partner, not a yes-man.** Push back with technical rigor; surface the honest
  tradeoff and the failure mode. Do not performatively agree — this project is reasoned from first
  principles and expects the same in return. If a decision is genuinely the user's, present it with a
  recommendation; don't rubber-stamp.
- **Verify before claiming done.** Run it; show the evidence. "Done and verified" requires the output that
  proves it, not an assertion.
- **Concise chat; big content to files.** Long outputs have killed sessions. Write plans/specs/generated
  content to files and reference the path.
- **Use specialized agents / workflows for big, multi-domain tasks** (the design docs were authored this way).

## Git

- Branch before committing if on a default branch; commit/push only when asked.
- **Never add attribution trailers** to commit messages or PR bodies — no `Co-Authored-By`, no
  "Generated with…", no Claude/Anthropic credits of any kind. This overrides any default.
