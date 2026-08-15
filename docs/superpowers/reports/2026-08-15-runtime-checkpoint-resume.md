# Runtime checkpoint and resume evidence

Date: 2026-08-15

Base: `eea06fd` (`origin/main`)

Branch: `agent/runtime-checkpoint`

## Outcome

`RuntimeSession` can save its last accepted state and complete scientific
configuration to one versioned safetensors checkpoint, then restore a new session
on CPU or CUDA. A CPU birth-capable fixture resumed with exact equality across
every authoritative tensor and produced an exact uninterrupted continuation.

This is a core headless API. It does not yet add checkpoint flags to
`tools/run_world.py` because that command's lifecycle-control branch is under
review separately.

## Authority boundary

Schema `sirrobin.runtime.checkpoint.v1` stores:

- population identity, lineage, allocator, material, and fractional carries;
- genotype tensors;
- developed-body and exact per-segment development state;
- motion position, velocity, yaw, gait clock, and controller state;
- all four field reservoirs, field carries, step, time, and buffer parity;
- the exact expected whole-world matter baseline; and
- every validated `LivingRuntimeConfig` component, including mortality and
  mutation seeds.

The checkpoint layout is frozen by
`e080f0c3feb51da73ff5a70de7a8a44362da17dfcf50c8280232d0fe68c584f4`.
Changing state or config field membership without an explicit schema decision
causes save and load to fail.

`FluidSample` remains an external interval input because the current
`RuntimeSession` does not own dynamic fluid state. Compilation and optimistic
execution settings are selected when restoring; they are execution policy, not
scientific state. Checkpoints occur only at accepted session boundaries, never in
the middle of a candidate chunk.

## Integrity and interrupted publication

The checkpoint is staged in the destination directory and published with one
atomic `os.replace`. An interrupted save therefore leaves the previous checkpoint
at the public path. A focused test injects a failure before publication, verifies
the earlier checkpoint is still exact and loadable, and verifies temporary state
is removed.

The single file embeds canonical configuration JSON, its SHA-256, the frozen
layout hash, bounded clock/capacity metadata, and a SHA-256 for every tensor's
dtype, shape, and bytes. Load rejects schema, layout, config, field-membership, or
tensor-integrity disagreement before returning state. It then runs the ordinary
runtime config and living-state validators, including exact whole-world matter
closure.

Live identity tensors can share storage across population, genotype, and developed
body views. The persistence boundary stores independent equal values rather than
making Python storage aliasing part of the public format; normal boundary
validation re-establishes their authoritative equality.

## Continuation evidence

The CPU fixture used one founder in three slots with the existing
`evolution-demo` profile. It advanced three authoritative intervals, saved and
restored, then compared:

- every tensor and scalar in the restored `LivingState` against the accepted
  pre-save state;
- four further uninterrupted intervals against four resumed intervals; and
- every field in the resulting chunk summaries.

All comparisons were exact and both final economy and whole-matter books closed.
The dedicated CUDA test loaded the same complete checkpoint directly to CUDA and
verified population, economy, and genotype authorities were device-resident.

## Verification

- initial RED collection: missing `sirrobin.runtime.checkpoint`;
- first implementation run: four failures exposed shared-storage identity views;
- final focused CPU checkpoint gate: 5 passed, 1 GPU test deselected;
- final direct CUDA checkpoint gate: 1 passed, 5 CPU tests deselected;
- runtime integration before the atomic single-file refinement: 26 passed, 2 GPU
  tests deselected;
- full non-GPU repository gate before that isolated refinement: 429 passed, 1
  expected CUDA-unavailable skip, 8 GPU tests deselected;
- final whole-tree Ruff and `git diff --check`: passed;
- Import Linter kept 6 contracts and reproduced only the existing forbidden
  `sirrobin.physics -> sirrobin.fields.geometry` imports in
  `ecological_motion.py` and `phase_response.py`.

## Next boundary

After the lifecycle-control PR is accepted, the next narrow slice can expose this
core contract through explicit headless CLI input/output paths and test a real
command-level stop-and-resume run. That integration should not broaden the v1
state schema or make the viewer an owner of persistence.
