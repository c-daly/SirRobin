# Creature material accounting

**Date:** 2026-08-09

**Branch:** `restart/original-baseline`

**Base commit:** `f395ab0`

**Status:** Independently accepted after correcting `int64` wraparound and runtime
schema-replacement closure bypasses.

**Uncommitted candidate manifest SHA-256:**
`0082e75742281ee1514a532ff3198310f8de43f998fa869853a99814ca0fdbda`

The manifest is the SHA-256 of the ordered `sha256sum` output for the changed
source, tool, and focused-test files. This report is excluded to avoid a
self-referential hash.

## Scope and authority

The slice adds two authoritative fixed-capacity `int64` creature reservoirs:

- structural limiting nutrient, `structure_q[world, capacity]`; and
- reserve limiting nutrient, `reserve_q[world, capacity]`.

They are separate from the developed body's physical mass. No conversion from
physical kilograms to nutrient quanta is claimed yet, and no feeding,
maintenance, birth, or death transfer is implemented in this slice.

`HeadlessWorld` captures one per-world persistent baseline:

```text
expected_total_q =
    ND_q + BP_q + BD_q + BM_q + structure_q + reserve_q
```

Every composed tick records each component before and after and arrests the
runner unless both totals equal that baseline exactly. The economy ledger now
has the narrower subsystem role: it proves that reaction and transport conserve
the field inventory that entered that economy step. It is not a competing
persistent field-only baseline, because such a baseline would reject a future
exact field-to-creature transfer.

## Validity and adversarial controls

Creature reservoirs must have the exact live population shape and device, use
`int64`, remain in `[0, 2^62)`, and contain zero in inactive slots. On both sides of
every tick, the runtime census rechecks dtype, exact shape, device, element domain,
and inactive-slot validity for every raw field and creature reservoir. It independently
sums valid raw reservoirs in `float64` to require the combined inventory below the
configured `<2^62` safe reduction bound before trusting the exact `int64` total. A
wrapped total, fractional replacement, or broadcastable wrong shape cannot close the
books; invalid pre-step authority arrests the runner before any mechanism advances.

Focused tests independently recompute raw tensor sums and cover:

- exact before/after closure with nonzero structure and reserve;
- explicit zero-valued stores with fixed-capacity shape;
- dtype, shape, negative-value, inactive-slot, and whole-inventory rejection;
- detection and runner arrest after a one-quantum creature mint; and
- rejection of a post-initialization `2^64` mint whose naive wrapped `int64` total
  exactly equals the expected baseline; and
- pre-step arrest for fractional creature stock, wrong creature shape, and non-int64
  field replacement; and
- acceptance of an equal producer-field debit and reserve credit, while the
  economy step still closes its own lower field input.

## Operational result

The existing 128-swimmer command now starts from explicitly declared fixture
stocks of 1,000 structural and 500 reserve quanta per live creature. These are
starting-condition demonstration values, not a biological calibration.

The command reported:

- initial and final field total: 44,500,000 quanta;
- initial and final creature total: 128,000 structural plus 64,000 reserve
  quanta;
- initial and final whole-world total: 44,692,000 quanta; and
- exact whole-world books closed.

Three complete CPU runs took 29.71, 30.15, and 28.19 advance-wall seconds, or
290.8, 286.5, and 306.4 simulated seconds per wall second. The latest run clears
the provisional 300 target, but all three are slower than Slice 1.5's 19.62-second
run. A direct breakdown measured only 0.00024 seconds in the new before/after
accounting and 27.97 seconds in the unchanged periodic mechanics path. This rules
out the ledger as the direct hotspot, but does not establish why the same mechanics
path is currently slower. The operational variance remains recorded rather than
being attributed to the new mechanism or silently discarded.

## Verification

- Dedicated material file: 13 passed after both review corrections.
- Complete post-correction suite: 143 passed, 5 CUDA-dependent skips, and the one
  known strict F12 xfail in 140.97 seconds.
- Ruff and `git diff --check`: clean.
- Import boundaries: 71 files, 139 dependencies, 7 contracts kept.
- Lock check: clean.

## Limits and next consumer

- The fixture stock sizes are not morphology-derived and cannot be used as a
  birth cost, reserve capacity, or maintenance calibration.
- The persistent baseline is not checkpointed yet; save/resume is Milestone
  3.4 and must include it when persistence becomes a consumer.
- The next authorized consumer is Slice 2.2: one-creature local feeding must
  debit the actual producer stock and credit reserve plus an explicitly routed
  remainder without changing this exact total.
