# One-creature local feeding

**Date:** 2026-08-09

**Branch:** `restart/original-baseline`

**Base commit:** `c5ead89`

**Status:** Independently accepted after correcting all Critical and Important
locality, transaction, numerical-carry, authority, and sequencing findings.

**Accepted candidate manifest SHA-256:**
`f0b887a5845ef3cc963fb04c3a4c94fcc6fb0f481b6c487a9be81665c5054521`

The manifest is the SHA-256 of the ordered `sha256sum` output for all changed
source, tool, and focused/inherited test files. This report is excluded to avoid
a self-referential hash.

## Scope and causality

This Slice 2.2 candidate adds one deliberately narrow field-to-creature
transaction. It is disabled by default and rejects any enabled world that does
not contain exactly one live creature. Population contention and deterministic
shared-stock allocation remain deferred to Slice 3.1; iteration order therefore
cannot become an undeclared feeding advantage in this slice.

For the one live creature, the transaction reads only:

- developed intake area;
- water-relative three-dimensional speed;
- producer concentration sampled trilinearly at the creature's continuous ENU
  position;
- the declared ecological interval; and
- explicit capture and assimilation parameters; and
- immutable world-owned producer/reserve chemical-energy densities.

The requested producer quanta are:

```text
clearance_m3 = intake_area_m2 * relative_speed_m_s * dt_s * capture_efficiency
request_total_mol = clearance_m3 * sampled_BP_mol_m3 + intake_carry_before_mol
requested_q = floor(request_total_mol / q_mass_mol)
intake_carry_after_mol = request_total_mol - requested_q * q_mass_mol
actual_debit_q = min(requested_q, positive_weight_local_stock_q)
```

There is no independent request clamp. The committed debit is only the actual
producer stock available on the positive-weight cells of that same local trilinear
stencil. Availability shortfall never becomes future debt. Zero speed, zero developed
intake, and zero local producer concentration with zero carry produce an exact no-op.

## Exact matter and explicit energy boundary

The actual producer debit is partitioned once:

```text
energy_fraction = min(1, producer_j_per_q / reserve_j_per_q)
effective_fraction = min(assimilation_efficiency, energy_fraction)
(reserve_credit_q, _, assimilation_carry_after_q) =
    deterministic_fraction(
        actual_debit_q, effective_fraction, assimilation_carry_before_q
    )
dissolved_return_q = actual_debit_q - reserve_credit_q
```

The remainder is returned locally to dissolved nutrient (`ND`), which is the
declared non-energetic waste pool for this first lean transaction. Every debited
nutrient quantum therefore reaches either creature reserve or dissolved nutrient,
and the existing field-plus-creature persistent baseline remains exact.

Intake molar carry and fractional reserve-credit carry are fixed-capacity authoritative
creature state. They remove timestep-dependent starvation without storing material:
each remains strictly below one quantum in its own units and must be zero in inactive
slots.

Producer and reserve energy densities are mandatory immutable world configuration.
The energy boundary includes the reserve-credit entitlement carried between events:

```text
producer_chemical_input_j = actual_debit_q * producer_j_per_q
reserve_chemical_credit_j = reserve_credit_q * reserve_j_per_q
carry_energy_j = assimilation_carry_q * reserve_j_per_q
producer_input_j + carry_before_j =
    reserve_credit_j + assimilation_heat_j + carry_after_j
```

The effective fraction prevents negative cumulative heat even when reserve energy
density exceeds producer energy density. Mechanical work remains observation and is
not yet a reserve debit; maintenance and locomotion payment belong to Slice 2.3.

## Composition and operation

The runner revalidates the one-live-creature scope before any mechanics or clock
advance. Its sequence is mechanics, the field reaction/transport step, mandatory
field-subsystem closure, optional feeding, then authoritative whole-world closure.
Any mechanism exception arrests the runner. The economy ledger certifies only its
own pre-feeding field step; the whole-world ledger certifies the combined transfer.

`tools/run_world.py --feed-one` is the explicit operational opt-in and requires
`--bodies 1`. Its feeding parameters are fixture-only demonstration anchors, not
a biological calibration. The existing default multi-body command remains feeding
disabled.

A one-second CPU run reported:

- 10 feeding opportunities;
- 290 producer quanta actually debited;
- 145 reserve quanta credited;
- 145 quanta returned to dissolved nutrient;
- 79.75 J declared assimilation heat; and
- final intake carry `1.59100099e-10` mol and zero final assimilation carry;
- declared producer/reserve densities of 0.5/0.45 J per quantum; and
- identical initial and final whole-world totals of 44,501,500 quanta.

## Adversarial controls

Focused tests independently recompute raw reservoir totals and attack:

- local request causality and exact debit/credit consequences;
- scarcity, proving the actual debit cannot exceed local producer stock;
- zero-weight center, surface, and periodic neighbors containing abundant stock;
- staged destination capacity failure and finite-energy failure before source debit;
- float64 apportionment mint/overdebit attacks near `2^62`, rejected before mutation;
- missing speed, intake, and producer causes;
- repeated subquantum intake, assimilation, and energy-limited conversion;
- nonfinite dynamic causes before any reservoir mutation;
- malformed efficiencies, carry state, and energy densities;
- immutable world energy authority and self-reconstructing report inputs;
- multi-creature use before a shared allocator exists;
- pre-tick population drift and inactive out-of-domain capacity positions;
- rejection of a broken economy ledger before feeding from its output;
- exact local integer deposit; mechanism-exception arrest; and
- runner and command composition through the whole-world ledger.

## Verification

- Corrected feeding, material, field, command, composition, and periodic tests:
  86 passed.
- Directly affected inherited runner, periodic mechanics, field, economy, and energy
  tests: 56 passed, 1 CUDA-dependent skip.
- Complete final suite: 182 passed, 5 CUDA-dependent skips, and the one known
  strict F12 xfail in 204.40 seconds.
- Ruff and `git diff --check`: clean before final re-review.
- Import boundaries: 72 files, 144 dependencies, 7 contracts kept.

## Limits and next consumer

- This is not a population feeding algorithm. Slice 3.1 must allocate contended
  stock in one deterministic transaction before feeding can be enabled for more
  than one live creature.
- Capture, assimilation, and world energy-density values are declared model
  parameters; the operational fixture values establish mechanism behavior, not
  empirical calibration or universal viability.
- Reserve capacity, growth, maintenance, starvation, death, birth, and persistence
  are not introduced here.
- Slice 2.3 death must explicitly dissipate or otherwise settle the persistent
  `assimilation_carry_q * reserve_j_per_q` energy entitlement before zeroing an
  inactive slot.
- The next authorized slice is 2.3: morphology/mass-derived maintenance and
  starvation death with exact matter return and explicit heat/work accounting.
