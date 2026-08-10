"""Acceptance contract for the first composed material lifecycle scenario."""

from __future__ import annotations

import pytest

from sirrobin.core.feeding import FeedingConfig
from tools.run_lifecycle import run_first_lifecycle_scenario


def test_paired_lifecycle_proves_viable_birth_starvation_and_field_recycling() -> None:
    report = run_first_lifecycle_scenario(device_name="cpu")

    viable = report.viable
    assert viable.feeding_events > 0
    assert viable.feeding_producer_debit_q == (
        viable.feeding_reserve_credit_q + viable.feeding_dissolved_return_q
    )
    assert viable.maintenance_reserve_debit_q == viable.maintenance_return_q
    assert viable.birth.born is True
    assert viable.birth.parent_id == 1
    assert viable.birth.child_id == 2
    assert viable.birth.total_debit_q == (
        viable.birth.structure_q + viable.birth.initial_reserve_q
    )
    assert viable.birth.parent_reserve_before_q - viable.birth.parent_reserve_after_q == (
        viable.birth.total_debit_q
    )
    assert viable.final_population == 2
    assert viable.initial_whole_world_q == viable.final_whole_world_q
    assert viable.books_closed is True

    # Independent represented-energy census across feeding carry, maintenance,
    # and construction. No production validator supplies this expectation.
    assert viable.feeding_producer_input_j == pytest.approx(
        viable.feeding_reserve_credit_j
        + viable.feeding_assimilation_heat_j
        + viable.final_assimilation_carry_j,
        abs=1.0e-9,
    )
    assert viable.maintenance_heat_j == pytest.approx(
        viable.maintenance_reserve_debit_q * report.reserve_j_per_q,
        abs=1.0e-12,
    )
    assert viable.birth.construction_heat_j == pytest.approx(
        viable.birth.structure_q * report.reserve_j_per_q,
        abs=1.0e-12,
    )

    starved = report.starved
    assert starved.starved is True
    assert starved.final_population == 0
    assert starved.maintenance_return_q + starved.death_return_q == (
        starved.initial_structure_q + starved.initial_reserve_q
    )
    assert starved.maintenance_heat_j == pytest.approx(
        starved.maintenance_return_q * report.reserve_j_per_q,
        abs=1.0e-12,
    )
    assert starved.death_dissipation_j >= 0.0
    assert starved.predeath_producer_recycling_q == 0
    assert starved.post_death_producer_recycling_q > 0
    assert starved.post_death_recycling_steps > 0
    assert starved.initial_whole_world_q == starved.final_whole_world_q
    assert starved.books_closed is True


def test_viable_arm_cannot_birth_when_feeding_causality_is_removed() -> None:
    with pytest.raises(RuntimeError, match="did not fund a paid birth"):
        run_first_lifecycle_scenario(
            device_name="cpu",
            feeding_config=FeedingConfig(
                capture_efficiency=0.0,
                assimilation_efficiency=1.0,
            ),
            max_viable_steps=30,
        )
