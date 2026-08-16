"""Foraging diagnostics characterize behavior without turning it into a gate."""

from __future__ import annotations

from tools.diagnose_foraging import INTERPRETATION, run_foraging_diagnostic


def test_foraging_diagnostic_is_deterministic_and_accounts_for_live_intent() -> None:
    arguments = {
        "device_name": "cpu",
        "duration_s": 0.2,
        "seed": 101,
        "compile_domains": False,
    }

    first = run_foraging_diagnostic(**arguments)
    second = run_foraging_diagnostic(**arguments)

    assert first == second
    assert first["interpretation"] == INTERPRETATION
    assert "passed" not in first
    assert "failed" not in first
    assert first["configuration"]["intervals"] == 2
    assert first["conservation"]["books_closed"] is True
    producer = first["producer_accounting"]
    assert producer["balance_closed"] is True
    assert producer["final_q"] == producer["expected_final_q"]
    assert producer["feeding_debit_q"] == first["aggregate"]["feeding_debit_q"]
    assert first["aggregate"]["locomoting_intervals"] == first["aggregate"]["behavior_intervals"]
    assert first["aggregate"]["food_gradient_intervals"] <= first["aggregate"]["behavior_intervals"]
    assert first["aggregate"]["behavior_intervals"] == 2 * first["configuration"]["initial_bodies"]
    for creature in first["creatures"]:
        assert creature["locomoting_intervals"] == creature["behavior_intervals"]
        assert creature["food_gradient_intervals"] <= creature["behavior_intervals"]
        assert creature["path_length_m"] >= creature["displacement_m"]
