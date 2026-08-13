"""Deterministic replicated lifecycle diagnostics report outcomes, not verdicts."""

from __future__ import annotations

from tools.diagnose_lifecycle_replicates import (
    run_replicated_lifecycle_diagnostic,
)


def test_replicated_lifecycle_diagnostic_is_deterministic_and_causal() -> None:
    arguments = {
        "device_name": "cpu",
        "profile_name": "evolution-demo",
        "replicates": 2,
        "duration_s": 0.1,
        "sample_every_s": 0.1,
        "base_seed": 101,
        "compile_domains": False,
    }

    first = run_replicated_lifecycle_diagnostic(**arguments)
    second = run_replicated_lifecycle_diagnostic(**arguments)

    assert first == second
    assert first["interpretation"] == (
        "observed lifecycle outcomes; extinction and persistence are not pass/fail"
    )
    assert "passed" not in first
    assert "failed" not in first
    assert first["configuration"]["replicate_seeds"] == [101, 102]
    assert len(first["replicates"]) == 2
    assert sum(first["aggregate"]["terminal_reason_counts"].values()) == 2
    for index, replicate in enumerate(first["replicates"]):
        assert replicate["replicate"] == index
        assert replicate["causal_inputs"]["position_seed"] == 101 + index
        assert replicate["causal_inputs"]["mortality_seed"] == 101 + index
        assert replicate["causal_inputs"]["mutation_seed"] == 101 + index
        assert [sample["time_s"] for sample in replicate["samples"]] == [0.0, 0.1]
        final = replicate["samples"][-1]
        assert final["food_availability"]["producer_q"] >= 0
        assert (
            final["reproduction"]["requested_births"] >= final["reproduction"]["births"]
        )
        assert final["mutation"]["events"] >= final["mutation"]["mutated_births"]
        assert sum(final["behavior"].values()) == replicate["causal_inputs"][
            "initial_population"
        ]
        assert final["conservation"]["economy_books_closed"] is True
        assert final["conservation"]["matter_books_closed"] is True
