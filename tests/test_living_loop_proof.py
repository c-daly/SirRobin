from __future__ import annotations

import pytest

from tools.prove_living_loop import _whole_intervals, run_living_loop_proof


def test_proof_horizon_must_contain_whole_authoritative_intervals() -> None:
    assert _whole_intervals(0.1) == 1
    with pytest.raises(ValueError, match="whole authoritative intervals"):
        _whole_intervals(0.15)


def test_short_proof_fails_closed_when_the_generational_loop_is_incomplete() -> None:
    report = run_living_loop_proof(
        device_name="cpu",
        max_duration_s=0.1,
        compile_domains=False,
    )

    assert report["schema"] == "sirrobin.living-loop-proof.v1"
    assert report["verdict"]["passed"] is False
    assert "mutated_paid_birth" in report["verdict"]["missing_claims"]
    assert report["claims"]["raw_matter_census_closed_every_interval"] is True
    assert report["execution"]["completed_intervals"] == 1
