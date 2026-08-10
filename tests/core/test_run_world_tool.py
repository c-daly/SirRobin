"""Operational contract for the small composed-world command."""

from __future__ import annotations

import os
import subprocess
import sys

import pytest


def _run_world(*arguments: str) -> subprocess.CompletedProcess[str]:
    environment = os.environ.copy()
    environment["PYTHONPATH"] = "src"
    environment["PYTHONDONTWRITEBYTECODE"] = "1"
    return subprocess.run(
        [sys.executable, "tools/run_world.py", *arguments],
        capture_output=True,
        check=False,
        env=environment,
        text=True,
    )


def test_run_world_reports_authoritative_state_and_measured_cost() -> None:
    completed = _run_world("--seconds", "0.2", "--bodies", "2", "--device", "cpu")

    assert completed.returncode == 0, completed.stderr
    assert "SirRobin composed-world run (operational output; not a stable schema)" in completed.stdout
    assert "requested simulated time s: 0.2" in completed.stdout
    assert "actual simulated time s: 0.2" in completed.stdout
    assert "economy steps: 2" in completed.stdout
    assert "mechanics steps: 24" in completed.stdout
    assert "mechanics steps / economy step: 12" in completed.stdout
    assert "shipped mechanics steps / economy step: 1036800" in completed.stdout
    assert "population: 2" in completed.stdout
    assert "initial field totals q: ND=40000000 BP=4000000 BD=500000 BM=0" in completed.stdout
    assert "initial field total q: 44500000" in completed.stdout
    assert "final field total q: 44500000" in completed.stdout
    assert "initial creature totals q: structure=2000 reserve=1000" in completed.stdout
    assert "final creature totals q: structure=2000 reserve=1000" in completed.stdout
    assert "initial whole-world total q: 44503000" in completed.stdout
    assert "final whole-world total q: 44503000" in completed.stdout
    assert "exact whole-world books closed: yes" in completed.stdout
    assert "mechanics clock range s: 0.2 .. 0.2" in completed.stdout
    assert "positions sample ENU m (2/2):" in completed.stdout
    assert "wall time s:" in completed.stdout
    assert "simulated seconds / wall second:" in completed.stdout


@pytest.mark.parametrize(
    ("seconds", "bodies", "message"),
    [
        ("0", "2", "seconds must be positive"),
        ("0.15", "2", "exact multiple"),
        ("0.1", "0", "bodies must be positive"),
    ],
)
def test_run_world_rejects_malformed_requests(
    seconds: str, bodies: str, message: str
) -> None:
    completed = _run_world("--seconds", seconds, "--bodies", bodies)

    assert completed.returncode != 0
    assert message in completed.stderr
