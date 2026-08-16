"""Operational contract for the small composed-world command."""

from __future__ import annotations

import os
import re
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


def test_run_world_defaults_to_runtime_session_and_reports_bounded_evidence() -> None:
    completed = _run_world("--seconds", "0.2", "--bodies", "2", "--device", "cpu")

    assert completed.returncode == 0, completed.stderr
    assert "SirRobin RuntimeSession run (operational output; not a stable schema)" in completed.stdout
    assert "runtime: cohesive device state and domain kernels" in completed.stdout
    assert "compiled domains: no" in completed.stdout
    assert "optimistic candidates: yes" in completed.stdout
    assert "runtime profile: causal" in completed.stdout
    assert "requested simulated time s: 0.2" in completed.stdout
    assert "actual simulated time s: 0.2" in completed.stdout
    assert "authoritative intervals: 2" in completed.stdout
    assert "slot capacity: 2" in completed.stdout
    assert "initial population: 2" in completed.stdout
    assert "host chunks: 1" in completed.stdout
    assert "maximum intervals / host chunk: 2" in completed.stdout
    assert "population: 2" in completed.stdout
    assert "initial field totals q: ND=40000000 BP=4000000 BD=500000 BM=0" in completed.stdout
    assert "initial creature totals q: structure=2000 reserve=10000" in completed.stdout
    assert "initial whole-world total q: 44512000" in completed.stdout
    assert "final whole-world total q: 44512000" in completed.stdout
    assert "exact whole-world books closed: yes" in completed.stdout
    assert "behavior intervals: " in completed.stdout
    assert "feeding actual debit q: " in completed.stdout
    assert "mechanics clock range s: 0.2 .. 0.2" in completed.stdout
    assert "positions sample ENU m (2/2):" in completed.stdout
    assert "warmup wall time s: 0.000000" in completed.stdout
    assert "simulated seconds / wall second:" in completed.stdout


def test_runtime_can_run_a_birth_capable_population_in_explicit_chunks() -> None:
    completed = _run_world(
        "--seconds",
        "0.2",
        "--bodies",
        "3",
        "--live-bodies",
        "1",
        "--profile",
        "evolution-demo",
        "--chunk-intervals",
        "1",
        "--dense-candidates",
        "--device",
        "cpu",
    )

    assert completed.returncode == 0, completed.stderr
    assert "runtime profile: evolution-demo" in completed.stdout
    assert "optimistic candidates: no" in completed.stdout
    assert "slot capacity: 3" in completed.stdout
    assert "initial population: 1" in completed.stdout
    assert "host chunks: 2" in completed.stdout
    assert "maximum intervals / host chunk: 1" in completed.stdout
    assert "population: 3" in completed.stdout
    assert int(re.search(r"births: (\d+)", completed.stdout).group(1)) > 0
    assert "exact whole-world books closed: yes" in completed.stdout


def test_run_world_preserves_the_reference_runner_explicitly() -> None:
    completed = _run_world(
        "--runtime",
        "reference",
        "--seconds",
        "0.2",
        "--bodies",
        "2",
        "--device",
        "cpu",
    )

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


def test_run_world_can_explicitly_enable_one_creature_feeding() -> None:
    completed = _run_world(
        "--runtime",
        "reference",
        "--seconds",
        "0.1",
        "--bodies",
        "1",
        "--device",
        "cpu",
        "--feed-one",
    )

    assert completed.returncode == 0, completed.stderr
    assert "one-creature feeding enabled: yes" in completed.stdout
    assert "feeding events: 1" in completed.stdout
    debit = int(re.search(r"feeding producer debit q: (\d+)", completed.stdout).group(1))
    reserve = int(re.search(r"feeding reserve credit q: (\d+)", completed.stdout).group(1))
    dissolved = int(
        re.search(r"feeding dissolved return q: (\d+)", completed.stdout).group(1)
    )
    assert debit > 0
    assert reserve > 0
    assert dissolved > 0
    assert debit == reserve + dissolved
    assert "feeding assimilation heat J: " in completed.stdout
    assert "final feeding intake carry mol: " in completed.stdout
    assert "final feeding assimilation carry q: " in completed.stdout
    assert "producer chemical energy density J/q: 0.5" in completed.stdout
    assert "reserve chemical energy density J/q: 0.45" in completed.stdout
    assert "final feeding assimilation carry energy J: " in completed.stdout
    assert "exact whole-world books closed: yes" in completed.stdout


def test_run_world_rejects_feeding_without_the_scoped_population() -> None:
    completed = _run_world(
        "--runtime", "reference", "--seconds", "0.1", "--bodies", "2", "--feed-one"
    )

    assert completed.returncode != 0
    assert "exactly one body" in completed.stderr


def test_run_world_exposes_mass_derived_maintenance() -> None:
    completed = _run_world(
        "--runtime",
        "reference",
        "--seconds",
        "0.1",
        "--economy-interval",
        "0.1",
        "--bodies",
        "1",
        "--maintain-one",
    )

    assert completed.returncode == 0, completed.stderr
    assert "one-creature maintenance enabled: yes" in completed.stdout
    assert "maintenance events: 1" in completed.stdout
    debit = int(re.search(r"maintenance reserve debit q: (\d+)", completed.stdout).group(1))
    returned = int(
        re.search(r"maintenance dissolved return q: (\d+)", completed.stdout).group(1)
    )
    assert debit > 0
    assert returned == debit
    assert "starvation deaths: 0" in completed.stdout
    assert "maintenance reserve chemical debit J: " in completed.stdout
    assert "exact whole-world books closed: yes" in completed.stdout


def test_run_world_exposes_one_paid_exact_clone_birth() -> None:
    completed = _run_world(
        "--runtime",
        "reference",
        "--seconds",
        "0.1",
        "--economy-interval",
        "0.1",
        "--bodies",
        "1",
        "--birth-one",
    )

    assert completed.returncode == 0, completed.stderr
    assert "one paid exact-clone birth requested: yes" in completed.stdout
    assert "birth succeeded: yes" in completed.stdout
    assert "birth parent/child IDs: 1 -> 2" in completed.stdout
    assert "birth structure q: 1000" in completed.stdout
    assert "birth initial reserve q: 100" in completed.stdout
    assert "birth total parent debit q: 1100" in completed.stdout
    assert "birth construction heat J: 450" in completed.stdout
    assert "population: 2" in completed.stdout
    assert "exact whole-world books closed: yes" in completed.stdout


def test_default_runtime_rejects_reference_only_probes() -> None:
    completed = _run_world("--seconds", "0.1", "--bodies", "1", "--feed-one")

    assert completed.returncode != 0
    assert "require --runtime reference" in completed.stderr


@pytest.mark.parametrize(
    ("arguments", "message"),
    [
        (("--bodies", "2", "--live-bodies", "3"), "fit inside body capacity"),
        (("--chunk-intervals", "0"), "chunk intervals must be positive"),
    ],
)
def test_runtime_rejects_malformed_population_and_chunk_controls(
    arguments: tuple[str, ...], message: str
) -> None:
    completed = _run_world(*arguments)

    assert completed.returncode != 0
    assert message in completed.stderr


def test_reference_runtime_rejects_device_lifecycle_controls() -> None:
    completed = _run_world(
        "--runtime",
        "reference",
        "--live-bodies",
        "1",
    )

    assert completed.returncode != 0
    assert "require --runtime device" in completed.stderr


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
