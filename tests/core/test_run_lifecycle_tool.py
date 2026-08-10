"""Operational CLI contract for the first lifecycle scenario."""

from __future__ import annotations

import os
import subprocess
import sys


def test_run_lifecycle_cli_reports_both_controlled_arms() -> None:
    environment = os.environ.copy()
    environment["PYTHONPATH"] = "src:."
    completed = subprocess.run(
        [sys.executable, "tools/run_lifecycle.py", "--device", "cpu"],
        check=False,
        capture_output=True,
        text=True,
        env=environment,
    )

    assert completed.returncode == 0, completed.stderr
    assert "paired controlled arms: yes" in completed.stdout
    assert "viable feeding events: " in completed.stdout
    assert "viable birth succeeded: yes" in completed.stdout
    assert "viable population: 2" in completed.stdout
    assert "viable assimilation heat J: " in completed.stdout
    assert "viable maintenance heat J: " in completed.stdout
    assert "starved death occurred: yes" in completed.stdout
    assert "starved population: 0" in completed.stdout
    assert "starved death dissipation J: " in completed.stdout
    assert "recycling kinetics: accelerated causal fixture" in completed.stdout
    assert "post-death producer recycling q: " in completed.stdout
    assert "viable exact books closed: yes" in completed.stdout
    assert "starved exact books closed: yes" in completed.stdout
