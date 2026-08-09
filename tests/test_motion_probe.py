"""Focused contract for the pure, full-physics motion probe."""

from __future__ import annotations

import json
import math
from dataclasses import FrozenInstanceError, fields
from pathlib import Path

import pytest
import torch

from sirrobin.benchmarks.motion_probe import probe_motion
from sirrobin.genetics.develop import develop
from sirrobin.genetics.genotype import GenotypeBatch

FIXTURE = Path("oracle/fixtures/live/donor_development_live.json")


def _body(body_id: str = "swimmer"):
    rows = json.loads(FIXTURE.read_text(encoding="utf-8"))["bodies"]
    row = next(item for item in rows if item["id"] == body_id)
    return develop(GenotypeBatch.from_donor_rows([row], dtype=torch.float64))


def test_probe_is_repeatable_immutable_and_does_not_mutate_the_body() -> None:
    body = _body()
    before = {field.name: getattr(body, field.name).clone() for field in fields(body)}

    first = probe_motion(
        body,
        effort_fraction=0.75,
        turn_fraction=0.25,
        warmup_cycles=0.25,
        measurement_cycles=0.5,
    )
    second = probe_motion(
        body,
        effort_fraction=0.75,
        turn_fraction=0.25,
        warmup_cycles=0.25,
        measurement_cycles=0.5,
    )

    assert first == second
    assert first.stable_id == 1
    assert first.measurement_steps == 30
    assert first.measurement_phase_cycles == pytest.approx(0.5)
    assert math.isfinite(first.cycle_mean_surge_m_s)
    assert math.isfinite(first.cycle_mean_yaw_response_rad_s)
    assert first.straight_mechanical_work_j >= 0.0
    assert first.turning_mechanical_work_j >= 0.0
    assert first.intervention_count == 0
    for field in fields(body):
        assert torch.equal(getattr(body, field.name), before[field.name]), field.name
    with pytest.raises(FrozenInstanceError):
        first.effort_fraction = 0.5  # type: ignore[misc]


def test_opposite_bounded_turns_have_opposite_paired_yaw_response() -> None:
    body = _body()
    arguments = {
        "effort_fraction": 1.0,
        "warmup_cycles": 0.25,
        "measurement_cycles": 0.5,
    }

    left = probe_motion(body, turn_fraction=0.25, **arguments)
    right = probe_motion(body, turn_fraction=-0.25, **arguments)

    assert left.cycle_mean_yaw_response_rad_s * right.cycle_mean_yaw_response_rad_s < 0.0
    assert left.cycle_mean_surge_m_s == right.cycle_mean_surge_m_s
    assert left.straight_mechanical_work_j == right.straight_mechanical_work_j


def test_zero_effort_is_a_structural_zero_response() -> None:
    result = probe_motion(
        _body(),
        effort_fraction=0.0,
        turn_fraction=1.0,
        warmup_cycles=0.0,
        measurement_cycles=0.1,
    )

    assert result.cycle_mean_surge_m_s == 0.0
    assert result.cycle_mean_yaw_response_rad_s == 0.0
    assert result.straight_mechanical_work_j == 0.0
    assert result.turning_mechanical_work_j == 0.0


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"effort_fraction": -0.01}, "effort_fraction"),
        ({"effort_fraction": 1.01}, "effort_fraction"),
        ({"turn_fraction": -1.01}, "turn_fraction"),
        ({"turn_fraction": 1.01}, "turn_fraction"),
        ({"warmup_cycles": -0.01}, "warmup_cycles"),
        ({"measurement_cycles": 0.0}, "measurement_cycles"),
    ],
)
def test_probe_rejects_out_of_domain_requests(overrides: dict[str, float], message: str) -> None:
    arguments = {
        "effort_fraction": 0.5,
        "turn_fraction": 0.25,
        "warmup_cycles": 0.0,
        "measurement_cycles": 0.1,
    }
    arguments.update(overrides)

    with pytest.raises(ValueError, match=message):
        probe_motion(_body(), **arguments)
