import json
from dataclasses import replace
from pathlib import Path

import torch

from sirrobin.core.live_world import initialize_live_state
from sirrobin.genetics.develop import develop
from sirrobin.genetics.genotype import GenotypeBatch
from sirrobin.physics.contracts import FluidSample
from sirrobin.physics.live_config import LiveLocomotionConfig
from sirrobin.physics.live_step import step_live
from sirrobin.physics.morphology import query_morphology

FIXTURE = Path("oracle/fixtures/live/donor_development_live.json")


def test_intake_is_geometry_derived_and_not_a_hydrodynamic_switch():
    rows = json.loads(FIXTURE.read_text(encoding="utf-8"))["bodies"]
    row = next(row for row in rows if row["id"] == "swimmer")
    body = develop(GenotypeBatch.from_donor_rows([row], dtype=torch.float64))
    no_intake = replace(body, intake=torch.zeros_like(body.intake))
    config = LiveLocomotionConfig()
    original = query_morphology(body, config)
    removed = query_morphology(no_intake, config)
    assert original.intake_area_m2 > 0
    assert removed.intake_area_m2 == 0
    assert torch.equal(original.structural_mass_kg, removed.structural_mass_kg)
    fluid = FluidSample(
        torch.full(body.alive.shape, 1000.0, dtype=torch.float64),
        torch.zeros((*body.alive.shape, 3), dtype=torch.float64),
    )
    first = step_live(body, initialize_live_state(body), fluid, config)
    second = step_live(no_intake, initialize_live_state(no_intake), fluid, config)
    assert torch.equal(first.total.force_enu_n, second.total.force_enu_n)
    assert torch.equal(first.total.torque_yaw_nm, second.total.torque_yaw_nm)
