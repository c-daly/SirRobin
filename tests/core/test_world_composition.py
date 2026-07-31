from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import torch

from sirrobin.core.live_world import initialize_live_state
from sirrobin.core.world import HeadlessWorld
from sirrobin.economy.config import EconomyConfig
from sirrobin.economy.state import EconomyState
from sirrobin.economy.step import EconomyKernel
from sirrobin.fields.geometry import GridGeometry
from sirrobin.genetics.develop import develop
from sirrobin.genetics.genotype import GenotypeBatch
from sirrobin.physics.contracts import FluidSample
from sirrobin.physics.live_config import LiveLocomotionConfig

FIXTURE = Path("oracle/fixtures/live/donor_development_live.json")


def test_headless_world_advances_one_composed_tick() -> None:
    rows = json.loads(FIXTURE.read_text(encoding="utf-8"))["bodies"]
    swimmer = next(row for row in rows if row["id"] == "swimmer")
    body = develop(GenotypeBatch.from_donor_rows([swimmer], dtype=torch.float64))
    live_state = initialize_live_state(body)
    fluid = FluidSample(
        torch.full(body.alive.shape, 1000.0, dtype=torch.float64),
        torch.zeros((*body.alive.shape, 3), dtype=torch.float64),
    )
    live_config = LiveLocomotionConfig()

    economy_config = replace(
        EconomyConfig(),
        gx=1,
        gy=1,
        gz=2,
        lx_m=10.0,
        ly_m=10.0,
        lz_m=10.0,
    )
    economy_state = EconomyState.zeros(economy_config)
    expected_total_q = economy_state.total_per_world().clone()
    world = HeadlessWorld(
        body=body,
        live_state=live_state,
        fluid=fluid,
        live_config=live_config,
        geometry=GridGeometry.from_config(economy_config),
        economy=EconomyKernel(economy_state, economy_config),
    )

    mechanics, economy = world.advance()

    assert torch.equal(live_state.gait_time_s, torch.full(body.alive.shape, live_config.dt, dtype=torch.float64))
    assert economy_state.step.item() == 1
    assert economy_state.time_s.item() == economy_config.dt_eco_s
    assert bool(economy.books_closed.all())
    assert torch.equal(economy_state.total_per_world(), expected_total_q)
    assert not bool(mechanics.nonfinite.any())
