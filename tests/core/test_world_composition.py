"""Tranche A composition smoke test.

Claim protected — the next consumer cannot function safely or honestly unless there is
one headless place where mechanics and economy advance on one declared schedule and the
complete tick verifies the books. Immediate consumer: Tranche B steps 4-5, whose reduced
surge/yaw step must have its complete-world cost measured inside a composed runner.

Class: invariant (plan §5.1 — books close exactly after every complete tick) plus
validity (finite mechanical state). This does NOT claim the subsystems are biologically
coupled; feeding, metabolism, and the whole-world creature/field ledger are Tranche C/D.
"""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import pytest
import torch

from sirrobin.core.runner import HeadlessRunner, WorldSchedule
from sirrobin.core.world import HeadlessWorld
from sirrobin.economy.config import EconomyConfig
from sirrobin.economy.state import EconomyState
from sirrobin.genetics.genotype import GenotypeBatch
from sirrobin.physics.contracts import FluidSample
from sirrobin.physics.live_config import LiveLocomotionConfig

FIXTURE = Path("oracle/fixtures/live/donor_development_live.json")


def _world() -> HeadlessWorld:
    """A deliberately cheap fixture: two donor swimmers on a 1x1x4 grid.

    `dt_eco_s` is shrunk so one composed interval is 12 mechanics substeps rather than
    the shipped 1,036,800; `remin_floor_s` is raised only to keep the config's
    deep-detritus residence check satisfied at that timestep. Both are declared fixture
    choices, not claims about ocean chemistry, and the config remains legal — the
    kernel validates it on construction.
    """
    rows = json.loads(FIXTURE.read_text(encoding="utf-8"))["bodies"]
    swimmer = next(row for row in rows if row["id"] == "swimmer")
    genotype = GenotypeBatch.from_donor_rows([swimmer] * 2, dtype=torch.float64)
    config = replace(
        EconomyConfig(),
        gx=1,
        gy=1,
        gz=4,
        lx_m=10,
        ly_m=10,
        lz_m=20,
        dt_eco_s=0.1,
        remin_floor_s=1.0e-4,
    )
    state = EconomyState.zeros(config)
    state.nd_q.fill_(10_000_000)
    state.bp_q.fill_(1_000_000)
    state.bd_q[..., 0] = 500_000
    shape = (2,)
    return HeadlessWorld(
        genotype=genotype,
        fluid=FluidSample(
            torch.full((1, *shape), 1000.0, dtype=torch.float64),
            torch.zeros((1, *shape, 3), dtype=torch.float64),
        ),
        live_config=LiveLocomotionConfig(),
        economy_state=state,
        economy_config=config,
    )


def test_composed_world_advances_both_clocks_and_closes_its_books_without_unity() -> None:
    runner = HeadlessRunner(_world())
    world = runner.world
    expected_total = world.economy_state.total_per_world().clone()
    substeps = runner.schedule.mechanics_steps_per_economy_step

    for interval in range(1, 4):
        gait_before = world.live_state.gait_time_s.clone()
        tick = runner.advance()

        # The declared schedule is what actually ran. `tick.mechanics_steps` is only a
        # config echo, so the substep count is checked against the mechanics clock,
        # which `step_live` accumulates independently. When Tranche D adds mid-run
        # births this must be scoped to creatures present at interval start; a newborn
        # legitimately advances less.
        gait_delta = world.live_state.gait_time_s - gait_before
        assert torch.allclose(
            gait_delta, torch.full_like(gait_delta, world.economy_config.dt_eco_s)
        )
        assert tick.sim_time_s == pytest.approx(interval * world.economy_config.dt_eco_s)
        assert int(world.economy_state.step) == interval

        # Composition disturbs neither subsystem's own invariants. `advance()` has
        # already required the books to close; these assert the same facts the runner
        # enforces, plus finiteness of the mechanical state it advanced.
        assert tick.economy.books_closed.all()
        assert torch.equal(world.economy_state.total_per_world(), expected_total)
        world.economy_state.validate(world.economy_config)
        assert torch.isfinite(world.live_state.position_enu_m).all()
        assert torch.isfinite(world.live_state.velocity_rel_water_enu_m_s).all()
        assert torch.isfinite(world.live_state.yaw_rad).all()

    # The shipped cadence is stated rather than hidden behind the cheap fixture.
    assert substeps == 12
    assert WorldSchedule.from_configs(
        LiveLocomotionConfig(), EconomyConfig()
    ).mechanics_steps_per_economy_step == round(EconomyConfig().dt_eco_s / LiveLocomotionConfig().dt)
