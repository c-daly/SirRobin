import json
from pathlib import Path

import torch

from sirrobin.core.controller import update_heading_controller
from sirrobin.core.live_snapshot import load_live_snapshot, save_live_snapshot
from sirrobin.core.live_world import advance_live_world, initialize_live_state
from sirrobin.fields.geometry import GridGeometry
from sirrobin.genetics.develop import develop
from sirrobin.genetics.genotype import GenotypeBatch
from sirrobin.physics.contracts import FluidSample
from sirrobin.physics.live_config import LiveLocomotionConfig
from sirrobin.physics.live_step import step_live

FIXTURE = Path("oracle/fixtures/live/donor_development_live.json")


def _genotypes(count: int = 1):
    rows = json.loads(FIXTURE.read_text(encoding="utf-8"))["bodies"]
    swimmer = next(row for row in rows if row["id"] == "swimmer")
    return GenotypeBatch.from_donor_rows([swimmer] * count, dtype=torch.float64)


def _fluid(body, velocities=None):
    if velocities is None:
        velocities = torch.zeros((*body.alive.shape, 3), dtype=torch.float64)
    return FluidSample(torch.full(body.alive.shape, 1000.0, dtype=torch.float64), velocities)


def test_controller_latches_and_commands_opposite_signs():
    body = develop(_genotypes(2))
    state = initialize_live_state(body)
    requested = torch.tensor([[[0.0, 1.0], [0.0, -1.0]]], dtype=torch.float64)
    update_heading_controller(body, state, requested, LiveLocomotionConfig())
    assert torch.equal(state.heading_initialized, body.alive)
    assert state.turn_bias_rad_per_depth[0, 0] > 0
    assert state.turn_bias_rad_per_depth[0, 1] < 0
    assert torch.allclose(state.desired_heading_enu, requested)


def test_uniform_current_is_transport_only_and_depth_is_unchanged():
    body = develop(_genotypes(2))
    state = initialize_live_state(body)
    state.position_enu_m[..., 2] = -12.5
    currents = torch.tensor([[[0.0, 0.0, 3.0], [2.0, -1.0, -4.0]]], dtype=torch.float64)
    fluid = _fluid(body, currents)
    config = LiveLocomotionConfig()
    geometry = GridGeometry(8, 8, 4, 100.0, 80.0, 40.0)
    ledger = advance_live_world(body, state, fluid, config, geometry)
    assert torch.allclose(
        state.velocity_rel_water_enu_m_s[0, 0], state.velocity_rel_water_enu_m_s[0, 1]
    )
    expected_offset = torch.tensor([2.0 * config.dt, 80.0 - config.dt], dtype=torch.float64)
    toroidal_offset = torch.remainder(
        state.position_enu_m[0, 1, :2] - state.position_enu_m[0, 0, :2],
        torch.tensor([100.0, 80.0], dtype=torch.float64),
    )
    assert torch.allclose(
        toroidal_offset,
        expected_offset,
        atol=1e-12,
    )
    assert torch.equal(state.position_enu_m[..., 2], torch.full_like(state.yaw_rad, -12.5))
    assert torch.allclose(ledger.total.force_enu_n[0], ledger.total.force_enu_n[1])


def test_snapshot_regenerates_body_and_continues_exactly(tmp_path):
    genotype = _genotypes()
    body = develop(genotype)
    state = initialize_live_state(body)
    fluid = _fluid(body)
    config = LiveLocomotionConfig()
    for _ in range(4):
        step_live(body, state, fluid, config)
    path = tmp_path / "live.safetensors"
    save_live_snapshot(path, genotype, state, config)
    restored_genotype, restored_body, restored_state, restored_config = load_live_snapshot(path)
    assert restored_config == config
    for field in body.__dataclass_fields__:
        assert torch.equal(getattr(body, field), getattr(restored_body, field)), field
    for field in state.__dataclass_fields__:
        assert torch.equal(getattr(state, field), getattr(restored_state, field)), field
    first = step_live(body, state, fluid, config)
    second = step_live(restored_body, restored_state, fluid, restored_config)
    assert torch.equal(first.total.force_enu_n, second.total.force_enu_n)
    for field in state.__dataclass_fields__:
        assert torch.equal(getattr(state, field), getattr(restored_state, field)), field
    assert restored_genotype.stable_id.tolist() == genotype.stable_id.tolist()


def test_ninety_degree_heading_command_homes_and_settles():
    body = develop(_genotypes())
    state = initialize_live_state(body)
    fluid = _fluid(body)
    config = LiveLocomotionConfig()
    geometry = GridGeometry(8, 8, 4, 100.0, 100.0, 40.0)
    requested = torch.tensor([[[0.0, 1.0]]], dtype=torch.float64)
    for _ in range(1200):
        ledger = advance_live_world(
            body,
            state,
            fluid,
            config,
            geometry,
            requested_heading_enu=requested,
        )
        assert not torch.any(ledger.omega_backstop_hit)
    final_error = torch.atan2(torch.sin(0.5 * torch.pi - state.yaw_rad),
                              torch.cos(0.5 * torch.pi - state.yaw_rad)).abs()
    assert final_error < torch.deg2rad(torch.tensor(15.0, dtype=torch.float64)), (
        state.yaw_rad,
        state.velocity_rel_water_enu_m_s,
        state.turn_bias_rad_per_depth,
    )
    assert state.yaw_rad > 0
