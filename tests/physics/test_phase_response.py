from __future__ import annotations

import json
from dataclasses import fields

import pytest
import torch

from sirrobin.core.live_world import advance_live_world, initialize_live_state
from sirrobin.genetics.develop import develop
from sirrobin.genetics.genotype import GenotypeBatch
from sirrobin.physics.contracts import LiveState
from sirrobin.physics.phase_response import (
    PhaseWindowConfig,
    advance_phase_window,
)
from tools.run_world import FIXTURE, _build_fixture_world


def _clone_state(state: LiveState) -> LiveState:
    return LiveState(
        **{field.name: getattr(state, field.name).clone() for field in fields(state)}
    )


def _canonical_second(world, turn_bias: float) -> tuple[LiveState, float, float]:
    state = initialize_live_state(world.body)
    state.turn_bias_rad_per_depth.fill_(turn_bias)
    positive_work_j = 0.0
    dissipated_work_j = 0.0
    for _ in range(round(1.0 / world.live_config.dt)):
        ledger = advance_live_world(
            world.body,
            state,
            world.fluid,
            world.live_config,
            world.geometry,
        )
        positive_work_j += float(
            ledger.total.input_power_w.clamp_min(0.0).sum()
        ) * world.live_config.dt
        dissipated_work_j += float(
            ledger.total.dissipated_power_w.sum()
        ) * world.live_config.dt
    return state, positive_work_j, dissipated_work_j


def _response_second(world, turn_bias: float):
    state = initialize_live_state(world.body)
    state.turn_bias_rad_per_depth.fill_(turn_bias)
    config = PhaseWindowConfig(interval_s=0.1, stages=4, phase_samples=3)
    config.validate()
    positive_work_j = 0.0
    dissipated_work_j = 0.0
    effort = torch.ones_like(world.body.alive, dtype=world.body.mass_sim.dtype)
    last = None
    for _ in range(10):
        last = advance_phase_window(
            world.body,
            state,
            world.fluid,
            world.live_config,
            world.geometry,
            config,
            effort_fraction=effort,
        )
        state = last.state
        positive_work_j += float(last.ledger.positive_actuator_work_j.sum())
        dissipated_work_j += float(last.ledger.dissipated_work_j.sum())
    assert last is not None
    return state, positive_work_j, dissipated_work_j, last.ledger


@pytest.mark.parametrize("turn_bias", [0.0, 0.025, -0.025])
def test_phase_window_preserves_founder_motion_and_named_work(turn_bias: float) -> None:
    world = _build_fixture_world(
        bodies=1,
        live_bodies=1,
        device=torch.device("cpu"),
        economy_interval_s=0.1,
        physics_dtype=torch.float32,
    )
    exact, exact_work, exact_dissipation = _canonical_second(world, turn_bias)
    approximate, work, dissipation, ledger = _response_second(world, turn_bias)

    position_error = torch.linalg.vector_norm(
        exact.position_enu_m - approximate.position_enu_m
    )
    velocity_error = torch.linalg.vector_norm(
        exact.velocity_rel_water_enu_m_s
        - approximate.velocity_rel_water_enu_m_s
    )
    yaw_error = torch.atan2(
        torch.sin(exact.yaw_rad - approximate.yaw_rad),
        torch.cos(exact.yaw_rad - approximate.yaw_rad),
    ).abs()
    assert float(position_error) < 0.01
    assert float(velocity_error) < 0.10
    assert float(yaw_error) < 0.005
    if turn_bias != 0.0:
        assert torch.sign(approximate.yaw_rad).item() == torch.sign(exact.yaw_rad).item()
    assert 0.94 < work / exact_work < 1.02
    assert 0.97 < dissipation / exact_dissipation < 1.02
    assert not bool(ledger.nonfinite.any())
    assert not bool(ledger.yaw_backstop_hit.any())


def test_phase_window_uses_the_actual_gait_window() -> None:
    world = _build_fixture_world(
        bodies=1,
        live_bodies=1,
        device=torch.device("cpu"),
        economy_interval_s=0.1,
        physics_dtype=torch.float32,
    )
    first = initialize_live_state(world.body)
    shifted = _clone_state(first)
    shifted.gait_time_s.fill_(0.125)
    config = PhaseWindowConfig(interval_s=0.05, stages=1, phase_samples=2)
    effort = torch.ones_like(world.body.alive, dtype=world.body.mass_sim.dtype)

    first_advance = advance_phase_window(
        world.body,
        first,
        world.fluid,
        world.live_config,
        world.geometry,
        config,
        effort_fraction=effort,
    )
    shifted_advance = advance_phase_window(
        world.body,
        shifted,
        world.fluid,
        world.live_config,
        world.geometry,
        config,
        effort_fraction=effort,
    )

    assert not torch.allclose(
        first_advance.state.velocity_rel_water_enu_m_s,
        shifted_advance.state.velocity_rel_water_enu_m_s,
    )
    assert not torch.allclose(
        first_advance.ledger.positive_actuator_work_j,
        shifted_advance.ledger.positive_actuator_work_j,
    )


def test_phase_window_retunes_turn_against_existing_yaw_rate() -> None:
    world = _build_fixture_world(
        bodies=1,
        live_bodies=1,
        device=torch.device("cpu"),
        economy_interval_s=0.1,
        physics_dtype=torch.float32,
    )
    state = initialize_live_state(world.body)
    state.heading_initialized.fill_(True)
    state.yaw_momentum_kg_m2_s.fill_(1.0)

    advance = advance_phase_window(
        world.body,
        state,
        world.fluid,
        world.live_config,
        world.geometry,
        PhaseWindowConfig(interval_s=0.1, stages=4, phase_samples=2),
        effort_fraction=torch.zeros_like(world.body.mass_sim[..., 0]),
    )

    assert advance.state.turn_bias_rad_per_depth.item() < 0.0


def test_phase_window_passive_drag_cannot_reverse_and_amplify_high_speed() -> None:
    world = _build_fixture_world(
        bodies=1,
        live_bodies=1,
        device=torch.device("cpu"),
        economy_interval_s=0.1,
        physics_dtype=torch.float32,
    )
    state = initialize_live_state(world.body)
    state.velocity_rel_water_enu_m_s[..., 0] = 50.0

    advance = advance_phase_window(
        world.body,
        state,
        world.fluid,
        world.live_config,
        world.geometry,
        PhaseWindowConfig(interval_s=0.1, stages=4, phase_samples=2),
        effort_fraction=torch.zeros_like(world.body.mass_sim[..., 0]),
    )

    final_velocity = advance.state.velocity_rel_water_enu_m_s[0, 0]
    assert torch.isfinite(final_velocity).all()
    assert final_velocity[0] >= 0.0
    assert torch.linalg.vector_norm(final_velocity) < 50.0
    assert not bool(advance.ledger.yaw_backstop_hit.any())


def test_phase_window_does_not_promote_a_structural_zero_into_motion() -> None:
    rows = json.loads(FIXTURE.read_text(encoding="utf-8"))["bodies"]
    root_only = next(row for row in rows if row["id"] == "root-only")
    genotype = GenotypeBatch.from_donor_rows([root_only], dtype=torch.float32)
    body = develop(genotype)
    fixture_world = _build_fixture_world(
        bodies=1,
        live_bodies=1,
        device=torch.device("cpu"),
        economy_interval_s=0.1,
        physics_dtype=torch.float32,
    )
    state = initialize_live_state(body)
    effort = torch.ones_like(body.alive, dtype=torch.float32)

    advance = advance_phase_window(
        body,
        state,
        fixture_world.fluid,
        fixture_world.live_config,
        fixture_world.geometry,
        PhaseWindowConfig(interval_s=0.1, stages=4, phase_samples=2),
        effort_fraction=effort,
    )

    assert torch.count_nonzero(advance.state.position_enu_m) == 0
    assert torch.count_nonzero(advance.state.velocity_rel_water_enu_m_s) == 0
    assert torch.count_nonzero(advance.state.yaw_rad) == 0
    assert torch.count_nonzero(advance.ledger.positive_actuator_work_j) == 0
    assert torch.count_nonzero(advance.ledger.dissipated_work_j) == 0


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
def test_phase_window_cpu_and_cuda_agree_for_the_same_founder() -> None:
    worlds = {
        device: _build_fixture_world(
            bodies=1,
            live_bodies=1,
            device=torch.device(device),
            economy_interval_s=0.1,
            physics_dtype=torch.float32,
        )
        for device in ("cpu", "cuda")
    }
    config = PhaseWindowConfig(interval_s=0.1, stages=4, phase_samples=2)
    advances = {}
    for device, world in worlds.items():
        state = initialize_live_state(world.body)
        state.turn_bias_rad_per_depth.fill_(-0.025)
        effort = torch.ones_like(world.body.alive, dtype=torch.float32)
        advances[device] = advance_phase_window(
            world.body,
            state,
            world.fluid,
            world.live_config,
            world.geometry,
            config,
            effort_fraction=effort,
        )

    cpu = advances["cpu"]
    cuda = advances["cuda"]
    for field in (
        "position_enu_m",
        "velocity_rel_water_enu_m_s",
        "yaw_rad",
        "yaw_momentum_kg_m2_s",
    ):
        assert torch.allclose(
            getattr(cpu.state, field),
            getattr(cuda.state, field).cpu(),
            rtol=2.0e-4,
            atol=2.0e-5,
        )
    assert torch.allclose(
        cpu.ledger.positive_actuator_work_j,
        cuda.ledger.positive_actuator_work_j.cpu(),
        rtol=2.0e-4,
        atol=2.0e-5,
    )
    assert not bool(cuda.ledger.nonfinite.any())
