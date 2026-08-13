from __future__ import annotations

from dataclasses import replace

import pytest
import torch

from sirrobin.economy.config import EconomyConfig
from sirrobin.organisms.behavior import BehaviorConfig, request_living_intent
from sirrobin.organisms.state import PopulationState
from tools.run_world import _build_fixture_world


def _fixture(*, gradient: bool = True, live_bodies: int = 1):
    economy = replace(
        EconomyConfig(),
        gx=2,
        gy=1,
        gz=2,
        lx_m=2.0,
        ly_m=1.0,
        lz_m=2.0,
        dt_eco_s=0.1,
        remin_floor_s=1.0e-4,
    )
    world = _build_fixture_world(
        bodies=2,
        live_bodies=live_bodies,
        device=torch.device("cpu"),
        economy_interval_s=0.1,
        economy_config=economy,
        physics_dtype=torch.float32,
    )
    if gradient:
        world.economy_state.bp_q[0, 0, 0, 0] = 500_000
        world.economy_state.bp_q[0, 1, 0, 0] = 1_500_000
    world.live_state.position_enu_m[..., 0] = 1.0
    alive = world.body.alive
    zeros_i64 = torch.zeros_like(alive, dtype=torch.int64)
    zeros_f64 = torch.zeros_like(alive, dtype=torch.float64)
    population = PopulationState(
        alive,
        world.body.stable_id,
        zeros_i64,
        zeros_i64,
        zeros_f64,
        world.creature_material.structure_q,
        world.creature_material.reserve_q,
        world.creature_material.intake_carry_mol,
        world.creature_material.assimilation_carry_q,
        world.creature_material.maintenance_carry_j,
        world.next_stable_id,
    )
    return world, population


def test_device_behavior_requests_gradient_intent_without_assigning_motion() -> None:
    world, population = _fixture()
    position = world.live_state.position_enu_m.clone()
    velocity = world.live_state.velocity_rel_water_enu_m_s.clone()
    yaw = world.live_state.yaw_rad.clone()

    step = request_living_intent(
        population,
        world.body,
        world.live_state,
        world.economy_state.bp_q,
        world.geometry,
        world.live_config,
        BehaviorConfig(0.6),
        q_mass_mol=world.economy_config.q_mass_mol,
    )

    assert step.horizontal_gradient_present.tolist() == [[True, False]]
    assert step.requested_effort_fraction.tolist() == [[pytest.approx(0.6), 0.0]]
    assert step.birth_requested.tolist() == [[True, False]]
    assert step.invalid.tolist() == [[False, False]]
    assert torch.equal(step.motion.position_enu_m, position)
    assert torch.equal(step.motion.velocity_rel_water_enu_m_s, velocity)
    assert torch.equal(step.motion.yaw_rad, yaw)
    assert step.motion is not world.live_state
    assert world.live_state.heading_initialized.tolist() == [[False, False]]


def test_flat_field_requests_no_effort_or_heading() -> None:
    world, population = _fixture(gradient=False)

    step = request_living_intent(
        population,
        world.body,
        world.live_state,
        world.economy_state.bp_q,
        world.geometry,
        world.live_config,
        BehaviorConfig(1.0),
        q_mass_mol=world.economy_config.q_mass_mol,
    )

    assert not bool(step.horizontal_gradient_present.any())
    assert not bool(step.requested_effort_fraction.any())
    assert not bool(step.requested_heading_enu.any())


def test_flat_field_can_request_explicit_search_effort_without_assigning_motion() -> None:
    world, population = _fixture(gradient=False)
    world.live_state.yaw_rad.fill_(torch.pi / 2.0)
    position = world.live_state.position_enu_m.clone()
    velocity = world.live_state.velocity_rel_water_enu_m_s.clone()
    yaw = world.live_state.yaw_rad.clone()

    step = request_living_intent(
        population,
        world.body,
        world.live_state,
        world.economy_state.bp_q,
        world.geometry,
        world.live_config,
        BehaviorConfig(0.6, search_effort_fraction=0.25),
        q_mass_mol=world.economy_config.q_mass_mol,
    )

    assert step.horizontal_gradient_present.tolist() == [[False, False]]
    assert step.requested_effort_fraction.tolist() == [
        [pytest.approx(0.25), 0.0]
    ]
    assert not bool(step.requested_heading_enu.any())
    assert torch.equal(step.motion.position_enu_m, position)
    assert torch.equal(step.motion.velocity_rel_water_enu_m_s, velocity)
    assert torch.equal(step.motion.yaw_rad, yaw)
    assert torch.equal(
        step.motion.turn_bias_rad_per_depth,
        world.live_state.turn_bias_rad_per_depth,
    )
    assert not bool(step.motion.heading_initialized.any())


def test_flat_field_search_uses_paused_straight_legs() -> None:
    world, population = _fixture(gradient=False, live_bodies=2)
    config = BehaviorConfig(
        0.6,
        search_effort_fraction=0.25,
        search_leg_duration_s=8.0,
        search_duty_fraction=0.8,
    )

    initial = request_living_intent(
        population,
        world.body,
        world.live_state,
        world.economy_state.bp_q,
        world.geometry,
        world.live_config,
        config,
        q_mass_mol=world.economy_config.q_mass_mol,
    )
    same_leg_motion = replace(
        world.live_state,
        gait_time_s=world.live_state.gait_time_s + 1.0,
    )
    same_leg = request_living_intent(
        population,
        world.body,
        same_leg_motion,
        world.economy_state.bp_q,
        world.geometry,
        world.live_config,
        config,
        q_mass_mol=world.economy_config.q_mass_mol,
    )
    paused = request_living_intent(
        population,
        world.body,
        replace(
            world.live_state,
            gait_time_s=world.live_state.gait_time_s + 4.0,
        ),
        world.economy_state.bp_q,
        world.geometry,
        world.live_config,
        config,
        q_mass_mol=world.economy_config.q_mass_mol,
    )
    next_leg = request_living_intent(
        population,
        world.body,
        replace(
            world.live_state,
            gait_time_s=world.live_state.gait_time_s + 8.0,
        ),
        world.economy_state.bp_q,
        world.geometry,
        world.live_config,
        config,
        q_mass_mol=world.economy_config.q_mass_mol,
    )

    assert initial.horizontal_gradient_present.tolist() == [[False, False]]
    assert initial.requested_effort_fraction.tolist() == [
        [pytest.approx(0.25), pytest.approx(0.25)]
    ]
    heading_norm = torch.linalg.vector_norm(
        initial.requested_heading_enu,
        dim=-1,
    )
    assert torch.allclose(heading_norm, torch.ones_like(heading_norm))
    assert not torch.allclose(
        initial.requested_heading_enu[0, 0],
        initial.requested_heading_enu[0, 1],
    )
    assert not torch.allclose(
        initial.requested_heading_enu,
        next_leg.requested_heading_enu,
    )
    assert torch.allclose(
        initial.requested_heading_enu[0, 0],
        same_leg.requested_heading_enu[0, 0],
    )
    assert not bool(paused.requested_heading_enu[0, 0].any())
    assert paused.requested_effort_fraction[0, 0].item() == 0.0
    assert initial.motion.heading_initialized.tolist() == [[True, True]]


def test_food_rich_gradient_cruises_forward_instead_of_milling() -> None:
    world, population = _fixture()
    world.live_state.yaw_rad.fill_(torch.pi / 2.0)

    step = request_living_intent(
        population,
        world.body,
        world.live_state,
        world.economy_state.bp_q,
        world.geometry,
        world.live_config,
        BehaviorConfig(
            0.6,
            search_effort_fraction=0.25,
            search_leg_duration_s=8.0,
            search_duty_fraction=0.5,
            food_sufficient_peak_fraction=0.5,
            food_cruise_effort_fraction=0.1,
        ),
        q_mass_mol=world.economy_config.q_mass_mol,
    )

    assert step.horizontal_gradient_present.tolist() == [[True, False]]
    assert step.requested_effort_fraction.tolist() == [
        [pytest.approx(0.1), 0.0]
    ]
    assert torch.allclose(
        step.requested_heading_enu[0, 0],
        step.requested_heading_enu.new_tensor([0.0, 1.0]),
        atol=1.0e-6,
    )


@pytest.mark.parametrize("duration", [-0.1, float("inf"), float("nan"), True])
def test_behavior_rejects_malformed_search_leg_duration(duration: float) -> None:
    with pytest.raises((TypeError, ValueError)):
        BehaviorConfig(0.5, search_leg_duration_s=duration).validate()


def test_behavior_stops_powering_gait_above_the_body_wave_speed() -> None:
    world, population = _fixture()
    world.live_state.velocity_rel_water_enu_m_s[..., 0] = 100.0

    step = request_living_intent(
        population,
        world.body,
        world.live_state,
        world.economy_state.bp_q,
        world.geometry,
        world.live_config,
        BehaviorConfig(0.6, search_effort_fraction=0.25),
        q_mass_mol=world.economy_config.q_mass_mol,
    )

    assert step.horizontal_gradient_present.tolist() == [[True, False]]
    assert step.requested_effort_fraction.tolist() == [[0.0, 0.0]]


def test_device_behavior_is_one_full_graph() -> None:
    world, population = _fixture()
    compiled = torch.compile(
        request_living_intent,
        fullgraph=True,
        dynamic=False,
        backend="eager",
    )

    actual = compiled(
        population,
        world.body,
        world.live_state,
        world.economy_state.bp_q,
        world.geometry,
        world.live_config,
        BehaviorConfig(0.75),
        q_mass_mol=world.economy_config.q_mass_mol,
    )

    assert actual.requested_effort_fraction.tolist() == [
        [pytest.approx(0.75), 0.0]
    ]
    assert actual.invalid.tolist() == [[False, False]]
