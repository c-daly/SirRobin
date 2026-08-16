from __future__ import annotations

from dataclasses import replace

import pytest
import torch

from sirrobin.economy.config import EconomyConfig
from sirrobin.organisms.behavior import BehaviorConfig, request_living_intent
from sirrobin.organisms.state import PopulationState
from tools.run_world import _build_fixture_world


def _fixture(
    *,
    gradient: bool = True,
    live_bodies: int = 1,
    gx: int = 2,
    gy: int = 1,
):
    economy = replace(
        EconomyConfig(),
        gx=gx,
        gy=gy,
        gz=2,
        lx_m=float(gx),
        ly_m=float(gy),
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
    world.live_state.position_enu_m[..., 1] = 0.5
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


def _request(world, population, effort: float = 0.6):
    return request_living_intent(
        population,
        world.body,
        world.live_state,
        world.economy_state.bp_q,
        world.geometry,
        world.live_config,
        BehaviorConfig(effort),
        q_mass_mol=world.economy_config.q_mass_mol,
    )


def test_generic_food_state_does_not_require_a_named_sense_node() -> None:
    world, population = _fixture()
    assert not bool(world.body.sense.any())

    step = _request(world, population)

    assert step.sampled_producer_mol_m3[0, 0] > 0.0
    assert step.horizontal_gradient_present.tolist() == [[True, False]]
    assert torch.allclose(
        step.requested_heading_enu[0, 0],
        step.requested_heading_enu.new_tensor([1.0, 0.0]),
        atol=1.0e-6,
    )


def test_food_state_changes_intent_without_assigning_motion() -> None:
    world, population = _fixture()
    position = world.live_state.position_enu_m.clone()
    velocity = world.live_state.velocity_rel_water_enu_m_s.clone()
    yaw = world.live_state.yaw_rad.clone()

    step = _request(world, population)

    assert step.horizontal_gradient_present.tolist() == [[True, False]]
    assert step.locomoting.tolist() == [[True, False]]
    assert step.requested_effort_fraction.tolist() == [[pytest.approx(0.6), 0.0]]
    assert step.birth_requested.tolist() == [[True, False]]
    assert step.invalid.tolist() == [[False, False]]
    assert torch.equal(step.motion.position_enu_m, position)
    assert torch.equal(step.motion.velocity_rel_water_enu_m_s, velocity)
    assert torch.equal(step.motion.yaw_rad, yaw)
    assert step.motion is not world.live_state
    assert world.live_state.heading_initialized.tolist() == [[False, False]]


def test_food_gradient_state_is_expressed_in_body_forward_left_axes() -> None:
    world, population = _fixture()
    world.live_state.yaw_rad[0, 0] = torch.pi / 2.0

    step = _request(world, population)

    body_gradient = step.food_gradient_body_forward_left_mol_m4[0, 0]
    assert abs(float(body_gradient[0])) < 1.0e-6 * abs(float(body_gradient[1]))
    assert body_gradient[1] < 0.0
    assert torch.allclose(
        step.requested_heading_enu[0, 0],
        step.requested_heading_enu.new_tensor([1.0, 0.0]),
        atol=1.0e-6,
    )


def test_flat_food_field_preserves_drive_without_inventing_a_heading() -> None:
    world, population = _fixture(gradient=False)
    position = world.live_state.position_enu_m.clone()
    velocity = world.live_state.velocity_rel_water_enu_m_s.clone()
    yaw = world.live_state.yaw_rad.clone()

    step = _request(world, population, effort=0.25)

    assert not bool(step.horizontal_gradient_present.any())
    assert step.locomoting.tolist() == [[True, False]]
    assert step.requested_effort_fraction.tolist() == [[pytest.approx(0.25), 0.0]]
    assert not bool(step.requested_heading_enu.any())
    assert torch.equal(step.motion.position_enu_m, position)
    assert torch.equal(step.motion.velocity_rel_water_enu_m_s, velocity)
    assert torch.equal(step.motion.yaw_rad, yaw)
    assert not bool(step.motion.heading_initialized.any())


def test_internal_reserve_does_not_switch_food_behavior_modes() -> None:
    world, population = _fixture(live_bodies=2)
    different_reserves = replace(
        population,
        reserve_q=population.reserve_q.new_tensor([[1, 1_000_000]]),
    )

    step = _request(world, different_reserves)

    assert step.horizontal_gradient_present.tolist() == [[True, True]]
    assert step.locomoting.tolist() == [[True, True]]
    assert step.requested_effort_fraction.tolist() == [
        [pytest.approx(0.6), pytest.approx(0.6)]
    ]
    assert torch.allclose(
        step.requested_heading_enu[0, 0],
        step.requested_heading_enu[0, 1],
    )


def test_remote_world_peak_does_not_change_local_food_state() -> None:
    world, population = _fixture(gx=4, gy=4)
    remote_peak = world.economy_state.bp_q.clone()
    remote_peak[0, 2, 2, :] = 2_000_000_000

    local = _request(world, population)
    remote = request_living_intent(
        population,
        world.body,
        world.live_state,
        remote_peak,
        world.geometry,
        world.live_config,
        BehaviorConfig(0.6),
        q_mass_mol=world.economy_config.q_mass_mol,
    )

    assert torch.equal(local.sampled_producer_mol_m3, remote.sampled_producer_mol_m3)
    assert torch.equal(local.producer_gradient_mol_m4, remote.producer_gradient_mol_m4)
    assert torch.equal(
        local.horizontal_gradient_present,
        remote.horizontal_gradient_present,
    )
    assert torch.equal(local.requested_heading_enu, remote.requested_heading_enu)


def test_out_of_bounds_food_state_is_invalid_and_not_steering() -> None:
    world, population = _fixture()
    world.live_state.position_enu_m[0, 0, 2] = 1.0

    step = _request(world, population)

    assert step.invalid.tolist() == [[True, False]]
    assert not bool(step.horizontal_gradient_present.any())
    assert not bool(step.sampled_producer_mol_m3.any())
    assert not bool(step.producer_gradient_mol_m4.any())


@pytest.mark.parametrize("effort", [-0.1, 1.1, float("inf"), float("nan"), True])
def test_behavior_rejects_malformed_locomotor_effort(effort: float) -> None:
    with pytest.raises((TypeError, ValueError)):
        BehaviorConfig(effort).validate()


def test_behavior_stops_powering_gait_above_body_wave_speed() -> None:
    world, population = _fixture()
    world.live_state.velocity_rel_water_enu_m_s[..., 0] = 100.0

    step = _request(world, population)

    assert step.horizontal_gradient_present.tolist() == [[True, False]]
    assert step.requested_effort_fraction.tolist() == [[0.0, 0.0]]
    assert not bool(step.locomoting.any())


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

    assert actual.requested_effort_fraction.tolist() == [[0.75, 0.0]]
    assert actual.horizontal_gradient_present.tolist() == [[True, False]]
    assert actual.invalid.tolist() == [[False, False]]
