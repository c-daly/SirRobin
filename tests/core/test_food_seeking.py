"""Food-gradient intent requests action without assigning physical success."""

from __future__ import annotations

import math
from dataclasses import fields, replace

import pytest
import torch

from sirrobin.core.controller import turn_authority
from sirrobin.core.feeding import FeedingConfig
from sirrobin.core.foraging import FoodSeekingConfig, apply_food_seeking_intent
from sirrobin.core.runner import HeadlessRunner
from sirrobin.economy.config import EconomyConfig
from sirrobin.physics.controller import retune_heading_controller_state
from tools.run_world import _build_fixture_world


def _world(*, bodies: int = 1, live_bodies: int | None = None, gradient: bool = True):
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
        bodies=bodies,
        live_bodies=live_bodies,
        device=torch.device("cpu"),
        economy_interval_s=0.1,
        economy_config=economy,
    )
    if gradient:
        world.economy_state.bp_q[0, 0, 0, 0] = 500_000
        world.economy_state.bp_q[0, 1, 0, 0] = 1_500_000
    world.live_state.position_enu_m[..., 0] = 1.0
    return world


def test_positive_local_gradient_requests_bounded_effort_without_moving_body() -> None:
    world = _world()
    position_before = world.live_state.position_enu_m.clone()
    velocity_before = world.live_state.velocity_rel_water_enu_m_s.clone()
    yaw_before = world.live_state.yaw_rad.clone()

    report = apply_food_seeking_intent(world, FoodSeekingConfig(0.6))

    assert report.horizontal_gradient_present.tolist() == [[True]]
    assert torch.allclose(
        report.requested_heading_enu,
        torch.tensor([[[1.0, 0.0]]], dtype=torch.float64),
    )
    assert report.requested_effort_fraction.tolist() == [[0.6]]
    assert torch.equal(world.live_state.position_enu_m, position_before)
    assert torch.equal(world.live_state.velocity_rel_water_enu_m_s, velocity_before)
    assert torch.equal(world.live_state.yaw_rad, yaw_before)


def test_heading_request_uses_bounded_controller_and_does_not_teleport_yaw() -> None:
    world = _world()
    world.live_state.yaw_rad.fill_(math.pi / 2.0)
    yaw_before = world.live_state.yaw_rad.clone()

    apply_food_seeking_intent(world, FoodSeekingConfig(1.0))

    authority = turn_authority(world.body, world.live_config)
    assert world.live_state.turn_bias_rad_per_depth.item() < 0.0
    assert torch.all(world.live_state.turn_bias_rad_per_depth.abs() <= authority)
    assert torch.equal(world.live_state.yaw_rad, yaw_before)


def test_heading_controller_brakes_existing_yaw_momentum_at_zero_error() -> None:
    world = _world()
    world.live_state.yaw_momentum_kg_m2_s.fill_(1.0)
    yaw_before = world.live_state.yaw_rad.clone()
    momentum_before = world.live_state.yaw_momentum_kg_m2_s.clone()

    apply_food_seeking_intent(world, FoodSeekingConfig(1.0))

    assert world.live_state.turn_bias_rad_per_depth.item() < 0.0
    assert torch.equal(world.live_state.yaw_rad, yaw_before)
    assert torch.equal(
        world.live_state.yaw_momentum_kg_m2_s,
        momentum_before,
    )


def test_heading_controller_reduces_curvature_as_flow_speed_rises() -> None:
    slow = _world()
    fast = _world()
    slow.live_state.yaw_rad.fill_(math.pi / 2.0)
    fast.live_state.yaw_rad.fill_(math.pi / 2.0)
    fast.live_state.velocity_rel_water_enu_m_s[..., 1] = 10.0

    apply_food_seeking_intent(slow, FoodSeekingConfig(1.0))
    apply_food_seeking_intent(fast, FoodSeekingConfig(1.0))
    slow_motion = retune_heading_controller_state(
        slow.body,
        slow.live_state,
        slow.live_config,
    )
    fast_motion = retune_heading_controller_state(
        fast.body,
        fast.live_state,
        fast.live_config,
    )

    assert fast_motion.turn_bias_rad_per_depth.abs().item() < (
        slow_motion.turn_bias_rad_per_depth.abs().item()
    )


def test_flat_food_field_requests_no_heading_or_effort() -> None:
    world = _world(gradient=False)

    report = apply_food_seeking_intent(world, FoodSeekingConfig(1.0))

    assert report.horizontal_gradient_present.tolist() == [[False]]
    assert torch.equal(
        report.requested_heading_enu,
        torch.zeros_like(report.requested_heading_enu),
    )
    assert report.requested_effort_fraction.tolist() == [[0.0]]
    assert world.live_state.heading_initialized.tolist() == [[False]]


def test_inactive_invalid_position_cannot_abort_live_gradient_sample() -> None:
    world = _world(bodies=2, live_bodies=1)
    world.live_state.position_enu_m[0, 1, 2] = 1.0e9

    report = apply_food_seeking_intent(world, FoodSeekingConfig(0.5))

    assert report.horizontal_gradient_present.tolist() == [[True, False]]
    assert report.requested_effort_fraction.tolist() == [[0.5, 0.0]]


def test_runner_canonical_mechanics_consumes_requested_effort() -> None:
    resting = _world()
    swimming = _world()
    resting_tick = HeadlessRunner(
        resting,
        food_seeking_config=FoodSeekingConfig(0.0),
    ).advance()
    swimming_tick = HeadlessRunner(
        swimming,
        food_seeking_config=FoodSeekingConfig(1.0),
    ).advance()

    assert resting_tick.food_seeking is not None
    assert swimming_tick.food_seeking is not None
    assert resting_tick.food_seeking.requested_effort_fraction.tolist() == [[0.0]]
    assert swimming_tick.food_seeking.requested_effort_fraction.tolist() == [[1.0]]
    assert float(resting_tick.mechanical_work_j.sum().item()) == 0.0
    assert float(swimming_tick.mechanical_work_j.sum().item()) > 0.0
    assert torch.equal(
        resting.live_state.position_enu_m,
        torch.tensor([[[1.0, 0.0, 0.0]]], dtype=torch.float64),
    )
    assert not torch.equal(
        swimming.live_state.position_enu_m,
        resting.live_state.position_enu_m,
    )
    assert resting_tick.matter.books_closed.tolist() == [True]
    assert swimming_tick.matter.books_closed.tolist() == [True]


def test_unsettled_food_heading_does_not_gate_runner_progress() -> None:
    world = _world()
    world.live_state.yaw_rad.fill_(math.pi / 2.0)

    tick = HeadlessRunner(
        world,
        food_seeking_config=FoodSeekingConfig(1.0),
    ).advance()

    assert tick.food_seeking is not None
    assert torch.allclose(
        tick.food_seeking.requested_heading_enu,
        torch.tensor([[[1.0, 0.0]]], dtype=torch.float64),
    )
    assert world.live_state.yaw_rad.item() != 0.0
    assert world.sim_time_s == 0.1
    assert tick.matter.books_closed.tolist() == [True]


def test_runner_composes_food_intent_with_transactional_feeding() -> None:
    world = _world()

    tick = HeadlessRunner(
        world,
        food_seeking_config=FoodSeekingConfig(1.0),
        feeding_config=FeedingConfig(0.5, 0.5),
    ).advance()

    assert tick.food_seeking is not None
    assert tick.feeding is not None
    assert tick.feeding.actual_debit_q > 0
    assert tick.feeding.actual_debit_q == (
        tick.feeding.reserve_credit_q + tick.feeding.dissolved_return_q
    )
    assert tick.matter.books_closed.tolist() == [True]


@pytest.mark.parametrize("effort", [-0.1, 1.1, math.nan])
def test_mechanics_rejects_malformed_effort_before_state_mutation(
    effort: float,
) -> None:
    world = _world()
    before = tuple(
        getattr(world.live_state, field.name).clone()
        for field in fields(world.live_state)
    )

    with pytest.raises(ValueError, match="effort_fraction"):
        world._step_mechanics(torch.tensor([[effort]], dtype=torch.float64))

    assert all(
        torch.equal(getattr(world.live_state, field.name), expected)
        for field, expected in zip(fields(world.live_state), before, strict=True)
    )


@pytest.mark.parametrize("effort", [-0.1, 1.1, math.inf, math.nan, True])
def test_food_seeking_config_rejects_malformed_effort(effort: float) -> None:
    with pytest.raises((TypeError, ValueError)):
        FoodSeekingConfig(effort)
