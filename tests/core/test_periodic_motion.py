"""Focused contract for error-controlled periodic full-physics fast-forward."""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import pytest
import torch

import sirrobin.core.periodic_motion as periodic_motion
from sirrobin.core.periodic_motion import (
    PeriodicErrorEstimate,
    PeriodicMotionPolicy,
    repeat_transform,
)
from sirrobin.core.runner import HeadlessRunner
from sirrobin.core.world import HeadlessWorld
from sirrobin.economy.config import EconomyConfig
from sirrobin.economy.state import EconomyState
from sirrobin.genetics.genotype import GenotypeBatch
from sirrobin.physics.contracts import FluidSample
from sirrobin.physics.live_config import LiveLocomotionConfig

FIXTURE = Path("oracle/fixtures/live/donor_development_live.json")


def _world(
    body_ids: tuple[str, ...], *, interval_s: float, worlds: int = 1
) -> HeadlessWorld:
    rows = json.loads(FIXTURE.read_text(encoding="utf-8"))["bodies"]
    by_id = {row["id"]: row for row in rows}
    genotype = GenotypeBatch.from_donor_rows(
        [by_id[body_id] for body_id in body_ids], worlds=worlds, dtype=torch.float64
    )
    config = replace(
        EconomyConfig(),
        worlds=worlds,
        gx=1,
        gy=1,
        gz=4,
        lx_m=10.0,
        ly_m=10.0,
        lz_m=20.0,
        dt_eco_s=interval_s,
        remin_floor_s=1.0e-4,
    )
    state = EconomyState.zeros(config)
    state.nd_q.fill_(10_000_000)
    state.bp_q.fill_(1_000_000)
    state.bd_q[..., 0] = 500_000
    lead = (worlds, len(body_ids) // worlds)
    return HeadlessWorld(
        genotype=genotype,
        fluid=FluidSample(
            torch.full(lead, 1000.0, dtype=torch.float64),
            torch.zeros((*lead, 3), dtype=torch.float64),
        ),
        live_config=LiveLocomotionConfig(),
        economy_state=state,
        economy_config=config,
    )


def _test_policy() -> PeriodicMotionPolicy:
    return PeriodicMotionPolicy(
        max_detection_cycles=3,
        required_consecutive_cycles=1,
        relative_tolerance=1.0e-12,
        max_accumulated_translation_error_m=1.0e-9,
        max_accumulated_yaw_error_rad=1.0e-12,
        max_projected_relative_state_error=1.0e-9,
        max_projected_velocity_error_m_s=1.0e-9,
        max_projected_yaw_momentum_error_kg_m2_s=1.0e-9,
        max_projected_relative_work_error=1.0e-12,
    )


def test_exact_zero_clones_cover_every_step_and_fast_forward_without_motion() -> None:
    world = _world(("root-only", "root-only"), interval_s=4.0)
    runner = HeadlessRunner(world, periodic_policy=_test_policy())
    before_total = world.economy_state.total_per_world().clone()

    tick = runner.advance()

    assert tick.mechanics_steps == 480
    assert tick.fast_forwarded_mechanics_steps > 0
    assert tick.representative_mechanics_steps > 0
    assert tick.full_batch_mechanics_steps < tick.mechanics_steps
    assert (
        tick.fast_forwarded_mechanics_steps
        + tick.representative_mechanics_steps
        + tick.full_batch_mechanics_steps
        == tick.mechanics_steps
    )
    assert torch.allclose(
        world.live_state.gait_time_s,
        torch.full_like(world.live_state.gait_time_s, 4.0),
    )
    assert torch.equal(world.live_state.position_enu_m, torch.zeros_like(world.live_state.position_enu_m))
    assert torch.equal(tick.mechanical_work_j, torch.zeros_like(tick.mechanical_work_j))
    assert tick.periodic_error is not None
    assert tick.periodic_error.accumulated_translation_error_m == 0.0
    assert tick.periodic_error.accumulated_yaw_error_rad == 0.0
    assert tick.periodic_error.projected_relative_state_error == 0.0
    assert tick.periodic_error.projected_velocity_error_m_s == 0.0
    assert tick.periodic_error.projected_yaw_momentum_error_kg_m2_s == 0.0
    assert tick.periodic_error.projected_relative_work_error == 0.0
    assert torch.equal(world.economy_state.total_per_world(), before_total)
    assert tick.economy.books_closed.all()


def test_canonical_mechanics_is_the_runner_default_until_policy_is_explicit() -> None:
    world = _world(("root-only", "root-only"), interval_s=4.0)

    tick = HeadlessRunner(world).advance()

    assert tick.full_batch_mechanics_steps == 480
    assert tick.representative_mechanics_steps == 0
    assert tick.fast_forwarded_mechanics_steps == 0
    assert tick.periodic_error is None


@pytest.mark.parametrize(
    "mismatch",
    ["body", "control", "vertical_velocity", "density", "ambient_flow"],
)
def test_non_equivalent_creatures_use_the_complete_canonical_fallback(mismatch: str) -> None:
    ids = ("root-only", "swimmer") if mismatch == "body" else ("root-only", "root-only")
    world = _world(ids, interval_s=1.0)
    if mismatch == "control":
        world.live_state.turn_bias_rad_per_depth[0, 1] = 0.1
    if mismatch == "vertical_velocity":
        world.live_state.velocity_rel_water_enu_m_s[0, 1, 2] = 0.1
    if mismatch == "density":
        world.fluid.density_kg_m3[0, 1] = 999.0
    if mismatch == "ambient_flow":
        world.fluid.velocity_enu_m_s[..., 0] = 0.1
    runner = HeadlessRunner(world, periodic_policy=_test_policy())

    tick = runner.advance()

    assert tick.mechanics_steps == 120
    assert tick.full_batch_mechanics_steps == 120
    assert tick.representative_mechanics_steps == 0
    assert tick.fast_forwarded_mechanics_steps == 0
    assert tick.periodic_error is None
    assert torch.allclose(
        world.live_state.gait_time_s,
        torch.full_like(world.live_state.gait_time_s, 1.0),
    )
    assert torch.isfinite(tick.mechanical_work_j).all()


def test_multi_world_ineligibility_uses_shape_correct_canonical_work() -> None:
    world = _world(("root-only", "root-only"), interval_s=1.0, worlds=2)

    tick = HeadlessRunner(world, periodic_policy=_test_policy()).advance()

    assert tick.full_batch_mechanics_steps == 120
    assert tick.fast_forwarded_mechanics_steps == 0
    assert tick.mechanical_work_j.shape == (2, 1)
    assert torch.equal(tick.mechanical_work_j, torch.zeros((2, 1), dtype=torch.float64))
    assert torch.allclose(
        world.live_state.gait_time_s,
        torch.ones_like(world.live_state.gait_time_s),
    )


def test_repeated_transform_closes_a_full_circle_instead_of_teleporting() -> None:
    translation, yaw = repeat_transform(
        torch.tensor([[1.0, 0.0]], dtype=torch.float64),
        torch.tensor([torch.pi / 2], dtype=torch.float64),
        4,
    )

    assert torch.allclose(translation, torch.zeros_like(translation), atol=1.0e-15)
    assert torch.allclose(yaw, torch.tensor([2.0 * torch.pi], dtype=torch.float64))


def _synthetic_error(**overrides: float) -> PeriodicErrorEstimate:
    values = {
        "max_relative_recurrence_error": 0.0,
        "accumulated_translation_error_m": 0.0,
        "accumulated_yaw_error_rad": 0.0,
        "projected_relative_state_error": 0.0,
        "projected_velocity_error_m_s": 0.0,
        "projected_yaw_momentum_error_kg_m2_s": 0.0,
        "projected_relative_work_error": 0.0,
    }
    values.update(overrides)
    return PeriodicErrorEstimate(**values)


def test_every_pair_in_the_consecutive_window_must_meet_the_complete_policy(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    world = _world(("root-only", "root-only"), interval_s=4.0)
    errors = iter(
        [_synthetic_error(projected_relative_work_error=0.1)]
        + [_synthetic_error()] * 4
    )
    monkeypatch.setattr(
        periodic_motion,
        "_accumulated_error",
        lambda current, previous, skipped_cycles: next(errors),
    )
    policy = replace(
        _test_policy(),
        max_detection_cycles=6,
        required_consecutive_cycles=4,
    )

    tick = HeadlessRunner(world, periodic_policy=policy).advance()

    # The first bad-work pair clears the window. Authorization waits for the four
    # later complete-policy pairs instead of counting recurrence alone.
    assert tick.representative_mechanics_steps == 6 * 60
    assert tick.fast_forwarded_mechanics_steps == 60
    assert tick.full_batch_mechanics_steps == 60


def test_projected_dynamic_state_budget_blocks_periodic_authorization(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    world = _world(("root-only", "root-only"), interval_s=4.0)
    monkeypatch.setattr(
        periodic_motion,
        "_accumulated_error",
        lambda current, previous, skipped_cycles: _synthetic_error(
            projected_velocity_error_m_s=1.0
        ),
    )

    tick = HeadlessRunner(world, periodic_policy=_test_policy()).advance()

    assert tick.full_batch_mechanics_steps == 480
    assert tick.representative_mechanics_steps == 0
    assert tick.fast_forwarded_mechanics_steps == 0
    assert tick.periodic_error is None


def test_mature_swimmer_fast_forward_matches_a_complete_canonical_run() -> None:
    fast = _world(("swimmer",), interval_s=4.0)
    full = _world(("swimmer",), interval_s=4.0)
    # Cycle-64 body-frame state from the exploratory full-physics convergence probe.
    # It seeds a periodic boundary; the expected trajectory still comes from an
    # independent canonical run below, not from these values.
    for world in (fast, full):
        world.live_state.velocity_rel_water_enu_m_s[..., 0] = 7.04090886156998
        world.live_state.velocity_rel_water_enu_m_s[..., 1] = -0.2721492596680295
        world.live_state.yaw_momentum_kg_m2_s.fill_(36.82010625919977)
    policy = replace(
        _test_policy(),
        relative_tolerance=1.0e-8,
        max_accumulated_translation_error_m=1.0e-5,
        max_accumulated_yaw_error_rad=1.0e-7,
        max_projected_relative_state_error=1.0e-5,
        max_projected_velocity_error_m_s=1.0e-5,
        max_projected_yaw_momentum_error_kg_m2_s=1.0e-5,
        max_projected_relative_work_error=1.0e-8,
    )

    accelerated = HeadlessRunner(fast, periodic_policy=policy).advance()
    full_work = torch.zeros((1, 1), dtype=torch.float64)
    for _ in range(480):
        ledger = full._step_mechanics()
        full_work += ledger.total.dissipated_power_w * full.live_config.dt
    full._step_economy()

    assert accelerated.fast_forwarded_mechanics_steps > 0
    assert accelerated.periodic_error is not None
    assert torch.allclose(
        fast.live_state.position_enu_m,
        full.live_state.position_enu_m,
        rtol=0.0,
        atol=1.0e-8,
    )
    assert torch.allclose(
        fast.live_state.velocity_rel_water_enu_m_s,
        full.live_state.velocity_rel_water_enu_m_s,
        rtol=0.0,
        atol=1.0e-8,
    )
    assert torch.allclose(fast.live_state.yaw_rad, full.live_state.yaw_rad, atol=1.0e-9)
    assert torch.allclose(
        fast.live_state.yaw_momentum_kg_m2_s,
        full.live_state.yaw_momentum_kg_m2_s,
        rtol=0.0,
        atol=1.0e-8,
    )
    assert torch.allclose(accelerated.mechanical_work_j, full_work, rtol=1.0e-10)
    error = accelerated.periodic_error
    assert error.accumulated_translation_error_m >= float(
        torch.linalg.vector_norm(
            fast.live_state.position_enu_m - full.live_state.position_enu_m,
            dim=-1,
        ).max()
    )
    assert error.accumulated_yaw_error_rad >= float(
        (fast.live_state.yaw_rad - full.live_state.yaw_rad).abs().max()
    )
    assert error.projected_velocity_error_m_s >= float(
        torch.linalg.vector_norm(
            fast.live_state.velocity_rel_water_enu_m_s
            - full.live_state.velocity_rel_water_enu_m_s,
            dim=-1,
        ).max()
    )
    assert error.projected_yaw_momentum_error_kg_m2_s >= float(
        (
            fast.live_state.yaw_momentum_kg_m2_s
            - full.live_state.yaw_momentum_kg_m2_s
        ).abs().max()
    )
    assert error.projected_relative_work_error >= float(
        ((accelerated.mechanical_work_j - full_work).abs() / full_work.abs()).max()
    )


def test_rotated_translated_clones_and_remainder_match_canonical_mechanics() -> None:
    interval_s = 4.025  # Eight whole gait cycles plus three canonical substeps.
    fast = _world(("swimmer", "swimmer"), interval_s=interval_s)
    full = _world(("swimmer", "swimmer"), interval_s=interval_s)
    for world in (fast, full):
        world.live_state.position_enu_m[0, 0, :2] = world.live_state.position_enu_m.new_tensor(
            [1.0, 2.0]
        )
        world.live_state.position_enu_m[0, 1, :2] = world.live_state.position_enu_m.new_tensor(
            [7.0, 8.0]
        )
        world.live_state.yaw_rad[0, 1] = torch.pi / 2
        body_velocity = world.live_state.velocity_rel_water_enu_m_s.new_tensor(
            [7.04090886156998, -0.2721492596680295]
        )
        world.live_state.velocity_rel_water_enu_m_s[0, 0, :2] = body_velocity
        world.live_state.velocity_rel_water_enu_m_s[0, 1, :2] = body_velocity.new_tensor(
            [-body_velocity[1], body_velocity[0]]
        )
        world.live_state.yaw_momentum_kg_m2_s.fill_(36.82010625919977)
    policy = replace(
        _test_policy(),
        relative_tolerance=1.0e-8,
        max_accumulated_translation_error_m=1.0e-5,
        max_accumulated_yaw_error_rad=1.0e-7,
        max_projected_relative_state_error=1.0e-5,
        max_projected_velocity_error_m_s=1.0e-5,
        max_projected_yaw_momentum_error_kg_m2_s=1.0e-5,
        max_projected_relative_work_error=1.0e-8,
    )

    accelerated = HeadlessRunner(fast, periodic_policy=policy).advance()
    full_work = torch.zeros((1, 2), dtype=torch.float64)
    for _ in range(483):
        ledger = full._step_mechanics()
        full_work += ledger.total.dissipated_power_w.reshape_as(full_work) * full.live_config.dt
    full._step_economy()

    assert accelerated.fast_forwarded_mechanics_steps > 0
    assert torch.allclose(fast.live_state.position_enu_m, full.live_state.position_enu_m, atol=1.0e-8)
    assert torch.allclose(
        fast.live_state.velocity_rel_water_enu_m_s,
        full.live_state.velocity_rel_water_enu_m_s,
        atol=1.0e-8,
    )
    assert torch.allclose(fast.live_state.yaw_rad, full.live_state.yaw_rad, atol=1.0e-9)
    assert torch.allclose(
        fast.live_state.yaw_momentum_kg_m2_s,
        full.live_state.yaw_momentum_kg_m2_s,
        atol=1.0e-8,
    )
    assert torch.allclose(accelerated.mechanical_work_j, full_work, rtol=1.0e-10)


def test_failed_recurrence_probe_rolls_back_before_complete_fallback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    candidate = _world(("swimmer",), interval_s=4.0)
    canonical = _world(("swimmer",), interval_s=4.0)
    monkeypatch.setattr(
        periodic_motion,
        "_accumulated_error",
        lambda current, previous, skipped_cycles: _synthetic_error(
            projected_relative_state_error=1.0
        ),
    )

    fallback = HeadlessRunner(candidate, periodic_policy=_test_policy()).advance()
    canonical_work = torch.zeros((1, 1), dtype=torch.float64)
    for _ in range(480):
        ledger = canonical._step_mechanics()
        canonical_work += (
            ledger.total.dissipated_power_w.reshape_as(canonical_work)
            * canonical.live_config.dt
        )
    canonical._step_economy()

    assert fallback.full_batch_mechanics_steps == 480
    assert fallback.fast_forwarded_mechanics_steps == 0
    assert torch.equal(candidate.live_state.position_enu_m, canonical.live_state.position_enu_m)
    assert torch.equal(
        candidate.live_state.velocity_rel_water_enu_m_s,
        canonical.live_state.velocity_rel_water_enu_m_s,
    )
    assert torch.equal(candidate.live_state.yaw_rad, canonical.live_state.yaw_rad)
    assert torch.equal(
        candidate.live_state.yaw_momentum_kg_m2_s,
        canonical.live_state.yaw_momentum_kg_m2_s,
    )
    assert torch.equal(fallback.mechanical_work_j, canonical_work)


@pytest.mark.parametrize(
    "overrides",
    [
        {"max_detection_cycles": 1},
        {"required_consecutive_cycles": 0},
        {"relative_tolerance": 0.0},
        {"max_accumulated_translation_error_m": 0.0},
        {"max_accumulated_yaw_error_rad": 0.0},
        {"max_projected_relative_state_error": 0.0},
        {"max_projected_velocity_error_m_s": 0.0},
        {"max_projected_yaw_momentum_error_kg_m2_s": 0.0},
        {"max_projected_relative_work_error": 0.0},
    ],
)
def test_periodic_policy_rejects_malformed_error_contracts(overrides: dict[str, float]) -> None:
    values = {
        "max_detection_cycles": 64,
        "required_consecutive_cycles": 4,
        "relative_tolerance": 1.0e-9,
        "max_accumulated_translation_error_m": 0.1,
        "max_accumulated_yaw_error_rad": 1.0e-3,
        "max_projected_relative_state_error": 1.0e-4,
        "max_projected_velocity_error_m_s": 1.0e-3,
        "max_projected_yaw_momentum_error_kg_m2_s": 1.0e-3,
        "max_projected_relative_work_error": 1.0e-4,
    }
    values.update(overrides)

    with pytest.raises(ValueError):
        PeriodicMotionPolicy(**values)  # type: ignore[arg-type]
