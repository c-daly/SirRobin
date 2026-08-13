from __future__ import annotations

from dataclasses import replace

import torch

from sirrobin.observe.runtime_snapshot import (
    stage_runtime_events,
    stage_runtime_snapshot,
)
from sirrobin.organisms.mortality import MortalityConfig
from sirrobin.physics.pose_live import resolve_live_pose
from sirrobin.runtime.step import advance_living_interval
from tools.benchmark_device_runtime import build_fixture


def test_initial_runtime_snapshot_is_host_owned_and_capacity_bounded() -> None:
    state, _, config = build_fixture(3, torch.device("cpu"))

    snapshot = stage_runtime_snapshot(state, config, None)

    assert snapshot.step == 0
    assert snapshot.time_s == 0.0
    assert snapshot.alive.tolist() == [[True, True, True]]
    assert snapshot.position_enu_m.shape == (1, 3, 3)
    assert snapshot.segment_position_flu_m.shape[:2] == (1, 3)
    assert snapshot.producer_grid_q.shape == (
        1,
        config.economy.gy,
        config.economy.gx,
    )
    assert snapshot.position_enu_m.device.type == "cpu"
    assert not bool(snapshot.accepted_effort_fraction.any())


def test_runtime_snapshot_pose_uses_physically_selected_effort() -> None:
    state, inputs, config = build_fixture(1, torch.device("cpu"))
    inputs = replace(
        inputs,
        requested_effort=torch.zeros_like(inputs.requested_effort),
    )
    advance = advance_living_interval(state, inputs, config)

    snapshot = stage_runtime_snapshot(advance.state, config, advance.ledger)
    selected = advance.ledger.motion.ledger.selected.effort_fraction
    expected = resolve_live_pose(
        advance.state.body,
        advance.state.motion.gait_time_s,
        advance.state.motion.turn_bias_rad_per_depth,
        effort=selected,
    )
    misleading_full_effort = resolve_live_pose(
        advance.state.body,
        advance.state.motion.gait_time_s,
        advance.state.motion.turn_bias_rad_per_depth,
    )

    assert not bool(selected.any())
    assert torch.equal(snapshot.accepted_effort_fraction, selected.cpu())
    assert torch.allclose(
        snapshot.segment_position_flu_m,
        expected.pos_flu_m.reshape_as(snapshot.segment_position_flu_m),
    )
    assert not torch.allclose(
        snapshot.segment_position_flu_m,
        misleading_full_effort.pos_flu_m.reshape_as(
            snapshot.segment_position_flu_m
        ),
    )


def test_runtime_snapshot_retains_death_identity_and_named_energy() -> None:
    state, inputs, config = build_fixture(3, torch.device("cpu"))
    config = replace(config, mortality=MortalityConfig(0.05, 0.05, seed=7))
    inputs = replace(inputs, birth_requested=torch.zeros_like(inputs.birth_requested))
    advance = advance_living_interval(state, inputs, config)

    snapshot = stage_runtime_snapshot(advance.state, config, advance.ledger)
    events = stage_runtime_events(advance.state, advance.ledger)

    assert snapshot.alive.tolist() == [[False, False, False]]
    assert snapshot.died.tolist() == [[True, True, True]]
    assert snapshot.death_stable_id.tolist() == [[1, 2, 3]]
    assert snapshot.old_age.tolist() == [[True, True, True]]
    assert not bool(snapshot.starved.any())
    assert snapshot.interval_dissipation_j >= 0.0
    assert snapshot.interval_light_input_j >= 0.0
    assert events.death_stable_id.tolist() == [[1, 2, 3]]
    assert events.interval_dissipation_j == snapshot.interval_dissipation_j
