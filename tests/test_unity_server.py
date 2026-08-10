"""Read-only Unity snapshots report the live lifecycle rather than fixture constants."""

from __future__ import annotations

import torch

from sirrobin.core.metabolism import MaintenanceConfig
from sirrobin.core.reproduction import BirthConfig
from sirrobin.core.runner import HeadlessRunner
from tools.run_world import _build_fixture_world
from tools.serve_unity import (
    _build_server_runner,
    _build_server_world,
    _descriptor,
    _payload,
    _seed_visible_baseline,
)


def test_snapshot_reports_authoritative_population_birth_and_death() -> None:
    world = _build_fixture_world(
        bodies=3,
        live_bodies=2,
        reserve_q_per_creature=0,
        device=torch.device("cpu"),
        economy_interval_s=0.1,
    )
    world.creature_material.reserve_q[0, 0] = 2_000
    world.creature_material.reserve_q[0, 1] = 3
    world.economy_state.nd_q[0, 0, 0, 0] -= 2_003
    runner = HeadlessRunner(
        world,
        maintenance_config=MaintenanceConfig(10.0),
        birth_config=BirthConfig(initial_reserve_q=100),
    )

    tick = runner.advance()
    birth = next(report for report in tick.births if report.born)
    assert birth.child_id is not None
    payload = _payload(
        world,
        tick,
        parent_by_id={1: None, birth.child_id: birth.parent_id},
        born_at_s={1: 0.0, birth.child_id: world.sim_time_s},
    )

    assert payload["population"] == 2
    assert {creature["id"] for creature in payload["creatures"]} == {
        1,
        birth.child_id,
    }
    assert payload["births"] == 1
    assert payload["deaths"] == 1
    assert payload["events"] == [
        "creature 2 died: starvation",
        f"creature 1 reproduced: child {birth.child_id}",
    ]
    assert all(isinstance(event, str) for event in payload["events"])
    assert all(creature["reserve"] >= 0 for creature in payload["creatures"])
    assert payload["energy"]["stored_chemical_j"] > 0.0


def test_snapshot_does_not_render_inactive_capacity_slots() -> None:
    world = _build_fixture_world(
        bodies=8,
        live_bodies=1,
        device=torch.device("cpu"),
        economy_interval_s=0.1,
    )

    payload = _payload(
        world,
        None,
        parent_by_id={1: None},
        born_at_s={1: 0.0},
    )

    assert payload["population"] == 1
    assert len(payload["creatures"]) == 1
    assert payload["births"] == 0
    assert payload["deaths"] == 0


def test_descriptor_retains_the_existing_unity_protocol() -> None:
    world = _build_fixture_world(
        bodies=1,
        device=torch.device("cpu"),
        economy_interval_s=0.1,
    )

    descriptor = _descriptor(world)

    assert descriptor["protocol"] == "sirrobin-observability/1"
    assert descriptor["record_types"] == [
        {"key": "snapshot.render", "label": "Render snapshot", "priority": 0}
    ]


def test_live_world_is_spacious_without_diluting_the_local_field_cells() -> None:
    world = _build_server_world()
    descriptor = _descriptor(world)

    assert world.economy_config.shape == (1, 6, 6, 4)
    assert world.geometry.lx_m == 60.0
    assert world.geometry.ly_m == 60.0
    assert world.geometry.lz_m == 20.0
    assert world.geometry.cell_volume_m3 == 500.0
    assert descriptor["configuration"]["world"] == {
        "width_m": 60.0,
        "height_m": 60.0,
        "depth_m": 20.0,
        "grid_cols": 6,
        "grid_rows": 6,
        "grid_layers": 4,
    }


def test_live_world_has_exact_patchy_food_without_changing_inventory() -> None:
    world = _build_server_world()
    _seed_visible_baseline(world)
    expected_uniform_total_q = world.economy_state.bp_q.numel() * 1_000_000

    assert int(world.economy_state.bp_q.sum()) == expected_uniform_total_q
    assert int(world.economy_state.bp_q.min()) < 1_000_000
    assert int(world.economy_state.bp_q.max()) > 1_000_000
    assert torch.equal(world.matter_totals().total_q, world.expected_matter_total_q)

    tick = _build_server_runner(world).advance()
    payload = _payload(
        world,
        tick,
        parent_by_id={stable_id: None for stable_id in range(1, 9)},
        born_at_s={stable_id: 0.0 for stable_id in range(1, 9)},
    )

    assert tick.food_seeking is None
    assert tick.matter.books_closed.tolist() == [True]
    assert len(payload["producer_grid"]) == 6
    assert all(len(row) == 6 for row in payload["producer_grid"])
    assert len({value for row in payload["producer_grid"] for value in row}) > 1
    assert sum(sum(row) for row in payload["producer_grid"]) == int(
        world.economy_state.bp_q.sum()
    )
    assert len(payload["dissolved_grid"]) == 6
    assert all(len(row) == 6 for row in payload["dissolved_grid"])
