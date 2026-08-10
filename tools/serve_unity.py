#!/usr/bin/env python3
"""Stream the current headless world to the existing read-only Unity viewer."""

from __future__ import annotations

import json
import math
import socket
import time
from dataclasses import replace

import torch

from sirrobin.core.periodic_motion import DEFAULT_PERIODIC_MOTION_POLICY
from sirrobin.core.runner import HeadlessRunner, WorldTick
from sirrobin.economy.config import EconomyConfig
from sirrobin.physics.pose_live import resolve_live_pose
from tools.run_world import (
    FIXTURE_BIRTH_CONFIG,
    FIXTURE_FEEDING_CONFIG,
    FIXTURE_MAINTENANCE_CONFIG,
    _build_fixture_world,
)

HOST = "127.0.0.1"
PORT = 8765
CAPACITY = 32
INITIAL_BODIES = 8
DISPLAY_BODIES = 32
ECONOMY_INTERVAL_S = 0.1
SESSION_ID = "original-baseline-live"
MODULE_DISPLAY_SCALE = 1.0 / 35.0
VIEW_WIDTH_M = 60.0
VIEW_HEIGHT_M = 60.0
VIEW_DEPTH_M = 20.0


def _line(value: object) -> bytes:
    return (json.dumps(value, separators=(",", ":"), allow_nan=False) + "\n").encode()


def _descriptor(world) -> dict[str, object]:
    config = world.economy_config
    return {
        "kind": "session",
        "protocol": "sirrobin-observability/1",
        "session_id": SESSION_ID,
        "simulation": "sirrobin-original-baseline",
        "metrics": [],
        "record_types": [
            {"key": "snapshot.render", "label": "Render snapshot", "priority": 0}
        ],
        "relationship_types": [],
        "configuration": {
            "world": {
                "width_m": VIEW_WIDTH_M,
                "height_m": VIEW_HEIGHT_M,
                "depth_m": VIEW_DEPTH_M,
                "grid_cols": config.gx,
                "grid_rows": config.gy,
                "grid_layers": config.gz,
            },
            "notice": "live read-only compatibility stream; not a persistence schema",
        },
    }


def _module_sets(world) -> dict[int, list[dict[str, float]]]:
    pose = resolve_live_pose(
        world.body,
        world.live_state.gait_time_s,
        world.live_state.turn_bias_rad_per_depth,
    )
    result: dict[int, list[dict[str, float]]] = {}
    for slot in world.body.alive[0].nonzero().flatten().tolist():
        modules = []
        for index in world.body.seg_mask[0, slot].nonzero().flatten().tolist():
            qx, qy, qz, qw = pose.rot_flu[slot, index].detach().cpu().tolist()
            orient = math.atan2(
                2.0 * (qw * qz + qx * qy),
                1.0 - 2.0 * (qy * qy + qz * qz),
            )
            axes = world.body.semi_axes_flu_m[0, slot, index]
            position = pose.pos_flu_m[slot, index]
            modules.append(
                {
                    "a": float(axes[0]) * MODULE_DISPLAY_SCALE,
                    "b": float(axes[1]) * MODULE_DISPLAY_SCALE,
                    "c": float(axes[2]) * MODULE_DISPLAY_SCALE,
                    "cx": float(position[0]) * MODULE_DISPLAY_SCALE,
                    "cy": float(position[1]) * MODULE_DISPLAY_SCALE,
                    "cz": float(position[2]) * MODULE_DISPLAY_SCALE,
                    "orient": orient,
                }
            )
        result[slot] = modules
    return result


def _seed_visible_baseline(world) -> None:
    """Deterministically separate the initial clones in the periodic world."""
    generator = torch.Generator(device="cpu").manual_seed(20260809)
    dtype = world.live_state.position_enu_m.dtype
    samples = torch.rand((world.body.capacity, 3), dtype=dtype, generator=generator)
    positions = world.live_state.position_enu_m[0]
    positions[:, 0] = samples[:, 0] * world.geometry.lx_m
    positions[:, 1] = samples[:, 1] * world.geometry.ly_m
    world.live_state.yaw_rad[0] = (2.0 * samples[:, 2] - 1.0) * math.pi


def _build_server_world():
    """Build a spacious world without changing the fixture's local cell scale."""
    economy = replace(
        EconomyConfig(),
        gx=6,
        gy=6,
        gz=4,
        lx_m=VIEW_WIDTH_M,
        ly_m=VIEW_HEIGHT_M,
        lz_m=VIEW_DEPTH_M,
        dt_eco_s=ECONOMY_INTERVAL_S,
        remin_floor_s=1.0e-4,
    )
    world = _build_fixture_world(
        bodies=CAPACITY,
        live_bodies=INITIAL_BODIES,
        device=torch.device("cpu"),
        economy_interval_s=ECONOMY_INTERVAL_S,
        economy_config=economy,
    )
    # An exact zero-sum redistribution makes local gradients visible without
    # minting nutrient or changing the fixture's mean producer concentration.
    band = torch.tensor([-3, -2, -1, 1, 2, 3], dtype=torch.int64)
    pattern = torch.stack(
        [torch.roll(band, shifts=column) for column in range(economy.gy)],
        dim=1,
    )
    world.economy_state.bp_q.add_(pattern[None, :, :, None] * 100_000)
    world.economy_state.validate(world.economy_config)
    if not torch.equal(world.matter_totals().total_q, world.expected_matter_total_q):
        raise RuntimeError("server producer pattern changed the exact matter inventory")
    return world


def _build_server_runner(world) -> HeadlessRunner:
    return HeadlessRunner(
        world,
        periodic_policy=DEFAULT_PERIODIC_MOTION_POLICY,
        feeding_config=FIXTURE_FEEDING_CONFIG,
        maintenance_config=FIXTURE_MAINTENANCE_CONFIG,
        birth_config=FIXTURE_BIRTH_CONFIG,
    )


def _events(tick: WorldTick | None) -> list[str]:
    if tick is None:
        return []
    events: list[str] = []
    for maintenance in tick.maintenance:
        if maintenance.starved:
            events.append(f"creature {maintenance.creature_id} died: starvation")
    for birth in tick.births:
        if birth.born:
            events.append(
                f"creature {birth.parent_id} reproduced: child {birth.child_id}"
            )
    refused = sum(birth.reason == "slot_exhausted" for birth in tick.births)
    if refused:
        events.append(f"{refused} funded birth attempts refused: capacity exhausted")
    return events


def _horizontal_grid(reservoir: torch.Tensor) -> list[list[float]]:
    """Project one world's depth layers into viewer rows (y) by columns (x)."""
    return reservoir[0].sum(dim=-1, dtype=torch.int64).T.to(torch.float64).tolist()


def _payload(
    world,
    tick: WorldTick | None,
    *,
    parent_by_id: dict[int, int | None],
    born_at_s: dict[int, float],
) -> dict[str, object]:
    module_sets = _module_sets(world)
    free_slot_exists = bool((~world.body.alive).any())
    creatures = []
    for slot in world.body.alive[0].nonzero().flatten().tolist()[:DISPLAY_BODIES]:
        stable_id = int(world.body.stable_id[0, slot])
        position = world.live_state.position_enu_m[0, slot]
        velocity = world.live_state.velocity_rel_water_enu_m_s[0, slot]
        speed = float(torch.linalg.vector_norm(velocity).item())
        reserve_q = int(world.creature_material.reserve_q[0, slot])
        structure_q = int(world.creature_material.structure_q[0, slot])
        birth_cost_q = structure_q + FIXTURE_BIRTH_CONFIG.initial_reserve_q
        funded = reserve_q >= birth_cost_q
        parent_id = parent_by_id.get(stable_id)
        creatures.append(
            {
                "id": stable_id,
                "lineage": (
                    f"founder-{stable_id}"
                    if parent_id is None
                    else f"clone-of-{parent_id}"
                ),
                "mass": float(world.body.mass_sim[0, slot].sum().item()),
                "x": float(position[0]) * VIEW_WIDTH_M / world.geometry.lx_m,
                "y": float(position[1]) * VIEW_HEIGHT_M / world.geometry.ly_m,
                "z": 0.5 * VIEW_DEPTH_M,
                "on_seabed": False,
                "age_s": world.sim_time_s - born_at_s.get(stable_id, 0.0),
                "yaw": float(world.live_state.yaw_rad[0, slot]),
                "yaw_rate": 0.0,
                "sideslip": 0.0,
                "actuating": speed > 0.0,
                "turning": False,
                "breeder": funded,
                "birth_ready": funded and free_slot_exists,
                "reproductive": min(1.0, reserve_q / max(1, birth_cost_q)),
                "reserve": float(reserve_q),
                "modules": module_sets[slot],
            }
        )

    births = 0 if tick is None else sum(report.born for report in tick.births)
    deaths = 0 if tick is None else sum(report.starved for report in tick.maintenance)
    producer_grid = _horizontal_grid(world.economy_state.bp_q)
    dissolved_grid = _horizontal_grid(world.economy_state.nd_q)
    producer_q = int(world.economy_state.bp_q.sum().item())
    reserve_q = int(world.creature_material.reserve_q.sum().item())
    dissipation_j = 0.0
    if tick is not None:
        dissipation_j = math.fsum(
            (
                float(tick.mechanical_work_j.sum().item()),
                *(report.assimilation_heat_j for report in (() if tick.feeding is None else (tick.feeding,))),
                *(report.maintenance_heat_j for report in tick.maintenance),
                *(report.death_dissipation_j for report in tick.maintenance),
                *(report.construction_heat_j for report in tick.births),
            )
        )
    return {
        "step": int(world.economy_state.step.item()),
        "time_s": world.sim_time_s,
        "population": int(world.body.alive.sum().item()),
        "creatures": creatures,
        "producer_grid": producer_grid,
        "zooplankton_grid": [
            [0.0 for _ in range(world.economy_config.gx)]
            for _ in range(world.economy_config.gy)
        ],
        "dissolved_grid": dissolved_grid,
        "births": births,
        "deaths": deaths,
        "events": _events(tick),
        "energy": {
            "stored_chemical_j": (
                producer_q * world.material_energy_config.producer_j_per_q
                + reserve_q * world.material_energy_config.reserve_j_per_q
            ),
            "kinetic_j": 0.0,
            "dissipation_j": dissipation_j,
            "light_input_j": 0.0,
        },
    }


def _record(
    world,
    sequence: int,
    tick: WorldTick | None,
    *,
    parent_by_id: dict[int, int | None],
    born_at_s: dict[int, float],
) -> dict[str, object]:
    step = int(world.economy_state.step.item())
    return {
        "kind": "record",
        "session_id": SESSION_ID,
        "sequence": sequence,
        "record_id": f"render:{step}",
        "record_type": "snapshot.render",
        "step": step,
        "time_s": world.sim_time_s,
        "subjects": [{"kind": "world", "id": "water-column", "label": "Water column"}],
        "links": [],
        "payload": _payload(
            world,
            tick,
            parent_by_id=parent_by_id,
            born_at_s=born_at_s,
        ),
        "provenance": {"bridge": "original-baseline-live"},
    }


def main() -> None:
    world = _build_server_world()
    _seed_visible_baseline(world)
    runner = _build_server_runner(world)
    parent_by_id = {
        int(stable_id): None
        for stable_id in world.body.stable_id[world.body.alive].tolist()
    }
    born_at_s = {stable_id: 0.0 for stable_id in parent_by_id}
    sequence = 0
    with socket.create_server((HOST, PORT), reuse_port=False) as server:
        print(
            f"SirRobin Unity server listening on {HOST}:{PORT} "
            f"({INITIAL_BODIES}/{CAPACITY} live/capacity)",
            flush=True,
        )
        while True:
            connection, address = server.accept()
            print(f"Unity client connected from {address[0]}:{address[1]}", flush=True)
            try:
                connection.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
                connection.settimeout(2.0)
                request = connection.makefile("rb").readline()
                print(f"client request: {request.decode(errors='replace').strip()}", flush=True)
                after_sequence = int(json.loads(request)["after_sequence"])
                sequence = max(sequence, after_sequence)
                connection.settimeout(None)
                connection.sendall(_line(_descriptor(world)))
                sequence += 1
                connection.sendall(
                    _line(
                        _record(
                            world,
                            sequence,
                            None,
                            parent_by_id=parent_by_id,
                            born_at_s=born_at_s,
                        )
                    )
                )
                stream_started = time.perf_counter()
                stream_sim_started = world.sim_time_s
                frames_sent = 0
                while True:
                    tick = runner.advance()
                    for birth in tick.births:
                        if birth.born and birth.child_id is not None:
                            parent_by_id[birth.child_id] = birth.parent_id
                            born_at_s[birth.child_id] = world.sim_time_s
                    sequence += 1
                    frames_sent += 1
                    connection.sendall(
                        _line(
                            _record(
                                world,
                                sequence,
                                tick,
                                parent_by_id=parent_by_id,
                                born_at_s=born_at_s,
                            )
                        )
                    )
                    if frames_sent % 50 == 0:
                        elapsed = time.perf_counter() - stream_started
                        print(
                            f"step={int(world.economy_state.step)} "
                            f"sim={world.sim_time_s:.1f}s "
                            f"population={int(world.body.alive.sum())} "
                            f"rate={(world.sim_time_s - stream_sim_started) / elapsed:.2f} "
                            "sim-s/wall-s",
                            flush=True,
                        )
            except (BrokenPipeError, ConnectionResetError, TimeoutError, OSError) as error:
                print(f"Unity client disconnected: {error}", flush=True)
            finally:
                connection.close()


if __name__ == "__main__":
    main()
