#!/usr/bin/env python3
"""Stream the current headless world to the existing read-only Unity viewer."""

from __future__ import annotations

import argparse
import json
import math
import signal
import socket
import time
from dataclasses import dataclass, replace

import torch

from sirrobin.core.mortality import AgeMortalityConfig
from sirrobin.core.reproduction import ParametricMutationConfig
from sirrobin.core.runner import HeadlessRunner, WorldTick
from sirrobin.economy.config import EconomyConfig
from sirrobin.physics.pose_live import resolve_live_pose
from tools.run_world import (
    FIXTURE_BIRTH_CONFIG,
    FIXTURE_FEEDING_CONFIG,
    FIXTURE_MAINTENANCE_CONFIG,
    LIVING_MATERIAL_ENERGY_CONFIG,
    _build_fixture_world,
)
from tools.runtime_unity import (
    EVOLUTION_DEMO_RUNTIME_PROFILE,
    RUNTIME_UNITY_PROFILES,
    RuntimeObservationTotals,
    RuntimeUnityBackend,
    RuntimeUnityProfile,
    RuntimeVisualLineages,
    runtime_events,
    runtime_payload,
)

HOST = "127.0.0.1"
PORT = 8765
CAPACITY = 64
INITIAL_BODIES = 8
LIVE_INITIAL_RESERVE_Q = 2
LIVE_RICH_FOOD_CELL_Q = 2_000_000
DISPLAY_BODIES = CAPACITY
ECONOMY_INTERVAL_S = 0.1
STREAM_EVERY_STEPS = 1
HEARTBEAT_INTERVAL_S = 5.0
EXTINCTION_EVENT = "extinction: population reached zero"
SESSION_ID = "original-baseline-live"
MODULE_DISPLAY_SCALE = 1.0 / 35.0
VIEW_WIDTH_M = 60.0
VIEW_HEIGHT_M = 60.0
VIEW_DEPTH_M = 20.0
LIVE_MUTATION_CONFIG = ParametricMutationConfig(seed=20260810)
LIVE_AGE_MORTALITY_CONFIG = AgeMortalityConfig(
    min_lifespan_s=60.0,
    max_lifespan_s=100.0,
    seed=20260810,
)


@dataclass(frozen=True, slots=True)
class _PendingTerminalRecord:
    """Immutable terminal observation retained until transport accepts it."""

    reason: str
    record: bytes


class _TerminalDeliveryPending(RuntimeError):
    """Signal that a terminal record must remain pending across connections."""

    def __init__(self, reason: str, record: bytes) -> None:
        super().__init__(f"{reason} record delivery remains pending")
        self.pending = _PendingTerminalRecord(reason, record)


@dataclass(slots=True)
class _StreamCursor:
    """Monotonic live-record identity retained across client reconnects."""

    last_sequence: int = 0

    def resume_after(self, accepted_sequence: int) -> None:
        if (
            isinstance(accepted_sequence, bool)
            or not isinstance(accepted_sequence, int)
            or accepted_sequence < 0
        ):
            raise ValueError("after_sequence must be a nonnegative integer")
        self.last_sequence = max(self.last_sequence, accepted_sequence)

    def next(self) -> int:
        self.last_sequence += 1
        return self.last_sequence


def _line(value: object) -> bytes:
    return (json.dumps(value, separators=(",", ":"), allow_nan=False) + "\n").encode()


def _send_record(
    connection,
    message: bytes,
    *,
    terminal_reason: str | None = None,
) -> None:
    try:
        connection.sendall(message)
    except (BrokenPipeError, ConnectionResetError, TimeoutError, OSError) as error:
        if terminal_reason is not None:
            raise _TerminalDeliveryPending(terminal_reason, message) from error
        raise


def _retry_terminal_record(
    connection,
    pending: _PendingTerminalRecord,
) -> str:
    """Replay the exact terminal observation; never reconstruct interval facts."""

    _send_record(
        connection,
        pending.record,
        terminal_reason=pending.reason,
    )
    return pending.reason


def _descriptor(
    world,
    *,
    profile: RuntimeUnityProfile | None = None,
) -> dict[str, object]:
    config = world.economy_config if hasattr(world, "economy_config") else world
    configuration: dict[str, object] = {
        "world": {
            "width_m": VIEW_WIDTH_M,
            "height_m": VIEW_HEIGHT_M,
            "depth_m": VIEW_DEPTH_M,
            "grid_cols": config.gx,
            "grid_rows": config.gy,
            "grid_layers": config.gz,
        },
        "notice": "live read-only compatibility stream; not a persistence schema",
    }
    if profile is not None:
        configuration["runtime_profile"] = {
            "name": profile.name,
            "description": profile.description,
            "min_lifespan_s": profile.mortality.min_lifespan_s,
            "max_lifespan_s": profile.mortality.max_lifespan_s,
            "mutation_rate_per_locus": profile.mutation.mutation_rate_per_locus,
        }
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
        "configuration": configuration,
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


def _seed_visible_baseline(world, *, seed: int = 20260809) -> None:
    """Deterministically separate the initial clones in the periodic world."""
    generator = torch.Generator(device="cpu").manual_seed(seed)
    dtype = world.live_state.position_enu_m.dtype
    samples = torch.rand(
        (world.body.capacity, 3), dtype=dtype, generator=generator
    ).to(world.body.alive.device)
    positions = world.live_state.position_enu_m[0]
    positions[:, 0] = samples[:, 0] * world.geometry.lx_m
    positions[:, 1] = samples[:, 1] * world.geometry.ly_m
    world.live_state.yaw_rad[0] = (2.0 * samples[:, 2] - 1.0) * math.pi


def _build_server_world(*, device: torch.device | None = None):
    """Build a spacious world without changing the fixture's local cell scale."""
    resolved_device = torch.device("cpu") if device is None else device
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
        device=resolved_device,
        economy_interval_s=ECONOMY_INTERVAL_S,
        economy_config=economy,
        material_energy_config=LIVING_MATERIAL_ENERGY_CONFIG,
        reserve_q_per_creature=LIVE_INITIAL_RESERVE_Q,
        physics_dtype=torch.float32,
    )
    # Retain producer only in three full-depth food patches. Availability remains
    # a local concentration derived from stock and the declared 500 m3 cell volume;
    # the exact surplus becomes dissolved nutrient rather than disappearing.
    producer_total_q = int(world.economy_state.bp_q.sum().item())
    patch_columns = ((1, 1), (3, 4), (5, 2))
    rich_cells = len(patch_columns) * economy.gz
    world.economy_state.bp_q.zero_()
    for x_index, y_index in patch_columns:
        world.economy_state.bp_q[0, x_index, y_index, :].fill_(
            LIVE_RICH_FOOD_CELL_Q
        )
    producer_surplus_q = producer_total_q - rich_cells * LIVE_RICH_FOOD_CELL_Q
    if producer_surplus_q < 0:
        raise RuntimeError("food patches exceed the available producer inventory")
    world.economy_state.nd_q[0, 0, 0, 0] += producer_surplus_q
    world.economy_state.validate(world.economy_config)
    if not torch.equal(world.matter_totals().total_q, world.expected_matter_total_q):
        raise RuntimeError("server producer pattern changed the exact matter inventory")
    return world


def _build_server_runner(world) -> HeadlessRunner:
    return HeadlessRunner(
        world,
        feeding_config=FIXTURE_FEEDING_CONFIG,
        maintenance_config=FIXTURE_MAINTENANCE_CONFIG,
        birth_config=FIXTURE_BIRTH_CONFIG,
        mutation_config=LIVE_MUTATION_CONFIG,
        age_mortality_config=LIVE_AGE_MORTALITY_CONFIG,
    )


def _events(world, tick: WorldTick | None) -> list[str]:
    if tick is None:
        return []
    events: list[str] = []
    for maintenance in tick.maintenance:
        if maintenance.death_cause is not None:
            label = maintenance.death_cause.replace("_", " ")
            events.append(f"creature {maintenance.creature_id} died: {label}")
    for birth in tick.births:
        if birth.born:
            if birth.mutation is None:
                events.append(
                    f"creature {birth.parent_id} reproduced: clone child {birth.child_id}"
                )
            else:
                mutation = birth.mutation
                locus = "".join(f"[{index}]" for index in mutation.index)
                events.append(
                    f"creature {birth.parent_id} reproduced: mutant child "
                    f"{birth.child_id}; {mutation.field_name}{locus} "
                    f"{mutation.parent_value:.6g}->{mutation.child_value:.6g}"
                )
    refused = sum(birth.reason == "slot_exhausted" for birth in tick.births)
    if refused:
        events.append(f"{refused} funded birth attempts refused: capacity exhausted")
    heartbeat_multiple = round(tick.sim_time_s / HEARTBEAT_INTERVAL_S)
    if heartbeat_multiple > 0 and math.isclose(
        tick.sim_time_s,
        heartbeat_multiple * HEARTBEAT_INTERVAL_S,
        rel_tol=0.0,
        abs_tol=1.0e-9,
    ):
        events.append(
            "heartbeat: "
            f"population={int(world.body.alive.sum().item())} "
            f"reserve_q={int(world.creature_material.reserve_q.sum().item())}"
        )
    return events


def _horizontal_grid(reservoir: torch.Tensor) -> list[list[float]]:
    """Project one world's depth layers into viewer rows (y) by columns (x)."""
    return reservoir[0].sum(dim=-1, dtype=torch.int64).T.to(torch.float64).tolist()


def _payload(
    world,
    tick: WorldTick | None,
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
        lineage = world.lineage_record(0, stable_id)
        if lineage.parent_id is None:
            lineage_label = f"founder-{stable_id}"
        elif lineage.mutation is None:
            lineage_label = f"clone-of-{lineage.parent_id}"
        else:
            lineage_label = f"mutant-of-{lineage.parent_id}"
        creatures.append(
            {
                "id": stable_id,
                "lineage": lineage_label,
                "mass": float(world.body.mass_sim[0, slot].sum().item()),
                "x": float(position[0]) * VIEW_WIDTH_M / world.geometry.lx_m,
                "y": float(position[1]) * VIEW_HEIGHT_M / world.geometry.ly_m,
                "z": 0.5 * VIEW_DEPTH_M,
                "on_seabed": False,
                "age_s": world.sim_time_s - lineage.born_at_s,
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
    deaths = 0 if tick is None else sum(
        report.death_cause is not None for report in tick.maintenance
    )
    producer_grid = _horizontal_grid(world.economy_state.bp_q)
    dissolved_grid = _horizontal_grid(world.economy_state.nd_q)
    producer_q = int(world.economy_state.bp_q.sum().item())
    reserve_q = int(world.creature_material.reserve_q.sum().item())
    assimilation_carry_j = (
        float(world.creature_material.assimilation_carry_q.sum().item())
        * world.material_energy_config.reserve_j_per_q
    )
    maintenance_liability_j = float(
        world.creature_material.maintenance_carry_j.sum().item()
    )
    dissipation_j = 0.0
    if tick is not None:
        dissipation_j = math.fsum(
            (
                float(tick.mechanical_work_j.sum().item()),
                *(report.assimilation_heat_j for report in (() if tick.feeding is None else (tick.feeding,))),
                *(report.baseline_maintenance_demand_j for report in tick.maintenance),
                *(report.muscle_inefficiency_heat_j for report in tick.maintenance),
                *(report.actuator_braking_heat_j for report in tick.maintenance),
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
        "events": _events(world, tick),
        "energy": {
            "stored_chemical_j": (
                producer_q * world.material_energy_config.producer_j_per_q
                + reserve_q * world.material_energy_config.reserve_j_per_q
                + assimilation_carry_j
                - maintenance_liability_j
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
    interval_events: list[str] | None = None,
    interval_births: int | None = None,
    interval_deaths: int | None = None,
    terminal_reason: str | None = None,
) -> dict[str, object]:
    step = int(world.economy_state.step.item())
    payload = _payload(world, tick)
    if interval_events is not None:
        payload["events"] = list(interval_events)
    if interval_births is not None:
        payload["births"] = interval_births
    if interval_deaths is not None:
        payload["deaths"] = interval_deaths
    if terminal_reason is not None:
        payload["terminal"] = {"reason": terminal_reason}
    return {
        "kind": "record",
        "session_id": SESSION_ID,
        "sequence": sequence,
        "record_id": f"render:sequence:{sequence}",
        "record_type": "snapshot.render",
        "step": step,
        "time_s": world.sim_time_s,
        "subjects": [{"kind": "world", "id": "water-column", "label": "Water column"}],
        "links": [],
        "payload": payload,
        "provenance": {"bridge": "original-baseline-live"},
    }


def _runtime_record(
    snapshot,
    config,
    sequence: int,
    *,
    interval_events: list[str] | None = None,
    interval_births: int | None = None,
    interval_deaths: int | None = None,
    interval_dissipation_j: float | None = None,
    interval_light_input_j: float | None = None,
    observation: RuntimeObservationTotals | None = None,
    visual_lineages: RuntimeVisualLineages | None = None,
    terminal_reason: str | None = None,
) -> dict[str, object]:
    payload = runtime_payload(
        snapshot,
        config,
        display_bodies=DISPLAY_BODIES,
        module_display_scale=MODULE_DISPLAY_SCALE,
        view_width_m=VIEW_WIDTH_M,
        view_height_m=VIEW_HEIGHT_M,
        view_depth_m=VIEW_DEPTH_M,
        interval_events=interval_events,
        interval_births=interval_births,
        interval_deaths=interval_deaths,
        interval_dissipation_j=interval_dissipation_j,
        interval_light_input_j=interval_light_input_j,
        observation=observation,
        visual_lineages=visual_lineages,
    )
    if terminal_reason is not None:
        payload["terminal"] = {"reason": terminal_reason}
    return {
        "kind": "record",
        "session_id": SESSION_ID,
        "sequence": sequence,
        "record_id": f"render:sequence:{sequence}",
        "record_type": "snapshot.render",
        "step": snapshot.step,
        "time_s": snapshot.time_s,
        "subjects": [
            {"kind": "world", "id": "water-column", "label": "Water column"}
        ],
        "links": [],
        "payload": payload,
        "provenance": {"bridge": "original-gpu-living-runtime"},
    }


def _stream_reference(
    connection,
    world,
    runner,
    cursor: _StreamCursor,
    *,
    stream_every_steps: int = STREAM_EVERY_STEPS,
) -> str | None:
    initially_extinct = not bool(world.body.alive.any())
    initial_events = [EXTINCTION_EVENT] if initially_extinct else None
    _send_record(
        connection,
        _line(
            _record(
                world,
                cursor.next(),
                None,
                interval_events=initial_events,
                terminal_reason="extinction" if initially_extinct else None,
            )
        ),
        terminal_reason="extinction" if initially_extinct else None,
    )
    if initially_extinct:
        return "extinction"
    stream_started = time.perf_counter()
    stream_sim_started = world.sim_time_s
    frames_sent = 0
    steps_since_frame = 0
    interval_events: list[str] = []
    interval_births = 0
    interval_deaths = 0
    while True:
        tick = runner.advance()
        steps_since_frame += 1
        events = _events(world, tick)
        interval_events.extend(events)
        interval_births += sum(report.born for report in tick.births)
        interval_deaths += sum(
            report.death_cause is not None for report in tick.maintenance
        )
        extinct = not bool(world.body.alive.any())
        if extinct and EXTINCTION_EVENT not in interval_events:
            interval_events.append(EXTINCTION_EVENT)
        if steps_since_frame < stream_every_steps and not extinct:
            continue
        sequence = cursor.next()
        frames_sent += 1
        _send_record(
            connection,
            _line(
                _record(
                    world,
                    sequence,
                    tick,
                    interval_events=interval_events,
                    interval_births=interval_births,
                    interval_deaths=interval_deaths,
                    terminal_reason="extinction" if extinct else None,
                )
            ),
            terminal_reason="extinction" if extinct else None,
        )
        if extinct:
            return "extinction"
        steps_since_frame = 0
        interval_events.clear()
        interval_births = 0
        interval_deaths = 0
        if frames_sent % 10 == 0:
            elapsed = time.perf_counter() - stream_started
            print(
                f"step={int(world.economy_state.step)} "
                f"sim={world.sim_time_s:.1f}s "
                f"population={int(world.body.alive.sum())} "
                f"rate={(world.sim_time_s - stream_sim_started) / elapsed:.2f} "
                "sim-s/wall-s",
                flush=True,
            )


def _stream_runtime(
    connection,
    backend: RuntimeUnityBackend,
    cursor: _StreamCursor,
    *,
    stream_every_steps: int = STREAM_EVERY_STEPS,
) -> str | None:
    snapshot = backend.snapshot()
    initially_extinct = not bool(snapshot.alive.any())
    _send_record(
        connection,
        _line(
            _runtime_record(
                snapshot,
                backend.config,
                cursor.next(),
                interval_events=[EXTINCTION_EVENT] if initially_extinct else None,
                observation=backend.observation,
                visual_lineages=backend.visual_lineages,
                terminal_reason="extinction" if initially_extinct else None,
            )
        ),
        terminal_reason="extinction" if initially_extinct else None,
    )
    if initially_extinct:
        return "extinction"
    stream_started = time.perf_counter()
    stream_sim_started = snapshot.time_s
    frames_sent = 0
    steps_since_frame = 0
    interval_events: list[str] = []
    interval_births = 0
    interval_deaths = 0
    interval_dissipation_j = 0.0
    interval_light_input_j = 0.0
    while True:
        event_snapshot = backend.advance_events()
        steps_since_frame += 1
        events = runtime_events(
            event_snapshot,
            backend.config,
            backend.observation,
        )
        interval_events.extend(events)
        interval_births += int(event_snapshot.born.sum())
        interval_deaths += int(event_snapshot.died.sum())
        interval_dissipation_j += event_snapshot.interval_dissipation_j
        interval_light_input_j += event_snapshot.interval_light_input_j
        for event in events:
            print(event, flush=True)
        extinct = not bool(event_snapshot.alive.any())
        if extinct and EXTINCTION_EVENT not in interval_events:
            interval_events.append(EXTINCTION_EVENT)
            print(EXTINCTION_EVENT, flush=True)
        if steps_since_frame < stream_every_steps and not extinct:
            continue
        snapshot = backend.snapshot()
        sequence = cursor.next()
        frames_sent += 1
        _send_record(
            connection,
            _line(
                _runtime_record(
                    snapshot,
                    backend.config,
                    sequence,
                    interval_events=interval_events,
                    interval_births=interval_births,
                    interval_deaths=interval_deaths,
                    interval_dissipation_j=interval_dissipation_j,
                    interval_light_input_j=interval_light_input_j,
                    observation=backend.observation,
                    visual_lineages=backend.visual_lineages,
                    terminal_reason="extinction" if extinct else None,
                )
            ),
            terminal_reason="extinction" if extinct else None,
        )
        if extinct:
            return "extinction"
        steps_since_frame = 0
        interval_events.clear()
        interval_births = 0
        interval_deaths = 0
        interval_dissipation_j = 0.0
        interval_light_input_j = 0.0
        if frames_sent % 10 == 0:
            elapsed = time.perf_counter() - stream_started
            print(
                f"step={snapshot.step} "
                f"sim={snapshot.time_s:.1f}s "
                f"population={int(snapshot.alive.sum())} "
                f"rate={(snapshot.time_s - stream_sim_started) / elapsed:.2f} "
                "sim-s/wall-s",
                flush=True,
            )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--runtime",
        choices=("device", "reference"),
        default="device",
        help="device is the cohesive GPU runtime; reference preserves the old runner",
    )
    parser.add_argument(
        "--device",
        choices=("cuda", "cpu"),
        default="cuda",
        help="tensor device for the cohesive runtime",
    )
    parser.add_argument(
        "--profile",
        choices=tuple(RUNTIME_UNITY_PROFILES),
        default=None,
        help=(
            "named Unity observation calibration; device defaults to "
            "evolution-demo and reference defaults to baseline"
        ),
    )
    parser.add_argument(
        "--fast-forward-seconds",
        type=float,
        default=0.0,
        help=(
            "advance a finite whole-interval horizon before accepting Unity; "
            "ordinary render snapshots are suppressed"
        ),
    )
    parser.add_argument(
        "--fast-forward-chunk-intervals",
        type=int,
        default=32,
        help="exact intervals per cancellable fast-forward chunk",
    )
    parser.add_argument(
        "--stream-every-steps",
        type=int,
        default=STREAM_EVERY_STEPS,
        help="coalesce this many exact simulation intervals into one Unity frame",
    )
    args = parser.parse_args()
    if args.stream_every_steps < 1:
        parser.error("--stream-every-steps must be positive")
    if args.fast_forward_chunk_intervals < 1:
        parser.error("--fast-forward-chunk-intervals must be positive")
    if args.fast_forward_seconds < 0.0 or not math.isfinite(
        args.fast_forward_seconds
    ):
        parser.error("--fast-forward-seconds must be finite and nonnegative")
    if args.runtime == "reference" and args.fast_forward_seconds > 0.0:
        parser.error("finite fast-forward is available on the device runtime")
    if args.runtime == "reference" and args.profile not in (None, "baseline"):
        parser.error("the reference runtime supports only --profile baseline")
    torch.set_num_threads(1)
    with torch.inference_mode():
        if args.runtime == "device":
            device = torch.device(args.device)
            profile = RUNTIME_UNITY_PROFILES[
                EVOLUTION_DEMO_RUNTIME_PROFILE.name
                if args.profile is None
                else args.profile
            ]
            if device.type == "cuda" and not torch.cuda.is_available():
                raise RuntimeError("CUDA is unavailable; use --device cpu explicitly")
            print(
                f"initializing cohesive device runtime on {device}; "
                f"profile={profile.name} ({profile.description})",
                flush=True,
            )
            world = _build_server_world(device=device)
            _seed_visible_baseline(world)
            backend = RuntimeUnityBackend.from_reference_fixture(
                world,
                profile=profile,
            )
            print("prewarming configured autonomous execution paths", flush=True)
            warmup_started = time.perf_counter()
            backend.prewarm()
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            warmup_elapsed_s = time.perf_counter() - warmup_started
            print(
                f"runtime warmup complete in {warmup_elapsed_s:.1f}s; "
                "authoritative state remains at step 0",
                flush=True,
            )
            if args.fast_forward_seconds > 0.0:
                cancel_fast_forward = False
                cancellation_announced = False

                def request_fast_forward_cancel(_signum, _frame) -> None:
                    nonlocal cancel_fast_forward, cancellation_announced
                    cancel_fast_forward = True
                    if not cancellation_announced:
                        print(
                            "fast-forward cancellation requested; finishing current "
                            "exact chunk",
                            flush=True,
                        )
                        cancellation_announced = True

                def show_fast_forward_progress(report) -> None:
                    print(
                        "fast-forward "
                        f"{report.completed_intervals}/{report.requested_intervals} "
                        f"intervals; sim={report.end_time_s:.1f}s "
                        f"births={report.births} deaths={report.deaths} "
                        f"mutations={report.mutation_events}",
                        flush=True,
                    )

                previous_sigint = signal.getsignal(signal.SIGINT)
                signal.signal(signal.SIGINT, request_fast_forward_cancel)
                try:
                    report = backend.fast_forward(
                        args.fast_forward_seconds,
                        chunk_intervals=args.fast_forward_chunk_intervals,
                        should_cancel=lambda: cancel_fast_forward,
                        progress=show_fast_forward_progress,
                    )
                finally:
                    signal.signal(signal.SIGINT, previous_sigint)
                final_snapshot = backend.snapshot()
                print(
                    "fast-forward "
                    f"{'cancelled' if report.cancelled else 'complete'}; "
                    f"step={final_snapshot.step} sim={final_snapshot.time_s:.1f}s "
                    f"population={int(final_snapshot.alive.sum())} "
                    f"births={report.births} deaths={report.deaths} "
                    f"mutations={report.mutation_events}",
                    flush=True,
                )
            runner = None
            descriptor_source = backend.config.economy
            descriptor_profile = profile
            runtime_label = f"cohesive device runtime on {device}"
            startup_label = "configured kernels prewarmed before accepting Unity"
        else:
            print("initializing preserved reference runner on cpu", flush=True)
            if args.device != "cuda":
                raise ValueError("--device applies only to --runtime device")
            world = _build_server_world()
            _seed_visible_baseline(world)
            world._step_mechanics = torch.compile(
                world._step_mechanics,
                dynamic=False,
            )
            runner = _build_server_runner(world)
            backend = None
            descriptor_source = world
            descriptor_profile = None
            runtime_label = "preserved reference runner on cpu"
            startup_label = "reference kernel compilation remains lazy"
        cursor = _StreamCursor()
        pending_terminal: _PendingTerminalRecord | None = None
        with socket.create_server((HOST, PORT), reuse_port=False) as server:
            print(
                f"SirRobin Unity server listening on {HOST}:{PORT} "
                f"({INITIAL_BODIES}/{CAPACITY} live/capacity; {runtime_label})",
                flush=True,
            )
            print(
                startup_label,
                flush=True,
            )
            while True:
                connection, address = server.accept()
                terminal_reason = None
                print(
                    f"Unity client connected from {address[0]}:{address[1]}",
                    flush=True,
                )
                try:
                    connection.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
                    connection.settimeout(2.0)
                    request = connection.makefile("rb").readline()
                    print(
                        f"client request: {request.decode(errors='replace').strip()}",
                        flush=True,
                    )
                    after_sequence = int(json.loads(request)["after_sequence"])
                    cursor.resume_after(after_sequence)
                    connection.settimeout(None)
                    connection.sendall(
                        _line(
                            _descriptor(
                                descriptor_source,
                                profile=descriptor_profile,
                            )
                        )
                    )
                    if pending_terminal is not None:
                        terminal_reason = _retry_terminal_record(
                            connection,
                            pending_terminal,
                        )
                        pending_terminal = None
                    elif backend is not None:
                        terminal_reason = _stream_runtime(
                            connection,
                            backend,
                            cursor,
                            stream_every_steps=args.stream_every_steps,
                        )
                    else:
                        assert runner is not None
                        terminal_reason = _stream_reference(
                            connection,
                            world,
                            runner,
                            cursor,
                            stream_every_steps=args.stream_every_steps,
                        )
                except _TerminalDeliveryPending as error:
                    pending_terminal = error.pending
                    print(
                        f"Unity client disconnected: {error}; "
                        "awaiting reconnect without advancing simulation",
                        flush=True,
                    )
                except (
                    BrokenPipeError,
                    ConnectionResetError,
                    TimeoutError,
                    OSError,
                ) as error:
                    print(f"Unity client disconnected: {error}", flush=True)
                finally:
                    connection.close()
                if terminal_reason is not None:
                    print(
                        f"simulation terminal: {terminal_reason}; server stopping",
                        flush=True,
                    )
                    break


if __name__ == "__main__":
    main()
