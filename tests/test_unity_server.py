"""Read-only Unity snapshots report the live lifecycle rather than fixture constants."""

from __future__ import annotations

import json
import math
from dataclasses import replace

import pytest
import torch

from sirrobin.core.metabolism import MaintenanceConfig
from sirrobin.core.mortality import AgeMortalityConfig
from sirrobin.core.reproduction import BirthConfig, ParametricMutationConfig
from sirrobin.core.runner import HeadlessRunner
from sirrobin.economy.config import EconomyConfig
from sirrobin.organisms.behavior import request_living_intent
from sirrobin.organisms.mortality import MortalityConfig
from sirrobin.organisms.mutation import MutationConfig
from sirrobin.physics.phase_response import PhaseWindowConfig, advance_phase_window
from sirrobin.runtime.reference_adapter import living_state_from_reference
from tools.run_world import LIVING_MATERIAL_ENERGY_CONFIG, _build_fixture_world
from tools.runtime_unity import (
    BASELINE_RUNTIME_PROFILE,
    EVOLUTION_DEMO_RUNTIME_PROFILE,
    LIVE_BEHAVIOR_CONFIG,
    RuntimeUnityBackend,
    RuntimeUnityProfile,
    runtime_payload,
)
from tools.serve_unity import (
    CAPACITY,
    DISPLAY_BODIES,
    EXTINCTION_EVENT,
    INITIAL_BODIES,
    LIVE_INITIAL_RESERVE_Q,
    LIVE_RICH_FOOD_CELL_Q,
    _build_server_runner,
    _build_server_world,
    _descriptor,
    _events,
    _payload,
    _record,
    _retry_terminal_record,
    _runtime_record,
    _seed_visible_baseline,
    _stream_reference,
    _stream_runtime,
    _StreamCursor,
    _TerminalDeliveryPending,
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
    payload = _payload(world, tick)

    assert payload["population"] == 2
    assert {creature["id"] for creature in payload["creatures"]} == {
        1,
        birth.child_id,
    }
    assert payload["births"] == 1
    assert payload["deaths"] == 1
    assert payload["events"] == [
        "creature 2 died: starvation",
        f"creature 1 reproduced: clone child {birth.child_id}",
    ]
    assert all(isinstance(event, str) for event in payload["events"])
    assert all(creature["reserve"] >= 0 for creature in payload["creatures"])
    assert payload["energy"]["stored_chemical_j"] > 0.0


def test_snapshot_reports_world_owned_mutant_lineage_and_mutation_event() -> None:
    world = _build_fixture_world(
        bodies=2,
        live_bodies=1,
        reserve_q_per_creature=2_000,
        device=torch.device("cpu"),
        economy_interval_s=0.1,
    )
    tick = HeadlessRunner(
        world,
        birth_config=BirthConfig(initial_reserve_q=100),
        mutation_config=ParametricMutationConfig(
            seed=5,
            traits=("swim_frequency",),
        ),
    ).advance()
    birth = tick.births[0]
    assert birth.child_id is not None
    assert birth.mutation is not None

    payload = _payload(world, tick)

    child = next(
        creature
        for creature in payload["creatures"]
        if creature["id"] == birth.child_id
    )
    assert child["lineage"] == f"mutant-of-{birth.parent_id}"
    assert payload["events"] == [
        f"creature {birth.parent_id} reproduced: mutant child {birth.child_id}; "
        f"{birth.mutation.field_name} "
        f"{birth.mutation.parent_value:.6g}->{birth.mutation.child_value:.6g}"
    ]


def test_snapshot_reports_old_age_as_a_distinct_death_cause() -> None:
    world = _build_fixture_world(
        bodies=1,
        live_bodies=1,
        device=torch.device("cpu"),
        economy_interval_s=0.1,
    )
    tick = HeadlessRunner(
        world,
        maintenance_config=MaintenanceConfig(0.0),
        age_mortality_config=AgeMortalityConfig(0.05, 0.05),
    ).advance()

    payload = _payload(world, tick)

    assert payload["population"] == 0
    assert payload["deaths"] == 1
    assert payload["events"] == ["creature 1 died: old age"]


def test_live_event_stream_emits_periodic_progress_heartbeat() -> None:
    world = _build_fixture_world(
        bodies=1,
        live_bodies=1,
        device=torch.device("cpu"),
        economy_interval_s=0.1,
    )
    tick = HeadlessRunner(world).advance()

    events = _events(world, replace(tick, sim_time_s=5.0))

    assert events == ["heartbeat: population=1 reserve_q=500"]


def test_coalesced_render_record_retains_interval_lifecycle_summary() -> None:
    world = _build_fixture_world(
        bodies=1,
        live_bodies=1,
        device=torch.device("cpu"),
        economy_interval_s=0.1,
    )
    tick = HeadlessRunner(world).advance()

    record = _record(
        world,
        7,
        tick,
        interval_events=["first event", "second event"],
        interval_births=3,
        interval_deaths=2,
    )

    assert record["payload"]["events"] == ["first event", "second event"]
    assert record["payload"]["births"] == 3
    assert record["payload"]["deaths"] == 2


class _CaptureConnection:
    def __init__(self) -> None:
        self.messages: list[bytes] = []

    def sendall(self, message: bytes) -> None:
        self.messages.append(message)


class _DisconnectBeforeFinalConnection(_CaptureConnection):
    def sendall(self, message: bytes) -> None:
        if self.messages:
            raise BrokenPipeError("test client disconnected before final record")
        super().sendall(message)


def test_reference_stream_emits_one_final_extinction_record_before_return() -> None:
    world = _build_fixture_world(
        bodies=1,
        live_bodies=1,
        device=torch.device("cpu"),
        economy_interval_s=0.1,
    )
    runner = HeadlessRunner(
        world,
        maintenance_config=MaintenanceConfig(0.0),
        age_mortality_config=AgeMortalityConfig(0.05, 0.05),
    )
    connection = _CaptureConnection()

    terminal_reason = _stream_reference(
        connection,
        world,
        runner,
        _StreamCursor(),
        stream_every_steps=10,
    )
    records = [json.loads(message) for message in connection.messages]

    assert terminal_reason == "extinction"
    assert len(records) == 2
    assert records[-1]["payload"]["population"] == 0
    assert records[-1]["payload"]["terminal"] == {"reason": "extinction"}
    assert records[-1]["payload"]["events"].count(EXTINCTION_EVENT) == 1


def test_runtime_stream_emits_one_final_extinction_record_before_return() -> None:
    world = _build_server_world(device=torch.device("cpu"))
    _seed_visible_baseline(world)
    profile = replace(
        EVOLUTION_DEMO_RUNTIME_PROFILE,
        mortality=MortalityConfig(0.05, 0.05, seed=7),
    )
    backend = RuntimeUnityBackend.from_reference_fixture(
        world,
        compile_domains=False,
        profile=profile,
    )
    connection = _CaptureConnection()

    terminal_reason = _stream_runtime(
        connection,
        backend,
        _StreamCursor(),
        stream_every_steps=10,
    )
    records = [json.loads(message) for message in connection.messages]

    assert terminal_reason == "extinction"
    assert len(records) == 2
    assert records[-1]["payload"]["population"] == 0
    assert records[-1]["payload"]["deaths"] == INITIAL_BODIES
    assert records[-1]["payload"]["terminal"] == {"reason": "extinction"}
    assert records[-1]["payload"]["events"].count(EXTINCTION_EVENT) == 1


def test_runtime_stream_retries_final_extinction_record_after_reconnect() -> None:
    world = _build_server_world(device=torch.device("cpu"))
    _seed_visible_baseline(world)
    profile = replace(
        EVOLUTION_DEMO_RUNTIME_PROFILE,
        mortality=MortalityConfig(0.05, 0.05, seed=7),
    )
    backend = RuntimeUnityBackend.from_reference_fixture(
        world,
        compile_domains=False,
        profile=profile,
    )
    cursor = _StreamCursor()

    with pytest.raises(_TerminalDeliveryPending, match="extinction") as caught:
        _stream_runtime(
            _DisconnectBeforeFinalConnection(),
            backend,
            cursor,
            stream_every_steps=10,
        )

    assert not bool(backend.snapshot().alive.any())
    pending_record = json.loads(caught.value.pending.record)
    assert pending_record["payload"]["deaths"] == INITIAL_BODIES
    assert pending_record["payload"]["events"][-1] == EXTINCTION_EVENT
    assert len(pending_record["payload"]["events"]) == INITIAL_BODIES + 1
    assert all(
        event.endswith("died: old age")
        for event in pending_record["payload"]["events"][:-1]
    )
    assert pending_record["payload"]["energy"]["dissipation_j"] > 0.0
    reconnect = _CaptureConnection()
    terminal_reason = _retry_terminal_record(reconnect, caught.value.pending)
    records = [json.loads(message) for message in reconnect.messages]

    assert terminal_reason == "extinction"
    assert len(records) == 1
    assert reconnect.messages[0] == caught.value.pending.record
    assert records[0] == pending_record


def test_live_stream_identity_survives_reconnect_and_simulation_restart() -> None:
    world = _build_fixture_world(
        bodies=1,
        live_bodies=1,
        device=torch.device("cpu"),
        economy_interval_s=0.1,
    )
    cursor = _StreamCursor()

    first_sequence = cursor.next()
    first = _record(world, first_sequence, None)
    cursor.resume_after(3752)
    second_sequence = cursor.next()
    second = _record(world, second_sequence, None)
    cursor.resume_after(10)

    assert first["step"] == second["step"] == 0
    assert first["record_id"] == "render:sequence:1"
    assert second["record_id"] == "render:sequence:3753"
    assert first["record_id"] != second["record_id"]
    assert cursor.last_sequence == 3753


def test_snapshot_does_not_render_inactive_capacity_slots() -> None:
    world = _build_fixture_world(
        bodies=8,
        live_bodies=1,
        device=torch.device("cpu"),
        economy_interval_s=0.1,
    )

    payload = _payload(world, None)

    assert payload["population"] == 1
    assert len(payload["creatures"]) == 1
    assert payload["births"] == 0
    assert payload["deaths"] == 0


def test_snapshot_dissipation_uses_named_outputs_not_the_reserve_debit() -> None:
    world = _build_fixture_world(
        bodies=1,
        device=torch.device("cpu"),
        economy_interval_s=0.1,
    )
    tick = HeadlessRunner(
        world,
        maintenance_config=MaintenanceConfig(
            0.0,
            chemical_to_mechanical_efficiency=0.5,
        ),
    ).advance()

    payload = _payload(world, tick)

    maintenance = tick.maintenance[0]
    expected_j = float(tick.mechanical_work_j.sum().item())
    expected_j += maintenance.baseline_maintenance_demand_j
    expected_j += maintenance.muscle_inefficiency_heat_j
    expected_j += maintenance.actuator_braking_heat_j
    expected_j += maintenance.death_dissipation_j
    assert payload["energy"]["dissipation_j"] == pytest.approx(expected_j)
    assert payload["energy"]["dissipation_j"] != pytest.approx(
        float(tick.mechanical_work_j.sum().item())
        + maintenance.maintenance_heat_j
    )
    gross_stored_j = (
        int(world.economy_state.bp_q.sum())
        * world.material_energy_config.producer_j_per_q
        + int(world.creature_material.reserve_q.sum())
        * world.material_energy_config.reserve_j_per_q
    )
    carry_asset_j = (
        float(world.creature_material.assimilation_carry_q.sum())
        * world.material_energy_config.reserve_j_per_q
    )
    carry_liability_j = float(world.creature_material.maintenance_carry_j.sum())
    assert payload["energy"]["stored_chemical_j"] == pytest.approx(
        gross_stored_j + carry_asset_j - carry_liability_j,
        rel=0.0,
        abs=1.0e-12,
    )


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

    assert CAPACITY == 64
    assert DISPLAY_BODIES == CAPACITY
    assert world.body.capacity == CAPACITY
    assert world.body.mass_sim.dtype == torch.float32
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


def test_live_world_has_exact_sparse_food_and_low_founder_reserves() -> None:
    world = _build_server_world()
    _seed_visible_baseline(world)
    assert int(world.economy_state.bp_q.sum()) == 12 * LIVE_RICH_FOOD_CELL_Q
    positive = world.economy_state.bp_q[world.economy_state.bp_q > 0]
    assert positive.numel() == 12
    assert torch.unique(positive).tolist() == [LIVE_RICH_FOOD_CELL_Q]
    assert LIVE_RICH_FOOD_CELL_Q == 2_000_000
    assert int((world.economy_state.bp_q == 0).sum()) == 132
    assert world.creature_material.reserve_q[world.body.alive].tolist() == [
        LIVE_INITIAL_RESERVE_Q
    ] * INITIAL_BODIES
    assert LIVE_INITIAL_RESERVE_Q == 2
    assert torch.equal(world.matter_totals().total_q, world.expected_matter_total_q)

    tick = _build_server_runner(world).advance()
    payload = _payload(world, tick)

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


def test_device_runtime_backend_advances_and_formats_existing_protocol() -> None:
    world = _build_server_world(device=torch.device("cpu"))
    _seed_visible_baseline(world)
    backend = RuntimeUnityBackend.from_reference_fixture(
        world,
        compile_domains=False,
    )

    initial = backend.snapshot()
    event_snapshot = backend.advance_events()
    advanced = backend.snapshot()
    payload = runtime_payload(
        advanced,
        backend.config,
        display_bodies=DISPLAY_BODIES,
        module_display_scale=1.0 / 35.0,
        view_width_m=60.0,
        view_height_m=60.0,
        view_depth_m=20.0,
    )
    record = _runtime_record(advanced, backend.config, 3)

    assert initial.step == 0
    assert not bool(initial.accepted_effort_fraction.any())
    assert advanced.step == 1
    assert advanced.time_s == pytest.approx(0.1)
    assert event_snapshot.time_s == pytest.approx(0.1)
    assert payload["population"] == INITIAL_BODIES
    assert len(payload["creatures"]) == INITIAL_BODIES
    assert payload["events"] == []
    assert record["payload"] == payload
    assert record["provenance"] == {
        "bridge": "original-gpu-living-runtime"
    }
    assert record["record_type"] == "snapshot.render"
    assert record["record_id"] == "render:sequence:3"
    assert backend.session.state is not world
    assert not backend.session.optimistic_motion_enabled
    assert backend.config.motion.stages == 4
    assert backend.config.motion.phase_samples == 3
    assert backend.config.behavior.search_effort_fraction > 0.0
    assert backend.config.behavior.search_leg_duration_s > 0.0
    assert backend.config.behavior.search_duty_fraction < 1.0
    assert backend.config.behavior.food_sufficient_peak_fraction > 0.0
    assert torch.equal(
        advanced.accepted_effort_fraction,
        backend.last_interval.motion.ledger.selected.effort_fraction.cpu(),
    )


def test_live_flat_field_exploration_holds_a_physical_straight_run() -> None:
    """An aligned explorer must not be assigned a new turn before it settles."""

    economy = replace(
        EconomyConfig(),
        gx=6,
        gy=6,
        gz=4,
        lx_m=60.0,
        ly_m=60.0,
        lz_m=20.0,
        dt_eco_s=0.1,
        remin_floor_s=1.0e-4,
    )
    world = _build_fixture_world(
        bodies=1,
        live_bodies=1,
        device=torch.device("cpu"),
        economy_interval_s=0.1,
        economy_config=economy,
        physics_dtype=torch.float32,
    )
    world.economy_state.bp_q.zero_()
    state = living_state_from_reference(world)
    state.motion.position_enu_m[..., :2].fill_(30.0)

    initial = request_living_intent(
        state.population,
        state.body,
        state.motion,
        state.economy.bp_q,
        world.geometry,
        world.live_config,
        LIVE_BEHAVIOR_CONFIG,
        q_mass_mol=world.economy_config.q_mass_mol,
    )
    initial_heading = initial.requested_heading_enu[0, 0]
    state.motion.yaw_rad[0, 0] = torch.atan2(
        initial_heading[1], initial_heading[0]
    )

    previous_position = state.motion.position_enu_m[0, 0, :2].clone()
    unwrapped_displacement = torch.zeros_like(previous_position)
    path_length_m = 0.0
    absolute_yaw_change_rad = 0.0
    previous_yaw = float(state.motion.yaw_rad[0, 0])
    late_heading_errors: list[float] = []
    for interval in range(200):
        behavior = request_living_intent(
            state.population,
            state.body,
            state.motion,
            state.economy.bp_q,
            world.geometry,
            world.live_config,
            LIVE_BEHAVIOR_CONFIG,
            q_mass_mol=world.economy_config.q_mass_mol,
        )
        motion = advance_phase_window(
            state.body,
            behavior.motion,
            world.fluid,
            world.live_config,
            world.geometry,
            PhaseWindowConfig(0.1, stages=4, phase_samples=3),
            effort_fraction=behavior.requested_effort_fraction,
        )
        state = replace(state, motion=motion.state)

        position = state.motion.position_enu_m[0, 0, :2]
        delta = (
            torch.remainder(position - previous_position + 30.0, 60.0) - 30.0
        )
        unwrapped_displacement += delta
        path_length_m += float(torch.linalg.vector_norm(delta))
        previous_position = position.clone()

        yaw = float(state.motion.yaw_rad[0, 0])
        yaw_delta = math.atan2(
            math.sin(yaw - previous_yaw), math.cos(yaw - previous_yaw)
        )
        absolute_yaw_change_rad += abs(yaw_delta)
        previous_yaw = yaw

        if interval >= 160:
            velocity = state.motion.velocity_rel_water_enu_m_s[0, 0, :2]
            travel = velocity / torch.linalg.vector_norm(velocity).clamp_min(
                1.0e-9
            )
            desired = state.motion.desired_heading_enu[0, 0].to(travel.dtype)
            late_heading_errors.append(
                abs(
                    float(
                        torch.atan2(
                            travel[0] * desired[1] - travel[1] * desired[0],
                            torch.dot(travel, desired).clamp(-1.0, 1.0),
                        )
                    )
                )
            )

    assert path_length_m > 10.0
    assert (
        float(torch.linalg.vector_norm(unwrapped_displacement))
        > 0.98 * path_length_m
    )
    assert absolute_yaw_change_rad < 0.5
    assert sum(late_heading_errors) / len(late_heading_errors) < 0.15


def test_evolution_demo_is_an_explicit_fivefold_observation_profile() -> None:
    baseline = BASELINE_RUNTIME_PROFILE
    demo = EVOLUTION_DEMO_RUNTIME_PROFILE

    assert baseline.name == "baseline"
    assert demo.name == "evolution-demo"
    assert demo.mortality.min_lifespan_s == 5 * baseline.mortality.min_lifespan_s
    assert demo.mortality.max_lifespan_s == 5 * baseline.mortality.max_lifespan_s
    assert (
        demo.mutation.mutation_rate_per_locus
        == 5 * baseline.mutation.mutation_rate_per_locus
    )
    assert demo.mutation.parameter_event_weight == (
        baseline.mutation.parameter_event_weight
    )
    assert demo.mutation.topology_event_weight == (
        baseline.mutation.topology_event_weight
    )

    descriptor = _descriptor(
        _build_server_world(),
        profile=demo,
    )

    assert descriptor["configuration"]["runtime_profile"] == {
        "name": "evolution-demo",
        "description": demo.description,
        "min_lifespan_s": 300.0,
        "max_lifespan_s": 500.0,
        "mutation_rate_per_locus": 0.01,
    }


def test_runtime_diagnostics_report_exact_reproduction_funding_outcomes() -> None:
    world = _build_server_world(device=torch.device("cpu"))
    _seed_visible_baseline(world)
    backend = RuntimeUnityBackend.from_reference_fixture(
        world,
        compile_domains=False,
    )

    backend.advance_events()
    snapshot = backend.snapshot()
    payload = runtime_payload(
        snapshot,
        backend.config,
        display_bodies=DISPLAY_BODIES,
        module_display_scale=1.0 / 35.0,
        view_width_m=60.0,
        view_height_m=60.0,
        view_depth_m=20.0,
        observation=backend.observation,
    )
    totals = payload["diagnostics"]["observed_session"]

    assert totals["intervals"] == 1
    assert totals["requested_births"] == INITIAL_BODIES
    assert totals["births"] == 0
    assert totals["unfunded_birth_rejections"] == INITIAL_BODIES
    assert totals["capacity_birth_rejections"] == 0
    assert totals["id_birth_rejections"] == 0
    assert totals["deaths"] == 0
    assert totals["starvation_deaths"] == 0
    assert totals["old_age_deaths"] == 0
    assert totals["mutation_events"] == (
        totals["parameter_mutation_events"]
        + totals["topology_mutation_events"]
    )
    assert totals["feeding_requested_q"] >= totals["feeding_actual_debit_q"]
    assert payload["diagnostics"]["current"]["population"] == INITIAL_BODIES
    assert payload["diagnostics"]["current"]["generation"]["counts"] == [
        {"generation": 0, "population": INITIAL_BODIES}
    ]


def test_runtime_diagnostics_split_parameter_and_topology_mutations() -> None:
    world = _build_fixture_world(
        bodies=2,
        live_bodies=1,
        reserve_q_per_creature=5_000,
        device=torch.device("cpu"),
        economy_interval_s=0.1,
        material_energy_config=LIVING_MATERIAL_ENERGY_CONFIG,
        physics_dtype=torch.float32,
    )
    forced_mutation = RuntimeUnityProfile(
        name="test-forced-mutation",
        description="test-only certain mutation exposure",
        mortality=MortalityConfig(100.0, 100.0, seed=7),
        mutation=MutationConfig(seed=7, mutation_rate_per_locus=1.0),
    )
    backend = RuntimeUnityBackend.from_reference_fixture(
        world,
        compile_domains=False,
        profile=forced_mutation,
    )

    events = backend.advance_events()
    totals = backend.observation
    snapshot = backend.snapshot()
    payload = runtime_payload(
        snapshot,
        backend.config,
        display_bodies=2,
        module_display_scale=1.0 / 35.0,
        view_width_m=60.0,
        view_height_m=60.0,
        view_depth_m=20.0,
        visual_lineages=backend.visual_lineages,
    )
    mutant = next(creature for creature in payload["creatures"] if creature["id"] == 2)

    assert totals.births == 1
    assert totals.mutated_births == 1
    assert totals.mutation_events == 3
    assert (
        totals.parameter_mutation_events + totals.topology_mutation_events
        == totals.mutation_events
    )
    assert mutant["lineage"] == "mutation-2"
    assert mutant["generation"] == 1
    assert mutant["mutated_at_birth"]
    assert mutant["mutation_kind"] in ("parameter", "topology", "mixed")
    assert mutant["mutation_count"] == 3
    assert mutant["mutation_summary"]

    clone_event = replace(
        events,
        stable_id=torch.tensor([[2, 3]], dtype=torch.int64),
        parent_id=torch.tensor([[1, 2]], dtype=torch.int64),
        generation=torch.tensor([[1, 2]], dtype=torch.int64),
        born=torch.tensor([[False, True]]),
        mutation_count=torch.zeros_like(events.mutation_count),
        mutation_event_applied=torch.zeros_like(events.mutation_event_applied),
    )
    backend.visual_lineages.observe(clone_event)
    branch = backend.visual_lineages.creatures

    assert branch[3].lineage == branch[2].lineage == "mutation-2"
    assert not branch[3].mutated_at_birth
    assert branch[3].mutation_kind == "none"


def test_device_runtime_backend_prewarms_without_publishing_a_tick() -> None:
    world = _build_server_world(device=torch.device("cpu"))
    _seed_visible_baseline(world)
    backend = RuntimeUnityBackend.from_reference_fixture(
        world,
        compile_domains=False,
    )
    initial = backend.snapshot()

    backend.prewarm()
    after = backend.snapshot()

    assert after.step == initial.step == 0
    assert after.time_s == initial.time_s == 0.0
    assert torch.equal(after.position_enu_m, initial.position_enu_m)
    assert torch.equal(after.reserve_q, initial.reserve_q)
    assert backend.last_interval is None


def test_runtime_payload_does_not_label_passive_velocity_as_actuation() -> None:
    world = _build_server_world(device=torch.device("cpu"))
    _seed_visible_baseline(world)
    backend = RuntimeUnityBackend.from_reference_fixture(
        world,
        compile_domains=False,
    )
    snapshot = backend.snapshot()
    passive = replace(
        snapshot,
        velocity_enu_m_s=torch.ones_like(snapshot.velocity_enu_m_s),
        accepted_effort_fraction=torch.zeros_like(
            snapshot.accepted_effort_fraction
        ),
    )

    payload = runtime_payload(
        passive,
        backend.config,
        display_bodies=DISPLAY_BODIES,
        module_display_scale=1.0 / 35.0,
        view_width_m=60.0,
        view_height_m=60.0,
        view_depth_m=20.0,
    )

    assert all(not creature["actuating"] for creature in payload["creatures"])


def test_device_runtime_backend_sustains_multiple_render_intervals() -> None:
    world = _build_server_world(device=torch.device("cpu"))
    _seed_visible_baseline(world)
    backend = RuntimeUnityBackend.from_reference_fixture(
        world,
        compile_domains=False,
    )

    snapshots = [backend.advance_events() for _ in range(25)]
    rendered = backend.snapshot()

    assert rendered.step == 25
    assert rendered.time_s == pytest.approx(2.5)
    assert all(snapshot.time_s > 0.0 for snapshot in snapshots)
    assert torch.isfinite(backend.session.state.motion.position_enu_m).all()


def test_finite_fast_forward_suppresses_rendering_and_aggregates_outcomes() -> None:
    world = _build_fixture_world(
        bodies=4,
        live_bodies=1,
        reserve_q_per_creature=5_000,
        device=torch.device("cpu"),
        economy_interval_s=0.1,
        material_energy_config=LIVING_MATERIAL_ENERGY_CONFIG,
        physics_dtype=torch.float32,
    )
    backend = RuntimeUnityBackend.from_reference_fixture(
        world,
        compile_domains=False,
    )
    initial_population = int(backend.session.state.population.alive.sum())

    report = backend.fast_forward(0.3, chunk_intervals=2)
    final_snapshot = backend.snapshot()

    assert report.requested_intervals == 3
    assert report.completed_intervals == 3
    assert not report.cancelled
    assert report.start_time_s == 0.0
    assert report.end_time_s == pytest.approx(0.3)
    assert final_snapshot.step == 3
    assert final_snapshot.time_s == pytest.approx(0.3)
    assert int(final_snapshot.alive.sum()) == (
        initial_population + report.births - report.deaths
    )
    assert report.mutation_events >= 0
    assert report.dissipation_j >= 0.0
    assert report.light_input_j >= 0.0


def test_fast_forward_preserves_observed_mutation_lineage() -> None:
    world = _build_fixture_world(
        bodies=3,
        live_bodies=1,
        reserve_q_per_creature=5_000,
        device=torch.device("cpu"),
        economy_interval_s=0.1,
        material_energy_config=LIVING_MATERIAL_ENERGY_CONFIG,
        physics_dtype=torch.float32,
    )
    forced_mutation = RuntimeUnityProfile(
        name="test-fast-forward-mutation",
        description="test-only certain mutation exposure",
        mortality=MortalityConfig(100.0, 100.0, seed=7),
        mutation=MutationConfig(seed=7, mutation_rate_per_locus=1.0),
    )
    backend = RuntimeUnityBackend.from_reference_fixture(
        world,
        compile_domains=False,
        profile=forced_mutation,
    )

    report = backend.fast_forward(0.2, chunk_intervals=2)
    payload = runtime_payload(
        backend.snapshot(),
        backend.config,
        display_bodies=3,
        module_display_scale=1.0 / 35.0,
        view_width_m=60.0,
        view_height_m=60.0,
        view_depth_m=20.0,
        visual_lineages=backend.visual_lineages,
    )
    mutant = next(
        creature for creature in payload["creatures"] if creature["id"] == 2
    )

    assert report.mutated_births >= 1
    assert mutant["lineage"] == "mutation-2"
    assert mutant["mutated_at_birth"]
    assert mutant["mutation_count"] > 0


def test_fast_forward_can_be_cancelled_only_between_exact_intervals() -> None:
    world = _build_fixture_world(
        bodies=2,
        live_bodies=1,
        reserve_q_per_creature=5_000,
        device=torch.device("cpu"),
        economy_interval_s=0.1,
        material_energy_config=LIVING_MATERIAL_ENERGY_CONFIG,
        physics_dtype=torch.float32,
    )
    backend = RuntimeUnityBackend.from_reference_fixture(
        world,
        compile_domains=False,
    )
    cancel = False

    def progress(report) -> None:
        nonlocal cancel
        cancel = report.completed_intervals >= 2

    report = backend.fast_forward(
        1.0,
        chunk_intervals=2,
        should_cancel=lambda: cancel,
        progress=progress,
    )

    assert report.cancelled
    assert report.completed_intervals == 2
    assert report.end_time_s == pytest.approx(0.2)
    assert backend.snapshot().step == 2


def test_fast_forward_rejects_fractional_authoritative_intervals() -> None:
    world = _build_fixture_world(
        bodies=2,
        live_bodies=1,
        reserve_q_per_creature=5_000,
        device=torch.device("cpu"),
        economy_interval_s=0.1,
        material_energy_config=LIVING_MATERIAL_ENERGY_CONFIG,
        physics_dtype=torch.float32,
    )
    backend = RuntimeUnityBackend.from_reference_fixture(
        world,
        compile_domains=False,
    )

    with pytest.raises(ValueError, match="whole authoritative intervals"):
        backend.fast_forward(0.15)


@pytest.mark.gpu
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
def test_unity_device_backend_advances_compiled_cuda_runtime() -> None:
    world = _build_server_world(device=torch.device("cuda"))
    _seed_visible_baseline(world)
    backend = RuntimeUnityBackend.from_reference_fixture(world)

    event_snapshots = [backend.advance_events() for _ in range(25)]
    snapshot = backend.snapshot()
    payload = runtime_payload(
        snapshot,
        backend.config,
        display_bodies=DISPLAY_BODIES,
        module_display_scale=1.0 / 35.0,
        view_width_m=60.0,
        view_height_m=60.0,
        view_depth_m=20.0,
    )

    assert snapshot.step == 25
    assert event_snapshots[-1].time_s == pytest.approx(2.5)
    assert snapshot.alive.device.type == "cpu"
    assert payload["population"] == INITIAL_BODIES
    assert len(payload["creatures"]) == INITIAL_BODIES
    assert backend.session.state.population.alive.device.type == "cuda"
