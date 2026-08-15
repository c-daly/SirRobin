"""GPU living-runtime bootstrap and Unity snapshot formatting."""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass, field

import torch

from sirrobin.observe.runtime_snapshot import (
    RuntimeEventSnapshot,
    RuntimeSnapshot,
    stage_runtime_events,
    stage_runtime_snapshot,
)
from sirrobin.organisms.mutation import (
    ATTACHMENT_ANGLE,
    ATTACHMENT_POSITION,
    JOINT_AMPLITUDE,
    SEGMENT_BUD,
    SEGMENT_RESHAPE,
    SEGMENT_VESTIGIAL,
    SWIM_FREQUENCY,
    SWIM_WAVE,
)
from sirrobin.physics.contracts import FluidSample
from sirrobin.runtime.config import LivingRuntimeConfig
from sirrobin.runtime.profile import (
    BASELINE_RUNTIME_PROFILE,
    EVOLUTION_DEMO_RUNTIME_PROFILE,
    LIVE_BEHAVIOR_CONFIG,
    RUNTIME_PROFILES,
    RuntimeProfile,
    living_runtime_config_from_reference,
)
from sirrobin.runtime.reference_adapter import living_state_from_reference
from sirrobin.runtime.session import LivingChunkSummary, RuntimeSession
from sirrobin.runtime.step import LivingIntervalLedger

__all__ = (
    "BASELINE_RUNTIME_PROFILE",
    "EVOLUTION_DEMO_RUNTIME_PROFILE",
    "LIVE_BEHAVIOR_CONFIG",
)

RuntimeUnityProfile = RuntimeProfile
RUNTIME_UNITY_PROFILES = RUNTIME_PROFILES


@dataclass(slots=True)
class RuntimeObservationTotals:
    """Host-side cumulative observations; never simulation authority."""

    intervals: int = 0
    births: int = 0
    deaths: int = 0
    starvation_deaths: int = 0
    old_age_deaths: int = 0
    requested_births: int = 0
    unfunded_birth_rejections: int = 0
    capacity_birth_rejections: int = 0
    id_birth_rejections: int = 0
    mutated_births: int = 0
    mutation_events: int = 0
    parameter_mutation_events: int = 0
    topology_mutation_events: int = 0
    behavior_seeking_intervals: int = 0
    behavior_searching_intervals: int = 0
    behavior_cruising_intervals: int = 0
    behavior_idle_intervals: int = 0
    feeding_requested_q: int = 0
    feeding_actual_debit_q: int = 0
    feeding_reserve_credit_q: int = 0

    def include(self, summary: LivingChunkSummary) -> None:
        self.intervals += summary.intervals
        for name in (
            "births",
            "deaths",
            "starvation_deaths",
            "old_age_deaths",
            "requested_births",
            "unfunded_birth_rejections",
            "capacity_birth_rejections",
            "id_birth_rejections",
            "mutated_births",
            "mutation_events",
            "parameter_mutation_events",
            "topology_mutation_events",
            "behavior_seeking_intervals",
            "behavior_searching_intervals",
            "behavior_cruising_intervals",
            "behavior_idle_intervals",
            "feeding_requested_q",
            "feeding_actual_debit_q",
            "feeding_reserve_credit_q",
        ):
            value = getattr(summary, name)
            setattr(self, name, getattr(self, name) + int(value.sum().detach().cpu()))

    def as_dict(self) -> dict[str, int]:
        return {
            name: int(getattr(self, name))
            for name in self.__dataclass_fields__
        }


@dataclass(frozen=True, slots=True)
class CreatureVisualLineage:
    """Presentation ancestry derived only from observed authoritative births."""

    lineage: str
    generation: int
    mutated_at_birth: bool
    mutation_kind: str
    mutation_count: int
    mutation_summary: str


@dataclass(slots=True)
class RuntimeVisualLineages:
    """Bounded host observer that makes clone and mutant branches legible."""

    creatures: dict[int, CreatureVisualLineage] = field(default_factory=dict)

    @staticmethod
    def _unobserved(
        identity: int,
        parent_id: int,
        generation: int,
    ) -> CreatureVisualLineage:
        lineage = f"founder-{identity}" if parent_id == 0 else f"unobserved-{identity}"
        return CreatureVisualLineage(lineage, generation, False, "none", 0, "")

    def reconcile(self, snapshot: RuntimeSnapshot | RuntimeEventSnapshot) -> None:
        """Retain current identities and mark any unobserved history honestly."""

        current: set[int] = set()
        for slot in snapshot.alive[0].nonzero().flatten().tolist():
            identity = int(snapshot.stable_id[0, slot])
            current.add(identity)
            if identity not in self.creatures:
                self.creatures[identity] = self._unobserved(
                    identity,
                    int(snapshot.parent_id[0, slot]),
                    int(snapshot.generation[0, slot]),
                )
        self.creatures = {
            identity: ancestry
            for identity, ancestry in self.creatures.items()
            if identity in current
        }

    def observe(self, snapshot: RuntimeEventSnapshot) -> None:
        """Branch visual ancestry on real mutations; clones inherit parent color."""

        for slot in snapshot.born[0].nonzero().flatten().tolist():
            child_id = int(snapshot.stable_id[0, slot])
            parent_id = int(snapshot.parent_id[0, slot])
            generation = int(snapshot.generation[0, slot])
            mutation_kind, mutation_count, mutation_summary = _mutation_details(
                snapshot,
                slot,
            )
            parent = self.creatures.get(parent_id)
            parent_lineage = (
                parent.lineage if parent is not None else f"unobserved-{parent_id}"
            )
            mutated = mutation_count > 0
            self.creatures[child_id] = CreatureVisualLineage(
                lineage=f"mutation-{child_id}" if mutated else parent_lineage,
                generation=generation,
                mutated_at_birth=mutated,
                mutation_kind=mutation_kind,
                mutation_count=mutation_count,
                mutation_summary=mutation_summary,
            )
        self.reconcile(snapshot)


@dataclass(frozen=True, slots=True)
class FastForwardReport:
    requested_intervals: int
    completed_intervals: int
    start_time_s: float
    end_time_s: float
    cancelled: bool
    births: int
    deaths: int
    starvation_deaths: int
    old_age_deaths: int
    requested_births: int
    unfunded_birth_rejections: int
    capacity_birth_rejections: int
    mutated_births: int
    mutation_events: int
    parameter_mutation_events: int
    topology_mutation_events: int
    feeding_actual_debit_q: int
    feeding_reserve_credit_q: int
    dissipation_j: float
    light_input_j: float


@dataclass(slots=True)
class RuntimeUnityBackend:
    """Own one autonomous device session and stage read-only host snapshots."""

    session: RuntimeSession
    fluid: FluidSample
    last_interval: LivingIntervalLedger | None = None
    last_summary: LivingChunkSummary | None = None
    observation: RuntimeObservationTotals = field(
        default_factory=RuntimeObservationTotals
    )
    visual_lineages: RuntimeVisualLineages = field(
        default_factory=RuntimeVisualLineages
    )

    @classmethod
    def from_reference_fixture(
        cls,
        world,
        *,
        compile_domains: bool = True,
        profile: RuntimeUnityProfile = BASELINE_RUNTIME_PROFILE,
    ) -> RuntimeUnityBackend:
        state = living_state_from_reference(world)
        config = living_runtime_config_from_reference(
            world,
            state,
            profile=profile,
        )
        backend = cls(
            RuntimeSession(
                state,
                config,
                compile_motion=compile_domains,
                compile_domains=compile_domains,
                optimistic_motion=False,
            ),
            world.fluid,
        )
        backend.visual_lineages.reconcile(backend.snapshot())
        return backend

    @property
    def config(self) -> LivingRuntimeConfig:
        return self.session.config

    def snapshot(self) -> RuntimeSnapshot:
        snapshot = stage_runtime_snapshot(
            self.session.state,
            self.config,
            self.last_interval,
        )
        self.visual_lineages.reconcile(snapshot)
        return snapshot

    def prewarm(self) -> None:
        """Compile the autonomous path without advancing observable state."""

        preserved_interval = self.last_interval
        try:
            self.session.prewarm_autonomous(self.fluid)
        finally:
            self.last_interval = preserved_interval

    def advance_events(self) -> RuntimeEventSnapshot:
        """Advance once and copy only event fields; render state stays resident."""

        chunk = self.session.advance_autonomous_chunk(self.fluid, intervals=1)
        self.last_interval = chunk.last_interval
        self.last_summary = chunk.summary
        if chunk.summary is None:
            raise RuntimeError("runtime chunk did not return an aggregate summary")
        self.observation.include(chunk.summary)
        events = stage_runtime_events(chunk.state, chunk.last_interval)
        self.visual_lineages.observe(events)
        return events

    def fast_forward(
        self,
        duration_s: float,
        *,
        chunk_intervals: int = 32,
        should_cancel: Callable[[], bool] | None = None,
        progress: Callable[[FastForwardReport], None] | None = None,
    ) -> FastForwardReport:
        """Advance a finite exact horizon without staging render snapshots."""

        if (
            isinstance(duration_s, bool)
            or not isinstance(duration_s, (int, float))
            or not math.isfinite(duration_s)
            or duration_s < 0.0
        ):
            raise ValueError("fast-forward duration must be finite and nonnegative")
        if (
            isinstance(chunk_intervals, bool)
            or not isinstance(chunk_intervals, int)
            or chunk_intervals < 1
        ):
            raise ValueError("fast-forward chunk size must be a positive integer")
        interval_s = self.config.economy.dt_eco_s
        interval_ratio = duration_s / interval_s
        requested_intervals = round(interval_ratio)
        if not math.isclose(
            interval_ratio,
            requested_intervals,
            rel_tol=0.0,
            abs_tol=1.0e-9,
        ):
            raise ValueError("fast-forward duration must contain whole authoritative intervals")

        start_time_s = float(self.session.state.economy.time_s.detach().cpu())
        completed = 0
        start_totals = self.observation.as_dict()
        dissipation_j = 0.0
        light_input_j = 0.0
        cancelled = False

        def report() -> FastForwardReport:
            totals = self.observation.as_dict()

            def delta(name: str) -> int:
                return totals[name] - start_totals[name]

            return FastForwardReport(
                requested_intervals=requested_intervals,
                completed_intervals=completed,
                start_time_s=start_time_s,
                end_time_s=float(
                    self.session.state.economy.time_s.detach().cpu()
                ),
                cancelled=cancelled,
                births=delta("births"),
                deaths=delta("deaths"),
                starvation_deaths=delta("starvation_deaths"),
                old_age_deaths=delta("old_age_deaths"),
                requested_births=delta("requested_births"),
                unfunded_birth_rejections=delta("unfunded_birth_rejections"),
                capacity_birth_rejections=delta("capacity_birth_rejections"),
                mutated_births=delta("mutated_births"),
                mutation_events=delta("mutation_events"),
                parameter_mutation_events=delta("parameter_mutation_events"),
                topology_mutation_events=delta("topology_mutation_events"),
                feeding_actual_debit_q=delta("feeding_actual_debit_q"),
                feeding_reserve_credit_q=delta("feeding_reserve_credit_q"),
                dissipation_j=dissipation_j,
                light_input_j=light_input_j,
            )

        while completed < requested_intervals:
            if should_cancel is not None and should_cancel():
                cancelled = True
                break
            count = min(chunk_intervals, requested_intervals - completed)
            # Birth identities and mutation details live on each authoritative
            # interval ledger, not in the aggregate chunk summary. Stage those
            # bounded event fields for every interval so prewarming cannot erase
            # the provenance later shown by Unity. Progress and cancellation
            # remain chunked; render snapshots remain suppressed.
            for _ in range(count):
                self.advance_events()
                summary = self.last_summary
                if summary is None:
                    raise RuntimeError(
                        "runtime interval did not return an aggregate summary"
                    )
                completed += summary.intervals
                dissipation_j += float(
                    summary.dissipation_j.sum().detach().cpu()
                )
                light_input_j += float(
                    summary.light_input_j.sum().detach().cpu()
                )
            if progress is not None:
                progress(report())

        return report()


def _mutation_details(
    snapshot: RuntimeSnapshot | RuntimeEventSnapshot,
    slot: int,
) -> tuple[str, int, str]:
    mutation_count = int(snapshot.mutation_count[0, slot])
    if mutation_count == 0:
        return "none", 0, ""
    changes = []
    parameter = False
    topology = False
    for event_slot in (
        snapshot.mutation_event_applied[0, slot].nonzero().flatten().tolist()
    ):
        trait = int(snapshot.mutation_event_trait_code[0, slot, event_slot])
        locus_index = int(snapshot.mutation_event_locus[0, slot, event_slot])
        component = int(snapshot.mutation_event_component[0, slot, event_slot])
        field_name = {
            JOINT_AMPLITUDE: "node_joint_amp_rad",
            SWIM_FREQUENCY: "swim_freq_hz",
            SWIM_WAVE: "swim_wave_rad_per_depth",
            SEGMENT_RESHAPE: "node_axis_m",
            ATTACHMENT_POSITION: "edge_attach_parent_axes",
            ATTACHMENT_ANGLE: "edge_attach_angle_rad",
            SEGMENT_BUD: "segment_bud",
            SEGMENT_VESTIGIAL: "segment_vestigial",
        }.get(trait, "unknown")
        topology |= trait in (SEGMENT_BUD, SEGMENT_VESTIGIAL)
        parameter |= trait not in (SEGMENT_BUD, SEGMENT_VESTIGIAL)
        indices = ""
        if locus_index >= 0:
            indices += f"[{locus_index}]"
        if component >= 0:
            indices += f"[{component}]"
        parent_value = float(
            snapshot.mutation_event_parent_value[0, slot, event_slot]
        )
        child_value = float(
            snapshot.mutation_event_child_value[0, slot, event_slot]
        )
        changes.append(
            f"{field_name}{indices} {parent_value:.6g}->{child_value:.6g}"
        )
    mutation_kind = "mixed" if parameter and topology else (
        "topology" if topology else "parameter"
    )
    return mutation_kind, mutation_count, "; ".join(changes)


def runtime_events(
    snapshot: RuntimeSnapshot | RuntimeEventSnapshot,
    config: LivingRuntimeConfig | None = None,
    observation: RuntimeObservationTotals | None = None,
) -> list[str]:
    events: list[str] = []
    for slot in snapshot.died[0].nonzero().flatten().tolist():
        identity = int(snapshot.death_stable_id[0, slot])
        cause = "starvation" if bool(snapshot.starved[0, slot]) else "old age"
        events.append(f"creature {identity} died: {cause}")
    for slot in snapshot.born[0].nonzero().flatten().tolist():
        child_id = int(snapshot.stable_id[0, slot])
        parent_id = int(snapshot.parent_id[0, slot])
        _, mutation_count, mutation_summary = _mutation_details(snapshot, slot)
        if mutation_count == 0:
            events.append(
                f"creature {parent_id} reproduced: child {child_id}; no mutation"
            )
            continue
        events.append(
            f"creature {parent_id} reproduced: mutant child {child_id}; "
            f"mutations={mutation_count}; {mutation_summary}"
        )
    heartbeat_multiple = round(snapshot.time_s / 5.0)
    if heartbeat_multiple > 0 and math.isclose(
        snapshot.time_s,
        heartbeat_multiple * 5.0,
        rel_tol=0.0,
        abs_tol=1.0e-9,
    ):
        heartbeat = (
            "heartbeat: "
            f"population={int(snapshot.alive.sum())} "
            f"reserve_q={int(snapshot.reserve_q.sum())}"
        )
        if config is not None:
            funded = snapshot.alive & (
                snapshot.reserve_q
                >= snapshot.structure_q + config.child_initial_reserve_q
            )
            generation = snapshot.generation[snapshot.alive]
            generation_max = int(generation.max()) if generation.numel() else 0
            heartbeat += (
                f" generation_max={generation_max}"
                f" funded_parents={int(funded.sum())}"
            )
        if observation is not None:
            heartbeat += (
                f" births={observation.births}"
                f" unfunded={observation.unfunded_birth_rejections}"
                f" mutations={observation.mutation_events}"
                f" topology={observation.topology_mutation_events}"
                f" starvation={observation.starvation_deaths}"
                f" old_age={observation.old_age_deaths}"
            )
        events.append(heartbeat)
    return events


def _numeric_summary(value: torch.Tensor) -> dict[str, float | int | None]:
    if value.numel() == 0:
        return {"count": 0, "min": None, "mean": None, "max": None}
    numeric = value.to(torch.float64)
    return {
        "count": value.numel(),
        "min": float(numeric.min()),
        "mean": float(numeric.mean()),
        "max": float(numeric.max()),
    }


def runtime_diagnostics(
    snapshot: RuntimeSnapshot,
    config: LivingRuntimeConfig,
    observation: RuntimeObservationTotals | None,
) -> dict[str, object]:
    """Summarize current state and exact observed transactions for display."""

    alive = snapshot.alive[0]
    age_s = snapshot.time_s - snapshot.born_at_s[0, alive]
    generation = snapshot.generation[0, alive]
    reserve_q = snapshot.reserve_q[0, alive]
    structure_q = snapshot.structure_q[0, alive]
    clone_funded = reserve_q >= structure_q + config.child_initial_reserve_q
    generation_counts = []
    if generation.numel():
        values, counts = torch.unique(generation, sorted=True, return_counts=True)
        generation_counts = [
            {"generation": int(value), "population": int(count)}
            for value, count in zip(values, counts, strict=True)
        ]
    producer = snapshot.producer_grid_q[0]
    totals = (
        RuntimeObservationTotals().as_dict()
        if observation is None
        else observation.as_dict()
    )
    return {
        "current": {
            "population": int(alive.sum()),
            "free_slots": int((~snapshot.alive[0]).sum()),
            "clone_funded_parents": int(clone_funded.sum()),
            "age_s": _numeric_summary(age_s),
            "generation": {
                **_numeric_summary(generation),
                "counts": generation_counts,
            },
            "reserve_q": {
                **_numeric_summary(reserve_q),
                "total": int(reserve_q.sum()),
            },
            "producer_q": {
                "total": int(producer.sum()),
                "occupied_cells": int((producer > 0).sum()),
                "peak_cell": int(producer.max()) if producer.numel() else 0,
            },
        },
        "observed_session": totals,
        "configuration": {
            "min_lifespan_s": config.mortality.min_lifespan_s,
            "max_lifespan_s": config.mortality.max_lifespan_s,
            "mutation_rate_per_locus": config.mutation.mutation_rate_per_locus,
            "max_mutations_per_birth": config.mutation.max_mutations_per_birth,
            "parameter_event_weight": config.mutation.parameter_event_weight,
            "topology_event_weight": config.mutation.topology_event_weight,
        },
    }


def _module_sets(
    snapshot: RuntimeSnapshot,
    *,
    module_display_scale: float,
) -> dict[int, list[dict[str, float]]]:
    result: dict[int, list[dict[str, float]]] = {}
    for slot in snapshot.alive[0].nonzero().flatten().tolist():
        modules = []
        for index in snapshot.segment_mask[0, slot].nonzero().flatten().tolist():
            qx, qy, qz, qw = snapshot.segment_rotation_flu[0, slot, index].tolist()
            orient = math.atan2(
                2.0 * (qw * qz + qx * qy),
                1.0 - 2.0 * (qy * qy + qz * qz),
            )
            axes = snapshot.segment_axes_flu_m[0, slot, index]
            position = snapshot.segment_position_flu_m[0, slot, index]
            modules.append(
                {
                    "a": float(axes[0]) * module_display_scale,
                    "b": float(axes[1]) * module_display_scale,
                    "c": float(axes[2]) * module_display_scale,
                    "cx": float(position[0]) * module_display_scale,
                    "cy": float(position[1]) * module_display_scale,
                    "cz": float(position[2]) * module_display_scale,
                    "orient": orient,
                }
            )
        result[slot] = modules
    return result


def runtime_payload(
    snapshot: RuntimeSnapshot,
    config: LivingRuntimeConfig,
    *,
    display_bodies: int,
    module_display_scale: float,
    view_width_m: float,
    view_height_m: float,
    view_depth_m: float,
    interval_events: list[str] | None = None,
    interval_births: int | None = None,
    interval_deaths: int | None = None,
    interval_dissipation_j: float | None = None,
    interval_light_input_j: float | None = None,
    observation: RuntimeObservationTotals | None = None,
    visual_lineages: RuntimeVisualLineages | None = None,
) -> dict[str, object]:
    modules = _module_sets(
        snapshot,
        module_display_scale=module_display_scale,
    )
    free_slot_exists = bool((~snapshot.alive).any())
    creatures = []
    for slot in snapshot.alive[0].nonzero().flatten().tolist()[:display_bodies]:
        identity = int(snapshot.stable_id[0, slot])
        parent_id = int(snapshot.parent_id[0, slot])
        position = snapshot.position_enu_m[0, slot]
        reserve_q = int(snapshot.reserve_q[0, slot])
        structure_q = int(snapshot.structure_q[0, slot])
        birth_cost_q = structure_q + config.child_initial_reserve_q
        funded = reserve_q >= birth_cost_q
        ancestry = (
            None
            if visual_lineages is None
            else visual_lineages.creatures.get(identity)
        )
        lineage = (
            ancestry.lineage
            if ancestry is not None
            else (
                f"founder-{identity}"
                if parent_id == 0
                else f"offspring-of-{parent_id}"
            )
        )
        creatures.append(
            {
                "id": identity,
                "lineage": lineage,
                "generation": int(snapshot.generation[0, slot]),
                "mutated_at_birth": (
                    ancestry.mutated_at_birth if ancestry is not None else False
                ),
                "mutation_kind": (
                    ancestry.mutation_kind if ancestry is not None else "none"
                ),
                "mutation_count": (
                    ancestry.mutation_count if ancestry is not None else 0
                ),
                "mutation_summary": (
                    ancestry.mutation_summary if ancestry is not None else ""
                ),
                "mass": float(snapshot.segment_mass_sim[0, slot].sum()),
                "x": float(position[0]) * view_width_m / config.geometry.lx_m,
                "y": float(position[1]) * view_height_m / config.geometry.ly_m,
                "z": 0.5 * view_depth_m,
                "on_seabed": False,
                "age_s": snapshot.time_s - float(snapshot.born_at_s[0, slot]),
                "yaw": float(snapshot.yaw_rad[0, slot]),
                "yaw_rate": 0.0,
                "sideslip": 0.0,
                "actuating": (
                    float(snapshot.accepted_effort_fraction[0, slot]) > 0.0
                ),
                "turning": abs(float(snapshot.turn_bias_rad_per_depth[0, slot])) > 0.0,
                "breeder": funded,
                "birth_ready": funded and free_slot_exists,
                "reproductive": min(1.0, reserve_q / max(1, birth_cost_q)),
                "reserve": float(reserve_q),
                "modules": modules[slot],
            }
        )
    events = (
        runtime_events(snapshot, config, observation)
        if interval_events is None
        else interval_events
    )
    births = int(snapshot.born.sum()) if interval_births is None else interval_births
    deaths = int(snapshot.died.sum()) if interval_deaths is None else interval_deaths
    dissipation_j = (
        snapshot.interval_dissipation_j
        if interval_dissipation_j is None
        else interval_dissipation_j
    )
    light_input_j = (
        snapshot.interval_light_input_j
        if interval_light_input_j is None
        else interval_light_input_j
    )
    return {
        "step": snapshot.step,
        "time_s": snapshot.time_s,
        "population": int(snapshot.alive.sum()),
        "creatures": creatures,
        "producer_grid": snapshot.producer_grid_q[0].to(torch.float64).tolist(),
        "zooplankton_grid": [
            [0.0 for _ in range(config.economy.gx)]
            for _ in range(config.economy.gy)
        ],
        "dissolved_grid": snapshot.dissolved_grid_q[0].to(torch.float64).tolist(),
        "births": births,
        "deaths": deaths,
        "events": list(events),
        "diagnostics": runtime_diagnostics(snapshot, config, observation),
        "energy": {
            "stored_chemical_j": snapshot.stored_chemical_j,
            "kinetic_j": 0.0,
            "dissipation_j": dissipation_j,
            "light_input_j": light_input_j,
        },
    }
