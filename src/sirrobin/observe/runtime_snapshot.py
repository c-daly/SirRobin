"""Immutable host snapshot staged from the device living runtime."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from sirrobin.physics.pose_live import resolve_live_pose
from sirrobin.runtime.config import LivingRuntimeConfig
from sirrobin.runtime.state import LivingState
from sirrobin.runtime.step import LivingIntervalLedger


@dataclass(frozen=True, slots=True)
class RuntimeSnapshot:
    step: int
    time_s: float
    alive: torch.Tensor
    stable_id: torch.Tensor
    parent_id: torch.Tensor
    generation: torch.Tensor
    born_at_s: torch.Tensor
    structure_q: torch.Tensor
    reserve_q: torch.Tensor
    position_enu_m: torch.Tensor
    velocity_enu_m_s: torch.Tensor
    accepted_effort_fraction: torch.Tensor
    yaw_rad: torch.Tensor
    turn_bias_rad_per_depth: torch.Tensor
    segment_mask: torch.Tensor
    segment_position_flu_m: torch.Tensor
    segment_rotation_flu: torch.Tensor
    segment_axes_flu_m: torch.Tensor
    segment_mass_sim: torch.Tensor
    producer_grid_q: torch.Tensor
    dissolved_grid_q: torch.Tensor
    born: torch.Tensor
    died: torch.Tensor
    death_stable_id: torch.Tensor
    starved: torch.Tensor
    old_age: torch.Tensor
    parent_slot_for_child: torch.Tensor
    mutation_trait_code: torch.Tensor
    mutation_locus: torch.Tensor
    mutation_parent_value: torch.Tensor
    mutation_child_value: torch.Tensor
    mutation_count: torch.Tensor
    mutation_event_applied: torch.Tensor
    mutation_event_trait_code: torch.Tensor
    mutation_event_locus: torch.Tensor
    mutation_event_component: torch.Tensor
    mutation_event_parent_value: torch.Tensor
    mutation_event_child_value: torch.Tensor
    stored_chemical_j: float
    interval_dissipation_j: float
    interval_light_input_j: float


@dataclass(frozen=True, slots=True)
class RuntimeEventSnapshot:
    """Small per-interval host record retained between render snapshots."""

    time_s: float
    alive: torch.Tensor
    stable_id: torch.Tensor
    parent_id: torch.Tensor
    generation: torch.Tensor
    born_at_s: torch.Tensor
    structure_q: torch.Tensor
    reserve_q: torch.Tensor
    born: torch.Tensor
    died: torch.Tensor
    death_stable_id: torch.Tensor
    starved: torch.Tensor
    old_age: torch.Tensor
    mutation_trait_code: torch.Tensor
    mutation_locus: torch.Tensor
    mutation_parent_value: torch.Tensor
    mutation_child_value: torch.Tensor
    mutation_count: torch.Tensor
    mutation_event_applied: torch.Tensor
    mutation_event_trait_code: torch.Tensor
    mutation_event_locus: torch.Tensor
    mutation_event_component: torch.Tensor
    mutation_event_parent_value: torch.Tensor
    mutation_event_child_value: torch.Tensor
    interval_dissipation_j: float
    interval_light_input_j: float


def _host(value: torch.Tensor) -> torch.Tensor:
    return value.detach().cpu()


def _interval_energy(
    ledger: LivingIntervalLedger,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    energy = ledger.energy
    dissipation_j = sum(
        (
            value.sum(dtype=torch.float64)
            for value in (
                energy.producer_maintenance_dissipation_j,
                energy.producer_mortality_dissipation_j,
                energy.assimilation_heat_j,
                energy.actuator_braking_heat_j,
                energy.muscle_inefficiency_heat_j,
                ledger.organisms.metabolism.ledger.baseline_demand_j,
                energy.motion_dissipation_j,
                energy.death_kinetic_and_carry_dissipation_j,
                energy.death_reserve_dissipation_j,
                energy.birth_construction_heat_j,
            )
        ),
        start=torch.zeros((), dtype=torch.float64, device=device),
    )
    return dissipation_j, energy.light_chemical_input_j.sum(dtype=torch.float64)


def stage_runtime_events(
    state: LivingState,
    ledger: LivingIntervalLedger,
) -> RuntimeEventSnapshot:
    """Stage lifecycle/event fields without resolving or copying render geometry."""

    lifecycle = ledger.organisms.lifecycle.ledger
    metabolism = ledger.organisms.metabolism
    mutation = ledger.mutation.ledger
    died = lifecycle.died
    death_stable_id = torch.where(
        died,
        metabolism.state.stable_id,
        torch.zeros_like(metabolism.state.stable_id),
    )
    dissipation_j, light_input_j = _interval_energy(
        ledger, state.population.alive.device
    )
    population = state.population
    return RuntimeEventSnapshot(
        time_s=float(state.economy.time_s.detach().cpu()),
        alive=_host(population.alive),
        stable_id=_host(population.stable_id),
        parent_id=_host(population.parent_id),
        generation=_host(population.generation),
        born_at_s=_host(population.born_at_s),
        structure_q=_host(population.structure_q),
        reserve_q=_host(population.reserve_q),
        born=_host(lifecycle.born),
        died=_host(died),
        death_stable_id=_host(death_stable_id),
        starved=_host(metabolism.ledger.starved),
        old_age=_host(metabolism.ledger.old_age_due),
        mutation_trait_code=_host(mutation.trait_code),
        mutation_locus=_host(mutation.locus),
        mutation_parent_value=_host(mutation.parent_value),
        mutation_child_value=_host(mutation.child_value),
        mutation_count=_host(mutation.mutation_count),
        mutation_event_applied=_host(mutation.event_applied),
        mutation_event_trait_code=_host(mutation.event_trait_code),
        mutation_event_locus=_host(mutation.event_locus),
        mutation_event_component=_host(mutation.event_component),
        mutation_event_parent_value=_host(mutation.event_parent_value),
        mutation_event_child_value=_host(mutation.event_child_value),
        interval_dissipation_j=float(dissipation_j.detach().cpu()),
        interval_light_input_j=float(light_input_j.detach().cpu()),
    )


def stage_runtime_snapshot(
    state: LivingState,
    config: LivingRuntimeConfig,
    ledger: LivingIntervalLedger | None,
) -> RuntimeSnapshot:
    """Copy one bounded read-only render/event snapshot to host memory."""

    worlds, capacity = state.population.alive.shape
    segments = state.body.seg_mask.shape[-1]
    accepted_effort = (
        torch.zeros_like(state.body.swim_freq_hz)
        if ledger is None
        else ledger.motion.ledger.selected.effort_fraction
    )
    pose = resolve_live_pose(
        state.body,
        state.motion.gait_time_s,
        state.motion.turn_bias_rad_per_depth,
        effort=accepted_effort,
    )
    pose_position = pose.pos_flu_m.reshape(worlds, capacity, segments, 3)
    pose_rotation = pose.rot_flu.reshape(worlds, capacity, segments, 4)
    producer_q = state.economy.bp_q.sum(dim=-1, dtype=torch.int64).transpose(1, 2)
    dissolved_q = state.economy.nd_q.sum(dim=-1, dtype=torch.int64).transpose(1, 2)
    population = state.population
    stored_chemical_j = (
        state.economy.bp_q.sum(dtype=torch.int64).to(torch.float64)
        * config.feeding.producer_j_per_q
        + population.reserve_q.sum(dtype=torch.int64).to(torch.float64)
        * config.feeding.reserve_j_per_q
        + population.assimilation_carry_q.sum()
        * config.feeding.reserve_j_per_q
        - population.maintenance_carry_j.sum()
    )

    zeros_bool = torch.zeros_like(population.alive)
    zeros_i64 = torch.zeros_like(population.stable_id)
    zeros_float = torch.zeros_like(state.motion.yaw_rad)
    event_shape = (*population.alive.shape, config.mutation.max_mutations_per_birth)
    if ledger is None:
        born = died = starved = old_age = zeros_bool
        death_stable_id = parent_slot = trait = locus = zeros_i64
        mutation_parent = mutation_child = zeros_float
        mutation_count = zeros_i64
        mutation_event_applied = torch.zeros(
            event_shape,
            dtype=torch.bool,
            device=population.alive.device,
        )
        mutation_event_trait = torch.zeros(
            event_shape,
            dtype=torch.int64,
            device=population.alive.device,
        )
        mutation_event_locus = torch.full_like(mutation_event_trait, -1)
        mutation_event_component = torch.full_like(mutation_event_trait, -1)
        mutation_event_parent = torch.zeros(
            event_shape,
            dtype=zeros_float.dtype,
            device=zeros_float.device,
        )
        mutation_event_child = torch.zeros_like(mutation_event_parent)
        dissipation_j = torch.zeros((), dtype=torch.float64, device=population.alive.device)
        light_input_j = torch.zeros_like(dissipation_j)
    else:
        lifecycle = ledger.organisms.lifecycle.ledger
        metabolism = ledger.organisms.metabolism
        mutation = ledger.mutation.ledger
        born = lifecycle.born
        died = lifecycle.died
        starved = metabolism.ledger.starved
        old_age = metabolism.ledger.old_age_due
        death_stable_id = torch.where(
            died, metabolism.state.stable_id, zeros_i64
        )
        parent_slot = lifecycle.parent_slot_for_child
        trait = mutation.trait_code
        locus = mutation.locus
        mutation_parent = mutation.parent_value
        mutation_child = mutation.child_value
        mutation_count = mutation.mutation_count
        mutation_event_applied = mutation.event_applied
        mutation_event_trait = mutation.event_trait_code
        mutation_event_locus = mutation.event_locus
        mutation_event_component = mutation.event_component
        mutation_event_parent = mutation.event_parent_value
        mutation_event_child = mutation.event_child_value
        dissipation_j, light_input_j = _interval_energy(
            ledger,
            population.alive.device,
        )

    return RuntimeSnapshot(
        step=int(state.economy.step.detach().cpu()),
        time_s=float(state.economy.time_s.detach().cpu()),
        alive=_host(population.alive),
        stable_id=_host(population.stable_id),
        parent_id=_host(population.parent_id),
        generation=_host(population.generation),
        born_at_s=_host(population.born_at_s),
        structure_q=_host(population.structure_q),
        reserve_q=_host(population.reserve_q),
        position_enu_m=_host(state.motion.position_enu_m),
        velocity_enu_m_s=_host(state.motion.velocity_rel_water_enu_m_s),
        accepted_effort_fraction=_host(accepted_effort),
        yaw_rad=_host(state.motion.yaw_rad),
        turn_bias_rad_per_depth=_host(state.motion.turn_bias_rad_per_depth),
        segment_mask=_host(state.body.seg_mask),
        segment_position_flu_m=_host(pose_position),
        segment_rotation_flu=_host(pose_rotation),
        segment_axes_flu_m=_host(state.body.semi_axes_flu_m),
        segment_mass_sim=_host(state.body.mass_sim),
        producer_grid_q=_host(producer_q),
        dissolved_grid_q=_host(dissolved_q),
        born=_host(born),
        died=_host(died),
        death_stable_id=_host(death_stable_id),
        starved=_host(starved),
        old_age=_host(old_age),
        parent_slot_for_child=_host(parent_slot),
        mutation_trait_code=_host(trait),
        mutation_locus=_host(locus),
        mutation_parent_value=_host(mutation_parent),
        mutation_child_value=_host(mutation_child),
        mutation_count=_host(mutation_count),
        mutation_event_applied=_host(mutation_event_applied),
        mutation_event_trait_code=_host(mutation_event_trait),
        mutation_event_locus=_host(mutation_event_locus),
        mutation_event_component=_host(mutation_event_component),
        mutation_event_parent_value=_host(mutation_event_parent),
        mutation_event_child_value=_host(mutation_event_child),
        stored_chemical_j=float(stored_chemical_j.detach().cpu()),
        interval_dissipation_j=float(dissipation_j.detach().cpu()),
        interval_light_input_j=float(light_input_j.detach().cpu()),
    )
