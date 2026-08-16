#!/usr/bin/env python3
"""Prove one complete causal living loop on the cohesive runtime."""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from dataclasses import fields, replace

import torch

from sirrobin.genetics.develop import develop_unchecked
from sirrobin.organisms.behavior import request_living_intent
from sirrobin.organisms.metabolism import available_actuator_work_j
from sirrobin.physics.ecological_motion import advance_affordable_motion
from sirrobin.physics.morphology import query_morphology
from sirrobin.runtime.motion_state import developed_support_radius_m
from tools.runtime_unity import CAUSAL_RUNTIME_PROFILE, RuntimeUnityBackend
from tools.serve_unity import (
    ECONOMY_INTERVAL_S,
    INITIAL_BODIES,
    _build_server_world,
    _seed_visible_baseline,
)


def _whole_intervals(duration_s: float) -> int:
    if (
        isinstance(duration_s, bool)
        or not isinstance(duration_s, (int, float))
        or not math.isfinite(duration_s)
        or duration_s <= 0.0
    ):
        raise ValueError("max_duration_s must be finite and positive")
    ratio = duration_s / ECONOMY_INTERVAL_S
    intervals = round(ratio)
    if not math.isclose(ratio, intervals, rel_tol=0.0, abs_tol=1.0e-9):
        raise ValueError("max_duration_s must contain whole authoritative intervals")
    return intervals


def _direct_matter_census_tensor(state) -> torch.Tensor:
    """Sum raw reservoirs and organism inventories without the runtime oracle."""

    economy_q = sum(
        (reservoir.sum(dtype=torch.int64) for reservoir in state.economy.reservoirs),
        start=torch.zeros(
            (),
            dtype=torch.int64,
            device=state.population.alive.device,
        ),
    )
    organism_q = state.population.structure_q.sum(dtype=torch.int64)
    organism_q += state.population.reserve_q.sum(dtype=torch.int64)
    return economy_q + organism_q


def _direct_matter_census(state) -> int:
    return int(_direct_matter_census_tensor(state).detach().cpu())


def _minimum_image_xy(
    child_position: torch.Tensor,
    parent_position: torch.Tensor,
    *,
    lx_m: float,
    ly_m: float,
) -> torch.Tensor:
    periods = child_position.new_tensor([lx_m, ly_m])
    delta = child_position[:2] - parent_position[:2]
    return torch.remainder(delta + 0.5 * periods, periods) - 0.5 * periods


def _periodic_displacement_m(
    before: torch.Tensor,
    after: torch.Tensor,
    *,
    lx_m: float,
    ly_m: float,
) -> float:
    horizontal = _minimum_image_xy(after, before, lx_m=lx_m, ly_m=ly_m)
    delta = torch.cat((horizontal, (after[2:] - before[2:])))
    return float(torch.linalg.vector_norm(delta).detach().cpu())


def _developed_child_matches_genotype(state, child_slot: int) -> tuple[bool, list[str]]:
    redeveloped = develop_unchecked(state.genotype)
    mismatches: list[str] = []
    for field in fields(state.body):
        actual = getattr(state.body, field.name)[0, child_slot]
        expected = getattr(redeveloped, field.name)[0, child_slot]
        if actual.dtype.is_floating_point:
            equal = torch.allclose(actual, expected, rtol=1.0e-6, atol=1.0e-8)
        else:
            equal = torch.equal(actual, expected)
        if not equal:
            mismatches.append(field.name)
    return not mismatches, mismatches


def _mutated_genotype_fields(state, parent_slot: int, child_slot: int) -> list[str]:
    changed: list[str] = []
    for field in fields(state.genotype):
        if field.name in ("alive", "stable_id"):
            continue
        parent = getattr(state.genotype, field.name)[0, parent_slot]
        child = getattr(state.genotype, field.name)[0, child_slot]
        if not torch.equal(parent, child):
            changed.append(field.name)
    return changed


def _physical_behavior_state_unchanged(before, controlled) -> bool:
    return all(
        torch.equal(getattr(before, name), getattr(controlled, name))
        for name in (
            "position_enu_m",
            "velocity_rel_water_enu_m_s",
            "yaw_rad",
            "yaw_momentum_kg_m2_s",
            "gait_time_s",
        )
    )


def _static_actuator_control(before, behavior, backend) -> dict[str, object]:
    static_body = replace(
        before.body,
        joint_amp_rad=torch.zeros_like(before.body.joint_amp_rad),
    )
    morphology = query_morphology(static_body, backend.config.live)
    budget_j = available_actuator_work_j(
        before.population,
        morphology.structural_mass_kg,
        backend.config.metabolism,
    )
    advance = advance_affordable_motion(
        static_body,
        behavior.motion,
        backend.fluid,
        backend.config.live,
        backend.config.geometry,
        backend.config.motion,
        requested_effort=behavior.requested_effort_fraction,
        budget_j=budget_j,
    )
    alive = before.population.alive[0]
    displacements = [
        _periodic_displacement_m(
            before.motion.position_enu_m[0, slot],
            advance.state.position_enu_m[0, slot],
            lx_m=backend.config.geometry.lx_m,
            ly_m=backend.config.geometry.ly_m,
        )
        for slot in alive.nonzero().flatten().tolist()
    ]
    positive_work_j = float(
        advance.ledger.response.positive_actuator_work_j.sum().detach().cpu()
    )
    return {
        "actuator": "developed joint amplitudes set to zero in a read-only control",
        "positive_actuator_work_j": positive_work_j,
        "max_displacement_m": max(displacements, default=0.0),
        "stationary": positive_work_j == 0.0
        and max(displacements, default=0.0) <= 1.0e-9,
    }


def run_living_loop_proof(
    *,
    device_name: str,
    max_duration_s: float = 300.0,
    seed: int = 20260809,
    compile_domains: bool | None = None,
) -> dict[str, object]:
    """Run until one naturally mutated child independently moves and feeds."""

    max_intervals = _whole_intervals(max_duration_s)
    if isinstance(seed, bool) or not isinstance(seed, int) or not 0 <= seed < 2**63:
        raise ValueError("seed must be an integer in [0,2^63)")
    device = torch.device(device_name)
    if device.type not in ("cpu", "cuda"):
        raise ValueError("device must be cpu or cuda")
    if device.type == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA was requested but is not available")
    compile_runtime = device.type == "cuda" if compile_domains is None else compile_domains

    world = _build_server_world(device=device)
    _seed_visible_baseline(world, seed=seed)
    backend = RuntimeUnityBackend.from_reference_fixture(
        world,
        compile_domains=compile_runtime,
        profile=CAUSAL_RUNTIME_PROFILE,
    )
    warmup_s = 0.0
    if compile_runtime:
        print("living-loop proof: prewarming compiled CUDA domains", file=sys.stderr, flush=True)
        warmup_started = time.perf_counter()
        backend.prewarm()
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        warmup_s = time.perf_counter() - warmup_started
        print(
            f"living-loop proof: warmup complete in {warmup_s:.1f}s; starting step 1",
            file=sys.stderr,
            flush=True,
        )

    state = backend.session.state
    initial_matter_q = _direct_matter_census(state)
    expected_matter_q = int(state.expected_matter_q[0].detach().cpu())
    initial_producer_q = int(state.economy.bp_q.sum().detach().cpu())
    world_volume_m3 = (
        backend.config.geometry.lx_m
        * backend.config.geometry.ly_m
        * backend.config.geometry.lz_m
    )
    initial_producer_mol_m3 = (
        initial_producer_q
        * backend.config.economy.q_mass_mol
        / world_volume_m3
    )
    if initial_matter_q != expected_matter_q:
        raise RuntimeError("independent initial matter census differs from authority")

    movement_seen_slots = torch.zeros_like(state.population.alive)
    feeding_seen_slots = torch.zeros_like(state.population.alive)
    gradient_intervals = 0
    locomoting_intervals = 0
    requested_effort_sum = 0.0
    actuator_work_j = 0.0
    exact_local_debit = True
    direct_matter_closed = True
    runtime_books_closed = True
    energy_boundaries_closed = True
    behavior_physical_state_unchanged = True
    finite = True
    previous_debit = None
    previous_prefeeding_producer = None
    depletion_counterfactual = None
    static_control = None
    first_local_debit = None
    mutated_birth = None
    child_independence = {
        "first_active_interval_observed": False,
        "sampled_food_mol_m3": 0.0,
        "locomoting": False,
        "accepted_effort": 0.0,
        "positive_actuator_work_j": 0.0,
        "physical_displacement_m": 0.0,
        "feeding_debit_q": 0,
    }
    total_feeding_debit_q = 0
    completed_intervals = 0
    run_started = time.perf_counter()

    with torch.inference_mode():
        for interval_index in range(1, max_intervals + 1):
            before = backend.session.state
            chunk = backend.session.advance_autonomous_chunk(
                backend.fluid,
                intervals=1,
            )
            behavior = chunk.last_behavior
            if behavior is None:
                raise RuntimeError("autonomous proof did not receive behavior state")
            ledger = chunk.last_interval
            after = chunk.state
            completed_intervals = interval_index

            behavior_physical_state_unchanged &= _physical_behavior_state_unchanged(
                before.motion,
                behavior.motion,
            )
            if static_control is None:
                static_control = _static_actuator_control(before, behavior, backend)

            periods = before.motion.position_enu_m.new_tensor(
                [backend.config.geometry.lx_m, backend.config.geometry.ly_m]
            )
            position_delta_xy = (
                ledger.motion.state.position_enu_m[..., :2]
                - before.motion.position_enu_m[..., :2]
            )
            position_delta_xy = torch.remainder(
                position_delta_xy + 0.5 * periods,
                periods,
            ) - 0.5 * periods
            position_delta = torch.cat(
                (
                    position_delta_xy,
                    ledger.motion.state.position_enu_m[..., 2:]
                    - before.motion.position_enu_m[..., 2:],
                ),
                dim=-1,
            )
            movement_seen_slots |= before.population.alive & (
                torch.linalg.vector_norm(position_delta, dim=-1) > 0.0
            )
            feeding_seen_slots |= ledger.feeding.ledger.actual_debit_q > 0

            gradient_intervals += int(
                behavior.horizontal_gradient_present.sum().detach().cpu()
            )
            locomoting_intervals += int(behavior.locomoting.sum().detach().cpu())
            requested_effort_sum += float(
                behavior.requested_effort_fraction.sum().detach().cpu()
            )
            interval_work_j = float(
                ledger.motion.ledger.response.positive_actuator_work_j.sum().detach().cpu()
            )
            actuator_work_j += interval_work_j

            producer_before_feeding = ledger.economy.state.bp_q
            producer_after_feeding = ledger.feeding.producer_q
            debit_by_cell = ledger.feeding.ledger.producer_debit_by_cell_q
            exact_local_debit &= torch.equal(
                producer_before_feeding - producer_after_feeding,
                debit_by_cell,
            )
            interval_debit_q = int(debit_by_cell.sum().detach().cpu())
            total_feeding_debit_q += interval_debit_q
            if first_local_debit is None and interval_debit_q > 0:
                cell = (debit_by_cell[0] > 0).nonzero()[0]
                cell_index = [int(value) for value in cell.detach().cpu().tolist()]
                first_local_debit = {
                    "step": interval_index,
                    "time_s": interval_index * ECONOMY_INTERVAL_S,
                    "cell_xyz": cell_index,
                    "cell_debit_q": int(
                        debit_by_cell[(0, *cell_index)].detach().cpu()
                    ),
                    "interval_debit_q": interval_debit_q,
                }

            if previous_debit is not None and depletion_counterfactual is None:
                restored = before.economy.bp_q + previous_debit
                if not torch.equal(restored, previous_prefeeding_producer):
                    raise RuntimeError("previous local feeding debit did not reconstruct")
                counterfactual = request_living_intent(
                    before.population,
                    before.body,
                    before.motion,
                    restored,
                    backend.config.geometry,
                    backend.config.live,
                    backend.config.behavior,
                    q_mass_mol=backend.config.economy.q_mass_mol,
                )
                heading_delta = torch.linalg.vector_norm(
                    behavior.requested_heading_enu
                    - counterfactual.requested_heading_enu,
                    dim=-1,
                )
                gradient_delta = torch.linalg.vector_norm(
                    behavior.producer_gradient_mol_m4
                    - counterfactual.producer_gradient_mol_m4,
                    dim=-1,
                )
                changed = before.population.alive & (heading_delta > 1.0e-9)
                if bool(changed.any().detach().cpu()):
                    slot = int(changed[0].nonzero()[0].detach().cpu())
                    depletion_counterfactual = {
                        "step": interval_index,
                        "creature_id": int(
                            before.population.stable_id[0, slot].detach().cpu()
                        ),
                        "heading_delta": float(heading_delta[0, slot].detach().cpu()),
                        "gradient_delta_mol_m4": float(
                            gradient_delta[0, slot].detach().cpu()
                        ),
                        "claim": (
                            "restoring only the prior interval's exact feeding debit "
                            "changed the requested heading"
                        ),
                    }

            previous_debit = debit_by_cell.detach().clone()
            previous_prefeeding_producer = producer_before_feeding.detach().clone()

            direct_matter_closed &= bool(
                (
                    _direct_matter_census_tensor(after) == initial_matter_q
                ).detach().cpu()
            )
            runtime_books_closed &= bool(
                ledger.matter.books_closed.all().detach().cpu()
            ) and bool(ledger.economy.ledger.books_closed.all().detach().cpu())
            finite &= not bool(ledger.invalid.any().detach().cpu())

            feeding_energy = ledger.feeding.ledger
            reserve_j_per_q = backend.config.feeding.reserve_j_per_q
            feeding_lhs = (
                feeding_energy.producer_chemical_input_j
                + before.population.assimilation_carry_q * reserve_j_per_q
            )
            feeding_rhs = (
                feeding_energy.reserve_chemical_credit_j
                + feeding_energy.assimilation_heat_j
                + ledger.feeding.population.assimilation_carry_q * reserve_j_per_q
            )
            metabolism = ledger.organisms.metabolism.ledger
            metabolism_rhs = (
                metabolism.requested_q.to(torch.float64) * reserve_j_per_q
                + metabolism.carry_after_j
                + metabolism.quantization_residual_j
            )
            energy_boundaries_closed &= torch.allclose(
                feeding_lhs,
                feeding_rhs,
                rtol=0.0,
                atol=1.0e-9,
            ) and torch.allclose(
                metabolism.total_demand_j,
                metabolism_rhs,
                rtol=0.0,
                atol=1.0e-9,
            )

            lifecycle = ledger.organisms.lifecycle.ledger
            mutation = ledger.mutation.ledger
            for child_slot in lifecycle.born[0].nonzero().flatten().tolist():
                if mutated_birth is not None:
                    break
                if int(mutation.mutation_count[0, child_slot].detach().cpu()) == 0:
                    continue
                parent_slot = int(
                    lifecycle.parent_slot_for_child[0, child_slot].detach().cpu()
                )
                child_id = int(after.population.stable_id[0, child_slot].detach().cpu())
                parent_id = int(after.population.stable_id[0, parent_slot].detach().cpu())
                radii = developed_support_radius_m(after.body)
                actual_separation_m = float(
                    torch.linalg.vector_norm(
                        _minimum_image_xy(
                            after.motion.position_enu_m[0, child_slot],
                            after.motion.position_enu_m[0, parent_slot],
                            lx_m=backend.config.geometry.lx_m,
                            ly_m=backend.config.geometry.ly_m,
                        )
                    ).detach().cpu()
                )
                required_separation_m = float(
                    (
                        radii[0, parent_slot]
                        + radii[0, child_slot]
                        + backend.config.birth_separation_clearance_m
                    ).detach().cpu()
                )
                impulse_residual = (
                    ledger.release.parent_impulse_enu_ns[0, parent_slot]
                    + ledger.release.child_impulse_enu_ns[0, child_slot]
                )
                release_q = int(
                    lifecycle.birth_release_energy_return_q[0, parent_slot]
                    .detach()
                    .cpu()
                )
                release_chemical_j = float(
                    ledger.energy.birth_release_chemical_input_j[0, parent_slot]
                    .detach()
                    .cpu()
                )
                release_kinetic_j = float(
                    ledger.release.kinetic_delta_j[0, parent_slot].detach().cpu()
                )
                release_heat_j = float(
                    ledger.energy.birth_release_heat_j[0, parent_slot].detach().cpu()
                )
                developed_match, developed_mismatches = (
                    _developed_child_matches_genotype(after, child_slot)
                )
                changed_genotype_fields = _mutated_genotype_fields(
                    after,
                    parent_slot,
                    child_slot,
                )
                mutated_birth = {
                    "step": interval_index,
                    "time_s": interval_index * ECONOMY_INTERVAL_S,
                    "parent_id": parent_id,
                    "child_id": child_id,
                    "parent_generation": int(
                        after.population.generation[0, parent_slot].detach().cpu()
                    ),
                    "child_generation": int(
                        after.population.generation[0, child_slot].detach().cpu()
                    ),
                    "mutation_count": int(
                        mutation.mutation_count[0, child_slot].detach().cpu()
                    ),
                    "changed_genotype_fields": changed_genotype_fields,
                    "developed_body_matches_mutated_genotype": developed_match,
                    "developed_body_mismatches": developed_mismatches,
                    "structure_transfer_q": int(
                        lifecycle.birth_structure_transfer_q[0, parent_slot]
                        .detach()
                        .cpu()
                    ),
                    "child_structure_q": int(
                        after.population.structure_q[0, child_slot].detach().cpu()
                    ),
                    "reserve_transfer_q": int(
                        lifecycle.birth_reserve_transfer_q[0, parent_slot]
                        .detach()
                        .cpu()
                    ),
                    "release_energy_q": release_q,
                    "release_chemical_input_j": release_chemical_j,
                    "release_kinetic_delta_j": release_kinetic_j,
                    "release_heat_j": release_heat_j,
                    "release_energy_closed": math.isclose(
                        release_chemical_j,
                        release_kinetic_j + release_heat_j,
                        rel_tol=0.0,
                        abs_tol=1.0e-9,
                    ),
                    "impulse_residual_ns": [
                        float(value) for value in impulse_residual.detach().cpu().tolist()
                    ],
                    "impulse_conserved": bool(
                        torch.allclose(
                            impulse_residual,
                            torch.zeros_like(impulse_residual),
                            rtol=0.0,
                            atol=1.0e-9,
                        )
                    ),
                    "actual_separation_m": actual_separation_m,
                    "required_separation_m": required_separation_m,
                    "nonoverlapping": actual_separation_m >= required_separation_m,
                    "child_heading_uninitialized": not bool(
                        after.motion.heading_initialized[0, child_slot].detach().cpu()
                    ),
                }
                break

            if (
                mutated_birth is not None
                and not child_independence["first_active_interval_observed"]
                and interval_index > int(mutated_birth["step"])
            ):
                child_id = int(mutated_birth["child_id"])
                child_match = before.population.alive[0] & (
                    before.population.stable_id[0] == child_id
                )
                if bool(child_match.any().detach().cpu()):
                    child_slot = int(child_match.nonzero()[0, 0].detach().cpu())
                    child_independence["first_active_interval_observed"] = True
                    child_independence["sampled_food_mol_m3"] = float(
                        behavior.sampled_producer_mol_m3[0, child_slot].detach().cpu()
                    )
                    child_independence["locomoting"] = bool(
                        behavior.locomoting[0, child_slot].detach().cpu()
                    )
                    child_independence["accepted_effort"] = float(
                        ledger.motion.ledger.selected.effort_fraction[0, child_slot]
                        .detach()
                        .cpu()
                    )
                    child_independence["positive_actuator_work_j"] = float(
                        ledger.motion.ledger.response.positive_actuator_work_j[
                            0, child_slot
                        ]
                        .detach()
                        .cpu()
                    )
                    child_independence["physical_displacement_m"] = (
                        _periodic_displacement_m(
                            before.motion.position_enu_m[0, child_slot],
                            ledger.motion.state.position_enu_m[0, child_slot],
                            lx_m=backend.config.geometry.lx_m,
                            ly_m=backend.config.geometry.ly_m,
                        )
                    )
                    child_independence["feeding_debit_q"] = int(
                        ledger.feeding.ledger.actual_debit_q[0, child_slot]
                        .detach()
                        .cpu()
                    )

            child_complete = bool(
                child_independence["first_active_interval_observed"]
                and child_independence["sampled_food_mol_m3"] > 0.0
                and child_independence["locomoting"]
                and child_independence["accepted_effort"] > 0.0
                and child_independence["positive_actuator_work_j"] > 0.0
                and child_independence["physical_displacement_m"] > 0.0
                and child_independence["feeding_debit_q"] > 0
            )
            if (
                mutated_birth is not None
                and child_complete
                and depletion_counterfactual is not None
                and total_feeding_debit_q > 0
                and actuator_work_j > 0.0
            ):
                break
            if interval_index % 100 == 0:
                print(
                    "living-loop proof: "
                    f"step={interval_index} "
                    f"sim={interval_index * ECONOMY_INTERVAL_S:.1f}s "
                    f"population={int(after.population.alive.sum().detach().cpu())} "
                    f"mutated_birth={'yes' if mutated_birth is not None else 'no'}",
                    file=sys.stderr,
                    flush=True,
                )

    if device.type == "cuda":
        torch.cuda.synchronize(device)
    run_elapsed_s = time.perf_counter() - run_started
    final_state = backend.session.state
    final_time_s = float(final_state.economy.time_s.detach().cpu())
    final_population = int(final_state.population.alive.sum().detach().cpu())

    moved_slots = int(movement_seen_slots.sum().detach().cpu())
    feeding_slots = int(feeding_seen_slots.sum().detach().cpu())
    child_complete = bool(
        child_independence["first_active_interval_observed"]
        and child_independence["sampled_food_mol_m3"] > 0.0
        and child_independence["locomoting"]
        and child_independence["accepted_effort"] > 0.0
        and child_independence["positive_actuator_work_j"] > 0.0
        and child_independence["physical_displacement_m"] > 0.0
        and child_independence["feeding_debit_q"] > 0
    )
    birth_complete = bool(
        mutated_birth is not None
        and mutated_birth["mutation_count"] > 0
        and mutated_birth["changed_genotype_fields"]
        and mutated_birth["developed_body_matches_mutated_genotype"]
        and mutated_birth["structure_transfer_q"]
        == mutated_birth["child_structure_q"]
        and mutated_birth["reserve_transfer_q"]
        == backend.config.child_initial_reserve_q
        and mutated_birth["release_energy_q"] > 0
        and mutated_birth["release_energy_closed"]
        and mutated_birth["impulse_conserved"]
        and mutated_birth["nonoverlapping"]
        and mutated_birth["child_heading_uninitialized"]
        and mutated_birth["child_generation"]
        == mutated_birth["parent_generation"] + 1
    )
    claims = {
        "local_food_sampled_and_debited": total_feeding_debit_q > 0,
        "local_debit_exact_by_cell": exact_local_debit,
        "depletion_changed_later_heading": depletion_counterfactual is not None,
        "behavior_did_not_write_physical_state": behavior_physical_state_unchanged,
        "paid_physics_motion": actuator_work_j > 0.0 and moved_slots > 0,
        "actuator_absent_control_stationary": bool(
            static_control is not None and static_control["stationary"]
        ),
        "mutated_paid_birth": birth_complete,
        "mutated_child_independently_sensed_moved_and_fed": child_complete,
        "raw_matter_census_closed_every_interval": direct_matter_closed,
        "runtime_books_closed_every_interval": runtime_books_closed,
        "named_energy_boundaries_closed_every_interval": energy_boundaries_closed,
        "finite_valid_state_every_interval": finite,
    }
    missing = [name for name, passed in claims.items() if not passed]
    replay = None
    if mutated_birth is not None:
        replay = {
            "profile": CAUSAL_RUNTIME_PROFILE.name,
            "seed": seed,
            "fast_forward_seconds": max(
                0.0,
                float(mutated_birth["time_s"]) - ECONOMY_INTERVAL_S,
            ),
            "expected_mutated_birth_step": mutated_birth["step"],
            "expected_child_id": mutated_birth["child_id"],
        }
    return {
        "schema": "sirrobin.living-loop-proof.v1",
        "verdict": {"passed": not missing, "missing_claims": missing},
        "configuration": {
            "device": str(device),
            "compiled_domains": compile_runtime,
            "profile": CAUSAL_RUNTIME_PROFILE.name,
            "profile_description": CAUSAL_RUNTIME_PROFILE.description,
            "position_seed": seed,
            "mutation_seed": backend.config.mutation.seed,
            "mutation_rate_per_locus": (
                backend.config.mutation.mutation_rate_per_locus
            ),
            "authoritative_interval_s": ECONOMY_INTERVAL_S,
            "max_duration_s": max_duration_s,
            "initial_population": INITIAL_BODIES,
            "initial_environment": {
                "producer_q": initial_producer_q,
                "mean_producer_mol_m3": initial_producer_mol_m3,
                "world_volume_m3": world_volume_m3,
            },
        },
        "execution": {
            "warmup_wall_s": warmup_s,
            "run_wall_s": run_elapsed_s,
            "completed_intervals": completed_intervals,
            "final_time_s": final_time_s,
            "sim_s_per_wall_s": (
                final_time_s / run_elapsed_s if run_elapsed_s > 0.0 else None
            ),
            "final_population": final_population,
        },
        "claims": claims,
        "evidence": {
            "initial_and_expected_matter_q": initial_matter_q,
            "total_feeding_debit_q": total_feeding_debit_q,
            "first_local_debit": first_local_debit,
            "depletion_counterfactual": depletion_counterfactual,
            "behavior": {
                "food_gradient_intervals": gradient_intervals,
                "locomoting_intervals": locomoting_intervals,
                "requested_effort_sum": requested_effort_sum,
                "physical_state_unchanged_by_behavior": (
                    behavior_physical_state_unchanged
                ),
            },
            "motion": {
                "positive_actuator_work_j": actuator_work_j,
                "slots_with_observed_movement": moved_slots,
                "slots_with_observed_feeding": feeding_slots,
                "actuator_absent_control": static_control,
            },
            "mutated_birth": mutated_birth,
            "child_first_active_interval": child_independence,
            "conservation": {
                "direct_matter_census_closed": direct_matter_closed,
                "runtime_books_closed": runtime_books_closed,
                "named_energy_boundaries_closed": energy_boundaries_closed,
            },
        },
        "unity_replay": replay,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--max-duration-s", type=float, default=300.0)
    parser.add_argument("--seed", type=int, default=20260809)
    parser.add_argument(
        "--eager",
        action="store_true",
        help="disable torch compilation (useful only for short CPU checks)",
    )
    args = parser.parse_args()
    torch.set_num_threads(1)
    report = run_living_loop_proof(
        device_name=args.device,
        max_duration_s=args.max_duration_s,
        seed=args.seed,
        compile_domains=False if args.eager else None,
    )
    print(json.dumps(report, indent=2, sort_keys=True, allow_nan=False))
    if not report["verdict"]["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
