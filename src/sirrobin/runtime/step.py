"""Thin ordering of one complete device-oriented living interval."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, replace

import torch

from sirrobin.economy.step import EconomyAdvance, advance_economy_unchecked
from sirrobin.genetics.develop import develop_unchecked
from sirrobin.genetics.genotype import GenotypeBatch
from sirrobin.organisms.body_cache import commit_developed_births
from sirrobin.organisms.development import (
    DevelopmentConfig,
    DevelopmentState,
    settle_development_lifecycle,
    target_structure_cost_q,
)
from sirrobin.organisms.feeding import FeedingStep, feed_population
from sirrobin.organisms.metabolism import (
    MetabolismInputs,
    available_actuator_work_j,
)
from sirrobin.organisms.mutation import (
    MutationConfig,
    MutationStep,
    commit_offspring_mutations,
    empty_mutation_step,
    propose_offspring_mutations,
)
from sirrobin.physics.contracts import DevelopedBody, FluidSample
from sirrobin.physics.ecological_motion import (
    AffordableMotionAdvance,
    advance_affordable_motion,
)
from sirrobin.physics.morphology import query_morphology
from sirrobin.runtime.config import LivingRuntimeConfig
from sirrobin.runtime.material import (
    ReturnDeposit,
    RuntimeMatterLedger,
    close_runtime_matter,
    deposit_organism_returns,
    total_matter_q,
)
from sirrobin.runtime.motion_state import (
    BirthReleaseLedger,
    propose_birth_release,
    settle_motion_lifecycle,
)
from sirrobin.runtime.organism_step import (
    OrganismIntervalInputs,
    OrganismIntervalStep,
    advance_organism_interval,
)
from sirrobin.runtime.state import LivingState


@dataclass(frozen=True, slots=True)
class LivingIntervalInputs:
    fluid: FluidSample
    requested_effort: torch.Tensor
    birth_requested: torch.Tensor


@dataclass(frozen=True, slots=True)
class LivingEnergyLedger:
    light_chemical_input_j: torch.Tensor
    producer_maintenance_dissipation_j: torch.Tensor
    producer_mortality_dissipation_j: torch.Tensor
    feeding_producer_input_j: torch.Tensor
    feeding_reserve_credit_j: torch.Tensor
    assimilation_heat_j: torch.Tensor
    positive_actuator_work_j: torch.Tensor
    actuator_braking_heat_j: torch.Tensor
    muscle_inefficiency_heat_j: torch.Tensor
    maintenance_chemical_output_j: torch.Tensor
    motion_dissipation_j: torch.Tensor
    death_kinetic_and_carry_dissipation_j: torch.Tensor
    death_reserve_dissipation_j: torch.Tensor
    birth_construction_heat_j: torch.Tensor
    birth_release_chemical_input_j: torch.Tensor
    birth_release_kinetic_delta_j: torch.Tensor
    birth_release_heat_j: torch.Tensor


@dataclass(frozen=True, slots=True)
class LivingIntervalLedger:
    motion: AffordableMotionAdvance
    economy: EconomyAdvance
    feeding: FeedingStep
    organisms: OrganismIntervalStep
    returns: ReturnDeposit
    mutation: MutationStep
    release: BirthReleaseLedger
    candidate_work_deferred: torch.Tensor
    candidate_slots_evaluated: torch.Tensor
    candidate_replay_required: torch.Tensor
    matter: RuntimeMatterLedger
    energy: LivingEnergyLedger
    motion_funding_unresolved: torch.Tensor
    feeding_allocation_unresolved: torch.Tensor
    invalid: torch.Tensor


@dataclass(frozen=True, slots=True)
class LivingIntervalAdvance:
    state: LivingState
    ledger: LivingIntervalLedger


@dataclass(frozen=True, slots=True)
class BirthCandidateStep:
    """Parent-indexed mutation, development, and exact structural price."""

    mutation: MutationStep
    body: DevelopedBody
    structure_q: torch.Tensor
    truncated: torch.Tensor
    deferred: torch.Tensor
    evaluated_slots: torch.Tensor
    replay_required: torch.Tensor


def prepare_birth_candidates_with_kernels(
    genotype: GenotypeBatch,
    body: DevelopedBody,
    requested_birth: torch.Tensor,
    event_index: torch.Tensor,
    mutation_config: MutationConfig,
    development_config: DevelopmentConfig,
    *,
    mutation_kernel: Callable[..., MutationStep],
    development_kernel: Callable[[GenotypeBatch], DevelopedBody],
    structure_cost_kernel: Callable[..., torch.Tensor],
) -> BirthCandidateStep:
    """Evaluate every parent slot through explicit domain-kernel boundaries."""

    proposal = mutation_kernel(
        genotype,
        body,
        requested_birth,
        event_index,
        mutation_config,
    )
    candidate_body = development_kernel(proposal.genotype)
    candidate_structure_q = structure_cost_kernel(
        candidate_body,
        development_config,
    )
    worlds, capacity = requested_birth.shape
    return BirthCandidateStep(
        proposal,
        candidate_body,
        candidate_structure_q,
        candidate_body.truncated_candidate_count > 0,
        torch.zeros(worlds, dtype=torch.bool, device=requested_birth.device),
        torch.full(
            (worlds,),
            capacity,
            dtype=torch.int64,
            device=requested_birth.device,
        ),
        torch.zeros(worlds, dtype=torch.bool, device=requested_birth.device),
    )


def prepare_dense_birth_candidates(
    genotype: GenotypeBatch,
    body: DevelopedBody,
    requested_birth: torch.Tensor,
    event_index: torch.Tensor,
    mutation_config: MutationConfig,
    development_config: DevelopmentConfig,
) -> BirthCandidateStep:
    """Evaluate every parent slot as the exact reference candidate path."""

    return prepare_birth_candidates_with_kernels(
        genotype,
        body,
        requested_birth,
        event_index,
        mutation_config,
        development_config,
        mutation_kernel=propose_offspring_mutations,
        development_kernel=develop_unchecked,
        structure_cost_kernel=target_structure_cost_q,
    )


def defer_birth_candidates(
    genotype: GenotypeBatch,
    body: DevelopedBody,
    requested_birth: torch.Tensor,
    event_index: torch.Tensor,
    mutation_config: MutationConfig,
    development_config: DevelopmentConfig,
) -> BirthCandidateStep:
    """Skip candidate work and flag every live request for exact chunk replay."""

    del event_index, development_config
    worlds = requested_birth.shape[0]
    device = requested_birth.device
    replay_required = requested_birth & genotype.alive
    mutation = empty_mutation_step(
        genotype,
        mutation_config.max_mutations_per_birth,
    )
    return BirthCandidateStep(
        mutation,
        body,
        torch.zeros_like(requested_birth, dtype=torch.int64),
        replay_required,
        torch.ones(worlds, dtype=torch.bool, device=device),
        torch.zeros(worlds, dtype=torch.int64, device=device),
        replay_required.any(dim=1),
    )


@dataclass(frozen=True, slots=True)
class LivingKernels:
    motion: Callable[..., AffordableMotionAdvance]
    economy: Callable[..., EconomyAdvance]
    feeding: Callable[..., FeedingStep]
    organisms: Callable[..., OrganismIntervalStep]
    returns: Callable[..., ReturnDeposit]
    candidates: Callable[..., BirthCandidateStep]
    commit_mutation: Callable[..., MutationStep]
    body_cache: Callable[..., DevelopedBody]
    development: Callable[..., DevelopmentState]


EAGER_LIVING_KERNELS = LivingKernels(
    motion=advance_affordable_motion,
    economy=advance_economy_unchecked,
    feeding=feed_population,
    organisms=advance_organism_interval,
    returns=deposit_organism_returns,
    candidates=prepare_dense_birth_candidates,
    commit_mutation=commit_offspring_mutations,
    body_cache=commit_developed_births,
    development=settle_development_lifecycle,
)


def _runtime_energy_ledger(
    economy: EconomyAdvance,
    feeding: FeedingStep,
    organisms: OrganismIntervalStep,
    motion: AffordableMotionAdvance,
    release: BirthReleaseLedger,
    config: LivingRuntimeConfig,
) -> LivingEnergyLedger:
    producer_j = config.feeding.producer_j_per_q
    reserve_j = config.metabolism.reserve_j_per_q
    lifecycle = organisms.lifecycle.ledger
    metabolism = organisms.metabolism.ledger
    return LivingEnergyLedger(
        light_chemical_input_j=economy.ledger.production_q.to(torch.float64)
        * producer_j,
        producer_maintenance_dissipation_j=(
            economy.ledger.producer_maintenance_q.to(torch.float64) * producer_j
        ),
        producer_mortality_dissipation_j=(
            economy.ledger.producer_mortality_q.to(torch.float64) * producer_j
        ),
        feeding_producer_input_j=feeding.ledger.producer_chemical_input_j,
        feeding_reserve_credit_j=feeding.ledger.reserve_chemical_credit_j,
        assimilation_heat_j=feeding.ledger.assimilation_heat_j,
        positive_actuator_work_j=metabolism.positive_actuator_work_j,
        actuator_braking_heat_j=metabolism.actuator_braking_heat_j,
        muscle_inefficiency_heat_j=metabolism.muscle_inefficiency_heat_j,
        maintenance_chemical_output_j=metabolism.maintenance_heat_j,
        motion_dissipation_j=motion.ledger.response.dissipated_work_j,
        death_kinetic_and_carry_dissipation_j=metabolism.death_dissipation_j,
        death_reserve_dissipation_j=(
            lifecycle.death_reserve_return_q.to(torch.float64) * reserve_j
        ),
        birth_construction_heat_j=(
            lifecycle.birth_structure_transfer_q.to(torch.float64) * reserve_j
        ),
        birth_release_chemical_input_j=(
            lifecycle.birth_release_energy_return_q.to(torch.float64) * reserve_j
        ),
        birth_release_kinetic_delta_j=release.kinetic_delta_j,
        birth_release_heat_j=(
            lifecycle.birth_release_energy_return_q.to(torch.float64) * reserve_j
            - release.kinetic_delta_j
        ),
    )


def advance_living_interval(
    state: LivingState,
    inputs: LivingIntervalInputs,
    config: LivingRuntimeConfig,
) -> LivingIntervalAdvance:
    return advance_living_interval_with_motion(
        state,
        inputs,
        config,
        motion_kernel=advance_affordable_motion,
    )


def advance_living_interval_with_motion(
    state: LivingState,
    inputs: LivingIntervalInputs,
    config: LivingRuntimeConfig,
    *,
    motion_kernel: Callable[..., AffordableMotionAdvance],
) -> LivingIntervalAdvance:
    return advance_living_interval_with_kernels(
        state,
        inputs,
        config,
        replace(EAGER_LIVING_KERNELS, motion=motion_kernel),
    )


def advance_living_interval_with_kernels(
    state: LivingState,
    inputs: LivingIntervalInputs,
    config: LivingRuntimeConfig,
    kernels: LivingKernels,
) -> LivingIntervalAdvance:
    """Advance one field, motion, feeding, metabolism, and lifecycle interval.

    Validation and failure-policy decisions belong to the session boundary. This
    function performs no device-to-host read and contains no domain equation of its
    own; it only routes explicit outputs into the next domain transaction.
    """

    matter_before_q = total_matter_q(state.economy, state.population)
    morphology = query_morphology(state.body, config.live)
    budget_j = available_actuator_work_j(
        state.population,
        morphology.structural_mass_kg,
        config.metabolism,
    )
    motion = kernels.motion(
        state.body,
        state.motion,
        inputs.fluid,
        config.live,
        config.geometry,
        config.motion,
        requested_effort=inputs.requested_effort,
        budget_j=budget_j,
    )
    economy = kernels.economy(state.economy, config.economy)
    feeding = kernels.feeding(
        state.population,
        economy.state.bp_q,
        economy.state.nd_q,
        motion.state.position_enu_m,
        motion.state.velocity_rel_water_enu_m_s,
        morphology.intake_area_m2,
        config.geometry,
        config.feeding,
    )
    economy_after_feeding = replace(
        economy.state,
        nd_q=feeding.dissolved_q,
        bp_q=feeding.producer_q,
    )
    birth_requested = inputs.birth_requested & feeding.population.alive
    candidates = kernels.candidates(
        state.genotype,
        state.body,
        birth_requested,
        economy.state.step,
        config.mutation,
        config.development,
    )
    proposal = candidates.mutation
    candidate_body = candidates.body
    candidate_structure_q = candidates.structure_q
    candidate_truncated = candidates.truncated
    release_proposal = propose_birth_release(
        state.body,
        candidate_body,
        motion.state,
        motion.ledger.response.effective_mass_after_kg,
        birth_requested & ~candidate_truncated,
        config.geometry,
        config.live,
        impulse_ns=config.birth_release_impulse_ns,
        clearance_m=config.birth_separation_clearance_m,
        reserve_j_per_q=config.metabolism.reserve_j_per_q,
    )
    worlds = state.population.alive.shape[0]
    organism_inputs = OrganismIntervalInputs(
        metabolism=MetabolismInputs(
            structural_mass_kg=morphology.structural_mass_kg,
            positive_actuator_work_j=motion.ledger.response.positive_actuator_work_j,
            actuator_braking_work_j=motion.ledger.response.actuator_braking_work_j,
            old_age_due=torch.zeros_like(state.population.alive),
            velocity_enu_m_s=motion.state.velocity_rel_water_enu_m_s,
            yaw_momentum_kg_m2_s=motion.state.yaw_momentum_kg_m2_s,
            effective_mass_after_kg=(
                motion.ledger.response.effective_mass_after_kg
            ),
            yaw_inertia_after_kg_m2=(
                motion.ledger.response.yaw_inertia_after_kg_m2
            ),
        ),
        birth_requested=(
            birth_requested & ~candidate_truncated & ~release_proposal.invalid
        ),
        child_structure_q=candidate_structure_q,
        child_reserve_q=torch.full_like(
            feeding.population.reserve_q,
            config.child_initial_reserve_q,
        ),
        birth_release_energy_q=release_proposal.release_energy_q,
        time_s=economy.state.time_s.expand(worlds),
    )
    organisms = kernels.organisms(
        feeding.population,
        organism_inputs,
        config.metabolism,
        config.mortality,
    )
    metabolism_ledger = organisms.metabolism.ledger
    lifecycle_ledger = organisms.lifecycle.ledger
    return_q = (
        metabolism_ledger.maintenance_return_q
        + lifecycle_ledger.death_structure_return_q
        + lifecycle_ledger.death_reserve_return_q
        + lifecycle_ledger.birth_release_energy_return_q
    )
    returns = kernels.returns(
        economy_after_feeding.nd_q,
        motion.state.position_enu_m,
        return_q,
        config.geometry,
    )
    economy_after_returns = replace(
        economy_after_feeding,
        nd_q=returns.dissolved_q,
    )
    mutation = kernels.commit_mutation(
        state.genotype,
        proposal,
        organisms.state,
        lifecycle_ledger,
    )
    body = kernels.body_cache(
        state.body,
        candidate_body,
        mutation.genotype,
        organisms.state,
        lifecycle_ledger,
    )
    development = kernels.development(
        state.development,
        organisms.state,
        body,
        lifecycle_ledger,
    )
    settled_motion, release = settle_motion_lifecycle(
        motion.state,
        organisms.state,
        lifecycle_ledger,
        release_proposal,
    )
    next_state = LivingState(
        organisms.state,
        mutation.genotype,
        body,
        development,
        settled_motion,
        economy_after_returns,
        state.expected_matter_q,
    )
    matter = close_runtime_matter(
        state.expected_matter_q,
        matter_before_q,
        next_state.economy,
        next_state.population,
    )
    motion_funding_unresolved = (
        state.population.alive & ~motion.ledger.funding_resolved
    ).any(dim=1)
    feeding_allocation_unresolved = (
        feeding.ledger.allocation_rounds_exhausted.any(dim=1)
    )
    invalid = (
        ~economy.ledger.books_closed
        | ~feeding.ledger.transaction_committed
        | ~returns.transaction_committed
        | ~matter.books_closed
        | motion.ledger.option_nonfinite.any(dim=(1, 2))
        | motion.ledger.response.yaw_backstop_hit.any(dim=1)
        | metabolism_ledger.invalid_death_kinetics.any(dim=1)
        | (birth_requested & candidate_truncated).any(dim=1)
        | (birth_requested & release_proposal.invalid).any(dim=1)
        | motion_funding_unresolved
    )
    energy = _runtime_energy_ledger(
        economy,
        feeding,
        organisms,
        motion,
        release,
        config,
    )
    return LivingIntervalAdvance(
        next_state,
        LivingIntervalLedger(
            motion,
            economy,
            feeding,
            organisms,
            returns,
            mutation,
            release,
            candidates.deferred,
            candidates.evaluated_slots,
            candidates.replay_required,
            matter,
            energy,
            motion_funding_unresolved,
            feeding_allocation_unresolved,
            invalid,
        ),
    )
