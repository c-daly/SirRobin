"""Compiled-kernel scheduling and chunk acceptance for the device runtime."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import torch

from sirrobin.economy.step import advance_economy_unchecked
from sirrobin.genetics.develop import develop_unchecked
from sirrobin.organisms.behavior import (
    BehaviorStep,
    request_living_intent,
)
from sirrobin.organisms.body_cache import commit_developed_births
from sirrobin.organisms.development import (
    settle_development_lifecycle,
    target_structure_cost_q,
)
from sirrobin.organisms.feeding import feed_population
from sirrobin.organisms.mutation import (
    SEGMENT_BUD,
    SEGMENT_VESTIGIAL,
    commit_offspring_mutations,
    propose_offspring_mutations,
)
from sirrobin.physics.contracts import FluidSample
from sirrobin.physics.ecological_motion import (
    AffordableMotionAdvance,
    advance_affordable_motion_with_stage,
    advance_requested_motion_with_stage,
)
from sirrobin.physics.phase_response import advance_phase_stage
from sirrobin.runtime.config import (
    LivingRuntimeConfig,
    validate_living_runtime_config,
)
from sirrobin.runtime.material import deposit_organism_returns
from sirrobin.runtime.organism_step import advance_organism_interval
from sirrobin.runtime.state import LivingState, validate_living_state
from sirrobin.runtime.step import (
    EAGER_LIVING_KERNELS,
    LivingIntervalInputs,
    LivingIntervalLedger,
    LivingKernels,
    advance_living_interval_with_kernels,
    defer_birth_candidates,
    prepare_birth_candidates_with_kernels,
)


@dataclass(frozen=True, slots=True)
class LivingChunkSummary:
    intervals: int
    births: torch.Tensor
    deaths: torch.Tensor
    starvation_deaths: torch.Tensor
    old_age_deaths: torch.Tensor
    requested_births: torch.Tensor
    unfunded_birth_rejections: torch.Tensor
    capacity_birth_rejections: torch.Tensor
    id_birth_rejections: torch.Tensor
    birth_release_energy_q: torch.Tensor
    mutated_births: torch.Tensor
    mutation_events: torch.Tensor
    parameter_mutation_events: torch.Tensor
    topology_mutation_events: torch.Tensor
    behavior_food_gradient_intervals: torch.Tensor
    behavior_locomoting_intervals: torch.Tensor
    feeding_requested_q: torch.Tensor
    feeding_actual_debit_q: torch.Tensor
    feeding_reserve_credit_q: torch.Tensor
    dissipation_j: torch.Tensor
    light_input_j: torch.Tensor


@dataclass(frozen=True, slots=True)
class LivingChunkAdvance:
    state: LivingState
    last_interval: LivingIntervalLedger
    invalid: torch.Tensor
    last_behavior: BehaviorStep | None = None
    summary: LivingChunkSummary | None = None


class _StagedMotionKernel:
    def __init__(self, stage_kernel: Any, *, requested_only: bool = False):
        self.stage_kernel = stage_kernel
        self.requested_only = requested_only

    def __call__(
        self,
        body,
        state,
        fluid,
        live_config,
        geometry,
        response_config,
        *,
        requested_effort,
        budget_j,
    ) -> AffordableMotionAdvance:
        function = (
            advance_requested_motion_with_stage
            if self.requested_only
            else advance_affordable_motion_with_stage
        )
        return function(
            body,
            state,
            fluid,
            live_config,
            geometry,
            response_config,
            requested_effort=requested_effort,
            budget_j=budget_j,
            stage_kernel=self.stage_kernel,
        )


class _DenseCandidateKernel:
    def __init__(
        self,
        mutation_kernel: Any,
        development_kernel: Any,
        structure_cost_kernel: Any,
    ) -> None:
        self.mutation_kernel = mutation_kernel
        self.development_kernel = development_kernel
        self.structure_cost_kernel = structure_cost_kernel

    def __call__(
        self,
        genotype,
        body,
        requested_birth,
        event_index,
        mutation_config,
        development_config,
    ):
        return prepare_birth_candidates_with_kernels(
            genotype,
            body,
            requested_birth,
            event_index,
            mutation_config,
            development_config,
            mutation_kernel=self.mutation_kernel,
            development_kernel=self.development_kernel,
            structure_cost_kernel=self.structure_cost_kernel,
        )


class RuntimeSession:
    """Own current state, compiled kernels, and bounded host synchronization.

    Biological equations remain in domain modules. A candidate chunk is kept local
    until its aggregate status is copied once; an invalid chunk never replaces the
    session's last accepted state.
    """

    def __init__(
        self,
        state: LivingState,
        config: LivingRuntimeConfig,
        *,
        compile_motion: bool = True,
        compile_domains: bool = False,
        optimistic_motion: bool = True,
        optimistic_feeding: bool = True,
        optimistic_candidates: bool = True,
        compile_backend: str | None = None,
    ) -> None:
        validate_living_runtime_config(config)
        validate_living_state(state, config.economy)
        if not isinstance(optimistic_candidates, bool):
            raise TypeError("optimistic candidates must be boolean")
        compile_options: dict[str, Any] = {
            "fullgraph": True,
            "dynamic": False,
        }
        if compile_backend is not None:
            compile_options["backend"] = compile_backend

        def compiled(function):
            return torch.compile(function, **compile_options)

        stage_kernel: Any = (
            compiled(advance_phase_stage) if compile_motion else advance_phase_stage
        )
        robust_motion = _StagedMotionKernel(stage_kernel)
        fast_motion = _StagedMotionKernel(
            stage_kernel,
            requested_only=optimistic_motion,
        )
        kernels = replace(
            EAGER_LIVING_KERNELS,
            motion=fast_motion,
        )
        robust_candidates = kernels.candidates
        if compile_domains:
            candidate_mutation = compiled(propose_offspring_mutations)
            candidate_development = compiled(develop_unchecked)
            candidate_structure_cost = compiled(target_structure_cost_q)
            robust_candidates = _DenseCandidateKernel(
                candidate_mutation,
                candidate_development,
                candidate_structure_cost,
            )
            fast_candidates = (
                compiled(defer_birth_candidates)
                if optimistic_candidates
                else robust_candidates
            )
            kernels = LivingKernels(
                motion=kernels.motion,
                economy=compiled(advance_economy_unchecked),
                feeding=compiled(feed_population),
                organisms=compiled(advance_organism_interval),
                returns=compiled(deposit_organism_returns),
                candidates=fast_candidates,
                commit_mutation=compiled(commit_offspring_mutations),
                body_cache=compiled(commit_developed_births),
                development=compiled(settle_development_lifecycle),
            )
        self._behavior_kernel = (
            compiled(request_living_intent)
            if compile_domains
            else request_living_intent
        )
        self._kernels = kernels
        self._robust_kernels = replace(
            kernels,
            motion=robust_motion,
            candidates=robust_candidates,
        )
        self._optimistic_motion = optimistic_motion
        self._robust_config = config
        self._fast_config = (
            replace(
                config,
                feeding=replace(config.feeding, allocation_rounds=1),
            )
            if optimistic_feeding and config.feeding.allocation_rounds > 1
            else config
        )
        self._optimistic_feeding = self._fast_config is not config
        self._optimistic_candidates = optimistic_candidates and compile_domains
        self._state = state
        self.config = config

    @property
    def state(self) -> LivingState:
        return self._state

    def save_checkpoint(self, path: Path | str) -> None:
        """Save the last accepted state and its complete scientific config."""

        from sirrobin.runtime.checkpoint import save_runtime_checkpoint

        save_runtime_checkpoint(path, self._state, self.config)

    @classmethod
    def from_checkpoint(
        cls,
        path: Path | str,
        *,
        device: torch.device | str = "cpu",
        compile_motion: bool = True,
        compile_domains: bool = False,
        optimistic_motion: bool = True,
        optimistic_feeding: bool = True,
        optimistic_candidates: bool = True,
        compile_backend: str | None = None,
    ) -> RuntimeSession:
        """Restore authoritative state and select fresh execution policy."""

        from sirrobin.runtime.checkpoint import load_runtime_checkpoint

        state, config = load_runtime_checkpoint(path, device=device)
        return cls(
            state,
            config,
            compile_motion=compile_motion,
            compile_domains=compile_domains,
            optimistic_motion=optimistic_motion,
            optimistic_feeding=optimistic_feeding,
            optimistic_candidates=optimistic_candidates,
            compile_backend=compile_backend,
        )

    @property
    def optimistic_motion_enabled(self) -> bool:
        return self._optimistic_motion

    @property
    def optimistic_candidates_enabled(self) -> bool:
        return self._optimistic_candidates

    @staticmethod
    def _validate_intervals(intervals: int) -> None:
        if isinstance(intervals, bool) or not isinstance(intervals, int):
            raise TypeError("chunk interval count must be an integer")
        if intervals < 1:
            raise ValueError("chunk interval count must be positive")

    def _validate_fluid(self, fluid: FluidSample) -> None:
        expected = tuple(self._state.population.alive.shape)
        if (
            tuple(fluid.density_kg_m3.shape) != expected
            or tuple(fluid.velocity_enu_m_s.shape) != (*expected, 3)
        ):
            raise ValueError("fluid sample shapes differ from runtime capacity")
        if fluid.density_kg_m3.device != self._state.population.alive.device:
            raise ValueError("fluid sample must share the runtime device")
        if fluid.velocity_enu_m_s.device != self._state.population.alive.device:
            raise ValueError("fluid sample must share the runtime device")
        if bool(
            (
                ~torch.isfinite(fluid.density_kg_m3)
                | (fluid.density_kg_m3 <= 0.0)
            ).any()
        ) or bool((~torch.isfinite(fluid.velocity_enu_m_s)).any()):
            raise ValueError("fluid samples must be finite with positive density")

    def _validate_inputs(self, inputs: LivingIntervalInputs) -> None:
        expected = tuple(self._state.population.alive.shape)
        if (
            inputs.requested_effort.shape != expected
            or not torch.is_floating_point(inputs.requested_effort)
        ):
            raise TypeError(f"requested effort must be floating with shape {expected}")
        if inputs.requested_effort.device != self._state.population.alive.device:
            raise ValueError("requested effort must share the runtime device")
        if bool(
            (
                ~torch.isfinite(inputs.requested_effort)
                | (inputs.requested_effort < 0.0)
                | (inputs.requested_effort > 1.0)
            ).any()
        ):
            raise ValueError("requested effort must be finite and in [0,1]")
        if (
            inputs.birth_requested.dtype != torch.bool
            or tuple(inputs.birth_requested.shape) != expected
        ):
            raise TypeError(f"birth request must be bool with shape {expected}")
        if inputs.birth_requested.device != self._state.population.alive.device:
            raise ValueError("birth request must share the runtime device")
        self._validate_fluid(inputs.fluid)

    def _autonomous_inputs(
        self,
        state: LivingState,
        fluid: FluidSample,
    ) -> tuple[LivingState, LivingIntervalInputs, BehaviorStep]:
        behavior = self._behavior_kernel(
            state.population,
            state.body,
            state.motion,
            state.economy.bp_q,
            self.config.geometry,
            self.config.live,
            self.config.behavior,
            q_mass_mol=self.config.economy.q_mass_mol,
        )
        controlled = replace(state, motion=behavior.motion)
        inputs = LivingIntervalInputs(
            fluid,
            behavior.requested_effort_fraction,
            behavior.birth_requested,
        )
        return controlled, inputs, behavior

    def _advance_with_provider(
        self,
        provider: Callable[
            [LivingState],
            tuple[LivingState, LivingIntervalInputs, BehaviorStep | None],
        ],
        *,
        intervals: int,
    ) -> LivingChunkAdvance:
        def run_candidate(
            kernels: LivingKernels,
            run_config: LivingRuntimeConfig,
        ):
            candidate = self._state
            invalid = torch.zeros(
                candidate.population.alive.shape[0],
                dtype=torch.bool,
                device=candidate.population.alive.device,
            )
            funding_unresolved = torch.zeros_like(invalid)
            feeding_unresolved = torch.zeros_like(invalid)
            candidate_replay_required = torch.zeros_like(invalid)
            zeros_i64 = torch.zeros_like(invalid, dtype=torch.int64)
            zeros_f64 = torch.zeros_like(invalid, dtype=torch.float64)
            births = zeros_i64.clone()
            deaths = zeros_i64.clone()
            starvation_deaths = zeros_i64.clone()
            old_age_deaths = zeros_i64.clone()
            requested_births = zeros_i64.clone()
            unfunded_birth_rejections = zeros_i64.clone()
            capacity_birth_rejections = zeros_i64.clone()
            id_birth_rejections = zeros_i64.clone()
            birth_release_energy_q = zeros_i64.clone()
            mutated_births = zeros_i64.clone()
            mutation_events = zeros_i64.clone()
            parameter_mutation_events = zeros_i64.clone()
            topology_mutation_events = zeros_i64.clone()
            behavior_food_gradient_intervals = zeros_i64.clone()
            behavior_locomoting_intervals = zeros_i64.clone()
            feeding_requested_q = zeros_i64.clone()
            feeding_actual_debit_q = zeros_i64.clone()
            feeding_reserve_credit_q = zeros_i64.clone()
            dissipation_j = zeros_f64.clone()
            light_input_j = zeros_f64.clone()
            last = None
            last_behavior = None
            for _ in range(intervals):
                candidate, inputs, behavior = provider(candidate)
                advance = advance_living_interval_with_kernels(
                    candidate,
                    inputs,
                    run_config,
                    kernels,
                )
                candidate = advance.state
                last = advance.ledger
                last_behavior = behavior
                invalid |= advance.ledger.invalid
                if behavior is not None:
                    invalid |= behavior.invalid.any(dim=1)
                    behavior_food_gradient_intervals += (
                        behavior.horizontal_gradient_present.sum(
                            dim=1,
                            dtype=torch.int64,
                        )
                    )
                    behavior_locomoting_intervals += behavior.locomoting.sum(
                        dim=1,
                        dtype=torch.int64,
                    )
                funding_unresolved |= advance.ledger.motion_funding_unresolved
                feeding_unresolved |= (
                    advance.ledger.feeding_allocation_unresolved
                )
                candidate_replay_required |= (
                    advance.ledger.candidate_replay_required
                )
                lifecycle = advance.ledger.organisms.lifecycle.ledger
                metabolism = advance.ledger.organisms.metabolism.ledger
                mutation = advance.ledger.mutation.ledger
                feeding = advance.ledger.feeding.ledger
                births += lifecycle.accepted_births
                deaths += lifecycle.died.sum(
                    dim=1,
                    dtype=torch.int64,
                )
                starvation_deaths += metabolism.starved.sum(
                    dim=1,
                    dtype=torch.int64,
                )
                old_age_deaths += metabolism.old_age_due.sum(
                    dim=1,
                    dtype=torch.int64,
                )
                requested_births += lifecycle.requested_births
                unfunded_birth_rejections += lifecycle.unfunded_rejections
                capacity_birth_rejections += lifecycle.capacity_rejections
                id_birth_rejections += lifecycle.id_rejections
                birth_release_energy_q += (
                    lifecycle.birth_release_energy_return_q.sum(
                        dim=1,
                        dtype=torch.int64,
                    )
                )
                mutated_births += mutation.mutated.sum(
                    dim=1,
                    dtype=torch.int64,
                )
                mutation_events += mutation.mutation_count.sum(
                    dim=1,
                    dtype=torch.int64,
                )
                topology_event = mutation.event_applied & (
                    (mutation.event_trait_code == SEGMENT_BUD)
                    | (mutation.event_trait_code == SEGMENT_VESTIGIAL)
                )
                topology_mutation_events += topology_event.sum(
                    dim=(1, 2),
                    dtype=torch.int64,
                )
                parameter_mutation_events += (
                    mutation.event_applied & ~topology_event
                ).sum(
                    dim=(1, 2),
                    dtype=torch.int64,
                )
                feeding_requested_q += feeding.requested_q.sum(
                    dim=1,
                    dtype=torch.int64,
                )
                feeding_actual_debit_q += feeding.actual_debit_q.sum(
                    dim=1,
                    dtype=torch.int64,
                )
                feeding_reserve_credit_q += feeding.reserve_credit_q.sum(
                    dim=1,
                    dtype=torch.int64,
                )
                energy = advance.ledger.energy
                light_input_j += energy.light_chemical_input_j.reshape(
                    energy.light_chemical_input_j.shape[0], -1
                ).sum(dim=1, dtype=torch.float64)
                for value in (
                    energy.producer_maintenance_dissipation_j,
                    energy.producer_mortality_dissipation_j,
                    energy.assimilation_heat_j,
                    energy.actuator_braking_heat_j,
                    energy.muscle_inefficiency_heat_j,
                    advance.ledger.organisms.metabolism.ledger.baseline_demand_j,
                    energy.motion_dissipation_j,
                    energy.death_kinetic_and_carry_dissipation_j,
                    energy.death_reserve_dissipation_j,
                    energy.birth_construction_heat_j,
                    energy.birth_release_heat_j,
                ):
                    dissipation_j += value.reshape(value.shape[0], -1).sum(
                        dim=1,
                        dtype=torch.float64,
                    )
            assert last is not None
            summary = LivingChunkSummary(
                intervals,
                births,
                deaths,
                starvation_deaths,
                old_age_deaths,
                requested_births,
                unfunded_birth_rejections,
                capacity_birth_rejections,
                id_birth_rejections,
                birth_release_energy_q,
                mutated_births,
                mutation_events,
                parameter_mutation_events,
                topology_mutation_events,
                behavior_food_gradient_intervals,
                behavior_locomoting_intervals,
                feeding_requested_q,
                feeding_actual_debit_q,
                feeding_reserve_credit_q,
                dissipation_j,
                light_input_j,
            )
            return (
                candidate,
                last,
                last_behavior,
                invalid,
                funding_unresolved,
                feeding_unresolved,
                candidate_replay_required,
                summary,
            )

        result = run_candidate(self._kernels, self._fast_config)
        (
            candidate,
            last,
            last_behavior,
            invalid,
            funding_unresolved,
            feeding_unresolved,
            candidate_replay_required,
            summary,
        ) = result
        status = torch.stack(
            (
                invalid,
                funding_unresolved,
                feeding_unresolved,
                candidate_replay_required,
            )
        ).cpu()
        retry_motion = self._optimistic_motion and bool(status[1].any())
        retry_feeding = self._optimistic_feeding and bool(status[2].any())
        retry_candidates = self._optimistic_candidates and bool(status[3].any())
        if retry_motion or retry_feeding or retry_candidates:
            if retry_motion or retry_candidates:
                retry_kernels = self._robust_kernels
            elif self._optimistic_candidates:
                # Robust feeding can preserve a parent that makes a later birth
                # request in this chunk, so the only retry must start dense.
                retry_kernels = replace(
                    self._kernels,
                    candidates=self._robust_kernels.candidates,
                )
            else:
                retry_kernels = self._kernels
            retry_config = self._robust_config if retry_feeding else self._fast_config
            result = run_candidate(retry_kernels, retry_config)
            (
                candidate,
                last,
                last_behavior,
                invalid,
                funding_unresolved,
                feeding_unresolved,
                candidate_replay_required,
                summary,
            ) = result
            status = torch.stack(
                (
                    invalid,
                    funding_unresolved,
                    feeding_unresolved,
                    candidate_replay_required,
                )
            ).cpu()
        if bool(status[0].any()):
            failure_masks = {
                "economy_books": ~last.economy.ledger.books_closed,
                "feeding_transaction": ~last.feeding.ledger.transaction_committed,
                "return_transaction": ~last.returns.transaction_committed,
                "matter_books": ~last.matter.books_closed,
                "motion_nonfinite": last.motion.ledger.option_nonfinite.any(
                    dim=(1, 2)
                ),
                "yaw_backstop": last.motion.ledger.response.yaw_backstop_hit.any(
                    dim=1
                ),
                "death_kinetics": (
                    last.organisms.metabolism.ledger.invalid_death_kinetics.any(
                        dim=1
                    )
                ),
                "motion_funding": last.motion_funding_unresolved,
            }
            if last_behavior is not None:
                failure_masks["behavior"] = last_behavior.invalid.any(dim=1)
            reasons = [
                name
                for name, mask in failure_masks.items()
                if bool(mask.any().detach().cpu())
            ]
            raise RuntimeError(
                "candidate runtime chunk is invalid; state was not published: "
                + ", ".join(reasons or ("earlier chunk interval",))
            )
        self._state = candidate
        return LivingChunkAdvance(candidate, last, invalid, last_behavior, summary)

    def advance_chunk(
        self,
        inputs: LivingIntervalInputs,
        *,
        intervals: int,
    ) -> LivingChunkAdvance:
        self._validate_intervals(intervals)
        self._validate_inputs(inputs)

        def fixed_provider(state: LivingState):
            return state, inputs, None

        return self._advance_with_provider(fixed_provider, intervals=intervals)

    def advance_autonomous_chunk(
        self,
        fluid: FluidSample,
        *,
        intervals: int,
    ) -> LivingChunkAdvance:
        """Advance behavior and complete living transactions with one host check."""

        self._validate_intervals(intervals)
        self._validate_fluid(fluid)

        def behavior_provider(state: LivingState):
            return self._autonomous_inputs(state, fluid)

        return self._advance_with_provider(
            behavior_provider,
            intervals=intervals,
        )

    def prewarm_autonomous(self, fluid: FluidSample) -> None:
        """Compile every autonomous specialization without publishing state."""

        self._validate_fluid(fluid)
        controlled, inputs, _ = self._autonomous_inputs(self._state, fluid)
        advance_living_interval_with_kernels(
            controlled,
            inputs,
            self._fast_config,
            self._kernels,
        )
        if (
            self._optimistic_motion
            or self._optimistic_feeding
            or self._optimistic_candidates
        ):
            advance_living_interval_with_kernels(
                controlled,
                inputs,
                self._robust_config,
                self._robust_kernels,
            )
