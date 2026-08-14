"""Headless multi-rate schedule.

This module owns explicit cadence in simulation time. Canonical mechanics remains
defined at the frozen locomotion dt. A uniformly validated periodic clone orbit may
cover repeated canonical steps by rigid-transform composition; every ineligible or
nonrecurrent state executes all full steps. The ecological reaction/transport kernel
runs at its own interval. No cadence depends on a render frame.

Per plan section 4.2 the complete tick verifies: advance() raises on a tick whose books
do not close, naming the worlds that failed, rather than reporting closure as a flag no
caller reads.
"""

from __future__ import annotations

from dataclasses import dataclass, fields, replace

import torch

from sirrobin.core.feeding import (
    FeedingConfig,
    PopulationFeedingReport,
    feed_population,
)
from sirrobin.core.foraging import (
    FoodSeekingConfig,
    FoodSeekingReport,
    apply_food_seeking_intent,
)
from sirrobin.core.material import WholeWorldMatterLedger
from sirrobin.core.metabolism import (
    MaintenanceConfig,
    MaintenanceReport,
    funded_positive_actuator_work_j,
    maintain_population,
)
from sirrobin.core.mortality import AgeMortalityConfig, old_age_due_mask
from sirrobin.core.periodic_motion import (
    MechanicsAdvance,
    PeriodicErrorEstimate,
    PeriodicMotionPolicy,
    advance_mechanics_interval,
)
from sirrobin.core.reproduction import (
    BirthConfig,
    BirthReport,
    ParametricMutationConfig,
    attempt_paid_birth,
)
from sirrobin.core.world import HeadlessWorld
from sirrobin.economy.config import EconomyConfig
from sirrobin.economy.contracts import EconomyStepLedger
from sirrobin.physics.contracts import LiveStepLedger
from sirrobin.physics.live_config import LiveLocomotionConfig


@dataclass(frozen=True, slots=True)
class WorldSchedule:
    """Declared multi-rate cadence, in simulation time only."""

    mechanics_steps_per_economy_step: int

    def __post_init__(self) -> None:
        if self.mechanics_steps_per_economy_step < 1:
            raise ValueError("an economy interval must span at least one mechanics step")

    @classmethod
    def from_configs(
        cls, live_config: LiveLocomotionConfig, economy_config: EconomyConfig
    ) -> WorldSchedule:
        ratio = economy_config.dt_eco_s / live_config.dt
        count = round(ratio)
        if abs(ratio - count) > 1e-9 * max(1.0, ratio):
            raise ValueError("the economy interval must be an exact multiple of the mechanics dt")
        return cls(count)


@dataclass(frozen=True, slots=True)
class WorldTick:
    """What one composed economy interval did. Observation only; owns no state.

    `economy` covers the whole interval. `last_mechanics_substep` is the final canonical
    substep only. `mechanical_work_j` integrates named dissipated power across actual
    and verified repeated cycles; it is observation, not yet a creature-energy debit.
    `positive_actuator_work_j` and `actuator_braking_work_j` split the canonical
    signed actuator channel without treating hydrodynamic dissipation as creature
    demand. Accelerated motion leaves them unavailable and cannot settle energy.
    `economy` proves the field subsystem conserved its own reaction/transport step.
    `food_seeking` records the sampled cause and requested action before mechanics;
    it does not report or require navigation success.
    `feeding`, when explicitly enabled, records the later field-to-creature transfer.
    `maintenance` records every organism settled from the tick-start population;
    `births` records funded attempts by surviving tick-start parents. Newborns cannot
    reproduce until a later tick. `matter` is the authoritative field-plus-creature
    baseline ledger over every transaction.
    """

    mechanics_steps: int
    full_batch_mechanics_steps: int
    representative_mechanics_steps: int
    fast_forwarded_mechanics_steps: int
    sim_time_s: float
    economy: EconomyStepLedger
    food_seeking: FoodSeekingReport | None
    feeding: PopulationFeedingReport | None
    maintenance: tuple[MaintenanceReport, ...]
    births: tuple[BirthReport, ...]
    matter: WholeWorldMatterLedger
    last_mechanics_substep: LiveStepLedger
    mechanical_work_j: torch.Tensor
    positive_actuator_work_j: torch.Tensor | None
    actuator_braking_work_j: torch.Tensor | None
    periodic_error: PeriodicErrorEstimate | None


def _live_state_snapshot(world: HeadlessWorld) -> tuple[torch.Tensor, ...]:
    return tuple(
        getattr(world.live_state, field.name).clone()
        for field in fields(world.live_state)
    )


def _restore_live_state(
    world: HeadlessWorld,
    snapshot: tuple[torch.Tensor, ...],
) -> None:
    for field, value in zip(fields(world.live_state), snapshot, strict=True):
        getattr(world.live_state, field.name).copy_(value)


def _advance_funded_mechanics(
    world: HeadlessWorld,
    steps: int,
    config: MaintenanceConfig,
    effort_fraction: torch.Tensor | None,
) -> MechanicsAdvance:
    """Execute requested effort only when its interval cost is fully backed.

    The common funded path runs canonical mechanics once. If any creature cannot
    fund that request, the interval is replayed from its exact starting state with
    that creature at zero actuation. Passive zero-effort body-fluid power is not a
    muscle charge. There is no energy debt and no outcome branch that kills a
    creature merely because its requested effort was refused.
    """
    requested = (
        torch.ones_like(world.live_state.yaw_rad)
        if effort_fraction is None
        else effort_fraction
    )
    snapshot = _live_state_snapshot(world)

    def trial(effort: torch.Tensor) -> MechanicsAdvance:
        _restore_live_state(world, snapshot)
        return advance_mechanics_interval(
            world,
            steps,
            None,
            effort_fraction=effort,
        )

    candidate = trial(requested)
    if (
        candidate.positive_actuator_work_j is None
        or candidate.actuator_braking_work_j is None
    ):
        raise RuntimeError("canonical mechanics omitted actuator work")
    budget = funded_positive_actuator_work_j(world, config)
    requested_work = candidate.positive_actuator_work_j
    funded = requested_work <= budget
    funding_fraction = torch.where(
        requested_work > 0.0,
        (budget / requested_work).clamp(0.0, 1.0),
        1.0,
    ).to(dtype=requested.dtype)
    accepted_effort = requested * funding_fraction
    if not bool(funded.all()):
        candidate = trial(accepted_effort)
        if (
            candidate.positive_actuator_work_j is None
            or candidate.actuator_braking_work_j is None
        ):
            raise RuntimeError("canonical mechanics omitted actuator work")
        still_unfunded = candidate.positive_actuator_work_j > budget
        if bool(still_unfunded.any()):
            accepted_effort = torch.where(still_unfunded, 0.0, accepted_effort)
            candidate = trial(accepted_effort)
            if (
                candidate.positive_actuator_work_j is None
                or candidate.actuator_braking_work_j is None
            ):
                raise RuntimeError("canonical mechanics omitted actuator work")
    actuating = accepted_effort > 0.0
    positive = torch.where(
        actuating,
        candidate.positive_actuator_work_j,
        0.0,
    )
    braking = torch.where(
        actuating,
        candidate.actuator_braking_work_j,
        0.0,
    )
    if bool((positive > budget).any()):
        raise RuntimeError("reduced effort replay exceeded its chemical budget")
    return replace(
        candidate,
        positive_actuator_work_j=positive,
        actuator_braking_work_j=braking,
    )


class HeadlessRunner:
    """Drives a `HeadlessWorld` on the declared schedule. Owns cadence, never state."""

    def __init__(
        self,
        world: HeadlessWorld,
        *,
        periodic_policy: PeriodicMotionPolicy | None = None,
        food_seeking_config: FoodSeekingConfig | None = None,
        feeding_config: FeedingConfig | None = None,
        maintenance_config: MaintenanceConfig | None = None,
        birth_config: BirthConfig | None = None,
        mutation_config: ParametricMutationConfig | None = None,
        age_mortality_config: AgeMortalityConfig | None = None,
    ) -> None:
        if periodic_policy is not None and maintenance_config is not None:
            raise ValueError(
                "energy settlement requires canonical mechanics; periodic "
                "fast-forward does not publish actuator work"
            )
        if mutation_config is not None and birth_config is None:
            raise ValueError("mutation requires a paid birth configuration")
        if age_mortality_config is not None and maintenance_config is None:
            raise ValueError("age mortality requires maintenance death settlement")
        self.world = world
        self.schedule = WorldSchedule.from_configs(world.live_config, world.economy_config)
        self.periodic_policy = periodic_policy
        self.food_seeking_config = food_seeking_config
        self.feeding_config = feeding_config
        self.maintenance_config = maintenance_config
        self.birth_config = birth_config
        self.mutation_config = mutation_config
        self.age_mortality_config = age_mortality_config
        self._books_failed = False
        self._gait_checkpoint_s = world.live_state.gait_time_s.clone()

    def advance(self) -> WorldTick:
        """Advance one economy interval: mechanics substeps, one economy step, verify.

        Closure can only be checked once the step has run, so the check is a
        post-mortem. On failure the runner is arrested: the world is left mutated and
        is not resumable, and further calls refuse rather than advancing its clocks
        behind a stream of exceptions.
        """
        if self.periodic_policy is not None and self.maintenance_config is not None:
            raise ValueError(
                "energy settlement requires canonical mechanics; periodic "
                "fast-forward does not publish actuator work"
            )
        if self._books_failed:
            raise RuntimeError("this world is arrested; it is not resumable")
        if not torch.equal(
            self.world.body.alive, self.world.genotype.alive
        ) or not torch.equal(
            self.world.body.stable_id, self.world.genotype.stable_id
        ):
            self._books_failed = True
            raise RuntimeError(
                "developed body identity cache differs from genotype authority"
            )
        if not torch.equal(
            self.world.live_state.gait_time_s,
            self._gait_checkpoint_s,
        ):
            self._books_failed = True
            raise RuntimeError(
                "mechanics sub-clock desynchronised outside the headless runner"
            )
        alive_before = self.world.body.alive.clone()
        stable_id_before = self.world.body.stable_id.clone()
        gait_before_s = self.world.live_state.gait_time_s.clone()
        live_count = int(alive_before.sum().item())
        birth_parents = tuple(
            sorted(
                [
                    (
                        int(world_index),
                        int(creature_slot),
                        int(
                            self.world.genotype.stable_id[
                                world_index, creature_slot
                            ]
                        ),
                    )
                    for world_index, creature_slot in self.world.body.alive.nonzero(
                        as_tuple=False
                    ).tolist()
                ],
                key=lambda parent: (parent[0], parent[2], parent[1]),
            )
        )
        steps = self.schedule.mechanics_steps_per_economy_step
        matter_before = self.world.matter_totals()
        if not bool(matter_before.raw_reservoirs_valid.all()):
            self._books_failed = True
            failed = (~matter_before.raw_reservoirs_valid).nonzero().flatten().tolist()
            raise RuntimeError(
                "whole-world nutrient books do not close in worlds "
                f"{failed} (invalid raw reservoir state)"
            )
        try:
            food_seeking = (
                apply_food_seeking_intent(self.world, self.food_seeking_config)
                if self.food_seeking_config is not None and live_count > 0
                else None
            )
            effort_fraction = (
                None
                if food_seeking is None
                else food_seeking.requested_effort_fraction
            )
            mechanics = (
                _advance_funded_mechanics(
                    self.world,
                    steps,
                    self.maintenance_config,
                    effort_fraction,
                )
                if self.maintenance_config is not None
                else advance_mechanics_interval(
                    self.world,
                    steps,
                    self.periodic_policy,
                    effort_fraction=effort_fraction,
                )
            )
            economy_ledger = self.world._step_economy()
            if not bool(economy_ledger.books_closed.all()):
                failed = (~economy_ledger.books_closed).nonzero().flatten().tolist()
                raise RuntimeError(f"exact nutrient books do not close in worlds {failed}")
            feeding = (
                feed_population(self.world, self.feeding_config)
                if self.feeding_config is not None and live_count > 0
                else None
            )
            maintenance = (
                maintain_population(
                    self.world,
                    self.maintenance_config,
                    last_mechanics_substep=mechanics.last_ledger,
                    positive_actuator_work_j=mechanics.positive_actuator_work_j,
                    actuator_braking_work_j=mechanics.actuator_braking_work_j,
                    old_age_due=(
                        old_age_due_mask(self.world, self.age_mortality_config)
                        if self.age_mortality_config is not None
                        else None
                    ),
                )
                if self.maintenance_config is not None
                else ()
            )
            births: list[BirthReport] = []
            if self.birth_config is not None:
                for world_index, parent_slot, parent_id in birth_parents:
                    if not bool(self.world.genotype.alive[world_index, parent_slot]):
                        continue
                    if (
                        int(self.world.genotype.stable_id[world_index, parent_slot])
                        != parent_id
                    ):
                        continue
                    structure_q = int(
                        self.world.creature_material.structure_q[
                            world_index, parent_slot
                        ]
                    )
                    reserve_q = int(
                        self.world.creature_material.reserve_q[
                            world_index, parent_slot
                        ]
                    )
                    if reserve_q < structure_q + self.birth_config.initial_reserve_q:
                        continue
                    births.append(
                        attempt_paid_birth(
                            self.world,
                            self.birth_config,
                            world_index=world_index,
                            parent_slot=parent_slot,
                            mutation_config=self.mutation_config,
                        )
                    )
            matter_ledger = self.world.close_matter_step(matter_before)
            if not bool(matter_ledger.books_closed.all()):
                failed = (~matter_ledger.books_closed).nonzero().flatten().tolist()
                raise RuntimeError(
                    f"whole-world nutrient books do not close in worlds {failed}"
                )

            # Each surviving identity advances by exactly one ecological interval.
            # A birth happens after mechanics and therefore starts at gait age zero;
            # no organism is required to have survived since world time zero.
            sim_time_s = self.world.sim_time_s
            alive_after = self.world.body.alive
            same_identity = (
                alive_before
                & alive_after
                & (self.world.body.stable_id == stable_id_before)
            )
            interval_s = steps * self.world.live_config.dt
            if bool(same_identity.any()):
                expected_gait_s = gait_before_s[same_identity] + interval_s
                actual_gait_s = self.world.live_state.gait_time_s[same_identity]
                if not torch.allclose(
                    actual_gait_s,
                    expected_gait_s,
                    rtol=0.0,
                    atol=0.5 * self.world.live_config.dt,
                ):
                    raise RuntimeError(
                        "mechanics sub-clock desynchronised for a surviving identity"
                    )
            new_identity = alive_after & ~same_identity
            if bool(
                (
                    self.world.live_state.gait_time_s[new_identity].abs()
                    > 0.5 * self.world.live_config.dt
                ).any()
            ):
                raise RuntimeError("newborn mechanics sub-clock did not start at zero")
        except Exception:
            self._books_failed = True
            raise
        self._gait_checkpoint_s = self.world.live_state.gait_time_s.clone()
        return WorldTick(
            mechanics_steps=steps,
            full_batch_mechanics_steps=mechanics.full_batch_steps,
            representative_mechanics_steps=mechanics.representative_steps,
            fast_forwarded_mechanics_steps=mechanics.fast_forwarded_steps,
            sim_time_s=sim_time_s,
            economy=economy_ledger,
            food_seeking=food_seeking,
            feeding=feeding,
            maintenance=maintenance,
            births=tuple(births),
            matter=matter_ledger,
            last_mechanics_substep=mechanics.last_ledger,
            mechanical_work_j=mechanics.mechanical_work_j,
            positive_actuator_work_j=mechanics.positive_actuator_work_j,
            actuator_braking_work_j=mechanics.actuator_braking_work_j,
            periodic_error=mechanics.periodic_error,
        )
