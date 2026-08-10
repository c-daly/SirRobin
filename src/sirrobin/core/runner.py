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

from dataclasses import dataclass

import torch

from sirrobin.core.feeding import FeedingConfig, FeedingReport, feed_single_creature
from sirrobin.core.material import WholeWorldMatterLedger
from sirrobin.core.metabolism import (
    MaintenanceConfig,
    MaintenanceReport,
    maintain_single_creature,
)
from sirrobin.core.periodic_motion import (
    PeriodicErrorEstimate,
    PeriodicMotionPolicy,
    advance_mechanics_interval,
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
    `economy` proves the field subsystem conserved its own reaction/transport step.
    `feeding`, when explicitly enabled, records the later field-to-creature transfer;
    `matter` is the authoritative field-plus-creature baseline ledger over both.
    """

    mechanics_steps: int
    full_batch_mechanics_steps: int
    representative_mechanics_steps: int
    fast_forwarded_mechanics_steps: int
    sim_time_s: float
    economy: EconomyStepLedger
    feeding: FeedingReport | None
    maintenance: MaintenanceReport | None
    matter: WholeWorldMatterLedger
    last_mechanics_substep: LiveStepLedger
    mechanical_work_j: torch.Tensor
    periodic_error: PeriodicErrorEstimate | None


class HeadlessRunner:
    """Drives a `HeadlessWorld` on the declared schedule. Owns cadence, never state."""

    def __init__(
        self,
        world: HeadlessWorld,
        *,
        periodic_policy: PeriodicMotionPolicy | None = None,
        feeding_config: FeedingConfig | None = None,
        maintenance_config: MaintenanceConfig | None = None,
    ) -> None:
        self.world = world
        self.schedule = WorldSchedule.from_configs(world.live_config, world.economy_config)
        self.periodic_policy = periodic_policy
        if feeding_config is not None and int(world.body.alive.sum().item()) != 1:
            raise ValueError("feeding currently requires exactly one live creature")
        self.feeding_config = feeding_config
        if maintenance_config is not None and int(world.body.alive.sum().item()) > 1:
            raise ValueError("maintenance currently supports at most one live creature")
        self.maintenance_config = maintenance_config
        self._books_failed = False

    def advance(self) -> WorldTick:
        """Advance one economy interval: mechanics substeps, one economy step, verify.

        Closure can only be checked once the step has run, so the check is a
        post-mortem. On failure the runner is arrested: the world is left mutated and
        is not resumable, and further calls refuse rather than advancing its clocks
        behind a stream of exceptions.
        """
        if self._books_failed:
            raise RuntimeError("this world is arrested; it is not resumable")
        live_count = int(self.world.body.alive.sum().item())
        if self.feeding_config is not None and live_count > 1:
            self._books_failed = True
            raise RuntimeError("feeding requires exactly one live creature before the tick")
        if self.maintenance_config is not None and live_count > 1:
            self._books_failed = True
            raise RuntimeError("maintenance supports at most one live creature before the tick")
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
            mechanics = advance_mechanics_interval(self.world, steps, self.periodic_policy)
            economy_ledger = self.world._step_economy()
            if not bool(economy_ledger.books_closed.all()):
                failed = (~economy_ledger.books_closed).nonzero().flatten().tolist()
                raise RuntimeError(f"exact nutrient books do not close in worlds {failed}")
            feeding = (
                feed_single_creature(self.world, self.feeding_config)
                if self.feeding_config is not None and live_count == 1
                else None
            )
            maintenance = (
                maintain_single_creature(
                    self.world,
                    self.maintenance_config,
                    last_mechanics_substep=mechanics.last_ledger,
                )
                if self.maintenance_config is not None
                else None
            )
            matter_ledger = self.world.close_matter_step(matter_before)
            if not bool(matter_ledger.books_closed.all()):
                failed = (~matter_ledger.books_closed).nonzero().flatten().tolist()
                raise RuntimeError(
                    f"whole-world nutrient books do not close in worlds {failed}"
                )

            # The mechanics sub-clock must stay in step with the authoritative ecological
            # clock. A creature alive since t=0 has advanced by exactly the elapsed time,
            # so the oldest gait phase equals sim_time_s. Tolerance is half a substep.
            sim_time_s = self.world.sim_time_s
            if bool(self.world.body.alive.any()):
                oldest_gait_s = float(
                    self.world.live_state.gait_time_s[self.world.body.alive].max()
                )
                if abs(oldest_gait_s - sim_time_s) > 0.5 * self.world.live_config.dt:
                    raise RuntimeError(
                        f"mechanics sub-clock desynchronised: oldest gait {oldest_gait_s} s "
                        f"against ecological clock {sim_time_s} s"
                    )
        except Exception:
            self._books_failed = True
            raise
        return WorldTick(
            mechanics_steps=steps,
            full_batch_mechanics_steps=mechanics.full_batch_steps,
            representative_mechanics_steps=mechanics.representative_steps,
            fast_forwarded_mechanics_steps=mechanics.fast_forwarded_steps,
            sim_time_s=sim_time_s,
            economy=economy_ledger,
            feeding=feeding,
            maintenance=maintenance,
            matter=matter_ledger,
            last_mechanics_substep=mechanics.last_ledger,
            mechanical_work_j=mechanics.mechanical_work_j,
            periodic_error=mechanics.periodic_error,
        )
