"""Headless multi-rate schedule.

Per plan §4.3 this module owns the explicit cadence in simulation time. Tranche A
composes two rates: live mechanics at the frozen locomotion dt, and the existing
ecological reaction/transport kernel at its own interval. Feeding, animal metabolism,
and snapshot/telemetry cadences join this schedule in Tranche C/D/E; they are absent
because their mechanisms do not exist yet, not because they run at the mechanics rate.

All cadences are configuration data. No cadence depends on a render frame.

Per plan §4.2 the complete tick verifies: `advance()` requires the books to close
rather than reporting closure as a flag no caller reads.
"""

from __future__ import annotations

from dataclasses import dataclass

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
    """What one composed economy interval did. Observation only; owns no state."""

    mechanics_steps: int
    sim_time_s: float
    economy: EconomyStepLedger
    mechanics: LiveStepLedger


class HeadlessRunner:
    """Drives a `HeadlessWorld` on the declared schedule. Owns cadence, never state."""

    def __init__(self, world: HeadlessWorld, schedule: WorldSchedule | None = None) -> None:
        derived = WorldSchedule.from_configs(world.live_config, world.economy_config)
        if schedule is not None and schedule != derived:
            raise ValueError("an explicit schedule must match the one the configs imply")
        self.world = world
        self.schedule = derived

    def advance(self) -> WorldTick:
        """Advance one economy interval: mechanics substeps, one economy step, verify."""
        steps = self.schedule.mechanics_steps_per_economy_step
        mechanics_ledger = self.world._step_mechanics()
        for _ in range(steps - 1):
            mechanics_ledger = self.world._step_mechanics()
        economy_ledger = self.world._step_economy()
        self.world.economy.mass_ledger.require_closed(self.world.economy_state)
        return WorldTick(steps, self.world.sim_time_s, economy_ledger, mechanics_ledger)
