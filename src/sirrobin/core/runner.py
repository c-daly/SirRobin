"""Headless multi-rate schedule.

Per plan section 4.3 this module owns the explicit cadence in simulation time. Tranche
A composes two rates: live mechanics at the frozen locomotion dt, and the existing
ecological reaction/transport kernel at its own interval. Feeding, animal metabolism,
and snapshot/telemetry cadences join this schedule in Tranche C/D/E; they are absent
because their mechanisms do not exist yet, not because they run at the mechanics rate.

All cadences are configuration data. No cadence depends on a render frame.

Per plan section 4.2 the complete tick verifies: advance() raises on a tick whose books
do not close, naming the worlds that failed, rather than reporting closure as a flag no
caller reads.
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
    """What one composed economy interval did. Observation only; owns no state.

    `economy` covers the whole interval. `last_mechanics_substep` is the final substep
    only — the runner does not aggregate per-substep mechanics. Per-interval mechanical
    work and fault counts are Tranche B/D work, since nothing consumes them yet.
    """

    mechanics_steps: int
    sim_time_s: float
    economy: EconomyStepLedger
    last_mechanics_substep: LiveStepLedger


class HeadlessRunner:
    """Drives a `HeadlessWorld` on the declared schedule. Owns cadence, never state."""

    def __init__(self, world: HeadlessWorld) -> None:
        self.world = world
        self.schedule = WorldSchedule.from_configs(world.live_config, world.economy_config)
        self._books_failed = False

    def advance(self) -> WorldTick:
        """Advance one economy interval: mechanics substeps, one economy step, verify.

        Closure can only be checked once the step has run, so the check is a
        post-mortem. On failure the runner is arrested: the world is left mutated and
        is not resumable, and further calls refuse rather than advancing its clocks
        behind a stream of exceptions.
        """
        if self._books_failed:
            raise RuntimeError("this world's books failed to close; it is not resumable")
        steps = self.schedule.mechanics_steps_per_economy_step
        mechanics_ledger = self.world._step_mechanics()
        for _ in range(steps - 1):
            mechanics_ledger = self.world._step_mechanics()
        economy_ledger = self.world._step_economy()
        if not bool(economy_ledger.books_closed.all()):
            self._books_failed = True
            failed = (~economy_ledger.books_closed).nonzero().flatten().tolist()
            raise RuntimeError(f"exact nutrient books do not close in worlds {failed}")

        # The mechanics sub-clock must stay in step with the authoritative ecological
        # clock. A creature alive since t=0 has advanced by exactly the elapsed time, so
        # the oldest gait phase equals sim_time_s; stepping mechanics outside this method
        # would silently drift it. Tolerance is half a substep — float accumulation over
        # a long run is picometres of time, a real desync is at least one whole substep.
        sim_time_s = self.world.sim_time_s
        oldest_gait_s = float(self.world.live_state.gait_time_s.max())
        if abs(oldest_gait_s - sim_time_s) > 0.5 * self.world.live_config.dt:
            raise RuntimeError(
                f"mechanics sub-clock desynchronised: oldest gait {oldest_gait_s} s "
                f"against ecological clock {sim_time_s} s"
            )
        return WorldTick(steps, sim_time_s, economy_ledger, mechanics_ledger)
