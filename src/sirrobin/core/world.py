"""Authoritative composition root for the headless world.

Tranche A seam. The ecological economy and the live mechanics are composed here and
share one declared schedule. No matter, energy, or intent crosses between them yet;
that coupling is Tranche C/D work and must arrive as explicit transfers through the
whole-world ledger, never as a side effect of stepping.

Authority follows the recovery synthesis §3.2: one authoritative state per quantity,
with derived data allowed where it is a rebuildable function of that authority. The
genotype is the authority; the developed body is a rebuildable cache and is held
alongside the genotype it derives from. Simulation time has one authority,
`EconomyState.time_s`; `sim_time_s` is a read-only view of it.

Simulation time belongs to the core. Nothing here depends on a render frame and the
whole module runs without Unity.
"""

from __future__ import annotations

from sirrobin.core.live_world import advance_live_world, initialize_live_state
from sirrobin.economy.config import EconomyConfig
from sirrobin.economy.contracts import EconomyStepLedger
from sirrobin.economy.state import EconomyState
from sirrobin.economy.step import EconomyKernel
from sirrobin.fields.geometry import GridGeometry
from sirrobin.genetics.develop import develop
from sirrobin.genetics.genotype import GenotypeBatch
from sirrobin.physics.contracts import FluidSample, LiveStepLedger
from sirrobin.physics.live_config import LiveLocomotionConfig


class HeadlessWorld:
    """Owns every authoritative live quantity of the composed world."""

    def __init__(
        self,
        *,
        genotype: GenotypeBatch,
        fluid: FluidSample,
        live_config: LiveLocomotionConfig,
        economy_state: EconomyState,
        economy_config: EconomyConfig,
        geometry: GridGeometry | None = None,
    ) -> None:
        live_config.validate()
        self.genotype = genotype
        self.body = develop(genotype)
        self.live_state = initialize_live_state(self.body)
        self.fluid = fluid
        self.live_config = live_config
        self.economy = EconomyKernel(economy_state, economy_config)
        self.geometry = geometry if geometry is not None else GridGeometry.from_config(economy_config)

    @property
    def economy_state(self) -> EconomyState:
        return self.economy.state

    @property
    def economy_config(self) -> EconomyConfig:
        return self.economy.config

    @property
    def sim_time_s(self) -> float:
        """Read-only view of the one authoritative simulation clock."""
        return float(self.economy.state.time_s)

    def _step_mechanics(self) -> LiveStepLedger:
        """Advance live mechanics by one frozen locomotion dt. Driven by the runner."""
        return advance_live_world(
            self.body,
            self.live_state,
            self.fluid,
            self.live_config,
            self.geometry,
        )

    def _step_economy(self) -> EconomyStepLedger:
        """Advance the ecological kernel, and with it the authoritative clock."""
        return self.economy.step()
