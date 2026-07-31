"""Authoritative composition root for the headless world.

Tranche A seam. The ecological economy and the live mechanics are composed here and
share one declared schedule. No matter, energy, or intent crosses between them yet;
that coupling is Tranche C/D work and must arrive as explicit transfers through the
whole-world ledger, never as a side effect of stepping.

Authority follows the recovery synthesis section 3.2: one authoritative state per
quantity, with derived data allowed where it is a rebuildable function of that
authority AND one-way. The genotype is the authority; the developed body is a
rebuildable cache. `develop()` returns four of its fields by reference, so those are
cloned here — otherwise writing the cache would write the heredity authority, which
is exactly the write-back section 3.2 forbids.

Simulation time: `EconomyState.time_s` is the authoritative ecological clock and
`sim_time_s` is a read-only view of it. `LiveState.gait_time_s` is the mechanics
sub-clock, advanced once per substep; the runner keeps them in step and `advance()`
is atomic, so they are never observable diverged.

Simulation time belongs to the core. Nothing here depends on a render frame and the
whole module runs without Unity.
"""

from __future__ import annotations

from dataclasses import replace

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

# `develop()` returns these by reference from the genotype. Cloning them keeps the
# developed body a one-way cache; without it, in-place writes to the body (the
# established lifecycle idiom, see benchmarks/lifecycle.py) rewrite the genotype.
_ALIASED_BODY_FIELDS = ("alive", "stable_id", "swim_freq_hz", "swim_wave_rad_per_depth")


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
    ) -> None:
        live_config.validate()
        economy_config.validate()
        if genotype.alive.shape[0] != economy_config.worlds:
            raise ValueError(
                f"population spans {genotype.alive.shape[0]} worlds but the economy spans "
                f"{economy_config.worlds}"
            )

        self.genotype = genotype
        body = develop(genotype)
        self.body = replace(
            body, **{name: getattr(body, name).clone() for name in _ALIASED_BODY_FIELDS}
        )
        self.live_state = initialize_live_state(self.body)
        self._require_matching_fluid(fluid)
        self.fluid = fluid
        self.live_config = live_config
        self.economy = EconomyKernel(economy_state, economy_config)
        self.geometry = GridGeometry.from_config(economy_config)

    def _require_matching_fluid(self, fluid: FluidSample) -> None:
        """`FluidSample` carries no validate(); a mis-shaped one silently broadcasts.

        A (1,1) density against a (1,2) body gives creature 1 creature 0's water
        without error, which changes the physics and nothing detects it.
        """
        lead = tuple(self.body.alive.shape)
        checks = (
            ("density_kg_m3", fluid.density_kg_m3, lead),
            ("velocity_enu_m_s", fluid.velocity_enu_m_s, (*lead, 3)),
        )
        for name, tensor, expected in checks:
            if tuple(tensor.shape) != expected:
                raise ValueError(f"fluid {name} must have shape {expected}, got {tuple(tensor.shape)}")
            if tensor.dtype != self.body.mass_sim.dtype:
                raise ValueError(f"fluid {name} dtype must match the body's {self.body.mass_sim.dtype}")
            if tensor.device != self.body.alive.device:
                raise ValueError(f"fluid {name} device must match the body's {self.body.alive.device}")

    @property
    def economy_state(self) -> EconomyState:
        return self.economy.state

    @property
    def economy_config(self) -> EconomyConfig:
        return self.economy.config

    @property
    def sim_time_s(self) -> float:
        """Read-only view of the authoritative ecological clock."""
        return float(self.economy.state.time_s)

    def rebuild_body(self) -> None:
        """Regenerate the developed-body cache from the genotype authority."""
        body = develop(self.genotype)
        self.body = replace(
            body, **{name: getattr(body, name).clone() for name in _ALIASED_BODY_FIELDS}
        )

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
