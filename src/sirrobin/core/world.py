"""Headless composition root for the existing mechanics and economy clocks."""

from __future__ import annotations

from dataclasses import dataclass

from sirrobin.core.live_world import advance_live_world
from sirrobin.economy.contracts import EconomyStepLedger
from sirrobin.economy.step import EconomyKernel
from sirrobin.fields.geometry import GridGeometry
from sirrobin.physics.contracts import DevelopedBody, FluidSample, LiveState, LiveStepLedger
from sirrobin.physics.live_config import LiveLocomotionConfig


@dataclass(slots=True)
class HeadlessWorld:
    """Own the existing subsystem state without biologically coupling it."""

    body: DevelopedBody
    live_state: LiveState
    fluid: FluidSample
    live_config: LiveLocomotionConfig
    geometry: GridGeometry
    economy: EconomyKernel

    def advance(self) -> tuple[LiveStepLedger, EconomyStepLedger]:
        mechanics = advance_live_world(
            self.body,
            self.live_state,
            self.fluid,
            self.live_config,
            self.geometry,
        )
        ecology = self.economy.step()
        return mechanics, ecology
