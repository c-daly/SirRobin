"""Identity-bound age mortality for the device population."""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch

from sirrobin.organisms.random import identity_uniform
from sirrobin.organisms.state import PopulationState


@dataclass(frozen=True, slots=True)
class MortalityConfig:
    min_lifespan_s: float
    max_lifespan_s: float
    seed: int = 0

    def validate(self) -> None:
        bounds = (self.min_lifespan_s, self.max_lifespan_s)
        if any(
            isinstance(value, bool) or not isinstance(value, (int, float))
            for value in bounds
        ):
            raise TypeError("lifespan bounds must be real numbers")
        if any(not math.isfinite(value) or value <= 0.0 for value in bounds):
            raise ValueError("lifespan bounds must be finite and positive")
        if self.min_lifespan_s > self.max_lifespan_s:
            raise ValueError("minimum lifespan cannot exceed maximum lifespan")
        if isinstance(self.seed, bool) or not isinstance(self.seed, int):
            raise TypeError("mortality seed must be an integer")
        if not 0 <= self.seed < 2**63:
            raise ValueError("mortality seed must be in [0,2^63)")


def lifespan_s(
    state: PopulationState,
    config: MortalityConfig,
) -> torch.Tensor:
    """Return the immutable lifespan derived from each live stable identity."""

    worlds = state.alive.shape[0]
    world_index = torch.arange(
        worlds, dtype=torch.int64, device=state.alive.device
    )[:, None].expand_as(state.stable_id)
    unit = identity_uniform(
        state.stable_id,
        world_index,
        seed=config.seed,
        stream=0,
    )
    lifespan = config.min_lifespan_s + unit * (
        config.max_lifespan_s - config.min_lifespan_s
    )
    return torch.where(state.alive, lifespan, 0.0)


def old_age_due(
    state: PopulationState,
    time_s: torch.Tensor,
    config: MortalityConfig,
) -> torch.Tensor:
    """Return live identities whose deterministic lifespan has elapsed."""

    age_s = time_s[:, None] - state.born_at_s
    return state.alive & (age_s >= lifespan_s(state, config))
