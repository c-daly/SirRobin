"""Declared identity-derived lifespan for explicit age mortality."""

from __future__ import annotations

import hashlib
import math
import struct
from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from sirrobin.core.world import HeadlessWorld


@dataclass(frozen=True, slots=True)
class AgeMortalityConfig:
    """Exploratory lifespan range, sampled deterministically per identity.

    The range is a declared starting condition rather than an Earth-derived
    assumption. Stable identity hashing avoids mutable RNG state and makes a
    lifespan independent of population iteration order or slot reuse.
    """

    min_lifespan_s: float
    max_lifespan_s: float
    seed: int = 0

    def __post_init__(self) -> None:
        values = (self.min_lifespan_s, self.max_lifespan_s)
        if any(
            isinstance(value, bool) or not isinstance(value, (int, float))
            for value in values
        ):
            raise TypeError("lifespan bounds must be real numbers")
        if any(not math.isfinite(value) or value <= 0.0 for value in values):
            raise ValueError("lifespan bounds must be finite and positive")
        if self.min_lifespan_s > self.max_lifespan_s:
            raise ValueError("minimum lifespan cannot exceed maximum lifespan")
        if isinstance(self.seed, bool) or not isinstance(self.seed, int):
            raise TypeError("age mortality seed must be an integer")
        if not 0 <= self.seed < 2**63:
            raise ValueError("age mortality seed must be in [0, 2^63)")

    def lifespan_s(self, world_index: int, creature_id: int) -> float:
        """Return the immutable lifespan assigned to one stable identity."""
        for name, value in (
            ("world_index", world_index),
            ("creature_id", creature_id),
        ):
            if isinstance(value, bool) or not isinstance(value, int):
                raise TypeError(f"{name} must be an integer")
            if not 0 <= value < 2**63:
                raise ValueError(f"{name} must be in [0, 2^63)")
        if self.min_lifespan_s == self.max_lifespan_s:
            return float(self.min_lifespan_s)
        digest = hashlib.sha256(
            struct.pack(">QQQ", self.seed, world_index, creature_id)
        ).digest()
        unit_interval = int.from_bytes(digest[:8], "big") / (2**64 - 1)
        return float(
            self.min_lifespan_s
            + unit_interval * (self.max_lifespan_s - self.min_lifespan_s)
        )


def old_age_due_mask(
    world: HeadlessWorld,
    config: AgeMortalityConfig,
) -> torch.Tensor:
    """Census identities whose declared lifespan elapsed by this boundary."""
    if not isinstance(config, AgeMortalityConfig):
        raise TypeError("config must be AgeMortalityConfig")
    due = torch.zeros_like(world.body.alive)
    now_s = world.sim_time_s
    for world_index, creature_slot in world.body.alive.nonzero(
        as_tuple=False
    ).tolist():
        creature_id = int(world.genotype.stable_id[world_index, creature_slot])
        born_at_s = world.lineage_record(world_index, creature_id).born_at_s
        age_s = now_s - born_at_s
        if not math.isfinite(age_s) or age_s < 0.0:
            raise ValueError("lineage age must be finite and nonnegative")
        due[world_index, creature_slot] = age_s >= config.lifespan_s(
            world_index, creature_id
        )
    return due
