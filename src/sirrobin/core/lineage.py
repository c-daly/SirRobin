"""Immutable lineage and parametric-mutation records owned by the headless world."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class ParametricMutation:
    """One committed scalar change in authoritative hereditary state."""

    trait: str
    field_name: str
    index: tuple[int, ...]
    parent_value: float
    child_value: float


@dataclass(frozen=True, slots=True)
class LineageRecord:
    """Historical identity retained after a creature dies or its slot is reused."""

    world_index: int
    creature_id: int
    parent_id: int | None
    generation: int
    born_at_s: float
    mutation: ParametricMutation | None
