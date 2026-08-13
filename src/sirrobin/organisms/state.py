"""Fixed-capacity organism state for the device runtime.

The container owns no biological behavior. Device kernels accept and return this
state explicitly so they can be compiled without reaching through a world object.
"""

from __future__ import annotations

from dataclasses import dataclass, fields

import torch

from sirrobin.numerics.flux import INT64_SAFE_MAX


@dataclass(frozen=True, slots=True)
class PopulationState:
    """Authoritative per-slot identity, lineage, and tracked material."""

    alive: torch.Tensor
    stable_id: torch.Tensor
    parent_id: torch.Tensor
    generation: torch.Tensor
    born_at_s: torch.Tensor
    structure_q: torch.Tensor
    reserve_q: torch.Tensor
    intake_carry_mol: torch.Tensor
    assimilation_carry_q: torch.Tensor
    maintenance_carry_j: torch.Tensor
    next_stable_id: torch.Tensor

    @property
    def shape(self) -> tuple[int, int]:
        return int(self.alive.shape[0]), int(self.alive.shape[1])


def validate_population_state(state: PopulationState) -> None:
    """Validate a state at a session boundary, outside compiled device steps."""

    if state.alive.dtype != torch.bool or state.alive.ndim != 2:
        raise TypeError("alive must be bool [world, slot]")
    shape = tuple(state.alive.shape)
    device = state.alive.device
    int_fields = (
        "stable_id",
        "parent_id",
        "generation",
        "structure_q",
        "reserve_q",
    )
    for name in int_fields:
        value = getattr(state, name)
        if value.dtype != torch.int64 or tuple(value.shape) != shape:
            raise TypeError(f"{name} must be int64 with shape {shape}")
        if value.device != device:
            raise ValueError(f"{name} must be on {device}")
    if state.born_at_s.dtype != torch.float64 or tuple(state.born_at_s.shape) != shape:
        raise TypeError(f"born_at_s must be float64 with shape {shape}")
    if state.born_at_s.device != device:
        raise ValueError(f"born_at_s must be on {device}")
    carry_fields = (
        "intake_carry_mol",
        "assimilation_carry_q",
        "maintenance_carry_j",
    )
    for name in carry_fields:
        value = getattr(state, name)
        if value.dtype != torch.float64 or tuple(value.shape) != shape:
            raise TypeError(f"{name} must be float64 with shape {shape}")
        if value.device != device:
            raise ValueError(f"{name} must be on {device}")
        if bool((~torch.isfinite(value)).any()) or bool((value < 0.0).any()):
            raise ValueError(f"{name} must be finite and nonnegative")
    if state.next_stable_id.dtype != torch.int64 or tuple(
        state.next_stable_id.shape
    ) != (shape[0],):
        raise TypeError(f"next_stable_id must be int64 with shape {(shape[0],)}")
    if state.next_stable_id.device != device:
        raise ValueError(f"next_stable_id must be on {device}")

    inactive = ~state.alive
    inactive_fields = tuple(
        name for name in int_fields if name not in {"parent_id", "generation"}
    ) + (
        "parent_id",
        "generation",
        "born_at_s",
        *carry_fields,
    )
    for name in inactive_fields:
        if bool((getattr(state, name)[inactive] != 0).any()):
            raise ValueError(f"inactive slots must have zero {name}")
    if bool((state.stable_id[state.alive] <= 0).any()):
        raise ValueError("live stable IDs must be positive")
    if bool((state.parent_id < 0).any()) or bool((state.generation < 0).any()):
        raise ValueError("parent IDs and generations must be nonnegative")
    for name in ("structure_q", "reserve_q"):
        value = getattr(state, name)
        if bool(((value < 0) | (value >= INT64_SAFE_MAX)).any()):
            raise ValueError(f"{name} must remain in [0,2^62)")
    if bool((~torch.isfinite(state.born_at_s)).any()) or bool(
        (state.born_at_s < 0.0).any()
    ):
        raise ValueError("birth times must be finite and nonnegative")
    if bool((state.next_stable_id <= 0).any()):
        raise ValueError("next stable IDs must be positive")
    maximum_live_id = torch.where(
        state.alive,
        state.stable_id,
        torch.zeros_like(state.stable_id),
    ).max(dim=1).values
    if bool((state.next_stable_id <= maximum_live_id).any()):
        raise ValueError("next stable ID must exceed every live stable ID")

    for world_index in range(shape[0]):
        live_ids = state.stable_id[world_index, state.alive[world_index]]
        if live_ids.numel() != torch.unique(live_ids).numel():
            raise ValueError("live stable IDs must be unique within each world")

    # Catch accidental fields early when this state is adapted to compiled tuples.
    if tuple(field.name for field in fields(state)) != (
        "alive",
        "stable_id",
        "parent_id",
        "generation",
        "born_at_s",
        "structure_q",
        "reserve_q",
        "intake_carry_mol",
        "assimilation_carry_q",
        "maintenance_carry_j",
        "next_stable_id",
    ):
        raise RuntimeError("population state field order changed")
