"""Paid exact-clone birth into fixed creature capacity."""

from __future__ import annotations

from dataclasses import dataclass, fields
from typing import TYPE_CHECKING

import torch

from sirrobin.genetics.develop import develop
from sirrobin.genetics.genotype import GenotypeBatch
from sirrobin.numerics.flux import INT64_SAFE_MAX

if TYPE_CHECKING:
    from sirrobin.core.world import HeadlessWorld


@dataclass(frozen=True, slots=True)
class BirthConfig:
    """Declared reserve delivered to a newborn after paying its structure."""

    initial_reserve_q: int

    def __post_init__(self) -> None:
        if isinstance(self.initial_reserve_q, bool) or not isinstance(
            self.initial_reserve_q, int
        ):
            raise TypeError("initial_reserve_q must be an integer")
        if self.initial_reserve_q < 0:
            raise ValueError("initial_reserve_q must be nonnegative")
        if self.initial_reserve_q >= INT64_SAFE_MAX:
            raise ValueError("initial_reserve_q exceeds the exact-integer domain")


@dataclass(frozen=True, slots=True)
class BirthReport:
    """Requested cost and committed result of one exact-clone birth attempt."""

    world_index: int
    parent_slot: int
    parent_id: int
    child_slot: int | None
    child_id: int | None
    born: bool
    reason: str | None
    structure_q: int
    initial_reserve_q: int
    total_debit_q: int
    parent_reserve_before_q: int
    parent_reserve_after_q: int
    construction_heat_j: float


def _clone_genotype_with_child(
    source: GenotypeBatch,
    *,
    world_index: int,
    parent_slot: int,
    child_slot: int,
    child_id: int,
) -> GenotypeBatch:
    values = {
        field.name: getattr(source, field.name).clone() for field in fields(source)
    }
    candidate = GenotypeBatch(**values)
    for field in fields(candidate):
        if field.name in {"alive", "stable_id"}:
            continue
        tensor = getattr(candidate, field.name)
        tensor[world_index, child_slot].copy_(tensor[world_index, parent_slot])
    candidate.stable_id[world_index, child_slot] = child_id
    candidate.alive[world_index, child_slot] = True
    candidate.validate()
    # Prove the copied authority develops successfully before committing any state.
    develop(candidate)
    return candidate


def _refusal(
    *,
    world_index: int,
    parent_slot: int,
    parent_id: int,
    reason: str,
    structure_q: int,
    initial_reserve_q: int,
    parent_reserve_q: int,
) -> BirthReport:
    return BirthReport(
        world_index=world_index,
        parent_slot=parent_slot,
        parent_id=parent_id,
        child_slot=None,
        child_id=None,
        born=False,
        reason=reason,
        structure_q=structure_q,
        initial_reserve_q=initial_reserve_q,
        total_debit_q=0,
        parent_reserve_before_q=parent_reserve_q,
        parent_reserve_after_q=parent_reserve_q,
        construction_heat_j=0.0,
    )


def attempt_exact_clone_birth(
    world: HeadlessWorld,
    config: BirthConfig,
    *,
    world_index: int = 0,
    parent_slot: int = 0,
) -> BirthReport:
    """Pay for and initialize one genetically exact but state-independent child."""
    if not isinstance(config, BirthConfig):
        raise TypeError("config must be BirthConfig")
    if not 0 <= world_index < world.body.worlds:
        raise IndexError("world_index is outside world capacity")
    if not 0 <= parent_slot < world.body.capacity:
        raise IndexError("parent_slot is outside creature capacity")
    if not bool(world.genotype.alive[world_index, parent_slot]):
        raise ValueError("birth parent must be alive")

    parent_id = int(world.genotype.stable_id[world_index, parent_slot].item())
    structure_q = int(
        world.creature_material.structure_q[world_index, parent_slot].item()
    )
    if structure_q <= 0:
        raise ValueError("birth parent must have positive structure")
    parent_reserve_q = int(
        world.creature_material.reserve_q[world_index, parent_slot].item()
    )
    total_cost_q = structure_q + config.initial_reserve_q
    if total_cost_q >= INT64_SAFE_MAX:
        raise ValueError("birth cost exceeds the exact-integer domain")

    free_slots = (~world.genotype.alive[world_index]).nonzero().flatten()
    if free_slots.numel() == 0:
        return _refusal(
            world_index=world_index,
            parent_slot=parent_slot,
            parent_id=parent_id,
            reason="slot_exhausted",
            structure_q=structure_q,
            initial_reserve_q=config.initial_reserve_q,
            parent_reserve_q=parent_reserve_q,
        )
    if parent_reserve_q < total_cost_q:
        return _refusal(
            world_index=world_index,
            parent_slot=parent_slot,
            parent_id=parent_id,
            reason="insufficient_reserve",
            structure_q=structure_q,
            initial_reserve_q=config.initial_reserve_q,
            parent_reserve_q=parent_reserve_q,
        )

    child_slot = int(free_slots[0].item())
    child_id = int(world.next_stable_id[world_index].item())
    if child_id >= torch.iinfo(torch.int64).max:
        raise ValueError("stable ID allocator is exhausted")
    candidate = _clone_genotype_with_child(
        world.genotype,
        world_index=world_index,
        parent_slot=parent_slot,
        child_slot=child_slot,
        child_id=child_id,
    )

    # All refusal and development checks have completed. From here the transaction
    # consists only of bounded tensor copies and the monotonic ID commit.
    world.creature_material.reserve_q[world_index, parent_slot] -= total_cost_q
    world.creature_material.structure_q[world_index, child_slot] = structure_q
    world.creature_material.reserve_q[world_index, child_slot] = config.initial_reserve_q
    for carry in world.creature_material.carries:
        carry[world_index, child_slot] = 0.0
    for field in fields(world.genotype):
        getattr(world.genotype, field.name).copy_(getattr(candidate, field.name))

    state = world.live_state
    state.position_enu_m[world_index, child_slot].copy_(
        state.position_enu_m[world_index, parent_slot]
    )
    state.velocity_rel_water_enu_m_s[world_index, child_slot] = 0.0
    state.yaw_rad[world_index, child_slot] = state.yaw_rad[world_index, parent_slot]
    state.yaw_momentum_kg_m2_s[world_index, child_slot] = 0.0
    state.gait_time_s[world_index, child_slot] = 0.0
    state.desired_heading_enu[world_index, child_slot] = torch.tensor(
        [1.0, 0.0],
        dtype=state.desired_heading_enu.dtype,
        device=state.desired_heading_enu.device,
    )
    state.turn_bias_rad_per_depth[world_index, child_slot] = 0.0
    state.heading_initialized[world_index, child_slot] = False
    world.rebuild_body()
    allocated_id = world._allocate_stable_id(world_index)
    if allocated_id != child_id:
        raise RuntimeError("stable ID allocator changed during birth commit")

    return BirthReport(
        world_index=world_index,
        parent_slot=parent_slot,
        parent_id=parent_id,
        child_slot=child_slot,
        child_id=child_id,
        born=True,
        reason=None,
        structure_q=structure_q,
        initial_reserve_q=config.initial_reserve_q,
        total_debit_q=total_cost_q,
        parent_reserve_before_q=parent_reserve_q,
        parent_reserve_after_q=parent_reserve_q - total_cost_q,
        construction_heat_j=(
            structure_q * world.material_energy_config.reserve_j_per_q
        ),
    )
