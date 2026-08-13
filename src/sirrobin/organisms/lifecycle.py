"""One batched, fixed-shape organism lifecycle transaction."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from sirrobin.numerics.flux import INT64_SAFE_MAX
from sirrobin.organisms.state import PopulationState


@dataclass(frozen=True, slots=True)
class LifecycleRequest:
    """Per-slot decisions and paid child requirements for one device step."""

    death: torch.Tensor
    birth: torch.Tensor
    child_structure_q: torch.Tensor
    child_reserve_q: torch.Tensor
    time_s: torch.Tensor


@dataclass(frozen=True, slots=True)
class LifecycleLedger:
    """Named outcomes sufficient for genotype development and observation."""

    died: torch.Tensor
    born: torch.Tensor
    accepted_parent: torch.Tensor
    parent_slot_for_child: torch.Tensor
    death_structure_return_q: torch.Tensor
    death_reserve_return_q: torch.Tensor
    birth_structure_transfer_q: torch.Tensor
    birth_reserve_transfer_q: torch.Tensor
    requested_births: torch.Tensor
    accepted_births: torch.Tensor
    unfunded_rejections: torch.Tensor
    capacity_rejections: torch.Tensor
    id_rejections: torch.Tensor


@dataclass(frozen=True, slots=True)
class LifecycleStep:
    state: PopulationState
    ledger: LifecycleLedger


def validate_lifecycle_request(
    state: PopulationState,
    request: LifecycleRequest,
) -> None:
    """Validate external lifecycle inputs before entering a compiled chunk."""

    shape = tuple(state.alive.shape)
    device = state.alive.device
    for name in ("death", "birth"):
        value = getattr(request, name)
        if value.dtype != torch.bool or tuple(value.shape) != shape:
            raise TypeError(f"{name} must be bool with shape {shape}")
        if value.device != device:
            raise ValueError(f"{name} must be on {device}")
    for name in ("child_structure_q", "child_reserve_q"):
        value = getattr(request, name)
        if value.dtype != torch.int64 or tuple(value.shape) != shape:
            raise TypeError(f"{name} must be int64 with shape {shape}")
        if value.device != device:
            raise ValueError(f"{name} must be on {device}")
        if bool(((value < 0) | (value >= INT64_SAFE_MAX)).any()):
            raise ValueError(f"{name} must remain in [0,2^62)")
    if bool(
        (
            request.child_structure_q
            > (INT64_SAFE_MAX - 1) - request.child_reserve_q
        ).any()
    ):
        raise ValueError("combined child material cost exceeds the safe domain")
    worlds = shape[0]
    if request.time_s.dtype != torch.float64 or tuple(request.time_s.shape) != (
        worlds,
    ):
        raise TypeError(f"time_s must be float64 with shape {(worlds,)}")
    if request.time_s.device != device:
        raise ValueError(f"time_s must be on {device}")
    if bool((~torch.isfinite(request.time_s)).any()) or bool(
        (request.time_s < 0.0).any()
    ):
        raise ValueError("time_s must be finite and nonnegative")


def settle_lifecycle(
    state: PopulationState,
    request: LifecycleRequest,
) -> LifecycleStep:
    """Settle deaths and paid births without a host or per-creature branch.

    This function assumes its fixed-shape inputs were validated when the runtime
    session was constructed. It intentionally contains no `.item()`, `.tolist()`,
    or tensor-valued Python condition so it can execute inside a compiled chunk.

    Birth candidates are ordered by stable ID. Accepted children occupy the
    lowest free slots, including slots released by deaths in this same step. One
    child per parent can be accepted. A parent is debited only when both a slot and
    a stable ID are available.
    """

    alive0 = state.alive
    died = alive0 & request.death
    surviving = alive0 & ~died
    zeros_i64 = torch.zeros_like(state.stable_id)
    zeros_f64 = torch.zeros_like(state.born_at_s)

    stable_surviving = torch.where(surviving, state.stable_id, zeros_i64)
    parent_surviving = torch.where(surviving, state.parent_id, zeros_i64)
    generation_surviving = torch.where(surviving, state.generation, zeros_i64)
    born_at_surviving = torch.where(surviving, state.born_at_s, zeros_f64)
    structure_surviving = torch.where(surviving, state.structure_q, zeros_i64)
    reserve_surviving = torch.where(surviving, state.reserve_q, zeros_i64)
    intake_carry_surviving = torch.where(
        surviving, state.intake_carry_mol, zeros_f64
    )
    assimilation_carry_surviving = torch.where(
        surviving, state.assimilation_carry_q, zeros_f64
    )
    maintenance_carry_surviving = torch.where(
        surviving, state.maintenance_carry_j, zeros_f64
    )

    child_cost_q = request.child_structure_q + request.child_reserve_q
    requested = surviving & request.birth
    funded_candidate = requested & (reserve_surviving >= child_cost_q)
    free = ~surviving

    worlds, capacity = alive0.shape
    slots = torch.arange(capacity, dtype=torch.int64, device=alive0.device)
    ranks = slots.expand(worlds, capacity)
    max_i64 = torch.iinfo(torch.int64).max

    # Stable sorting makes the identity order explicit and device-independent when
    # two invalid sentinel keys compare equal.
    candidate_key = torch.where(funded_candidate, stable_surviving, max_i64)
    parent_order = torch.argsort(candidate_key, dim=1, stable=True)
    free_key = torch.where(free, ranks, capacity)
    destination_order = torch.argsort(free_key, dim=1, stable=True)

    candidate_count = funded_candidate.sum(dim=1, dtype=torch.int64)
    free_count = free.sum(dim=1, dtype=torch.int64)
    # IDs use [next, int64_max); excluding int64_max avoids overflow and preserves
    # the reference allocator's exhaustion boundary.
    id_room = max_i64 - state.next_stable_id
    capacity_accepted = torch.minimum(candidate_count, free_count)
    accepted_count = torch.minimum(capacity_accepted, id_room)
    accepted_rank = ranks < accepted_count[:, None]

    accepted_parent = torch.zeros_like(alive0).scatter(
        1, parent_order, accepted_rank
    )
    born = torch.zeros_like(alive0).scatter(
        1, destination_order, accepted_rank
    )
    parent_slot_for_child = torch.full_like(state.stable_id, -1).scatter(
        1, destination_order, parent_order
    )
    rank_for_child = torch.zeros_like(state.stable_id).scatter(
        1, destination_order, ranks
    )

    birth_structure_by_parent = torch.where(
        accepted_parent, request.child_structure_q, zeros_i64
    )
    birth_reserve_by_parent = torch.where(
        accepted_parent, request.child_reserve_q, zeros_i64
    )
    parent_debit_q = birth_structure_by_parent + birth_reserve_by_parent
    reserve_after_payment = reserve_surviving - parent_debit_q

    safe_parent_slot = parent_slot_for_child.clamp_min(0)
    child_structure = torch.gather(request.child_structure_q, 1, safe_parent_slot)
    child_reserve = torch.gather(request.child_reserve_q, 1, safe_parent_slot)
    child_parent_id = torch.gather(stable_surviving, 1, safe_parent_slot)
    child_generation = torch.gather(generation_surviving, 1, safe_parent_slot) + 1
    child_stable_id = state.next_stable_id[:, None] + rank_for_child
    child_born_at = request.time_s[:, None].expand_as(state.born_at_s)

    next_state = PopulationState(
        alive=surviving | born,
        stable_id=torch.where(born, child_stable_id, stable_surviving),
        parent_id=torch.where(born, child_parent_id, parent_surviving),
        generation=torch.where(born, child_generation, generation_surviving),
        born_at_s=torch.where(born, child_born_at, born_at_surviving),
        structure_q=torch.where(born, child_structure, structure_surviving),
        reserve_q=torch.where(born, child_reserve, reserve_after_payment),
        intake_carry_mol=torch.where(born, zeros_f64, intake_carry_surviving),
        assimilation_carry_q=torch.where(
            born, zeros_f64, assimilation_carry_surviving
        ),
        maintenance_carry_j=torch.where(
            born, zeros_f64, maintenance_carry_surviving
        ),
        next_stable_id=state.next_stable_id + accepted_count,
    )
    ledger = LifecycleLedger(
        died=died,
        born=born,
        accepted_parent=accepted_parent,
        parent_slot_for_child=torch.where(
            born, parent_slot_for_child, torch.full_like(parent_slot_for_child, -1)
        ),
        death_structure_return_q=torch.where(died, state.structure_q, zeros_i64),
        death_reserve_return_q=torch.where(died, state.reserve_q, zeros_i64),
        birth_structure_transfer_q=birth_structure_by_parent,
        birth_reserve_transfer_q=birth_reserve_by_parent,
        requested_births=requested.sum(dim=1, dtype=torch.int64),
        accepted_births=accepted_count,
        unfunded_rejections=(requested & ~funded_candidate).sum(
            dim=1, dtype=torch.int64
        ),
        capacity_rejections=(candidate_count - capacity_accepted),
        id_rejections=(capacity_accepted - accepted_count),
    )
    return LifecycleStep(next_state, ledger)
