"""Paid clone or bounded-parametric birth into fixed creature capacity."""

from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass, fields
from typing import TYPE_CHECKING

import torch

from sirrobin.core.lineage import ParametricMutation
from sirrobin.genetics.develop import develop
from sirrobin.genetics.genotype import GenotypeBatch
from sirrobin.numerics.flux import INT64_SAFE_MAX

if TYPE_CHECKING:
    from sirrobin.core.world import HeadlessWorld


_MUTATION_TRAITS = frozenset(
    {"joint_amplitude", "swim_frequency", "swim_wave"}
)
_MIN_JOINT_AMPLITUDE_RAD = 0.0
_MAX_JOINT_AMPLITUDE_RAD = 0.5 * math.pi
_MIN_SWIM_FREQUENCY_HZ = 0.0
_MAX_SWIM_FREQUENCY_HZ = 10.0
_MIN_SWIM_WAVE_RAD_PER_DEPTH = -math.pi
_MAX_SWIM_WAVE_RAD_PER_DEPTH = math.pi


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
class ParametricMutationConfig:
    """Exactly one bounded locomotion-locus mutation per committed paid birth.

    These are exploratory numerical operating bounds, not biological calibration.
    Shape and topology are excluded until their structural matter cost is defined.
    """

    seed: int
    traits: tuple[str, ...] = (
        "joint_amplitude",
        "swim_frequency",
        "swim_wave",
    )
    joint_amplitude_step_rad: float = math.radians(2.0)
    swim_frequency_step_hz: float = 0.1
    swim_wave_step_rad_per_depth: float = math.radians(2.0)

    def __post_init__(self) -> None:
        if isinstance(self.seed, bool) or not isinstance(self.seed, int):
            raise TypeError("mutation seed must be an integer")
        if self.seed < 0 or self.seed >= 2**63:
            raise ValueError("mutation seed must be in [0, 2^63)")
        if not isinstance(self.traits, tuple):
            raise TypeError("mutation traits must be a tuple")
        if not self.traits:
            raise ValueError("at least one mutation trait is required")
        if len(set(self.traits)) != len(self.traits):
            raise ValueError("mutation traits must not contain duplicates")
        unknown = set(self.traits) - _MUTATION_TRAITS
        if unknown:
            raise ValueError(f"unknown mutation traits: {sorted(unknown)}")
        steps = (
            self.joint_amplitude_step_rad,
            self.swim_frequency_step_hz,
            self.swim_wave_step_rad_per_depth,
        )
        if any(not math.isfinite(step) or step <= 0.0 for step in steps):
            raise ValueError("mutation steps must be positive and finite")


@dataclass(frozen=True, slots=True)
class BirthReport:
    """Requested cost and committed result of one paid birth attempt."""

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
    mutation: ParametricMutation | None


def _bounded_step(
    parent_value: float,
    *,
    step: float,
    direction: int,
    lower: float,
    upper: float,
    trait: str,
) -> float:
    if not lower <= parent_value <= upper:
        raise ValueError(f"parent {trait} is outside the declared mutation domain")
    child_value = min(upper, max(lower, parent_value + direction * step))
    if child_value == parent_value:
        child_value = min(upper, max(lower, parent_value - direction * step))
    if child_value == parent_value:
        raise ValueError(f"parent {trait} has no mutable range")
    return child_value


def _mutation_word(
    config: ParametricMutationConfig,
    *,
    world_index: int,
    parent_id: int,
    child_id: int,
) -> int:
    payload = (
        f"sirrobin-parametric-mutation-v1|{config.seed}|{world_index}|"
        f"{parent_id}|{child_id}"
    ).encode("ascii")
    return int.from_bytes(hashlib.sha256(payload).digest(), "big")


def _mutate_child(
    candidate: GenotypeBatch,
    config: ParametricMutationConfig,
    *,
    world_index: int,
    parent_slot: int,
    child_slot: int,
    parent_id: int,
    child_id: int,
) -> ParametricMutation:
    joint_nodes = (
        candidate.node_mask[world_index, parent_slot]
        & candidate.node_expressed[world_index, parent_slot]
    ).nonzero(as_tuple=False).flatten()
    joint_nodes = joint_nodes[joint_nodes > 0]
    available_traits = tuple(
        trait
        for trait in config.traits
        if trait != "joint_amplitude" or joint_nodes.numel() > 0
    )
    if not available_traits:
        raise ValueError("birth parent has no locus for the configured mutation traits")

    word = _mutation_word(
        config,
        world_index=world_index,
        parent_id=parent_id,
        child_id=child_id,
    )
    trait = available_traits[word % len(available_traits)]
    direction = 1 if ((word >> 16) & 1) else -1
    if trait == "joint_amplitude":
        node_index = int(joint_nodes[(word >> 32) % joint_nodes.numel()].item())
        field_name = "node_joint_amp_rad"
        index = (node_index,)
        tensor = candidate.node_joint_amp_rad
        parent_value = float(tensor[world_index, parent_slot, node_index].item())
        child_value = _bounded_step(
            parent_value,
            step=config.joint_amplitude_step_rad,
            direction=direction,
            lower=_MIN_JOINT_AMPLITUDE_RAD,
            upper=_MAX_JOINT_AMPLITUDE_RAD,
            trait=trait,
        )
        tensor[world_index, child_slot, node_index] = child_value
        child_value = float(tensor[world_index, child_slot, node_index].item())
    elif trait == "swim_frequency":
        field_name = "swim_freq_hz"
        index = ()
        tensor = candidate.swim_freq_hz
        parent_value = float(tensor[world_index, parent_slot].item())
        child_value = _bounded_step(
            parent_value,
            step=config.swim_frequency_step_hz,
            direction=direction,
            lower=_MIN_SWIM_FREQUENCY_HZ,
            upper=_MAX_SWIM_FREQUENCY_HZ,
            trait=trait,
        )
        tensor[world_index, child_slot] = child_value
        child_value = float(tensor[world_index, child_slot].item())
    else:
        field_name = "swim_wave_rad_per_depth"
        index = ()
        tensor = candidate.swim_wave_rad_per_depth
        parent_value = float(tensor[world_index, parent_slot].item())
        child_value = _bounded_step(
            parent_value,
            step=config.swim_wave_step_rad_per_depth,
            direction=direction,
            lower=_MIN_SWIM_WAVE_RAD_PER_DEPTH,
            upper=_MAX_SWIM_WAVE_RAD_PER_DEPTH,
            trait=trait,
        )
        tensor[world_index, child_slot] = child_value
        child_value = float(tensor[world_index, child_slot].item())
    return ParametricMutation(
        trait=trait,
        field_name=field_name,
        index=index,
        parent_value=parent_value,
        child_value=child_value,
    )


def _candidate_genotype_with_child(
    source: GenotypeBatch,
    *,
    world_index: int,
    parent_slot: int,
    child_slot: int,
    child_id: int,
    mutation_config: ParametricMutationConfig | None,
) -> tuple[GenotypeBatch, ParametricMutation | None]:
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
    mutation = (
        None
        if mutation_config is None
        else _mutate_child(
            candidate,
            mutation_config,
            world_index=world_index,
            parent_slot=parent_slot,
            child_slot=child_slot,
            parent_id=int(source.stable_id[world_index, parent_slot].item()),
            child_id=child_id,
        )
    )
    candidate.validate()
    # Prove the prospective hereditary authority develops before committing state.
    develop(candidate)
    return candidate, mutation


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
        mutation=None,
    )


def attempt_paid_birth(
    world: HeadlessWorld,
    config: BirthConfig,
    *,
    world_index: int = 0,
    parent_slot: int = 0,
    mutation_config: ParametricMutationConfig | None = None,
) -> BirthReport:
    """Pay for and initialize one state-independent clone or parametric mutant."""
    if not isinstance(config, BirthConfig):
        raise TypeError("config must be BirthConfig")
    if mutation_config is not None and not isinstance(
        mutation_config, ParametricMutationConfig
    ):
        raise TypeError("mutation_config must be ParametricMutationConfig or None")
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
    candidate, mutation = _candidate_genotype_with_child(
        world.genotype,
        world_index=world_index,
        parent_slot=parent_slot,
        child_slot=child_slot,
        child_id=child_id,
        mutation_config=mutation_config,
    )
    lineage = world._prepare_birth_lineage(
        world_index=world_index,
        parent_id=parent_id,
        child_id=child_id,
        mutation=mutation,
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
    world._commit_lineage(lineage)

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
        mutation=mutation,
    )


def attempt_exact_clone_birth(
    world: HeadlessWorld,
    config: BirthConfig,
    *,
    world_index: int = 0,
    parent_slot: int = 0,
) -> BirthReport:
    """Pay for one genetically exact child while recording its lineage."""
    return attempt_paid_birth(
        world,
        config,
        world_index=world_index,
        parent_slot=parent_slot,
        mutation_config=None,
    )
