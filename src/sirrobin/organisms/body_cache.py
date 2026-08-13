"""Exact developed-body cache updates for parametric paid births."""

from __future__ import annotations

from dataclasses import fields

import torch

from sirrobin.genetics.genotype import GenotypeBatch
from sirrobin.organisms.lifecycle import LifecycleLedger
from sirrobin.organisms.state import PopulationState
from sirrobin.physics.contracts import DevelopedBody


def _gather_parent(value: torch.Tensor, parent_slot: torch.Tensor) -> torch.Tensor:
    tail = value.shape[2:]
    index = parent_slot[(...,) + (None,) * len(tail)].expand(
        *parent_slot.shape, *tail
    )
    return torch.gather(value, 1, index)


def inherit_developed_births(
    body: DevelopedBody,
    genotype: GenotypeBatch,
    population: PopulationState,
    lifecycle: LifecycleLedger,
) -> DevelopedBody:
    """Update the cache exactly for the currently supported parametric mutations.

    Paid children inherit static developed topology and geometry from their parent.
    Joint amplitude, swim frequency, swim wave, phase, identity, and live masks are
    then re-derived from hereditary authority. Topology or shape mutation must use
    full development instead of this kernel.
    """

    born = lifecycle.born
    parent_slot = lifecycle.parent_slot_for_child.clamp_min(0)
    values: dict[str, torch.Tensor] = {}
    for field in fields(body):
        name = field.name
        current = getattr(body, name)
        if name == "alive":
            values[name] = population.alive
        elif name == "stable_id":
            values[name] = population.stable_id
        else:
            parent = _gather_parent(current, parent_slot)
            mask = born[(...,) + (None,) * (current.ndim - born.ndim)]
            values[name] = torch.where(mask, parent, current)

    source_node = values["source_node"].to(torch.int64)
    joint_amp = torch.gather(genotype.node_joint_amp_rad, -1, source_node)
    values["joint_amp_rad"] = torch.where(
        values["seg_mask"], joint_amp, 0.0
    )
    values["swim_freq_hz"] = genotype.swim_freq_hz
    values["swim_wave_rad_per_depth"] = genotype.swim_wave_rad_per_depth
    values["phase_rad"] = torch.where(
        values["seg_mask"],
        -values["depth"].to(genotype.swim_wave_rad_per_depth.dtype)
        * genotype.swim_wave_rad_per_depth[..., None],
        0.0,
    )
    return DevelopedBody(**values)


def commit_developed_births(
    body: DevelopedBody,
    candidate_body: DevelopedBody,
    genotype: GenotypeBatch,
    population: PopulationState,
    lifecycle: LifecycleLedger,
) -> DevelopedBody:
    """Gather fully developed accepted candidates into their assigned slots."""

    born = lifecycle.born
    parent_slot = lifecycle.parent_slot_for_child.clamp_min(0)
    values: dict[str, torch.Tensor] = {}
    for field in fields(body):
        name = field.name
        current = getattr(body, name)
        if name == "alive":
            values[name] = population.alive
        elif name == "stable_id":
            values[name] = population.stable_id
        else:
            parent_candidate = _gather_parent(
                getattr(candidate_body, name), parent_slot
            )
            mask = born[(...,) + (None,) * (current.ndim - born.ndim)]
            values[name] = torch.where(mask, parent_candidate, current)

    # The committed genotype is accepted as an explicit authority dependency;
    # identity is already taken from the population transaction above.
    torch._assert_async(
        (genotype.alive == population.alive).all(),
        "committed genotype identity must match population authority",
    )
    return DevelopedBody(**values)
