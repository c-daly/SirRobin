"""Ordered composition of one device organism transaction interval."""

from __future__ import annotations

from dataclasses import dataclass, replace

import torch

from sirrobin.organisms.lifecycle import (
    LifecycleRequest,
    LifecycleStep,
    settle_lifecycle,
)
from sirrobin.organisms.metabolism import (
    MetabolismConfig,
    MetabolismInputs,
    MetabolismStep,
    settle_metabolism,
)
from sirrobin.organisms.mortality import MortalityConfig, old_age_due
from sirrobin.organisms.state import PopulationState


@dataclass(frozen=True, slots=True)
class OrganismIntervalInputs:
    metabolism: MetabolismInputs
    birth_requested: torch.Tensor
    child_structure_q: torch.Tensor
    child_reserve_q: torch.Tensor
    birth_release_energy_q: torch.Tensor
    time_s: torch.Tensor


@dataclass(frozen=True, slots=True)
class OrganismIntervalStep:
    state: PopulationState
    metabolism: MetabolismStep
    lifecycle: LifecycleStep


def advance_organism_interval(
    state: PopulationState,
    inputs: OrganismIntervalInputs,
    metabolism_config: MetabolismConfig,
    mortality_config: MortalityConfig,
) -> OrganismIntervalStep:
    """Settle age, metabolism, death, paid births, IDs, and live lineage.

    This is deliberately a thin ordering function. It contains no biological
    equation of its own and no observation/report formatting. The field-runtime
    consumer must deposit the two exact return ledgers before publishing the next
    complete world state, and the genotype consumer must initialize every slot in
    ``lifecycle.ledger.born`` from ``parent_slot_for_child``.
    """

    age_due = old_age_due(state, inputs.time_s, mortality_config)
    metabolism = settle_metabolism(
        state,
        replace(inputs.metabolism, old_age_due=age_due),
        metabolism_config,
    )
    lifecycle = settle_lifecycle(
        metabolism.state,
        LifecycleRequest(
            death=metabolism.ledger.death,
            birth=inputs.birth_requested,
            child_structure_q=inputs.child_structure_q,
            child_reserve_q=inputs.child_reserve_q,
            birth_release_energy_q=inputs.birth_release_energy_q,
            time_s=inputs.time_s,
        ),
    )
    return OrganismIntervalStep(lifecycle.state, metabolism, lifecycle)
