"""Exact tracked-matter allocation across developed body segments.

The population's per-creature ``structure_q`` remains the tracked matter
reservoir.  ``DevelopmentState`` is its authoritative per-segment partition,
not an additional material account.  This separation lets later developmental
growth change current morphology without inventing or double-counting matter.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch

from sirrobin.numerics.flux import INT64_SAFE_MAX, apportion_integer
from sirrobin.organisms.lifecycle import LifecycleLedger
from sirrobin.organisms.state import PopulationState
from sirrobin.physics.contracts import DevelopedBody


@dataclass(frozen=True, slots=True)
class DevelopmentState:
    """Exact allocation of each creature's structure quanta by segment slot."""

    segment_structure_q: torch.Tensor


@dataclass(frozen=True, slots=True)
class DevelopmentConfig:
    """Declared conversion between developed simulation mass and structure q."""

    structure_q_per_mass_sim: float

    def validate(self) -> None:
        value = self.structure_q_per_mass_sim
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(value)
            or value <= 0.0
        ):
            raise ValueError("structure_q_per_mass_sim must be finite and positive")


def calibrate_development_config(
    population: PopulationState,
    body: DevelopedBody,
) -> DevelopmentConfig:
    """Bind the resumed fixture's existing paid structure to physical mass.

    Calibration is a one-time migration boundary. All live founders must encode
    the same conversion; descendants are subsequently priced from this fixed
    value rather than from their parent, avoiding lineage-dependent material cost.
    """

    total_mass = body.mass_sim.to(torch.float64).sum(dim=-1)
    live = population.alive
    if not bool(live.any()):
        raise ValueError("development calibration requires at least one live founder")
    if bool((total_mass[live] <= 0.0).any()):
        raise ValueError("live founder mass must be positive")
    ratio = population.structure_q[live].to(torch.float64) / total_mass[live]
    reference = float(ratio[0].detach().cpu())
    if not torch.allclose(
        ratio,
        torch.full_like(ratio, reference),
        rtol=1.0e-6,
        atol=1.0e-9,
    ):
        raise ValueError("live founders disagree on the structure-to-mass conversion")
    config = DevelopmentConfig(reference)
    config.validate()
    return config


def target_structure_cost_q(
    body: DevelopedBody,
    config: DevelopmentConfig,
) -> torch.Tensor:
    """Price each developed body from its physical structural mass."""

    total_mass = torch.where(
        body.seg_mask,
        body.mass_sim.to(torch.float64),
        torch.zeros_like(body.mass_sim, dtype=torch.float64),
    ).sum(dim=-1)
    raw_q = total_mass * config.structure_q_per_mass_sim
    torch._assert_async(
        (torch.isfinite(raw_q) & (raw_q >= 0.0) & (raw_q < INT64_SAFE_MAX)).all(),
        "developed structure cost exceeds the exact material domain",
    )
    rounded_q = torch.round(raw_q).to(torch.int64)
    return torch.where(
        body.alive,
        rounded_q.clamp_min(1),
        torch.zeros_like(rounded_q),
    )


def allocate_segment_structure_q(
    structure_q: torch.Tensor,
    segment_mass_sim: torch.Tensor,
    segment_mask: torch.Tensor,
    alive: torch.Tensor,
) -> torch.Tensor:
    """Partition aggregate structure exactly using developed mass as weights.

    Largest-remainder allocation gives deterministic low-slot tie-breaking.
    A uniform fallback over developed slots handles a zero-mass malformed input
    without losing quanta; session-boundary validation remains responsible for
    rejecting an impossible live body with no developed segment.
    """

    if structure_q.dtype != torch.int64:
        raise TypeError("structure_q must be int64")
    if segment_mask.dtype != torch.bool or alive.dtype != torch.bool:
        raise TypeError("segment_mask and alive must be boolean")
    if segment_mass_sim.shape != segment_mask.shape:
        raise ValueError("segment mass and mask shapes must match")
    if segment_mask.shape[:-1] != structure_q.shape or alive.shape != structure_q.shape:
        raise ValueError("segment tensors must add one slot dimension to creature tensors")
    if segment_mass_sim.device != structure_q.device:
        raise ValueError("segment mass and structure must share one device")
    if segment_mask.device != structure_q.device or alive.device != structure_q.device:
        raise ValueError("all allocation tensors must share one device")

    active = alive[..., None] & segment_mask
    weights = torch.where(
        active,
        segment_mass_sim.to(torch.float64),
        torch.zeros_like(segment_mass_sim, dtype=torch.float64),
    )
    torch._assert_async(
        (torch.isfinite(weights) & (weights >= 0.0)).all(),
        "active segment masses must be finite and nonnegative",
    )
    has_mass = weights.sum(dim=-1, keepdim=True) > 0.0
    weights = torch.where(has_mass, weights, active.to(torch.float64))
    allocated = apportion_integer(structure_q, weights)
    return torch.where(active, allocated, torch.zeros_like(allocated))


def initialize_development_state(
    population: PopulationState,
    body: DevelopedBody,
) -> DevelopmentState:
    """Create the exact initial segment partition without changing either input."""

    return DevelopmentState(
        segment_structure_q=allocate_segment_structure_q(
            population.structure_q,
            body.mass_sim,
            body.seg_mask,
            population.alive,
        )
    )


def settle_development_lifecycle(
    state: DevelopmentState,
    population: PopulationState,
    body: DevelopedBody,
    lifecycle: LifecycleLedger,
) -> DevelopmentState:
    """Clear deaths and exactly allocate the structure of committed newborns."""

    born_allocation = allocate_segment_structure_q(
        population.structure_q,
        body.mass_sim,
        body.seg_mask,
        population.alive,
    )
    next_allocation = torch.where(
        lifecycle.born[..., None],
        born_allocation,
        state.segment_structure_q,
    )
    active = population.alive[..., None] & body.seg_mask
    return DevelopmentState(
        segment_structure_q=torch.where(
            active,
            next_allocation,
            torch.zeros_like(next_allocation),
        )
    )


def validate_development_state(
    state: DevelopmentState,
    population: PopulationState,
    body: DevelopedBody,
) -> None:
    """Validate the segment allocation against body and population authorities."""

    value = state.segment_structure_q
    if value.dtype != torch.int64:
        raise TypeError("segment_structure_q must be int64")
    expected_shape = tuple(body.seg_mask.shape)
    if tuple(value.shape) != expected_shape:
        raise ValueError("segment_structure_q must match developed segment shape")
    if value.device != population.alive.device or body.seg_mask.device != value.device:
        raise ValueError("development, body, and population must share one device")
    if tuple(population.alive.shape) != expected_shape[:-1]:
        raise ValueError("developed body capacity differs from population capacity")
    if body.seg_mask.dtype != torch.bool:
        raise TypeError("developed segment mask must be boolean")
    if bool((value < 0).any()):
        raise ValueError("segment structure must be nonnegative")

    active = population.alive[..., None] & body.seg_mask
    if bool((value[~active] != 0).any()):
        raise ValueError("inactive or undeveloped segment must contain zero structure")
    if bool((population.alive & ~body.seg_mask.any(dim=-1)).any()):
        raise ValueError("every live creature requires a developed segment")
    if not torch.equal(value.sum(dim=-1, dtype=torch.int64), population.structure_q):
        raise ValueError("development must exactly partition population structure")
