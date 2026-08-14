"""Fallible local producer-gradient intent for ecological movement."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch

from sirrobin.core.controller import update_heading_controller
from sirrobin.fields.grid import ScalarGrid

if TYPE_CHECKING:
    from sirrobin.core.world import HeadlessWorld


@dataclass(frozen=True, slots=True)
class FoodSeekingConfig:
    """The bounded swim effort requested when a horizontal gradient exists."""

    effort_fraction: float

    def __post_init__(self) -> None:
        value = self.effort_fraction
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise TypeError("food-seeking effort must be a real number")
        if not math.isfinite(value) or not 0.0 <= value <= 1.0:
            raise ValueError("food-seeking effort must be finite and in [0,1]")


@dataclass(frozen=True, slots=True)
class FoodSeekingReport:
    """Observed local cause and requested action, never physical outcome."""

    sampled_producer_mol_m3: torch.Tensor
    producer_gradient_mol_m4: torch.Tensor
    horizontal_gradient_present: torch.Tensor
    requested_heading_enu: torch.Tensor
    requested_effort_fraction: torch.Tensor


def apply_food_seeking_intent(
    world: HeadlessWorld,
    config: FoodSeekingConfig,
) -> FoodSeekingReport:
    """Request uphill heading and fixed bounded effort from the local field.

    A flat horizontal field requests zero effort and no new heading. The existing
    controller bounds the turn request; this function never assigns position,
    velocity, or yaw and makes no claim that the organism reaches food.
    """
    if not isinstance(config, FoodSeekingConfig):
        raise TypeError("config must be FoodSeekingConfig")
    if not torch.equal(world.body.alive, world.genotype.alive) or not torch.equal(
        world.body.stable_id, world.genotype.stable_id
    ):
        raise ValueError("developed body identity cache differs from genotype authority")

    alive = world.body.alive
    safe_positions = torch.where(
        alive[..., None],
        world.live_state.position_enu_m,
        torch.zeros_like(world.live_state.position_enu_m),
    )
    producer = ScalarGrid(
        world.economy_state.bp_q,
        world.geometry,
        q_mass_mol=world.economy_config.q_mass_mol,
    )
    sample = producer.sample(safe_positions)
    finite = torch.isfinite(sample.value_mol_m3) & torch.isfinite(
        sample.gradient_mol_m4
    ).all(dim=-1)
    nonnegative = sample.value_mol_m3 >= 0.0
    if not bool((~alive | (finite & nonnegative)).all()):
        raise ValueError("live producer samples must be finite and nonnegative")

    horizontal = sample.gradient_mol_m4[..., :2]
    magnitude = torch.linalg.vector_norm(horizontal, dim=-1)
    gradient_present = alive & (magnitude > 0.0)
    heading = horizontal / magnitude[..., None].clamp_min(
        torch.finfo(horizontal.dtype).tiny
    )
    heading = torch.where(
        gradient_present[..., None], heading, torch.zeros_like(heading)
    )
    effort = torch.where(
        gradient_present,
        torch.full_like(magnitude, config.effort_fraction),
        torch.zeros_like(magnitude),
    )
    update_heading_controller(world.body, world.live_state, heading, world.live_config)

    return FoodSeekingReport(
        sampled_producer_mol_m3=torch.where(
            alive, sample.value_mol_m3, torch.zeros_like(sample.value_mol_m3)
        ).clone(),
        producer_gradient_mol_m4=torch.where(
            alive[..., None],
            sample.gradient_mol_m4,
            torch.zeros_like(sample.gradient_mol_m4),
        ).clone(),
        horizontal_gradient_present=gradient_present.clone(),
        requested_heading_enu=heading.clone(),
        requested_effort_fraction=effort.clone(),
    )
