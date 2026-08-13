"""Lifecycle consequences for fixed-capacity motion state."""

from __future__ import annotations

from dataclasses import fields

import torch

from sirrobin.organisms.lifecycle import LifecycleLedger
from sirrobin.organisms.state import PopulationState
from sirrobin.physics.contracts import LiveState


def _gather_parent(value: torch.Tensor, parent_slot: torch.Tensor) -> torch.Tensor:
    tail = value.shape[2:]
    index = parent_slot[(...,) + (None,) * len(tail)].expand(
        *parent_slot.shape, *tail
    )
    return torch.gather(value, 1, index)


def settle_motion_lifecycle(
    motion: LiveState,
    population: PopulationState,
    lifecycle: LifecycleLedger,
) -> LiveState:
    """Clear dead slots and initialize newborn motion from the live parent."""

    alive = population.alive
    born = lifecycle.born
    parent_slot = lifecycle.parent_slot_for_child.clamp_min(0)
    zeroed: dict[str, torch.Tensor] = {}
    for field in fields(motion):
        value = getattr(motion, field.name)
        mask = alive[(...,) + (None,) * (value.ndim - alive.ndim)]
        zeroed[field.name] = torch.where(mask, value, torch.zeros_like(value))

    born_vector = born[..., None]
    parent_position = _gather_parent(motion.position_enu_m, parent_slot)
    parent_yaw = _gather_parent(motion.yaw_rad, parent_slot)
    desired = torch.zeros_like(motion.desired_heading_enu)
    desired[..., 0] = 1.0
    return LiveState(
        position_enu_m=torch.where(
            born_vector, parent_position, zeroed["position_enu_m"]
        ),
        velocity_rel_water_enu_m_s=torch.where(
            born_vector,
            torch.zeros_like(motion.velocity_rel_water_enu_m_s),
            zeroed["velocity_rel_water_enu_m_s"],
        ),
        yaw_rad=torch.where(born, parent_yaw, zeroed["yaw_rad"]),
        yaw_momentum_kg_m2_s=torch.where(
            born,
            torch.zeros_like(motion.yaw_momentum_kg_m2_s),
            zeroed["yaw_momentum_kg_m2_s"],
        ),
        gait_time_s=torch.where(
            born, torch.zeros_like(motion.gait_time_s), zeroed["gait_time_s"]
        ),
        desired_heading_enu=torch.where(
            born_vector, desired, zeroed["desired_heading_enu"]
        ),
        turn_bias_rad_per_depth=torch.where(
            born,
            torch.zeros_like(motion.turn_bias_rad_per_depth),
            zeroed["turn_bias_rad_per_depth"],
        ),
        heading_initialized=torch.where(
            born,
            torch.zeros_like(motion.heading_initialized),
            zeroed["heading_initialized"],
        ),
    )
