"""Batched local-field intent for the device living runtime."""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch

from sirrobin.fields.geometry import GridGeometry
from sirrobin.fields.sample import sample_reservoir_device
from sirrobin.organisms.state import PopulationState
from sirrobin.physics.contracts import DevelopedBody, LiveState
from sirrobin.physics.controller import body_wave_speed_m_s, heading_controller_state
from sirrobin.physics.live_config import LiveLocomotionConfig
from sirrobin.physics.pose_live import forward_left


@dataclass(frozen=True, slots=True)
class BehaviorConfig:
    """One bounded locomotor drive steered only by local food state."""

    locomotor_effort_fraction: float

    def validate(self) -> None:
        value = self.locomotor_effort_fraction
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise TypeError("locomotor effort must be a real number")
        if not math.isfinite(value) or not 0.0 <= value <= 1.0:
            raise ValueError("locomotor effort must be finite and in [0,1]")


@dataclass(frozen=True, slots=True)
class BehaviorStep:
    motion: LiveState
    sampled_producer_mol_m3: torch.Tensor
    producer_gradient_mol_m4: torch.Tensor
    food_gradient_body_forward_left_mol_m4: torch.Tensor
    horizontal_gradient_present: torch.Tensor
    locomoting: torch.Tensor
    requested_heading_enu: torch.Tensor
    requested_effort_fraction: torch.Tensor
    birth_requested: torch.Tensor
    invalid: torch.Tensor


def request_living_intent(
    population: PopulationState,
    body: DevelopedBody,
    motion: LiveState,
    producer_q: torch.Tensor,
    geometry: GridGeometry,
    live_config: LiveLocomotionConfig,
    config: BehaviorConfig,
    *,
    q_mass_mol: float,
) -> BehaviorStep:
    """Request autonomous locomotion, local food steering, and lifecycle attempts.

    Every live organism samples the same generic local food state. Intent changes
    only heading-controller state; position, velocity, and yaw remain physical
    outputs. Gait power tapers to zero as forward speed reaches the developed body's
    traveling-wave speed. The later lifecycle transaction alone decides whether a
    requested birth is fundable and has capacity and identity.
    """

    alive = population.alive
    safe_position = torch.where(
        alive[..., None], motion.position_enu_m, torch.zeros_like(motion.position_enu_m)
    )
    sample = sample_reservoir_device(
        producer_q,
        safe_position,
        geometry,
        q_mass_mol=q_mass_mol,
    )
    finite = torch.isfinite(sample.value_mol_m3) & torch.isfinite(
        sample.gradient_mol_m4
    ).all(dim=-1)
    nonnegative = sample.value_mol_m3 >= 0.0
    horizontal = sample.gradient_mol_m4[..., :2]
    magnitude = torch.linalg.vector_norm(horizontal, dim=-1)
    valid_sample = finite & nonnegative & ~sample.vertical_out_of_bounds
    gradient_present = alive & valid_sample & (magnitude > 0.0)
    gradient_heading = horizontal / magnitude[..., None].clamp_min(
        torch.finfo(horizontal.dtype).tiny
    )
    forward, left = forward_left(motion.yaw_rad)
    body_gradient = torch.stack(
        (
            (horizontal * forward[..., :2]).sum(dim=-1),
            (horizontal * left[..., :2]).sum(dim=-1),
        ),
        dim=-1,
    )
    body_gradient_heading = body_gradient / magnitude[..., None].clamp_min(
        torch.finfo(horizontal.dtype).tiny
    )
    world_heading_from_body_state = (
        body_gradient_heading[..., :1] * forward[..., :2]
        + body_gradient_heading[..., 1:] * left[..., :2]
    )
    heading = torch.where(
        gradient_present[..., None],
        world_heading_from_body_state,
        torch.zeros_like(gradient_heading),
    )
    effort = torch.where(
        alive,
        torch.full_like(magnitude, config.locomotor_effort_fraction),
        torch.zeros_like(magnitude),
    )
    speed = torch.linalg.vector_norm(
        motion.velocity_rel_water_enu_m_s[..., :2],
        dim=-1,
    )
    wave_speed = body_wave_speed_m_s(body, live_config)
    propulsive_margin = (1.0 - (speed / wave_speed).square()).clamp(0.0, 1.0)
    effort = (effort * propulsive_margin).to(motion.yaw_rad.dtype)
    locomoting = alive & (effort > 0.0)
    next_motion = heading_controller_state(body, motion, heading, live_config)
    invalid = alive & (sample.vertical_out_of_bounds | ~finite | ~nonnegative)
    return BehaviorStep(
        next_motion,
        torch.where(alive & valid_sample, sample.value_mol_m3, 0.0),
        torch.where(
            (alive & valid_sample)[..., None],
            sample.gradient_mol_m4,
            0.0,
        ),
        torch.where(
            (alive & valid_sample)[..., None],
            body_gradient,
            0.0,
        ),
        gradient_present,
        locomoting,
        heading,
        effort,
        alive,
        invalid,
    )
