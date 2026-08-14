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
    """Bounded feeding-gradient and exploratory locomotion requests."""

    food_seeking_effort_fraction: float
    search_effort_fraction: float = 0.0
    search_leg_duration_s: float = 0.0
    search_duty_fraction: float = 1.0
    food_sufficient_reserve_ratio: float = 0.0
    food_cruise_effort_fraction: float = 0.0

    def validate(self) -> None:
        for label, value in (
            ("food-seeking", self.food_seeking_effort_fraction),
            ("search", self.search_effort_fraction),
            ("food-cruise", self.food_cruise_effort_fraction),
        ):
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise TypeError(f"{label} effort must be a real number")
            if not math.isfinite(value) or not 0.0 <= value <= 1.0:
                raise ValueError(f"{label} effort must be finite and in [0,1]")
        duration = self.search_leg_duration_s
        if isinstance(duration, bool) or not isinstance(duration, (int, float)):
            raise TypeError("search leg duration must be a real number")
        if not math.isfinite(duration) or duration < 0.0:
            raise ValueError("search leg duration must be finite and nonnegative")
        for label, value in (("search duty", self.search_duty_fraction),):
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise TypeError(f"{label} fraction must be a real number")
            if not math.isfinite(value) or not 0.0 <= value <= 1.0:
                raise ValueError(f"{label} fraction must be finite and in [0,1]")
        reserve_ratio = self.food_sufficient_reserve_ratio
        if isinstance(reserve_ratio, bool) or not isinstance(
            reserve_ratio, (int, float)
        ):
            raise TypeError("food sufficient reserve ratio must be a real number")
        if not math.isfinite(reserve_ratio) or reserve_ratio < 0.0:
            raise ValueError(
                "food sufficient reserve ratio must be finite and nonnegative"
            )


@dataclass(frozen=True, slots=True)
class BehaviorStep:
    motion: LiveState
    sampled_producer_mol_m3: torch.Tensor
    producer_gradient_mol_m4: torch.Tensor
    horizontal_gradient_present: torch.Tensor
    food_sufficient: torch.Tensor
    seeking: torch.Tensor
    searching: torch.Tensor
    cruising: torch.Tensor
    idle: torch.Tensor
    requested_heading_enu: torch.Tensor
    requested_effort_fraction: torch.Tensor
    birth_requested: torch.Tensor
    invalid: torch.Tensor


def _reserve_meets_ratio(
    reserve_q: torch.Tensor,
    structure_q: torch.Tensor,
    ratio: int | float,
) -> torch.Tensor:
    """Compare ``reserve_q / structure_q`` with ``ratio`` exactly.

    A float conversion cannot distinguish every permitted int64 inventory.  This
    uses the continued-fraction comparison algorithm instead, so neither operand
    is rounded and no cross-product can overflow int64.
    """

    if ratio <= 0.0:
        return torch.ones_like(reserve_q, dtype=torch.bool)
    if ratio == 1.0:
        return reserve_q >= structure_q

    if isinstance(ratio, int):
        right_num, right_den = ratio, 1
    else:
        right_num, right_den = ratio.as_integer_ratio()

    undecided = structure_q > 0
    less = torch.zeros_like(undecided)
    left_num = reserve_q
    left_den = structure_q
    inverted = False
    int64_max = torch.iinfo(torch.int64).max

    while True:
        safe_den = torch.where(undecided, left_den, torch.ones_like(left_den))
        left_whole = torch.div(left_num, safe_den, rounding_mode="floor")
        left_rem = torch.remainder(left_num, safe_den)
        right_whole, right_rem = divmod(right_num, right_den)

        if right_whole > int64_max:
            whole_less = undecided
            whole_greater = torch.zeros_like(undecided)
            whole_equal = torch.zeros_like(undecided)
        else:
            whole_less = undecided & (left_whole < right_whole)
            whole_greater = undecided & (left_whole > right_whole)
            whole_equal = undecided & (left_whole == right_whole)
        less = less | (whole_greater if inverted else whole_less)
        undecided = whole_equal

        if right_rem == 0:
            remainder_greater = undecided & (left_rem > 0)
            less = less | (
                remainder_greater
                if inverted
                else torch.zeros_like(remainder_greater)
            )
            break

        left_is_integer = undecided & (left_rem == 0)
        less = less | (
            left_is_integer if not inverted else torch.zeros_like(left_is_integer)
        )
        undecided = undecided & (left_rem > 0)
        left_num, left_den = (
            torch.where(undecided, left_den, torch.zeros_like(left_den)),
            torch.where(undecided, left_rem, torch.ones_like(left_rem)),
        )
        right_num, right_den = right_den, right_rem
        inverted = not inverted

    return ~less


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
    """Request local chemotaxis, bounded effort, and funded lifecycle attempts.

    Intent changes only heading-controller state. Position, velocity, and yaw remain
    physical outputs. Exploratory effort is explicit, and gait power tapers to zero
    as forward speed reaches the developed body's traveling-wave speed. Every live
    identity may request a birth; the later lifecycle transaction alone decides
    funding, capacity, and identity.
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
    valid_sample = finite & nonnegative
    gradient_present = alive & valid_sample & (magnitude > 0.0)
    gradient_heading = horizontal / magnitude[..., None].clamp_min(
        torch.finfo(horizontal.dtype).tiny
    )
    reserve_target_met = _reserve_meets_ratio(
        population.reserve_q,
        population.structure_q,
        config.food_sufficient_reserve_ratio,
    )
    food_sufficient = (
        alive
        & valid_sample
        & (config.food_sufficient_reserve_ratio > 0.0)
        & (sample.value_mol_m3 > 0.0)
        & reserve_target_met
    )
    seeking = gradient_present & ~food_sufficient

    safe_leg_duration_s = (
        config.search_leg_duration_s
        if config.search_leg_duration_s > 0.0
        else 1.0
    )
    leg_index = torch.floor(motion.gait_time_s / safe_leg_duration_s)
    golden_turn_fraction = 0.5 * (3.0 - math.sqrt(5.0))
    duty_phase = torch.remainder(
        motion.gait_time_s / safe_leg_duration_s
        + population.stable_id.to(motion.gait_time_s.dtype) * golden_turn_fraction,
        1.0,
    )
    leg_active = duty_phase < config.search_duty_fraction
    schedule_active = leg_active | (config.search_leg_duration_s <= 0.0)
    search_phase = (
        population.stable_id.to(motion.yaw_rad.dtype)
        * (math.pi * (3.0 - math.sqrt(5.0)))
        + leg_index * (math.pi * (3.0 - math.sqrt(5.0)))
    )
    search_heading = torch.stack(
        (torch.cos(search_phase), torch.sin(search_phase)),
        dim=-1,
    )
    search_powered = (
        alive
        & ~gradient_present
        & ~food_sufficient
        & (config.search_effort_fraction > 0.0)
        & schedule_active
    )
    search_heading_active = search_powered & (config.search_leg_duration_s > 0.0)
    forward, _ = forward_left(motion.yaw_rad)
    horizontal_velocity = motion.velocity_rel_water_enu_m_s[..., :2]
    horizontal_speed = torch.linalg.vector_norm(horizontal_velocity, dim=-1)
    travel_heading = horizontal_velocity / horizontal_speed[..., None].clamp_min(
        torch.finfo(horizontal_velocity.dtype).tiny
    )
    food_heading = torch.where(
        (horizontal_speed >= live_config.min_heading_speed_m_s)[..., None],
        travel_heading,
        forward[..., :2],
    )
    heading = torch.where(
        gradient_present[..., None],
        gradient_heading,
        torch.where(
            food_sufficient[..., None],
            food_heading,
            torch.where(
                search_heading_active[..., None],
                search_heading,
                torch.zeros_like(search_heading),
            ),
        ),
    )
    seeking_effort = torch.where(
        seeking,
        torch.full_like(magnitude, config.food_seeking_effort_fraction),
        torch.zeros_like(magnitude),
    )
    cruise_effort = torch.where(
        food_sufficient & schedule_active,
        torch.full_like(magnitude, config.food_cruise_effort_fraction),
        torch.zeros_like(magnitude),
    )
    search_effort = torch.where(
        search_powered,
        torch.full_like(magnitude, config.search_effort_fraction),
        torch.zeros_like(magnitude),
    )
    effort = (seeking_effort + cruise_effort + search_effort).to(
        motion.yaw_rad.dtype
    )
    searching = search_powered
    cruising = food_sufficient & schedule_active
    idle = alive & ~seeking & ~searching & ~cruising
    speed = torch.linalg.vector_norm(
        motion.velocity_rel_water_enu_m_s[..., :2],
        dim=-1,
    )
    wave_speed = body_wave_speed_m_s(body, live_config)
    propulsive_margin = (1.0 - (speed / wave_speed).square()).clamp(0.0, 1.0)
    effort = effort * propulsive_margin
    next_motion = heading_controller_state(body, motion, heading, live_config)
    invalid = alive & (
        sample.vertical_out_of_bounds | ~valid_sample
    )
    return BehaviorStep(
        next_motion,
        torch.where(alive, sample.value_mol_m3, 0.0),
        torch.where(alive[..., None], sample.gradient_mol_m4, 0.0),
        gradient_present,
        food_sufficient,
        seeking,
        searching,
        cruising,
        idle,
        heading,
        effort,
        alive,
        invalid,
    )
