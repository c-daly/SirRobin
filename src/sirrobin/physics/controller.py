"""Stateless desired-heading policy; physical yaw remains torque/drag set."""

from __future__ import annotations

from dataclasses import replace

import torch

from sirrobin.physics.contracts import DevelopedBody, LiveState
from sirrobin.physics.live_config import LiveLocomotionConfig
from sirrobin.physics.pose_live import forward_left, resolve_live_pose
from sirrobin.physics.yaw import yaw_inertia


def _normalized_xy(value: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    norm = torch.linalg.vector_norm(value, dim=-1, keepdim=True)
    return value / norm.clamp_min(torch.finfo(value.dtype).tiny), norm.squeeze(-1) > 0


def _slerp_heading(old: torch.Tensor, new: torch.Tensor, alpha: float) -> torch.Tensor:
    dot = (old * new).sum(-1).clamp(-1.0, 1.0)
    cross = old[..., 0] * new[..., 1] - old[..., 1] * new[..., 0]
    delta = torch.atan2(cross, dot)
    angle = torch.atan2(old[..., 1], old[..., 0]) + alpha * delta
    return torch.stack((torch.cos(angle), torch.sin(angle)), -1)


def turn_authority(body: DevelopedBody, config: LiveLocomotionConfig) -> torch.Tensor:
    depth = body.depth.to(body.joint_amp_rad.dtype)
    articulated = body.seg_mask & (depth > 0)
    remaining = (config.amp_max_rad - body.joint_amp_rad).clamp_min(0.0)
    per_segment = remaining / depth.clamp_min(1.0)
    infinity = torch.full_like(per_segment, torch.inf)
    cap = torch.where(articulated, per_segment, infinity).amin(-1)
    return torch.where(torch.isfinite(cap), cap, torch.zeros_like(cap))


def body_wave_speed_m_s(
    body: DevelopedBody,
    config: LiveLocomotionConfig,
) -> torch.Tensor:
    """Return morphology-derived traveling-wave speed for gait control."""

    articulated = body.seg_mask & (body.depth > 0)
    body_length = torch.where(
        articulated,
        torch.linalg.vector_norm(body.local_pos_flu_m, dim=-1),
        0.0,
    ).sum(dim=-1)
    depth_extent = torch.where(
        articulated,
        body.depth.to(body_length.dtype),
        0.0,
    ).amax(dim=-1)
    phase_extent = (
        depth_extent * body.swim_wave_rad_per_depth.abs()
    ).clamp_min(1.0)
    return (
        body_length
        * (2.0 * torch.pi)
        * body.swim_freq_hz.abs()
        / phase_extent
    ).clamp_min(config.min_heading_speed_m_s)


def _current_yaw_rate(
    body: DevelopedBody,
    state: LiveState,
    config: LiveLocomotionConfig,
) -> torch.Tensor:
    """Derive angular velocity from current momentum and developed morphology."""

    pose = resolve_live_pose(
        body,
        state.gait_time_s,
        state.turn_bias_rad_per_depth,
    )
    segments = body.seg_mask.shape[-1]
    mask = body.seg_mask.reshape(-1, segments)
    mass_sim = body.mass_sim.reshape(-1, segments)
    total_mass_sim = mass_sim.sum(dim=-1, keepdim=True)
    center_flu = (pose.pos_flu_m * mass_sim[..., None]).sum(dim=1) / (
        total_mass_sim.clamp_min(torch.finfo(mass_sim.dtype).tiny)
    )
    relative_flu = pose.pos_flu_m - center_flu[:, None, :]
    inertia = yaw_inertia(
        relative_flu,
        pose.rot_flu,
        mass_sim * config.kg_per_sim_mass,
        body.added_mass_flu_kg.reshape(-1, segments, 3),
        mask,
    ).reshape_as(state.yaw_rad)
    safe_inertia = inertia.clamp_min(config.inertia_floor_kg_m2)
    return torch.where(
        body.alive,
        state.yaw_momentum_kg_m2_s / safe_inertia,
        0.0,
    )


def _bounded_turn_command(
    body: DevelopedBody,
    state: LiveState,
    desired_heading: torch.Tensor,
    config: LiveLocomotionConfig,
    *,
    flow_limited: bool = False,
    slew_limited: bool = True,
) -> torch.Tensor:
    forward, _ = forward_left(state.yaw_rad)
    velocity_xy = state.velocity_rel_water_enu_m_s[..., :2]
    travel, moving = _normalized_xy(velocity_xy)
    speed = torch.linalg.vector_norm(velocity_xy, dim=-1)
    use_travel = moving & (speed >= config.min_heading_speed_m_s)
    reference = torch.where(
        use_travel[..., None],
        travel,
        forward[..., :2],
    )
    dot = (reference * desired_heading).sum(-1).clamp(-1.0, 1.0)
    cross = (
        reference[..., 0] * desired_heading[..., 1]
        - reference[..., 1] * desired_heading[..., 0]
    )
    error = torch.atan2(cross, dot)
    authority = turn_authority(body, config)
    desired_rate = config.heading_rate_target_rad_s * (
        error / config.full_authority_error_rad
    ).clamp(-1.0, 1.0)
    rate_error = desired_rate - _current_yaw_rate(body, state, config)
    target = authority * (
        rate_error / config.heading_rate_target_rad_s
    ).clamp(-1.0, 1.0)
    if flow_limited:
        body_wave_speed = body_wave_speed_m_s(body, config)
        unloading_band = 0.25 * body_wave_speed
        flow_scale = (
            (body_wave_speed - speed) / unloading_band
        ).clamp(0.0, 1.0)
        target = target * flow_scale
    if not slew_limited:
        return target.clamp(-authority, authority)
    max_delta = config.turn_slew_fraction * authority
    delta = (target - state.turn_bias_rad_per_depth).clamp(-max_delta, max_delta)
    return (state.turn_bias_rad_per_depth + delta).clamp(
        -authority, authority
    )


def heading_controller_state(
    body: DevelopedBody,
    state: LiveState,
    requested_heading_enu: torch.Tensor,
    config: LiveLocomotionConfig,
) -> LiveState:
    """Return a bounded heading/turn update without mutating the input state."""

    requested, valid_request = _normalized_xy(requested_heading_enu)
    current, current_valid = _normalized_xy(state.desired_heading_enu)
    latched = torch.where(
        state.heading_initialized[..., None] & current_valid[..., None],
        _slerp_heading(current, requested, config.heading_lowpass_alpha),
        requested,
    )
    use = body.alive & valid_request
    desired_heading = torch.where(
        use[..., None], latched, state.desired_heading_enu
    )
    heading_initialized = state.heading_initialized | use

    command = _bounded_turn_command(body, state, desired_heading, config)
    return replace(
        state,
        desired_heading_enu=desired_heading,
        turn_bias_rad_per_depth=torch.where(
            body.alive,
            torch.where(
                heading_initialized,
                command,
                state.turn_bias_rad_per_depth,
            ),
            0.0,
        ),
        heading_initialized=heading_initialized,
    )


def retune_heading_controller_state(
    body: DevelopedBody,
    state: LiveState,
    config: LiveLocomotionConfig,
) -> LiveState:
    """Update turn actuation from current yaw rate without accepting new intent."""

    forward, _ = forward_left(state.yaw_rad)
    desired_heading = torch.where(
        state.heading_initialized[..., None],
        state.desired_heading_enu,
        forward[..., :2],
    )
    command = _bounded_turn_command(
        body,
        state,
        desired_heading,
        config,
        flow_limited=True,
        slew_limited=False,
    )
    return replace(
        state,
        turn_bias_rad_per_depth=torch.where(
            body.alive,
            torch.where(
                state.heading_initialized,
                command,
                state.turn_bias_rad_per_depth,
            ),
            0.0,
        ),
    )
