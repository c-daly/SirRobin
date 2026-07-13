"""Donor-shaped desired-heading policy; physical yaw remains torque/drag set."""

from __future__ import annotations

import torch

from sirrobin.physics.contracts import DevelopedBody, LiveState
from sirrobin.physics.live_config import LiveLocomotionConfig
from sirrobin.physics.pose_live import forward_left


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


def update_heading_controller(
    body: DevelopedBody,
    state: LiveState,
    requested_heading_enu: torch.Tensor,
    config: LiveLocomotionConfig,
) -> None:
    requested, valid_request = _normalized_xy(requested_heading_enu)
    current = state.desired_heading_enu
    current, current_valid = _normalized_xy(current)
    latched = torch.where(
        state.heading_initialized[..., None] & current_valid[..., None],
        _slerp_heading(current, requested, config.heading_lowpass_alpha),
        requested,
    )
    use = body.alive & valid_request
    state.desired_heading_enu.copy_(torch.where(use[..., None], latched, state.desired_heading_enu))
    state.heading_initialized.copy_(state.heading_initialized | use)

    forward, _ = forward_left(state.yaw_rad)
    velocity_xy = state.velocity_rel_water_enu_m_s[..., :2]
    travel, moving = _normalized_xy(velocity_xy)
    use_travel = moving & (
        torch.linalg.vector_norm(velocity_xy, dim=-1) >= config.min_heading_speed_m_s
    )
    reference = torch.where(use_travel[..., None], travel, forward[..., :2])
    desired = state.desired_heading_enu
    dot = (reference * desired).sum(-1).clamp(-1.0, 1.0)
    cross = reference[..., 0] * desired[..., 1] - reference[..., 1] * desired[..., 0]
    error = torch.atan2(cross, dot)
    authority = turn_authority(body, config)
    target = authority * (error / config.full_authority_error_rad).clamp(-1.0, 1.0)
    max_delta = config.turn_slew_fraction * authority
    delta = (target - state.turn_bias_rad_per_depth).clamp(-max_delta, max_delta)
    command = (state.turn_bias_rad_per_depth + delta).clamp(-authority, authority)
    state.turn_bias_rad_per_depth.copy_(torch.where(body.alive, command, 0.0))
