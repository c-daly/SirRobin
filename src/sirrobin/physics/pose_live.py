"""Canonical FLU body pose and ENU yaw transforms."""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch

from sirrobin.numerics.quat import angle_axis_deg, identity, multiply, rotate
from sirrobin.physics.contracts import DevelopedBody


@dataclass(frozen=True, slots=True)
class LivePose:
    pos_flu_m: torch.Tensor
    rot_flu: torch.Tensor


def _flat(value: torch.Tensor, tail_dims: int) -> torch.Tensor:
    tail = value.shape[-tail_dims:] if tail_dims else ()
    return value.reshape(-1, *tail)


def _gather_axis1(values: torch.Tensor, index: torch.Tensor) -> torch.Tensor:
    extra = values.ndim - index.ndim
    gather_index = index[(...,) + (None,) * extra].expand(*index.shape, *values.shape[index.ndim:])
    return torch.gather(values, 1, gather_index)


def resolve_live_pose(
    body: DevelopedBody,
    time_s: torch.Tensor,
    turn_bias_rad_per_depth: torch.Tensor,
    *,
    effort: torch.Tensor | None = None,
) -> LivePose:
    mask = body.seg_mask.reshape(-1, body.seg_mask.shape[-1])
    parent = body.parent.reshape_as(mask).to(torch.int64)
    depth = body.depth.reshape_as(mask).to(torch.int64)
    local_pos = _flat(body.local_pos_flu_m, 2)
    local_rot = _flat(body.local_rot_flu, 2)
    amp = _flat(body.joint_amp_rad, 1)
    axis = _flat(body.hinge_axis_flu, 2)
    freq = body.swim_freq_hz.reshape(-1)
    phase = _flat(body.phase_rad, 1)
    time = time_s.reshape(-1).to(local_pos.dtype)
    turn = turn_bias_rad_per_depth.reshape(-1).to(local_pos.dtype)
    if effort is None:
        effort = torch.ones_like(turn)
    theta = effort.reshape(-1, 1) * amp * torch.sin(2 * math.pi * freq[:, None] * time[:, None] + phase)
    theta = theta + turn[:, None] * depth.to(local_pos.dtype)
    flex = angle_axis_deg(torch.rad2deg(theta), axis)
    local_flexed = multiply(local_rot, flex)
    b, s = mask.shape
    pos = torch.zeros((b, s, 3), dtype=local_pos.dtype, device=local_pos.device)
    rot = identity((b, s), dtype=local_pos.dtype, device=local_pos.device)
    for level in range(6):
        active = mask & (depth == level)
        p = parent.clamp(0, s - 1)
        p_pos = _gather_axis1(pos, p)
        p_rot = _gather_axis1(rot, p)
        next_pos = p_pos + rotate(p_rot, local_pos)
        next_rot = multiply(p_rot, local_flexed)
        pos = torch.where(active[..., None], next_pos, pos)
        rot = torch.where(active[..., None], next_rot, rot)
    torch._assert_async(torch.isfinite(pos).all(), "live pose position is nonfinite")
    torch._assert_async(torch.isfinite(rot).all(), "live pose rotation is nonfinite")
    return LivePose(pos, rot)


def yaw_quaternion(yaw_rad: torch.Tensor) -> torch.Tensor:
    half = 0.5 * yaw_rad
    zeros = torch.zeros_like(half)
    return torch.stack((zeros, zeros, torch.sin(half), torch.cos(half)), -1)


def forward_left(yaw_rad: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    cosine, sine = torch.cos(yaw_rad), torch.sin(yaw_rad)
    zeros = torch.zeros_like(cosine)
    forward = torch.stack((cosine, sine, zeros), -1)
    left = torch.stack((-sine, cosine, zeros), -1)
    return forward, left


def gather_slots(values: torch.Tensor, slots: torch.Tensor) -> torch.Tensor:
    index = slots[:, None]
    if values.ndim == 3:
        return torch.gather(values, 1, index[..., None].expand(-1, 1, values.shape[-1])).squeeze(1)
    return torch.gather(values, 1, index).squeeze(1)
