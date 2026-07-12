"""Sentinel-safe fixed six-pass body pose."""

from __future__ import annotations

import math

import torch

from sirrobin.numerics.quat import angle_axis_deg, identity, multiply, rotate
from sirrobin.physics.contracts import BodyBatch, Pose


def _gather_axis1(values: torch.Tensor, index: torch.Tensor) -> torch.Tensor:
    extra = values.ndim - index.ndim
    gather_index = index[(...,) + (None,) * extra].expand(*index.shape, *values.shape[index.ndim :])
    return torch.gather(values, 1, gather_index)


def resolve_pose(body: BodyBatch, time_s: torch.Tensor, *, apply_gait: bool = True) -> Pose:
    b, s = body.seg_mask.shape
    pos = torch.zeros((b, s, 3), dtype=body.local_pos.dtype, device=body.local_pos.device)
    rot = identity((b, s), dtype=body.local_pos.dtype, device=body.local_pos.device)
    axis_y = torch.zeros_like(body.local_pos)
    axis_y[..., 1] = 1
    theta = body.amp_deg * torch.sin(
        2.0 * math.pi * body.swim_freq[:, None] * time_s.to(body.local_pos.dtype)[:, None] + body.phase_rad
    )
    if not apply_gait:
        theta = torch.zeros_like(theta)
    flex = angle_axis_deg(theta, axis_y)
    local_flexed = multiply(body.local_rot, flex)
    for depth in range(6):
        active = body.seg_mask & (body.depth == depth)
        parent = body.parent.clamp(0, s - 1)
        parent_pos = _gather_axis1(pos, parent)
        parent_rot = _gather_axis1(rot, parent)
        next_pos = parent_pos + rotate(parent_rot, body.local_pos)
        next_rot = multiply(parent_rot, local_flexed)
        pos = torch.where(active[..., None], next_pos, pos)
        rot = torch.where(active[..., None], next_rot, rot)
    torch._assert_async(torch.isfinite(pos).all(), "pose position is non-finite")
    torch._assert_async(torch.isfinite(rot).all(), "pose rotation is non-finite")
    return Pose(pos=pos, rot=rot)


def tail_slots(body: BodyBatch) -> torch.Tensor:
    slots = torch.arange(body.seg_mask.shape[1], device=body.seg_mask.device)
    return torch.where(body.is_tail & body.seg_mask, slots[None, :], 0).amax(dim=1)


def gather_slots(values: torch.Tensor, slots: torch.Tensor) -> torch.Tensor:
    index = slots[:, None]
    if values.ndim == 3:
        return torch.gather(values, 1, index[..., None].expand(-1, 1, values.shape[-1])).squeeze(1)
    return torch.gather(values, 1, index).squeeze(1)


def tail_tip(body: BodyBatch, pose: Pose) -> torch.Tensor:
    slots = tail_slots(body)
    center = gather_slots(pose.pos, slots)
    rot = gather_slots(pose.rot, slots)
    c = gather_slots(body.abc[..., 2], slots)
    forward = torch.zeros_like(center)
    forward[..., 2] = 1
    return center + rotate(rot, forward) * c[:, None]
