"""Batched quaternions in `(x,y,z,w)` order, matching Unity storage."""

from __future__ import annotations

import math

import torch


def identity(shape: tuple[int, ...], *, dtype: torch.dtype, device: torch.device | str) -> torch.Tensor:
    out = torch.zeros((*shape, 4), dtype=dtype, device=device)
    out[..., 3] = 1
    return out


def conjugate(q: torch.Tensor) -> torch.Tensor:
    return torch.cat((-q[..., :3], q[..., 3:4]), dim=-1)


def multiply(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    ax, ay, az, aw = a.unbind(-1)
    bx, by, bz, bw = b.unbind(-1)
    return torch.stack(
        (
            aw * bx + ax * bw + ay * bz - az * by,
            aw * by - ax * bz + ay * bw + az * bx,
            aw * bz + ax * by - ay * bx + az * bw,
            aw * bw - ax * bx - ay * by - az * bz,
        ),
        dim=-1,
    )


def normalize(q: torch.Tensor) -> torch.Tensor:
    return q / torch.linalg.vector_norm(q, dim=-1, keepdim=True).clamp_min(torch.finfo(q.dtype).tiny)


def rotate(q: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    qv = q[..., :3]
    t = 2.0 * torch.linalg.cross(qv, v, dim=-1)
    return v + q[..., 3:4] * t + torch.linalg.cross(qv, t, dim=-1)


def angle_axis_deg(angle_deg: torch.Tensor, axis: torch.Tensor) -> torch.Tensor:
    half = angle_deg * (math.pi / 360.0)
    axis = axis / torch.linalg.vector_norm(axis, dim=-1, keepdim=True).clamp_min(torch.finfo(axis.dtype).tiny)
    return torch.cat((axis * torch.sin(half).unsqueeze(-1), torch.cos(half).unsqueeze(-1)), dim=-1)


def euler_unity_deg(euler_xyz_deg: torch.Tensor) -> torch.Tensor:
    """Unity `Quaternion.Euler(x,y,z)`: apply z, then x, then y."""
    half = euler_xyz_deg * (math.pi / 360.0)
    x, y, z = half.unbind(-1)
    zeros = torch.zeros_like(x)
    qx = torch.stack((torch.sin(x), zeros, zeros, torch.cos(x)), -1)
    qy = torch.stack((zeros, torch.sin(y), zeros, torch.cos(y)), -1)
    qz = torch.stack((zeros, zeros, torch.sin(z), torch.cos(z)), -1)
    return normalize(multiply(qy, multiply(qx, qz)))
