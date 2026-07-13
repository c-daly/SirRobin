"""Canonical ENU/FLU live hydrodynamic force contributor."""

from __future__ import annotations

import math

import torch

from sirrobin.numerics.quat import conjugate, multiply, rotate
from sirrobin.physics.contracts import (
    DevelopedBody,
    ForceTorquePower,
    HydrodynamicDiagnostics,
    HydrodynamicResult,
    LiveState,
)
from sirrobin.physics.force_fin import fin_channel
from sirrobin.physics.force_reactive import reactive_channel
from sirrobin.physics.live_config import LiveLocomotionConfig
from sirrobin.physics.pose_live import LivePose, forward_left, gather_slots, yaw_quaternion
from sirrobin.physics.yaw import yaw_drag_coefficient, yaw_inertia


def _flat(value: torch.Tensor, trailing: int) -> torch.Tensor:
    return value.reshape(-1, *value.shape[-trailing:]) if trailing else value.reshape(-1)


def _world_geometry(
    body: DevelopedBody, pose: LivePose, yaw_rad: torch.Tensor, config: LiveLocomotionConfig
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    mask = _flat(body.seg_mask, 1)
    seg_mass_sim = _flat(body.mass_sim, 1)
    total_mass_sim = seg_mass_sim.sum(-1)
    com_flu = (pose.pos_flu_m * seg_mass_sim[..., None]).sum(1) / total_mass_sim[:, None].clamp_min(1e-30)
    rel_flu = pose.pos_flu_m - com_flu[:, None, :]
    yaw_q = yaw_quaternion(yaw_rad)[:, None, :]
    rel_enu = rotate(yaw_q, rel_flu)
    rotation_enu = multiply(yaw_q.expand_as(pose.rot_flu), pose.rot_flu)
    mass_kg = total_mass_sim * config.kg_per_sim_mass
    basis = torch.eye(3, dtype=pose.pos_flu_m.dtype, device=pose.pos_flu_m.device)
    columns = torch.stack(
        [rotate(rotation_enu, basis[i].expand_as(pose.pos_flu_m)) for i in range(3)], dim=-1
    )
    added = _flat(body.added_mass_flu_kg, 2)
    matrix = torch.einsum("bsik,bsk,bsjk->bij", columns, added, columns)
    matrix = matrix + torch.diag_embed(mass_kg[:, None].expand(-1, 3))
    seg_mass_kg = seg_mass_sim * config.kg_per_sim_mass
    inertia = yaw_inertia(rel_enu, rotation_enu, seg_mass_kg, added, mask)
    broadside = _flat(body.drag_area_flu_m2, 2)[..., 1]
    return rel_flu, rel_enu, rotation_enu, matrix, inertia, broadside


def hydrodynamic_contribution(
    body: DevelopedBody,
    state: LiveState,
    pose0: LivePose,
    pose1: LivePose,
    density_kg_m3: torch.Tensor,
    config: LiveLocomotionConfig,
) -> HydrodynamicResult:
    mask = _flat(body.seg_mask, 1) & _flat(body.alive, 0)[:, None]
    yaw = _flat(state.yaw_rad, 0)
    velocity = _flat(state.velocity_rel_water_enu_m_s, 1)
    momentum = _flat(state.yaw_momentum_kg_m2_s, 0)
    density = _flat(density_kg_m3, 0).to(velocity.dtype)
    rel0_flu, _, _, matrix0, inertia0, _ = _world_geometry(body, pose0, yaw, config)
    rel1_flu, rel1, rot1, matrix1, inertia1, broadside = _world_geometry(body, pose1, yaw, config)
    forward, left = forward_left(yaw)
    omega = momentum / inertia1.clamp_min(config.inertia_floor_kg_m2)
    rigid = torch.stack(
        (-omega[:, None] * rel1[..., 1], omega[:, None] * rel1[..., 0], torch.zeros_like(rel1[..., 0])),
        dim=-1,
    )
    gait_velocity_flu = (rel1_flu - rel0_flu) / config.dt
    gait_velocity = rotate(yaw_quaternion(yaw)[:, None, :], gait_velocity_flu)
    segment_velocity = velocity[:, None, :] + rigid + gait_velocity

    tail = _flat(body.tail_slot, 0).to(torch.int64)
    axes = _flat(body.semi_axes_flu_m, 2)
    aft0 = torch.zeros_like(pose0.pos_flu_m)
    aft1 = torch.zeros_like(pose1.pos_flu_m)
    aft0[..., 0] = -axes[..., 0]
    aft1[..., 0] = -axes[..., 0]
    tail_tip0_flu = pose0.pos_flu_m + rotate(pose0.rot_flu, aft0)
    tail_tip1_flu = pose1.pos_flu_m + rotate(pose1.rot_flu, aft1)
    tail_gait_flu = (gather_slots(tail_tip1_flu, tail) - gather_slots(tail_tip0_flu, tail)) / config.dt
    tail_gait = rotate(yaw_quaternion(yaw), tail_gait_flu)
    tail_velocity = velocity + tail_gait
    u = (tail_velocity * forward).sum(-1)
    vt = (tail_velocity * left).sum(-1)
    local_aft = torch.zeros_like(forward)
    local_aft[..., 0] = -1.0
    tail_tangent = rotate(gather_slots(rot1, tail), local_aft)
    slope = (tail_tangent * left).sum(-1)
    tail_added_y = gather_slots(_flat(body.added_mass_flu_kg, 2)[..., 1], tail)
    tail_fin = gather_slots(_flat(body.fin_perpendicular_kg, 1), tail)
    tail_axis_x = gather_slots(axes[..., 0], tail).clamp_min(5e-5)
    reactive_mass = torch.where(tail_fin > 0, tail_fin, tail_added_y)
    mt = reactive_mass / (2.0 * tail_axis_x)
    t_react, p_reactive, p_wake, _ = reactive_channel(mt, u, vt, slope)
    p_wake_diss = torch.where(u >= 0, p_wake, torch.zeros_like(p_wake))

    surface = gather_slots(_flat(body.is_surface, 1), tail) & _flat(body.alive, 0)
    tail_axes = gather_slots(axes, tail)
    chord = (2.0 * tail_axes[..., 0]).clamp_min(1e-4)
    span = 2.0 * tail_axes[..., 2]
    area = chord * span
    aspect = span / chord
    lift_slope = 2.0 * math.pi * aspect / (aspect + 2.0).clamp_min(1e-4)
    t_fin, p_fin_in, p_fin = fin_channel(
        lift_slope,
        aspect,
        area,
        u,
        vt,
        slope,
        surface,
        rho_water=config.rho_water,
        profile_cd=config.fin_profile_cd,
        span_eff=config.fin_span_eff,
        stall_aoa=config.fin_stall_aoa,
    )
    thrust_force = (t_react + t_fin)[:, None] * forward

    local_velocity = rotate(conjugate(rot1), segment_velocity)
    axial = local_velocity[..., 0]
    local_drag = torch.zeros_like(local_velocity)
    area_x = _flat(body.drag_area_flu_m2, 2)[..., 0]
    local_drag[..., 0] = -0.5 * density[:, None] * config.drag_coeff * area_x * axial.abs() * axial
    drag_each = torch.where(mask[..., None], rotate(rot1, local_drag), 0.0)
    drag_force = drag_each.sum(1)
    p_drag = torch.clamp_min(-(drag_each * segment_velocity).sum(-1), 0.0).sum(-1)
    total_force = thrust_force + drag_force
    tail_center = gather_slots(rel1, tail)
    tail_torque = torch.linalg.cross(tail_center, thrust_force, dim=-1)[..., 2]
    drag_torque = torch.linalg.cross(rel1, drag_each, dim=-1)[..., 2].sum(-1)
    yaw_coeff = yaw_drag_coefficient(rel1, broadside, mask, density, config.yaw_drag_coeff)
    yaw_drag = -yaw_coeff * omega * omega.abs()
    torque = tail_torque + drag_torque + yaw_drag
    contribution = ForceTorquePower(
        total_force,
        torque,
        p_reactive + p_fin_in,
        p_wake_diss + p_fin + p_drag,
    )
    diagnostics = HydrodynamicDiagnostics(
        u,
        vt,
        slope,
        t_react,
        p_reactive,
        p_wake,
        p_wake_diss,
        t_fin,
        p_fin_in,
        p_fin,
        drag_force,
        p_drag,
        matrix0,
        matrix1,
        inertia0,
        inertia1,
        yaw_coeff,
    )
    return HydrodynamicResult(contribution, diagnostics)
