"""Complete batched frozen-heading S0 step."""

from __future__ import annotations

import math

import torch

from sirrobin.numerics.quat import rotate
from sirrobin.numerics.solve_constrained_xz import solve_constrained_xz
from sirrobin.physics.config import LocomotionConfig
from sirrobin.physics.contracts import BodyBatch, StepLedger
from sirrobin.physics.force_drag import drag_channel
from sirrobin.physics.force_fin import fin_channel
from sirrobin.physics.force_reactive import reactive_channel
from sirrobin.physics.mass_matrix import StaticMassData, mass_properties, prepare_mass_data
from sirrobin.physics.pose import gather_slots, resolve_pose, tail_slots, tail_tip


def _xz_matrix(matrix: torch.Tensor) -> torch.Tensor:
    return torch.stack(
        (
            torch.stack((matrix[:, 0, 0], matrix[:, 0, 2]), dim=-1),
            torch.stack((matrix[:, 2, 0], matrix[:, 2, 2]), dim=-1),
        ),
        dim=-2,
    )


class SwimKernel:
    def __init__(self, body: BodyBatch, config: LocomotionConfig):
        self.body = body
        self.config = config
        self.static_mass: StaticMassData = prepare_mass_data(body, config)
        self.tail = tail_slots(body)
        self.area_z = 4.0 * body.abc[..., 0] * body.abc[..., 1]
        rest_pose = resolve_pose(body, body.gait_time, apply_gait=False)
        mass_sim = self.static_mass.seg_mass_sim
        rest_com = (rest_pose.pos * mass_sim[..., None]).sum(1) / mass_sim.sum(
            1, keepdim=True
        ).clamp_min(1e-30)
        tail_center = gather_slots(rest_pose.pos, self.tail)
        heading = rest_com - tail_center
        heading[:, 1] = 0
        heading_norm = torch.linalg.vector_norm(heading, dim=-1, keepdim=True)
        fallback = torch.tensor(
            [0.0, 0.0, -1.0], dtype=body.abc.dtype, device=body.abc.device
        ).expand_as(heading)
        body.f_hat.copy_(
            torch.where(heading_norm > 1e-8, heading / heading_norm.clamp_min(1e-30), fallback)
        )
        up = torch.tensor([0.0, 1.0, 0.0], dtype=body.abc.dtype, device=body.abc.device).expand_as(
            heading
        )
        body.n_hat.copy_(torch.linalg.cross(up, body.f_hat, dim=-1))
        pose0 = resolve_pose(body, body.gait_time)
        tail_rot = gather_slots(pose0.rot, self.tail)
        local_up = torch.tensor([0.0, 1.0, 0.0], dtype=body.abc.dtype, device=body.abc.device).expand_as(
            body.f_hat
        )
        fin_normal = rotate(tail_rot, local_up)
        self.fin_align = (fin_normal * body.n_hat).sum(dim=-1).abs()
        tail_fin_mass = gather_slots(self.static_mass.fin_perpendicular_mass, self.tail)
        tail_added_x = gather_slots(self.static_mass.added_mass[..., 0], self.tail)
        tail_c = gather_slots(body.abc[..., 2], self.tail).clamp_min(5e-5)
        reactive_mass = torch.where(tail_fin_mass > 0, tail_fin_mass, tail_added_x)
        scale = (1.0 - config.fin_plane_gain) + config.fin_plane_gain * self.fin_align.square()
        reactive_mass = torch.where(tail_fin_mass > 0, reactive_mass * scale, reactive_mass)
        self.mt = reactive_mass / (2.0 * tail_c)
        self.fin_active = gather_slots(body.is_surface, self.tail) & body.alive
        span = gather_slots(body.fin_span, self.tail)
        chord = gather_slots(body.fin_chord, self.tail).clamp_min(0.03)
        self.fin_area = span * chord
        self.fin_ar = span / chord
        self.fin_lift_slope = 2.0 * math.pi * self.fin_ar / (self.fin_ar + 2.0).clamp_min(1e-4)

    def step(self) -> StepLedger:
        body, cfg = self.body, self.config
        t0 = body.gait_time
        t1 = t0 + cfg.dt
        pose0 = resolve_pose(body, t0)
        pose1 = resolve_pose(body, t1)
        mass0 = mass_properties(body, pose0, cfg, self.static_mass)
        mass1 = mass_properties(body, pose1, cfg, self.static_mass)
        prev_tip = tail_tip(body, pose0)
        next_tip = tail_tip(body, pose1)
        u_tail = body.v_com + (next_tip - prev_tip) / cfg.dt
        u = (u_tail * body.f_hat).sum(dim=-1)
        vt = (u_tail * body.n_hat).sum(dim=-1)
        local_forward = torch.zeros_like(body.f_hat)
        local_forward[..., 2] = 1
        tail_heading = rotate(gather_slots(pose1.rot, self.tail), local_forward)
        slope = (tail_heading * body.n_hat).sum(dim=-1)
        t_react, p_reactive_in, p_wake, _ = reactive_channel(self.mt, u, vt, slope)
        p_wake_dissipated = torch.where(u >= 0, p_wake, torch.zeros_like(p_wake))

        vt_fin = vt * ((1.0 - cfg.fin_plane_gain) + cfg.fin_plane_gain * self.fin_align)
        t_fin, p_fin_in, p_fin = fin_channel(
            self.fin_lift_slope,
            self.fin_ar,
            self.fin_area,
            u,
            vt_fin,
            slope,
            self.fin_active,
            rho_water=cfg.rho_water,
            profile_cd=cfg.fin_profile_cd,
            span_eff=cfg.fin_span_eff,
            stall_aoa=cfg.fin_stall_aoa,
        )
        segment_velocity = body.v_com[:, None, :] + (pose1.pos - pose0.pos) / cfg.dt
        f_drag, p_drag = drag_channel(
            segment_velocity,
            pose1.rot,
            self.area_z,
            body.seg_mask & body.alive[:, None],
            rho_water=cfg.rho_water,
            cd=cfg.drag_coeff,
        )
        f_stream = (t_react + t_fin)[:, None] * body.f_hat + f_drag
        valid = body.alive & (mass1.mass_kg > 0)
        solve = solve_constrained_xz(
            mass1.matrix,
            f_stream * cfg.dt,
            valid,
            kappa_max=cfg.kappa_max,
            lam_floor=cfg.lam_floor_kg,
            eps_spd=cfg.eps_spd,
        )
        v0 = body.v_com
        v1 = v0 + solve.dv
        v1 = torch.stack((v1[:, 0], torch.zeros_like(v1[:, 0]), v1[:, 2]), dim=-1)
        v0xz = torch.stack((v0[:, 0], v0[:, 2]), dim=-1)
        v1xz = torch.stack((v1[:, 0], v1[:, 2]), dim=-1)
        m0xz, m1xz = _xz_matrix(mass0.matrix), _xz_matrix(mass1.matrix)
        ke0 = 0.5 * torch.einsum("bi,bij,bj->b", v0xz, m0xz, v0xz)
        ke1 = 0.5 * torch.einsum("bi,bij,bj->b", v1xz, m1xz, v1xz)
        delta_ke = ke1 - ke0
        v_mid = 0.5 * (v0xz + v1xz)
        total_impulse = f_stream * cfg.dt + solve.j_reg
        impulse = torch.stack((total_impulse[:, 0], total_impulse[:, 2]), dim=-1)
        work_impulse = (v_mid * impulse).sum(dim=-1)
        delta_m = m1xz - m0xz
        work_delta_m = 0.5 * torch.einsum("bi,bij,bj->b", v0xz, delta_m, v0xz)
        r_step = delta_ke - work_impulse - work_delta_m

        body.v_com.copy_(torch.where(body.alive[:, None], v1, torch.zeros_like(v1)))
        body.x_com.add_(body.v_com * cfg.dt)
        body.gait_time.copy_(t1)
        return StepLedger(
            u=u,
            vt=vt,
            slope=slope,
            t_react=t_react,
            p_reactive_in=p_reactive_in,
            p_wake=p_wake,
            p_wake_dissipated=p_wake_dissipated,
            t_fin=t_fin,
            p_fin_in=p_fin_in,
            p_fin=p_fin,
            p_drag=p_drag,
            f_drag=f_drag,
            f_stream=f_stream,
            m_before=mass0.matrix,
            m_after=mass1.matrix,
            dv=solve.dv,
            j_reg=solve.j_reg,
            regularized=solve.regularized,
            delta_ke=delta_ke,
            work_impulse=work_impulse,
            work_delta_m=work_delta_m,
            r_step=r_step,
        )
