"""Persistent canonical live-locomotion step with linear and angular ledgers."""

from __future__ import annotations

import torch

from sirrobin.numerics.solve_constrained_xy import solve_constrained_xy
from sirrobin.physics.contracts import DevelopedBody, FluidSample, LiveState, LiveStepLedger
from sirrobin.physics.force_hydrodynamic import hydrodynamic_contribution
from sirrobin.physics.force_sum import sum_contributions
from sirrobin.physics.live_config import LiveLocomotionConfig
from sirrobin.physics.pose_live import resolve_live_pose
from sirrobin.physics.yaw import advance_yaw


def _flat(value: torch.Tensor, trailing: int) -> torch.Tensor:
    return value.reshape(-1, *value.shape[-trailing:]) if trailing else value.reshape(-1)


def _xy(matrix: torch.Tensor) -> torch.Tensor:
    return matrix[:, :2, :2]


def step_live(
    body: DevelopedBody,
    state: LiveState,
    fluid: FluidSample,
    config: LiveLocomotionConfig,
) -> LiveStepLedger:
    config.validate()
    alive = _flat(body.alive, 0)
    t0 = _flat(state.gait_time_s, 0)
    t1 = t0 + config.dt
    turn = _flat(state.turn_bias_rad_per_depth, 0)
    pose0 = resolve_live_pose(body, t0, turn)
    pose1 = resolve_live_pose(body, t1, turn)
    hydro = hydrodynamic_contribution(
        body,
        state,
        pose0,
        pose1,
        _flat(fluid.density_kg_m3, 0),
        config,
    )
    velocity0 = _flat(state.velocity_rel_water_enu_m_s, 1)
    total = sum_contributions((hydro.contribution,), velocity0)
    m0 = hydro.diagnostics.effective_mass_before_kg
    m1 = hydro.diagnostics.effective_mass_after_kg
    valid = alive & torch.isfinite(m1).all((-2, -1)) & (torch.diagonal(m1, dim1=-2, dim2=-1).sum(-1) > 0)
    solve = solve_constrained_xy(
        m1,
        total.force_enu_n * config.dt,
        valid,
        kappa_max=config.kappa_max,
        lam_floor=config.lam_floor_kg,
        eps_spd=config.eps_spd,
    )
    velocity1 = velocity0 + solve.dv
    velocity1 = torch.stack((velocity1[:, 0], velocity1[:, 1], torch.zeros_like(velocity1[:, 0])), -1)
    velocity1 = torch.where(alive[:, None], velocity1, 0.0)
    v0xy, v1xy = velocity0[:, :2], velocity1[:, :2]
    ke0 = 0.5 * torch.einsum("bi,bij,bj->b", v0xy, _xy(m0), v0xy)
    ke1 = 0.5 * torch.einsum("bi,bij,bj->b", v1xy, _xy(m1), v1xy)
    delta_ke = ke1 - ke0
    impulse_xy = (total.force_enu_n * config.dt + solve.j_reg)[:, :2]
    work_impulse = (0.5 * (v0xy + v1xy) * impulse_xy).sum(-1)
    work_delta_m = 0.5 * torch.einsum("bi,bij,bj->b", v0xy, _xy(m1 - m0), v0xy)
    residual_linear = delta_ke - work_impulse - work_delta_m

    yaw = advance_yaw(
        _flat(state.yaw_rad, 0),
        _flat(state.yaw_momentum_kg_m2_s, 0),
        hydro.diagnostics.yaw_inertia_before_kg_m2,
        hydro.diagnostics.yaw_inertia_after_kg_m2,
        total.torque_yaw_nm,
        config.dt,
        alive,
        inertia_floor=config.inertia_floor_kg_m2,
        emergency_omega=config.emergency_omega_rad_s,
    )
    finite = (
        torch.isfinite(velocity1).all(-1)
        & torch.isfinite(yaw.yaw)
        & torch.isfinite(yaw.momentum)
        & torch.isfinite(total.force_enu_n).all(-1)
        & torch.isfinite(total.torque_yaw_nm)
    )
    nonfinite = alive & ~finite
    torch._assert_async(~nonfinite.any(), "live locomotion produced nonfinite state")
    torch._assert_async(
        ~yaw.backstop_hit.any(), "live locomotion exceeded the emergency yaw threshold"
    )

    world_shape = body.alive.shape
    state.velocity_rel_water_enu_m_s.copy_(velocity1.reshape(*world_shape, 3))
    state.yaw_momentum_kg_m2_s.copy_(yaw.momentum.reshape(world_shape))
    state.yaw_rad.copy_(yaw.yaw.reshape(world_shape))
    state.gait_time_s.copy_(t1.reshape(world_shape))
    return LiveStepLedger(
        hydro.diagnostics,
        total,
        m0,
        m1,
        solve.dv,
        solve.j_reg,
        solve.regularized,
        yaw.floor_hit,
        yaw.backstop_hit,
        nonfinite,
        delta_ke,
        work_impulse,
        work_delta_m,
        residual_linear,
        yaw.delta_ke_j,
        yaw.work_impulse_j,
        yaw.work_delta_inertia_j,
        yaw.residual_j,
    )
