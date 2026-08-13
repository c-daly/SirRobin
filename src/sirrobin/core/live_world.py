"""Core composition of controller, live physics, fluid transport, and wrapping."""

from __future__ import annotations

import torch

from sirrobin.core.controller import update_heading_controller
from sirrobin.fields.geometry import GridGeometry
from sirrobin.physics.contracts import DevelopedBody, FluidSample, LiveState, LiveStepLedger
from sirrobin.physics.live_config import LiveLocomotionConfig
from sirrobin.physics.live_step import step_live


def initialize_live_state(body: DevelopedBody) -> LiveState:
    lead = body.alive.shape
    dtype = body.mass_sim.dtype
    device = body.alive.device
    position = torch.zeros((*lead, 3), dtype=dtype, device=device)
    velocity = torch.zeros_like(position)
    yaw = torch.zeros(lead, dtype=dtype, device=device)
    desired = torch.zeros((*lead, 2), dtype=dtype, device=device)
    desired[..., 0] = 1.0
    return LiveState(
        position,
        velocity,
        yaw,
        torch.zeros_like(yaw),
        torch.zeros(lead, dtype=torch.float64, device=device),
        desired,
        torch.zeros_like(yaw),
        torch.zeros(lead, dtype=torch.bool, device=device),
    )


def advance_live_world(
    body: DevelopedBody,
    state: LiveState,
    fluid: FluidSample,
    config: LiveLocomotionConfig,
    geometry: GridGeometry,
    *,
    requested_heading_enu: torch.Tensor | None = None,
    effort_fraction: torch.Tensor | None = None,
    _effort_fraction_validated: bool = False,
) -> LiveStepLedger:
    if requested_heading_enu is not None:
        update_heading_controller(body, state, requested_heading_enu, config)
    ledger = step_live(
        body,
        state,
        fluid,
        config,
        effort_fraction=effort_fraction,
        _effort_fraction_validated=_effort_fraction_validated,
    )
    transport = state.velocity_rel_water_enu_m_s + fluid.velocity_enu_m_s
    next_xy = state.position_enu_m[..., :2] + transport[..., :2] * config.dt
    wrapped_x = torch.remainder(next_xy[..., 0], geometry.lx_m)
    wrapped_y = torch.remainder(next_xy[..., 1], geometry.ly_m)
    state.position_enu_m[..., 0].copy_(torch.where(body.alive, wrapped_x, state.position_enu_m[..., 0]))
    state.position_enu_m[..., 1].copy_(torch.where(body.alive, wrapped_y, state.position_enu_m[..., 1]))
    return ledger
