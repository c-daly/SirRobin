"""Mutable reference-world adapter for the stateless heading controller."""

from __future__ import annotations

import torch

from sirrobin.physics import controller as _controller
from sirrobin.physics.contracts import DevelopedBody, LiveState
from sirrobin.physics.live_config import LiveLocomotionConfig

turn_authority = _controller.turn_authority


def update_heading_controller(
    body: DevelopedBody,
    state: LiveState,
    requested_heading_enu: torch.Tensor,
    config: LiveLocomotionConfig,
) -> None:
    """Mutable adapter retained for the reference world."""

    updated = _controller.heading_controller_state(
        body,
        state,
        requested_heading_enu,
        config,
    )
    state.desired_heading_enu.copy_(updated.desired_heading_enu)
    state.turn_bias_rad_per_depth.copy_(updated.turn_bias_rad_per_depth)
    state.heading_initialized.copy_(updated.heading_initialized)
