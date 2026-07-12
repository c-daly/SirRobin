"""Lighthill trailing-edge reactive channel."""

import torch


def reactive_channel(
    mt: torch.Tensor, u: torch.Tensor, vt: torch.Tensor, slope: torch.Tensor
) -> tuple[torch.Tensor, ...]:
    wt = vt + u * slope
    thrust = 0.5 * mt * (vt.square() - u.square() * slope.square())
    p_wake = 0.5 * mt * u * wt.square()
    p_input = mt * u * vt * wt
    return thrust, p_input, p_wake, wt
