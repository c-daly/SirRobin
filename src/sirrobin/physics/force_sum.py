"""Typed additive force/torque/power composition."""

from __future__ import annotations

from collections.abc import Iterable

import torch

from sirrobin.physics.contracts import ForceTorquePower


def zero_force(reference: torch.Tensor) -> ForceTorquePower:
    scalar = torch.zeros(reference.shape[:-1], dtype=reference.dtype, device=reference.device)
    return ForceTorquePower(torch.zeros_like(reference), scalar, scalar.clone(), scalar.clone())


def sum_contributions(
    contributions: Iterable[ForceTorquePower], reference: torch.Tensor
) -> ForceTorquePower:
    total = zero_force(reference)
    force = total.force_enu_n
    torque = total.torque_yaw_nm
    input_power = total.input_power_w
    dissipated = total.dissipated_power_w
    for item in contributions:
        force = force + item.force_enu_n
        torque = torque + item.torque_yaw_nm
        input_power = input_power + item.input_power_w
        dissipated = dissipated + item.dissipated_power_w
    return ForceTorquePower(force, torque, input_power, dissipated)
