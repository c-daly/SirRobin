"""Donor-shaped 3 s warmup + 5 s measurement episode."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from sirrobin.physics.pose import resolve_pose
from sirrobin.physics.swim_step import SwimKernel


@dataclass(frozen=True, slots=True)
class EpisodeResult:
    cruise_speed: torch.Tensor
    cost_of_transport: torch.Tensor
    reactive_ratio: torch.Tensor
    mechanical_work: torch.Tensor
    regularization_count: int


def run_episode(kernel: SwimKernel, *, warmup_steps: int = 360, measure_steps: int = 600) -> EpisodeResult:
    body = kernel.body

    def com_now() -> torch.Tensor:
        pose = resolve_pose(body, body.gait_time)
        mass = kernel.static_mass.seg_mass_sim
        body_frame = (pose.pos * mass[..., None]).sum(1) / mass.sum(1, keepdim=True).clamp_min(1e-30)
        return body.x_com + body_frame

    start = com_now()
    for _ in range(warmup_steps):
        kernel.step()
    warm = com_now()
    axis = warm - start
    axis[:, 1] = 0
    magnitude = torch.linalg.vector_norm(axis, dim=-1, keepdim=True)
    axis = torch.where(magnitude > 1e-8, axis / magnitude.clamp_min(1e-30), body.f_hat)
    measure_start = com_now()
    mech = torch.zeros(body.batch_size, dtype=body.v_com.dtype, device=body.v_com.device)
    reactive_impulse = torch.zeros_like(mech)
    drag_impulse = torch.zeros_like(mech)
    regularized = 0
    for _ in range(measure_steps):
        ledger = kernel.step()
        mech += (
            ledger.p_wake_dissipated + ledger.p_fin + ledger.p_drag
        ) * kernel.config.dt
        reactive_impulse += ledger.t_react.abs() * kernel.config.dt
        drag_impulse += (ledger.f_drag * body.f_hat).sum(-1).abs() * kernel.config.dt
        regularized += int(ledger.regularized.sum().item())
    displacement = ((com_now() - measure_start) * axis).sum(-1)
    measure_time = measure_steps * kernel.config.dt
    cruise = displacement / measure_time
    mass_sim = kernel.static_mass.seg_mass_sim.sum(1)
    cot = mech / (mass_sim * displacement.abs()).clamp_min(1e-6)
    ratio = reactive_impulse / (reactive_impulse + drag_impulse).clamp_min(1e-9)
    return EpisodeResult(cruise, cot, ratio, mech, regularized)
