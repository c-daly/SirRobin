"""Fixed-slot S0 body and step contracts."""

from __future__ import annotations

from dataclasses import dataclass, fields
from typing import Any

import torch

from sirrobin.numerics.quat import euler_unity_deg, identity
from sirrobin.physics.config import LocomotionConfig


@dataclass(slots=True)
class BodyBatch:
    alive: torch.Tensor
    stable_id: torch.Tensor
    seg_mask: torch.Tensor
    local_pos: torch.Tensor
    local_rot: torch.Tensor
    abc: torch.Tensor
    density_gene: torch.Tensor
    amp_deg: torch.Tensor
    phase_rad: torch.Tensor
    is_surface: torch.Tensor
    is_tail: torch.Tensor
    parent: torch.Tensor
    depth: torch.Tensor
    fin_span: torch.Tensor
    fin_chord: torch.Tensor
    swim_freq: torch.Tensor
    swim_wave: torch.Tensor
    f_hat: torch.Tensor
    n_hat: torch.Tensor
    x_com: torch.Tensor
    v_com: torch.Tensor
    gait_time: torch.Tensor

    @property
    def batch_size(self) -> int:
        return int(self.alive.numel())

    @classmethod
    def from_rows(
        cls,
        rows: list[dict[str, Any]],
        config: LocomotionConfig,
        *,
        dtype: torch.dtype = torch.float32,
        device: torch.device | str = "cpu",
    ) -> BodyBatch:
        b = len(rows)
        s = config.s_slot

        target_device = torch.device(device)
        storage_device = torch.device("cpu") if target_device.type == "cuda" else target_device

        def zeros(*tail: int, dtype_: torch.dtype = dtype) -> torch.Tensor:
            return torch.zeros((b, s, *tail), dtype=dtype_, device=storage_device)

        seg_mask = zeros(dtype_=torch.bool)
        local_pos = zeros(3)
        local_rot = identity((b, s), dtype=dtype, device=storage_device)
        abc = zeros(3)
        density = zeros()
        amp = zeros()
        phase = zeros()
        is_surface = zeros(dtype_=torch.bool)
        is_tail = zeros(dtype_=torch.bool)
        parent = zeros(dtype_=torch.int64)
        depth = torch.full((b, s), -1, dtype=torch.int64, device=device)
        fin_span = zeros()
        fin_chord = zeros()
        for bi, row in enumerate(rows):
            for seg in row["segments"]:
                k = int(seg["slot"])
                seg_mask[bi, k] = True
                local_pos[bi, k] = torch.tensor(seg["local_pos_m"], dtype=dtype)
                local_rot[bi, k] = euler_unity_deg(
                    torch.tensor(seg["local_euler_deg_xyz"], dtype=dtype)
                )
                abc[bi, k] = torch.tensor(seg["abc_m"], dtype=dtype)
                density[bi, k] = float(seg["density_gene_sim_mass_m3"])
                amp[bi, k] = float(seg["amp_deg"])
                phase[bi, k] = float(seg["phase_rad"])
                is_surface[bi, k] = bool(seg["is_surface"])
                is_tail[bi, k] = bool(seg["is_tail"])
                parent[bi, k] = int(seg["parent"])
                depth[bi, k] = int(seg["depth"])
                fin_span[bi, k] = float(seg["fin_span_m"])
                fin_chord[bi, k] = float(seg["fin_chord_m"])
        result = cls(
            alive=torch.ones(b, dtype=torch.bool, device=storage_device),
            stable_id=torch.arange(1, b + 1, dtype=torch.int64, device=storage_device),
            seg_mask=seg_mask,
            local_pos=local_pos,
            local_rot=local_rot,
            abc=abc,
            density_gene=density,
            amp_deg=amp,
            phase_rad=phase,
            is_surface=is_surface,
            is_tail=is_tail,
            parent=parent,
            depth=depth,
            fin_span=fin_span,
            fin_chord=fin_chord,
            swim_freq=torch.tensor([row["swim_freq_hz"] for row in rows], dtype=dtype, device=storage_device),
            swim_wave=torch.tensor(
                [row["swim_wave_rad_per_depth"] for row in rows], dtype=dtype, device=storage_device
            ),
            f_hat=torch.tensor([0.0, 0.0, -1.0], dtype=dtype, device=storage_device).expand(b, 3).clone(),
            n_hat=torch.tensor([-1.0, 0.0, 0.0], dtype=dtype, device=storage_device).expand(b, 3).clone(),
            x_com=torch.zeros((b, 3), dtype=dtype, device=storage_device),
            v_com=torch.zeros((b, 3), dtype=dtype, device=storage_device),
            gait_time=torch.zeros(b, dtype=torch.float64, device=storage_device),
        )
        if target_device != storage_device:
            for field in fields(result):
                value = getattr(result, field.name)
                if isinstance(value, torch.Tensor):
                    setattr(result, field.name, value.to(target_device))
        return result


@dataclass(slots=True)
class Pose:
    pos: torch.Tensor
    rot: torch.Tensor


@dataclass(slots=True)
class MassProperties:
    mass_sim: torch.Tensor
    mass_kg: torch.Tensor
    added_mass: torch.Tensor
    matrix: torch.Tensor


@dataclass(slots=True)
class StepLedger:
    u: torch.Tensor
    vt: torch.Tensor
    slope: torch.Tensor
    t_react: torch.Tensor
    p_reactive_in: torch.Tensor
    # Signed wake-energy flux used by the reactive-channel identity.
    p_wake: torch.Tensor
    # Nonnegative wake power dissipated into the fluid, matching donor work accounting.
    p_wake_dissipated: torch.Tensor
    t_fin: torch.Tensor
    p_fin_in: torch.Tensor
    p_fin: torch.Tensor
    p_drag: torch.Tensor
    f_drag: torch.Tensor
    f_stream: torch.Tensor
    m_before: torch.Tensor
    m_after: torch.Tensor
    dv: torch.Tensor
    j_reg: torch.Tensor
    regularized: torch.Tensor
    delta_ke: torch.Tensor
    work_impulse: torch.Tensor
    work_delta_m: torch.Tensor
    r_step: torch.Tensor
