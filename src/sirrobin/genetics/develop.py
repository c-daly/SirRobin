"""Fixed 16-emission batched DFS development into the canonical body contract."""

from __future__ import annotations

import math

import torch

from sirrobin.genetics.genotype import E_MAX, N_MAX, SURFACE, GenotypeBatch
from sirrobin.numerics.ellipsoid_added_mass import added_mass
from sirrobin.numerics.quat import identity, multiply, rotate
from sirrobin.physics.contracts import DevelopedBody

S_SLOT = 17
STACK = 16
MAX_DEPTH = 5


def _flat(value: torch.Tensor, trailing: int) -> torch.Tensor:
    return value.reshape(-1, *value.shape[-trailing:]) if trailing else value.reshape(-1)


def _gather(values: torch.Tensor, index: torch.Tensor) -> torch.Tensor:
    extra = values.ndim - 2
    gather_index = index[:, None, *([None] * extra)].expand(-1, 1, *values.shape[2:])
    return torch.gather(values, 1, gather_index).squeeze(1)


def _gather_order(values: torch.Tensor, order: torch.Tensor) -> torch.Tensor:
    extra = values.ndim - order.ndim
    gather_index = order[(...,) + (None,) * extra].expand(*order.shape, *values.shape[order.ndim:])
    return torch.gather(values, 1, gather_index)


def _merge_stack(candidate: torch.Tensor, remaining: torch.Tensor, order: torch.Tensor) -> torch.Tensor:
    return _gather_order(torch.cat((candidate, remaining), dim=1), order)


def _mirror_quat(q: torch.Tensor, side: torch.Tensor) -> torch.Tensor:
    reflected = torch.stack((-q[..., 0], q[..., 1], -q[..., 2], q[..., 3]), dim=-1)
    return torch.where((side < 0)[..., None], reflected, q)


def develop(genotype: GenotypeBatch) -> DevelopedBody:
    genotype.validate()
    w, n = genotype.alive.shape
    b = w * n
    dtype = genotype.node_log_axes_flu_m.dtype
    device = genotype.alive.device
    alive = genotype.alive.reshape(b)

    node_mask = genotype.node_mask.reshape(b, N_MAX)
    node_expressed = genotype.node_expressed.reshape(b, N_MAX)
    node_type = genotype.node_type.reshape(b, N_MAX)
    node_axes = genotype.node_log_axes_flu_m.reshape(b, N_MAX, 3).exp()
    node_density = genotype.node_density_gene.reshape(b, N_MAX)
    node_intake = genotype.node_intake.reshape(b, N_MAX)
    node_sense = genotype.node_sense.reshape(b, N_MAX)
    node_amp = genotype.node_joint_amp_rad.reshape(b, N_MAX)
    node_hinge = genotype.node_hinge_axis_flu.reshape(b, N_MAX, 3)
    edge_mask = genotype.edge_mask.reshape(b, E_MAX)
    edge_src = genotype.edge_src.reshape(b, E_MAX).to(torch.int64)
    edge_dst = genotype.edge_dst.reshape(b, E_MAX).to(torch.int64)
    edge_attach = genotype.edge_attach_parent_axes.reshape(b, E_MAX, 3)
    edge_rot = genotype.edge_rot_flu.reshape(b, E_MAX, 4)
    edge_scale = genotype.edge_scale.reshape(b, E_MAX)
    edge_mirror = genotype.edge_mirror.reshape(b, E_MAX)
    edge_recursion = genotype.edge_recursion.reshape(b, E_MAX).to(torch.int64)

    shape_s = (b, S_SLOT)
    seg_mask = torch.zeros(shape_s, dtype=torch.bool, device=device)
    parent_out = torch.zeros(shape_s, dtype=torch.int16, device=device)
    depth_out = torch.zeros(shape_s, dtype=torch.int8, device=device)
    local_pos_out = torch.zeros((*shape_s, 3), dtype=dtype, device=device)
    local_rot_out = identity(shape_s, dtype=dtype, device=device)
    axes_out = torch.zeros((*shape_s, 3), dtype=dtype, device=device)
    density_out = torch.zeros(shape_s, dtype=dtype, device=device)
    surface_out = torch.zeros(shape_s, dtype=torch.bool, device=device)
    intake_out = torch.zeros(shape_s, dtype=torch.bool, device=device)
    sense_out = torch.zeros(shape_s, dtype=torch.bool, device=device)
    amp_out = torch.zeros(shape_s, dtype=dtype, device=device)
    hinge_out = torch.zeros((*shape_s, 3), dtype=dtype, device=device)
    phase_out = torch.zeros(shape_s, dtype=dtype, device=device)
    rest_pos_out = torch.zeros((*shape_s, 3), dtype=dtype, device=device)
    rest_rot_out = identity(shape_s, dtype=dtype, device=device)

    stack_valid = torch.zeros((b, STACK), dtype=torch.bool, device=device)
    stack_valid[:, 0] = alive & node_mask[:, 0] & node_expressed[:, 0]
    stack_node = torch.zeros((b, STACK), dtype=torch.int64, device=device)
    stack_parent = torch.zeros((b, STACK), dtype=torch.int16, device=device)
    stack_depth = torch.zeros((b, STACK), dtype=torch.int64, device=device)
    stack_side = torch.ones((b, STACK), dtype=dtype, device=device)
    stack_local_pos = torch.zeros((b, STACK, 3), dtype=dtype, device=device)
    stack_local_rot = identity((b, STACK), dtype=dtype, device=device)
    stack_rest_pos = torch.zeros((b, STACK, 3), dtype=dtype, device=device)
    stack_rest_rot = identity((b, STACK), dtype=dtype, device=device)
    stack_scale = torch.ones((b, STACK), dtype=dtype, device=device)
    stack_incoming = torch.full((b, STACK), -1, dtype=torch.int64, device=device)
    stack_repeat = torch.zeros((b, STACK), dtype=torch.int64, device=device)
    truncated = torch.zeros(b, dtype=torch.int64, device=device)
    edge_slots = torch.arange(E_MAX, device=device)[None, :].expand(b, -1)

    for emission in range(16):
        valid = stack_valid[:, 0]
        node = stack_node[:, 0].clamp(0, N_MAX - 1)
        node_valid = valid & _gather(node_mask[..., None], node).squeeze(-1)
        node_valid &= _gather(node_expressed[..., None], node).squeeze(-1)
        slot = emission + 1
        axes = _gather(node_axes, node) * stack_scale[:, 0, None]
        density = _gather(node_density[..., None], node).squeeze(-1)
        seg_mask[:, slot] = node_valid
        parent_out[:, slot] = torch.where(node_valid, stack_parent[:, 0], 0)
        depth_out[:, slot] = torch.where(node_valid, stack_depth[:, 0], 0).to(torch.int8)
        local_pos_out[:, slot] = torch.where(node_valid[:, None], stack_local_pos[:, 0], 0.0)
        local_rot_out[:, slot] = torch.where(
            node_valid[:, None], stack_local_rot[:, 0], local_rot_out[:, slot]
        )
        axes_out[:, slot] = torch.where(node_valid[:, None], axes, 0.0)
        density_out[:, slot] = torch.where(node_valid, density, 0.0)
        surface_out[:, slot] = node_valid & (_gather(node_type[..., None], node).squeeze(-1) == SURFACE)
        intake_out[:, slot] = node_valid & _gather(node_intake[..., None], node).squeeze(-1)
        sense_out[:, slot] = node_valid & _gather(node_sense[..., None], node).squeeze(-1)
        amp_out[:, slot] = torch.where(node_valid, _gather(node_amp[..., None], node).squeeze(-1), 0.0)
        hinge = _gather(node_hinge, node)
        mirrored_hinge = torch.stack((-hinge[:, 0], hinge[:, 1], -hinge[:, 2]), dim=-1)
        hinge = torch.where((stack_side[:, 0] < 0)[:, None], mirrored_hinge, hinge)
        hinge_out[:, slot] = torch.where(node_valid[:, None], hinge, 0.0)
        phase_out[:, slot] = torch.where(
            node_valid,
            -stack_depth[:, 0].to(dtype) * genotype.swim_wave_rad_per_depth.reshape(b),
            0.0,
        )
        rest_pos_out[:, slot] = torch.where(node_valid[:, None], stack_rest_pos[:, 0], 0.0)
        rest_rot_out[:, slot] = torch.where(
            node_valid[:, None], stack_rest_rot[:, 0], rest_rot_out[:, slot]
        )

        current_depth = stack_depth[:, 0]
        current_side = stack_side[:, 0]
        incoming = stack_incoming[:, 0]
        repeat = stack_repeat[:, 0]
        outgoing = edge_mask & (edge_src == node[:, None]) & node_valid[:, None]
        outgoing &= current_depth[:, None] < MAX_DEPTH
        is_self = edge_src == edge_dst
        same_self = incoming[:, None] == edge_slots
        self_ok = torch.where(same_self, repeat[:, None] > 0, edge_recursion > 1)
        candidate_valid = outgoing & torch.where(is_self, self_ok, torch.ones_like(self_ok))
        next_repeat = torch.where(same_self, repeat[:, None] - 1, edge_recursion - 2).clamp_min(0)
        next_repeat = torch.where(is_self, next_repeat, torch.zeros_like(next_repeat))

        positive_side = current_side[:, None].expand(-1, E_MAX)
        sides = torch.stack((positive_side, -positive_side), -1)
        variant_valid = torch.stack((candidate_valid, candidate_valid & edge_mirror), dim=-1)
        candidate_node = edge_dst[..., None].expand(-1, -1, 2)
        candidate_parent = torch.full_like(candidate_node, slot, dtype=torch.int16)
        candidate_depth = (current_depth[:, None, None] + 1).expand(-1, E_MAX, 2)
        candidate_scale = (stack_scale[:, 0, None] * edge_scale)[..., None].expand(-1, -1, 2)
        candidate_incoming = edge_slots[..., None].expand(-1, -1, 2)
        candidate_repeat = next_repeat[..., None].expand(-1, -1, 2)
        local_pos = edge_attach * axes[:, None, :]
        local_pos = local_pos[..., None, :].expand(-1, -1, 2, -1).clone()
        local_pos[..., 1] *= sides
        local_rot = edge_rot[..., None, :].expand(-1, -1, 2, -1)
        local_rot = _mirror_quat(local_rot, sides)
        parent_rest_pos = stack_rest_pos[:, 0, None, None, :]
        parent_rest_rot = stack_rest_rot[:, 0, None, None, :]
        candidate_rest_pos = parent_rest_pos + rotate(parent_rest_rot, local_pos)
        candidate_rest_rot = multiply(parent_rest_rot.expand_as(local_rot), local_rot)

        def candidates(value: torch.Tensor) -> torch.Tensor:
            return value.reshape(b, E_MAX * 2, *value.shape[3:])

        cand_valid = candidate_valid.new_zeros((b, E_MAX, 2)) | variant_valid
        combined_valid = torch.cat((cand_valid.reshape(b, -1), stack_valid[:, 1:]), dim=1)
        priorities = torch.arange(combined_valid.shape[1], device=device)[None, :].expand(b, -1)
        priorities = torch.where(combined_valid, priorities, priorities + 10_000)
        order = torch.argsort(priorities, dim=1, stable=True)[:, :STACK]
        truncated += (combined_valid.sum(1) - STACK).clamp_min(0)

        stack_valid = _merge_stack(cand_valid.reshape(b, -1), stack_valid[:, 1:], order)
        stack_node = _merge_stack(candidates(candidate_node), stack_node[:, 1:], order)
        stack_parent = _merge_stack(candidates(candidate_parent), stack_parent[:, 1:], order)
        stack_depth = _merge_stack(candidates(candidate_depth), stack_depth[:, 1:], order)
        stack_side = _merge_stack(candidates(sides), stack_side[:, 1:], order)
        stack_local_pos = _merge_stack(candidates(local_pos), stack_local_pos[:, 1:], order)
        stack_local_rot = _merge_stack(candidates(local_rot), stack_local_rot[:, 1:], order)
        stack_rest_pos = _merge_stack(candidates(candidate_rest_pos), stack_rest_pos[:, 1:], order)
        stack_rest_rot = _merge_stack(candidates(candidate_rest_rot), stack_rest_rot[:, 1:], order)
        stack_scale = _merge_stack(candidates(candidate_scale), stack_scale[:, 1:], order)
        stack_incoming = _merge_stack(candidates(candidate_incoming), stack_incoming[:, 1:], order)
        stack_repeat = _merge_stack(candidates(candidate_repeat), stack_repeat[:, 1:], order)

    truncated += stack_valid.sum(1)
    volume = (4.0 / 3.0) * math.pi * axes_out.prod(-1)
    mass_sim = volume * density_out
    drag_area = torch.stack(
        (
            4 * axes_out[..., 1] * axes_out[..., 2],
            4 * axes_out[..., 0] * axes_out[..., 2],
            4 * axes_out[..., 0] * axes_out[..., 1],
        ),
        dim=-1,
    )
    safe_axes = torch.where(seg_mask[..., None], axes_out, torch.ones_like(axes_out))
    madd = added_mass(safe_axes, 1000.0)
    madd = torch.where(seg_mask[..., None], madd, 0.0)
    fin_perp = torch.where(surface_out, madd[..., 1], 0.0)
    aft_local = torch.zeros_like(rest_pos_out)
    aft_local[..., 0] = -axes_out[..., 0]
    endpoints = rest_pos_out + rotate(rest_rot_out, aft_local)
    endpoint_x = torch.where(seg_mask, endpoints[..., 0], torch.full_like(endpoints[..., 0], torch.inf))
    reverse_index = torch.argmin(endpoint_x.flip(1), dim=1)
    tail = torch.where(seg_mask.any(1), S_SLOT - 1 - reverse_index, 0).to(torch.int16)

    out_shape = (w, n, S_SLOT)

    def reshape(value: torch.Tensor, *tail_shape: int) -> torch.Tensor:
        return value.reshape(*out_shape, *tail_shape)
    return DevelopedBody(
        alive=genotype.alive,
        stable_id=genotype.stable_id,
        seg_mask=reshape(seg_mask),
        parent=reshape(parent_out),
        depth=reshape(depth_out),
        local_pos_flu_m=reshape(local_pos_out, 3),
        local_rot_flu=reshape(local_rot_out, 4),
        semi_axes_flu_m=reshape(axes_out, 3),
        density_gene=reshape(density_out),
        mass_sim=reshape(mass_sim),
        volume_m3=reshape(volume),
        drag_area_flu_m2=reshape(drag_area, 3),
        added_mass_flu_kg=reshape(madd, 3),
        fin_perpendicular_kg=reshape(fin_perp),
        is_surface=reshape(surface_out),
        intake=reshape(intake_out),
        sense=reshape(sense_out),
        joint_amp_rad=reshape(amp_out),
        hinge_axis_flu=reshape(hinge_out, 3),
        phase_rad=reshape(phase_out),
        tail_slot=tail.reshape(w, n),
        swim_freq_hz=genotype.swim_freq_hz,
        swim_wave_rad_per_depth=genotype.swim_wave_rad_per_depth,
        truncated_candidate_count=truncated.to(torch.int16).reshape(w, n),
    )
