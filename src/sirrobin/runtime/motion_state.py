"""Lifecycle consequences for fixed-capacity motion state."""

from __future__ import annotations

import math
from dataclasses import dataclass, fields

import torch

from sirrobin.fields.geometry import GridGeometry
from sirrobin.numerics.flux import INT64_SAFE_MAX
from sirrobin.organisms.lifecycle import LifecycleLedger
from sirrobin.organisms.state import PopulationState
from sirrobin.physics.contracts import DevelopedBody, LiveState
from sirrobin.physics.force_hydrodynamic import effective_mass_and_yaw_inertia
from sirrobin.physics.live_config import LiveLocomotionConfig
from sirrobin.physics.pose_live import forward_left, resolve_live_pose
from sirrobin.physics.yaw import wrap_pi


@dataclass(frozen=True, slots=True)
class BirthReleaseProposal:
    """Parent-indexed physical state and cost for a possible birth release."""

    child_position_enu_m: torch.Tensor
    child_velocity_rel_water_enu_m_s: torch.Tensor
    child_yaw_rad: torch.Tensor
    child_impulse_enu_ns: torch.Tensor
    parent_velocity_after_enu_m_s: torch.Tensor
    kinetic_delta_j: torch.Tensor
    release_energy_q: torch.Tensor
    invalid: torch.Tensor


@dataclass(frozen=True, slots=True)
class BirthReleaseLedger:
    """Accepted equal-and-opposite release consequences."""

    parent_impulse_enu_ns: torch.Tensor
    child_impulse_enu_ns: torch.Tensor
    kinetic_delta_j: torch.Tensor


def developed_support_radius_m(body: DevelopedBody) -> torch.Tensor:
    """Conservative rotation-independent radius around the body origin."""

    mask = body.seg_mask & body.alive[..., None]
    link_length = torch.linalg.vector_norm(body.local_pos_flu_m, dim=-1)
    longest_axis = body.semi_axes_flu_m.amax(dim=-1)
    path_bound = torch.where(mask, link_length, 0.0).sum(dim=-1)
    segment_bound = torch.where(mask, longest_axis, 0.0).amax(dim=-1)
    return path_bound + segment_bound


def propose_birth_release(
    parent_body: DevelopedBody,
    child_body: DevelopedBody,
    motion: LiveState,
    parent_effective_mass_kg: torch.Tensor,
    requested_birth: torch.Tensor,
    geometry: GridGeometry,
    live_config: LiveLocomotionConfig,
    *,
    impulse_ns: float,
    clearance_m: float,
    reserve_j_per_q: float,
) -> BirthReleaseProposal:
    """Propose a paid, momentum-conserving release for each requesting parent."""

    active = requested_birth & parent_body.alive & child_body.alive
    parent_forward, _ = forward_left(motion.yaw_rad)
    away = -parent_forward
    child_yaw = wrap_pi(motion.yaw_rad + math.pi)

    zeros = torch.zeros_like(motion.gait_time_s)
    child_pose = resolve_live_pose(
        child_body,
        zeros,
        zeros.to(motion.yaw_rad.dtype),
        effort=zeros.to(motion.yaw_rad.dtype),
    )
    child_effective_mass, _ = effective_mass_and_yaw_inertia(
        child_body,
        child_pose,
        child_yaw,
        live_config,
    )
    eye = torch.eye(
        2,
        dtype=parent_effective_mass_kg.dtype,
        device=parent_effective_mass_kg.device,
    ).expand(*active.shape, 2, 2)
    parent_xy_mass = torch.where(
        active[..., None, None],
        parent_effective_mass_kg[..., :2, :2],
        eye,
    )
    child_xy_mass = torch.where(
        active[..., None, None],
        child_effective_mass[..., :2, :2],
        eye,
    )
    parent_velocity_xy = motion.velocity_rel_water_enu_m_s[..., :2]
    parent_momentum_xy = torch.einsum(
        "...ij,...j->...i", parent_xy_mass, parent_velocity_xy
    )
    child_impulse_xy = away[..., :2] * impulse_ns
    parent_momentum_after_xy = parent_momentum_xy - child_impulse_xy
    parent_velocity_after_xy, parent_info = torch.linalg.solve_ex(
        parent_xy_mass,
        parent_momentum_after_xy[..., None],
        check_errors=False,
    )
    child_velocity_xy, child_info = torch.linalg.solve_ex(
        child_xy_mass,
        child_impulse_xy[..., None],
        check_errors=False,
    )
    parent_velocity_after_xy = parent_velocity_after_xy.squeeze(-1)
    child_velocity_xy = child_velocity_xy.squeeze(-1)

    def horizontal(value: torch.Tensor) -> torch.Tensor:
        return torch.cat((value, torch.zeros_like(value[..., :1])), dim=-1)

    parent_velocity_after = horizontal(parent_velocity_after_xy)
    child_velocity = horizontal(child_velocity_xy)
    child_impulse = horizontal(child_impulse_xy)
    parent_ke_before = 0.5 * (
        parent_momentum_xy.to(torch.float64)
        * parent_velocity_xy.to(torch.float64)
    ).sum(dim=-1)
    parent_ke_after = 0.5 * (
        parent_momentum_after_xy.to(torch.float64)
        * parent_velocity_after_xy.to(torch.float64)
    ).sum(dim=-1)
    child_ke = 0.5 * (
        child_impulse_xy.to(torch.float64)
        * child_velocity_xy.to(torch.float64)
    ).sum(dim=-1)
    kinetic_delta_j = torch.where(
        active,
        parent_ke_after + child_ke - parent_ke_before,
        0.0,
    )
    chemical_q_float = torch.ceil(kinetic_delta_j.clamp_min(0.0) / reserve_j_per_q)
    energy_domain_invalid = (
        ~torch.isfinite(chemical_q_float)
        | (chemical_q_float >= INT64_SAFE_MAX)
    )
    release_energy_q = torch.where(
        active & ~energy_domain_invalid,
        chemical_q_float.clamp(0.0, INT64_SAFE_MAX - 1).to(torch.int64),
        0,
    )

    separation_m = (
        developed_support_radius_m(parent_body)
        + developed_support_radius_m(child_body)
        + clearance_m
    )
    # The authoritative motion state is commonly float32.  Forming a child
    # position several metres from a parent whose world coordinate may be tens
    # of metres can round the requested separation inward by a few micrometres.
    # Add a bound for the finite-precision coordinate operations so the stored
    # state still provides at least the declared geometric clearance.  This is
    # numerical representation margin, not biological release distance.
    coordinate_roundoff_m = (
        8.0
        * torch.finfo(motion.position_enu_m.dtype).eps
        * max(geometry.lx_m, geometry.ly_m)
    )
    stored_separation_m = separation_m + coordinate_roundoff_m
    raw_child_position = (
        motion.position_enu_m + away * stored_separation_m[..., None]
    )
    child_position = torch.stack(
        (
            torch.remainder(raw_child_position[..., 0], geometry.lx_m),
            torch.remainder(raw_child_position[..., 1], geometry.ly_m),
            raw_child_position[..., 2],
        ),
        dim=-1,
    )
    finite = (
        torch.isfinite(parent_velocity_after).all(dim=-1)
        & torch.isfinite(child_velocity).all(dim=-1)
        & torch.isfinite(child_position).all(dim=-1)
        & torch.isfinite(child_yaw)
        & torch.isfinite(kinetic_delta_j)
        & torch.isfinite(separation_m)
    )
    fits_periodic_domain = separation_m < 0.5 * min(geometry.lx_m, geometry.ly_m)
    invalid = active & (
        ~finite
        | ~fits_periodic_domain
        | energy_domain_invalid
        | (parent_info != 0)
        | (child_info != 0)
    )
    return BirthReleaseProposal(
        child_position,
        child_velocity,
        child_yaw,
        child_impulse,
        parent_velocity_after,
        kinetic_delta_j,
        release_energy_q,
        invalid,
    )


def _gather_parent(value: torch.Tensor, parent_slot: torch.Tensor) -> torch.Tensor:
    tail = value.shape[2:]
    index = parent_slot[(...,) + (None,) * len(tail)].expand(
        *parent_slot.shape, *tail
    )
    return torch.gather(value, 1, index)


def settle_motion_lifecycle(
    motion: LiveState,
    population: PopulationState,
    lifecycle: LifecycleLedger,
    release: BirthReleaseProposal,
) -> tuple[LiveState, BirthReleaseLedger]:
    """Clear dead slots and commit accepted independent newborn releases."""

    alive = population.alive
    born = lifecycle.born
    parent_slot = lifecycle.parent_slot_for_child.clamp_min(0)
    zeroed: dict[str, torch.Tensor] = {}
    for field in fields(motion):
        value = getattr(motion, field.name)
        mask = alive[(...,) + (None,) * (value.ndim - alive.ndim)]
        zeroed[field.name] = torch.where(mask, value, torch.zeros_like(value))

    born_vector = born[..., None]
    accepted_parent = lifecycle.accepted_parent
    parent_velocity = torch.where(
        accepted_parent[..., None],
        release.parent_velocity_after_enu_m_s,
        zeroed["velocity_rel_water_enu_m_s"],
    )
    child_position = _gather_parent(release.child_position_enu_m, parent_slot)
    child_velocity = _gather_parent(
        release.child_velocity_rel_water_enu_m_s,
        parent_slot,
    )
    child_yaw = _gather_parent(release.child_yaw_rad, parent_slot)
    child_impulse = _gather_parent(release.child_impulse_enu_ns, parent_slot)
    desired = torch.zeros_like(motion.desired_heading_enu)
    next_motion = LiveState(
        position_enu_m=torch.where(
            born_vector, child_position, zeroed["position_enu_m"]
        ),
        velocity_rel_water_enu_m_s=torch.where(
            born_vector,
            child_velocity,
            parent_velocity,
        ),
        yaw_rad=torch.where(born, child_yaw, zeroed["yaw_rad"]),
        yaw_momentum_kg_m2_s=torch.where(
            born,
            torch.zeros_like(motion.yaw_momentum_kg_m2_s),
            zeroed["yaw_momentum_kg_m2_s"],
        ),
        gait_time_s=torch.where(
            born, torch.zeros_like(motion.gait_time_s), zeroed["gait_time_s"]
        ),
        desired_heading_enu=torch.where(
            born_vector, desired, zeroed["desired_heading_enu"]
        ),
        turn_bias_rad_per_depth=torch.where(
            born,
            torch.zeros_like(motion.turn_bias_rad_per_depth),
            zeroed["turn_bias_rad_per_depth"],
        ),
        heading_initialized=torch.where(
            born,
            torch.zeros_like(motion.heading_initialized),
            zeroed["heading_initialized"],
        ),
    )
    ledger = BirthReleaseLedger(
        parent_impulse_enu_ns=torch.where(
            accepted_parent[..., None],
            -release.child_impulse_enu_ns,
            0.0,
        ),
        child_impulse_enu_ns=torch.where(born_vector, child_impulse, 0.0),
        kinetic_delta_j=torch.where(
            accepted_parent,
            release.kinetic_delta_j,
            0.0,
        ),
    )
    return next_motion, ledger
