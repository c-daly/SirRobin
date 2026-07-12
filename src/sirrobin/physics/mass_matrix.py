"""Structural plus rotated Lamb added-mass effective inertia."""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch

from sirrobin.numerics.quat import rotate
from sirrobin.physics.config import LocomotionConfig
from sirrobin.physics.contracts import BodyBatch, MassProperties, Pose
from sirrobin.physics.lamb import added_mass, donor_added_mass


@dataclass(slots=True)
class StaticMassData:
    seg_mass_sim: torch.Tensor
    added_mass: torch.Tensor
    fin_perpendicular_mass: torch.Tensor


def prepare_mass_data(body: BodyBatch, config: LocomotionConfig) -> StaticMassData:
    mask = body.seg_mask & body.alive[:, None]
    abc_safe = torch.where(mask[..., None], body.abc, torch.ones_like(body.abc))
    box_mass = (8.0 * body.abc.prod(dim=-1) * body.density_gene).clamp_min(0.1)
    mass_scale = 1.0 + config.ellipsoid_mass_gain * (math.pi / 6.0 - 1.0)
    seg_mass_sim = torch.where(mask, box_mass * mass_scale, torch.zeros_like(box_mass))
    mass_rule = donor_added_mass if config.ellipsoid_mass_gain == 0.0 else added_mass
    madd = mass_rule(abc_safe, config.rho_water).to(body.abc.dtype)
    madd = torch.where(mask[..., None], madd, torch.zeros_like(madd))

    # At gain1 a Surface is a plate whose broadside added mass lies on local Y.
    surface = mask & body.is_surface & (body.fin_span > 0)
    true_abc = torch.stack(
        (body.abc[..., 0], (body.fin_span * 0.5).clamp_min(1e-4), body.abc[..., 2]), dim=-1
    )
    true_abc = torch.where(surface[..., None], true_abc, torch.ones_like(true_abc))
    m_perp = mass_rule(true_abc, config.rho_water)[..., 0].to(body.abc.dtype)
    moved = madd.clone()
    natural_x, natural_y = madd[..., 0], madd[..., 1]
    gain = config.fin_plane_gain
    moved[..., 0] = torch.where(surface, m_perp + gain * (natural_x - m_perp), moved[..., 0])
    moved[..., 1] = torch.where(surface, natural_y + gain * (m_perp - natural_y), moved[..., 1])
    madd = torch.where(mask[..., None], moved, torch.zeros_like(moved))

    return StaticMassData(
        seg_mass_sim=seg_mass_sim, added_mass=madd, fin_perpendicular_mass=torch.where(surface, m_perp, 0.0)
    )


def mass_properties(
    body: BodyBatch, pose: Pose, config: LocomotionConfig, static: StaticMassData | None = None
) -> MassProperties:
    static = prepare_mass_data(body, config) if static is None else static
    mass_sim = static.seg_mass_sim.sum(dim=1)
    mass_kg = mass_sim * config.kg_per_sim_mass
    basis = torch.eye(3, dtype=pose.pos.dtype, device=pose.pos.device)
    columns = torch.stack([rotate(pose.rot, basis[i].expand_as(pose.pos)) for i in range(3)], dim=-1)
    matrix = torch.einsum("bsik,bsk,bsjk->bij", columns, static.added_mass, columns)
    matrix = matrix + torch.diag_embed(mass_kg[:, None].expand(-1, 3))
    torch._assert_async(torch.isfinite(matrix).all(), "effective mass matrix is non-finite")
    return MassProperties(mass_sim=mass_sim, mass_kg=mass_kg, added_mass=static.added_mass, matrix=matrix)


def fin_perpendicular_mass(body: BodyBatch, config: LocomotionConfig) -> torch.Tensor:
    return prepare_mass_data(body, config).fin_perpendicular_mass
