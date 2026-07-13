"""Stateless physical morphology queries; no cached capability scores."""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch

from sirrobin.physics.contracts import DevelopedBody
from sirrobin.physics.live_config import LiveLocomotionConfig


@dataclass(frozen=True, slots=True)
class MorphologyReadout:
    segment_count: torch.Tensor
    structural_mass_sim: torch.Tensor
    structural_mass_kg: torch.Tensor
    displaced_volume_m3: torch.Tensor
    projected_area_flu_m2: torch.Tensor
    intake_area_m2: torch.Tensor


def query_morphology(
    body: DevelopedBody, config: LiveLocomotionConfig
) -> MorphologyReadout:
    mask = body.seg_mask & body.alive[..., None]
    mass_sim = torch.where(mask, body.mass_sim, 0.0).sum(-1)
    volume = torch.where(mask, body.volume_m3, 0.0).sum(-1)
    projected = torch.where(mask[..., None], body.drag_area_flu_m2, 0.0).sum(-2)
    axes = body.semi_axes_flu_m
    intake_face = math.pi * axes[..., 1] * axes[..., 2]
    intake_area = torch.where(mask & body.intake, intake_face, 0.0).sum(-1)
    return MorphologyReadout(
        mask.sum(-1),
        mass_sim,
        mass_sim * config.kg_per_sim_mass,
        volume,
        projected,
        intake_area,
    )
