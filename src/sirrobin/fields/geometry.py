"""Rectilinear periodic-horizontal, closed-vertical grid geometry."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol


class GeometryConfig(Protocol):
    gx: int
    gy: int
    gz: int
    lx_m: float
    ly_m: float
    lz_m: float


@dataclass(frozen=True, slots=True)
class GridGeometry:
    gx: int
    gy: int
    gz: int
    lx_m: float
    ly_m: float
    lz_m: float

    @classmethod
    def from_config(cls, config: GeometryConfig) -> GridGeometry:
        return cls(config.gx, config.gy, config.gz, config.lx_m, config.ly_m, config.lz_m)

    @property
    def dx_m(self) -> float:
        return self.lx_m / self.gx

    @property
    def dy_m(self) -> float:
        return self.ly_m / self.gy

    @property
    def dz_m(self) -> float:
        return self.lz_m / self.gz

    @property
    def z_min_m(self) -> float:
        return -self.lz_m

    @property
    def z_max_m(self) -> float:
        return 0.0

    @property
    def cell_volume_m3(self) -> float:
        return self.dx_m * self.dy_m * self.dz_m
