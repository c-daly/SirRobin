"""Compatibility API for ellipsoid Lamb added-mass quadrature."""

from sirrobin.numerics.ellipsoid_added_mass import (
    added_mass,
    donor_added_mass,
    donor_lamb_factors,
    gl_nodes_weights,
    lamb_coefficients,
    lamb_factors,
)

__all__ = [
    "added_mass",
    "donor_added_mass",
    "donor_lamb_factors",
    "gl_nodes_weights",
    "lamb_coefficients",
    "lamb_factors",
]
