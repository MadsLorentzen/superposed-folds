"""Cylindrical drill-core geometry: parametrize a drill core in world
coordinates and sample superposed-fold layer indices on its curved surface.

The local frame convention used here:

  * The cylinder axis is along local +z'. The axial parameter `s` runs
    from 0 at the collar to `length` at the toe.
  * The local +z' direction in world coordinates is the down-plunge unit
    vector (so increasing `s` moves down into the borehole).
  * The local +x' direction in world coordinates is east-of-trend. This
    pins where theta = 0 lands on the unrolled view, so the unrolled
    strip's horizontal axis has a consistent geological meaning across
    different orientations.
  * The local +y' direction completes a right-handed basis
    (e_y' = e_z' x e_x').
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from .geometry import FoldParameters, initial_z_at
from .viz import layer_index_from_z


@dataclass(frozen=True)
class DrillCoreParameters:
    """Parameters of a single drill core in world coordinates.

    All angles are in degrees. `azimuth_deg` is the compass bearing the
    core trends toward (0 = north, 90 = east). `plunge_deg` is the angle
    below horizontal (0 = horizontal, 90 = straight down).
    """

    collar_x: float
    collar_y: float
    collar_z: float
    azimuth_deg: float
    plunge_deg: float
    length: float
    diameter: float
    n_axial: int = 200
    n_circ: int = 200


def cylinder_surface_points(
    p: DrillCoreParameters,
) -> tuple[
    NDArray[np.floating], NDArray[np.floating], NDArray[np.floating]
]:
    """Return three `(n_axial, n_circ)` arrays of world-frame X, Y, Z
    coordinates on the curved surface of the cylinder."""
    s = np.linspace(0.0, p.length, p.n_axial)
    theta = np.linspace(0.0, 2.0 * np.pi, p.n_circ)
    S, T = np.meshgrid(s, theta, indexing="ij")
    r = p.diameter / 2.0

    Xl = r * np.cos(T)
    Yl = r * np.sin(T)
    Zl = S

    az = np.deg2rad(p.azimuth_deg)
    pl = np.deg2rad(p.plunge_deg)
    cp = np.cos(pl)
    sp = np.sin(pl)

    # Down-plunge unit vector in world coordinates.
    trend = np.array(
        [
            np.sin(az) * cp,
            np.cos(az) * cp,
            -sp,
        ]
    )
    # East-of-trend unit vector (azimuth + 90 degrees, projected onto the
    # horizontal plane). Stays well-defined for plunge = 90.
    e_x_local = np.array([np.cos(az), -np.sin(az), 0.0])
    # Right-handed completion.
    e_y_local = np.cross(trend, e_x_local)

    Xw = (
        p.collar_x
        + Xl * e_x_local[0]
        + Yl * e_y_local[0]
        + Zl * trend[0]
    )
    Yw = (
        p.collar_y
        + Xl * e_x_local[1]
        + Yl * e_y_local[1]
        + Zl * trend[1]
    )
    Zw = (
        p.collar_z
        + Xl * e_x_local[2]
        + Yl * e_y_local[2]
        + Zl * trend[2]
    )
    return Xw, Yw, Zw


def sample_layers_on_cylinder(
    p: DrillCoreParameters,
    f1: FoldParameters,
    f2: FoldParameters,
    n_layers: int,
    extent: float,
) -> tuple[
    NDArray[np.floating],
    NDArray[np.floating],
    NDArray[np.floating],
    NDArray[np.integer],
    NDArray[np.floating],
]:
    """Sample superposed-fold layer indices on the cylinder surface.

    Returns `(X, Y, Z, layer_idx, Z0)` where the first three are the
    world-frame surface coordinates from `cylinder_surface_points`,
    `layer_idx` is the discrete layer index at each surface point
    (computed by inverting the superposed-fold map via `initial_z_at`
    and binning via `layer_index_from_z`), and `Z0` is the raw initial-z
    value at each surface point. Callers use `Z0` to detect whether
    each cylinder vertex lies in the original layer stack
    (`Z0 ∈ [-extent/2, extent/2]` modulo half a layer spacing) or in
    the model's periodic continuation.
    """
    X, Y, Z = cylinder_surface_points(p)
    Z0 = initial_z_at(X, Y, Z, f1, f2)
    layer_idx = layer_index_from_z(Z0, n_layers, extent)
    return X, Y, Z, layer_idx, Z0
