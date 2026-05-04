"""Conversions between dip-direction/dip/rake and the three rotation angles
used by the MATLAB script `fold_interference_pattern.m`.

The convention (Schöpfer, UCD) is:
    alpha_x = (90 - dip)        (radians)
    alpha_y = rake               (radians)
    alpha_z = -dip_dir           (radians)

`rotate_xyz` applies the composite rotation R_y · R_x · R_z used by the
MATLAB script (see comment block in `fold_interference_pattern.m`).
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


def dipdir_dip_rake_to_rotation_angles(
    dip_dir: float, dip: float, rake: float
) -> tuple[float, float, float]:
    """Convert (dip_dir, dip, rake) in degrees to (alpha_x, alpha_y, alpha_z) in radians."""
    alpha_x = np.deg2rad(90.0 - dip)
    alpha_y = np.deg2rad(rake)
    alpha_z = np.deg2rad(-dip_dir)
    return float(alpha_x), float(alpha_y), float(alpha_z)


def rotate_xyz(
    X: NDArray[np.floating],
    Y: NDArray[np.floating],
    Z: NDArray[np.floating],
    alpha_x: float,
    alpha_y: float,
    alpha_z: float,
    *,
    inverse: bool = False,
) -> tuple[NDArray[np.floating], NDArray[np.floating], NDArray[np.floating]]:
    """Apply R_y(alpha_y) · R_x(alpha_x) · R_z(alpha_z) to the points.

    With `inverse=True`, applies the inverse rotation R_z(-alpha_z) · R_x(-alpha_x) · R_y(-alpha_y).
    """
    if inverse:
        # inverse: apply transpose, in reverse order
        X, Y, Z = _rot_y(X, Y, Z, -alpha_y)
        X, Y, Z = _rot_x(X, Y, Z, -alpha_x)
        X, Y, Z = _rot_z(X, Y, Z, -alpha_z)
        return X, Y, Z
    X, Y, Z = _rot_z(X, Y, Z, alpha_z)
    X, Y, Z = _rot_x(X, Y, Z, alpha_x)
    X, Y, Z = _rot_y(X, Y, Z, alpha_y)
    return X, Y, Z


def _rot_z(X, Y, Z, a):
    s, c = np.sin(a), np.cos(a)
    return Y * s + c * X, c * Y - X * s, Z


def _rot_x(X, Y, Z, a):
    s, c = np.sin(a), np.cos(a)
    return X, Z * s + c * Y, c * Z - Y * s


def _rot_y(X, Y, Z, a):
    s, c = np.sin(a), np.cos(a)
    return Z * s + c * X, Y, c * Z - X * s
