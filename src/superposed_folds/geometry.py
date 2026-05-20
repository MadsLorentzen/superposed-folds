"""Plane-strain superposed-folding geometry (Ramsay & Lisle 2000, p. 955)."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from .layers import N_LAYERS
from .transforms import dipdir_dip_rake_to_rotation_angles, rotate_xyz


@dataclass(frozen=True)
class FoldParameters:
    """Parameters of a single folding event.

    `A` and `B` set the fold (amplitude is A*B); `C` is the orthogonal stretch
    (defaults to 1/B for constant-volume deformation).
    `dip_dir`, `dip`, `rake` orient the fold's axial plane and fold axis in
    degrees, following the convention used by Schöpfer's UCD MATLAB script.
    `wavelength` is the pre-fold wavelength of the sinusoidal fold (default
    2π preserves Schöpfer's UCD MATLAB behavior, where `sin(Y)` has period
    2π in pre-fold Y units).
    """

    A: float
    B: float
    dip_dir: float
    dip: float
    rake: float
    C: float | None = None
    wavelength: float = 2 * np.pi

    def __post_init__(self) -> None:
        if self.C is None:
            object.__setattr__(self, "C", 1.0 / self.B)

    @property
    def amplitude(self) -> float:
        return self.A * self.B


def apply_fold(
    X: NDArray[np.floating],
    Y: NDArray[np.floating],
    Z: NDArray[np.floating],
    p: FoldParameters,
) -> tuple[NDArray[np.floating], NDArray[np.floating], NDArray[np.floating]]:
    """Apply the Ramsay & Lisle (2000) plane-strain fold map (p. 955).

        x_f = X
        y_f = C * Y
        z_f = A * B * sin(2π Y / λ) + B * Z

    Operates element-wise; orientation parameters on `p` are unused here
    (orientation handling lives in `apply_superposed_fold`).
    """
    Xf = X
    Yf = p.C * Y
    Zf = p.A * p.B * np.sin(2.0 * np.pi * Y / p.wavelength) + p.B * Z
    return Xf, Yf, Zf


def initial_z_at(
    X: NDArray[np.floating],
    Y: NDArray[np.floating],
    Z: NDArray[np.floating],
    f1: FoldParameters,
    f2: FoldParameters,
) -> NDArray[np.floating]:
    """Return the initial z-position whose superposed-fold image is (X, Y, Z).

    Mirrors `fold_interference_pattern.m`: rotate into F2 frame, undo F2 fold,
    rotate from F2 frame to F1 frame, undo F1 fold, return resulting z.

    The returned z-coordinate is in **F1's reference frame**, not world
    coordinates. For F1 at the standard UCD orientation (dip_dir=0, dip=90,
    rake=0) the two coincide, which is the case for every preset shipped
    here. If you call this with a non-standard F1, interpret the returned
    value as the layer index in F1's local frame.
    """
    bx, by, bz = dipdir_dip_rake_to_rotation_angles(f2.dip_dir, f2.dip, f2.rake)
    ax, ay, az = dipdir_dip_rake_to_rotation_angles(f1.dip_dir, f1.dip, f1.rake)

    # Rotate into F2 reference frame
    x, y, z = rotate_xyz(X, Y, Z, bx, by, bz)

    # Undo F2 fold:  y_pre = y / C2;  z_pre = (z - A2*B2*sin(2π y_pre / λ2)) / B2
    y_pre = y / f2.C
    z_pre = (z - f2.A * f2.B * np.sin(2.0 * np.pi * y_pre / f2.wavelength)) / f2.B

    # Undo F2's frame rotation (back to world coordinates).
    x, y, z = rotate_xyz(x, y_pre, z_pre, bx, by, bz, inverse=True)
    # Then rotate into F1's reference frame.
    x, y, z = rotate_xyz(x, y, z, ax, ay, az)

    # Undo F1 fold
    y_pre = y / f1.C
    z_pre = (z - f1.A * f1.B * np.sin(2.0 * np.pi * y_pre / f1.wavelength)) / f1.B

    return z_pre


def apply_superposed_fold(
    X: NDArray[np.floating],
    Y: NDArray[np.floating],
    Z: NDArray[np.floating],
    f1: FoldParameters,
    f2: FoldParameters,
) -> tuple[NDArray[np.floating], NDArray[np.floating], NDArray[np.floating]]:
    """Forward superposed-fold map: F1 then F2 (each in its own oriented frame)."""
    ax, ay, az = dipdir_dip_rake_to_rotation_angles(f1.dip_dir, f1.dip, f1.rake)
    bx, by, bz = dipdir_dip_rake_to_rotation_angles(f2.dip_dir, f2.dip, f2.rake)

    # Rotate into F1 frame, fold, rotate back
    x, y, z = rotate_xyz(X, Y, Z, ax, ay, az)
    x, y, z = apply_fold(x, y, z, f1)
    x, y, z = rotate_xyz(x, y, z, ax, ay, az, inverse=True)

    # Rotate into F2 frame, fold, rotate back
    x, y, z = rotate_xyz(x, y, z, bx, by, bz)
    x, y, z = apply_fold(x, y, z, f2)
    x, y, z = rotate_xyz(x, y, z, bx, by, bz, inverse=True)

    return x, y, z


def make_layer_stack(
    n_layers: int = N_LAYERS,
    extent: float = 5.0,
    n_grid: int = 64,
    z_span: float | None = None,
) -> list[tuple[NDArray[np.floating], NDArray[np.floating], NDArray[np.floating]]]:
    """Build `n_layers` flat horizon meshes covering [-extent, extent]² in (X, Y).

    Returns a list of `(X, Y, Z)` triples, one per horizon. Layers are stacked
    evenly across `[-z_span/2, z_span/2]` (default `z_span = extent`).
    """
    if z_span is None:
        z_span = extent
    xs = np.linspace(-extent, extent, n_grid)
    ys = np.linspace(-extent, extent, n_grid)
    X, Y = np.meshgrid(xs, ys, indexing="xy")
    z_levels = np.linspace(-z_span / 2, z_span / 2, n_layers)
    return [(X, Y, np.full_like(X, z)) for z in z_levels]
