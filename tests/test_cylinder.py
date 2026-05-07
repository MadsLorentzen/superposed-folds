"""Pin cylinder geometry: shape, range, orientation, theta=0 convention."""

import numpy as np
import pytest

from superposed_folds.cylinder import (
    DrillCoreParameters,
    cylinder_surface_points,
    sample_layers_on_cylinder,
)
from superposed_folds.geometry import FoldParameters, initial_z_at
from superposed_folds.viz import layer_index_from_z


def _make_params(**overrides):
    """Helper: build a DrillCoreParameters with sensible defaults for tests."""
    base = dict(
        collar_x=0.0,
        collar_y=0.0,
        collar_z=0.0,
        azimuth_deg=0.0,
        plunge_deg=90.0,
        length=5.0,
        diameter=0.4,
        n_axial=20,
        n_circ=20,
    )
    base.update(overrides)
    return DrillCoreParameters(**base)


def test_drill_core_parameters_defaults():
    """All user-controlled fields are required; n_axial and n_circ default
    to 200; the dataclass is frozen."""
    p = DrillCoreParameters(
        collar_x=0.0,
        collar_y=0.0,
        collar_z=2.5,
        azimuth_deg=0.0,
        plunge_deg=90.0,
        length=5.0,
        diameter=0.4,
    )
    assert p.n_axial == 200
    assert p.n_circ == 200
    with pytest.raises(AttributeError):
        p.length = 6.0


def test_cylinder_surface_points_shape():
    """Returns three (n_axial, n_circ) numpy arrays."""
    p = _make_params(n_axial=10, n_circ=20)
    X, Y, Z = cylinder_surface_points(p)
    assert X.shape == (10, 20)
    assert Y.shape == (10, 20)
    assert Z.shape == (10, 20)


def test_vertical_core_at_origin():
    """Plunge=90, collar=(0,0,0): every surface point sits at horizontal
    distance r from the z-axis, and z spans [-length, 0]."""
    p = _make_params(collar_z=0.0, plunge_deg=90.0, length=5.0, diameter=0.4)
    X, Y, Z = cylinder_surface_points(p)
    r = 0.4 / 2
    radial = np.sqrt(X**2 + Y**2)
    np.testing.assert_allclose(radial, r, atol=1e-9)
    assert np.isclose(Z.min(), -5.0, atol=1e-9)
    assert np.isclose(Z.max(), 0.0, atol=1e-9)


def test_horizontal_core_along_north():
    """Plunge=0, azimuth=0: axis is +Y; surface points sit at distance r
    from the y-axis (in the xz plane), and y spans [0, length]."""
    p = _make_params(plunge_deg=0.0, azimuth_deg=0.0, length=5.0, diameter=0.4)
    X, Y, Z = cylinder_surface_points(p)
    r = 0.4 / 2
    radial = np.sqrt(X**2 + Z**2)
    np.testing.assert_allclose(radial, r, atol=1e-9)
    assert np.isclose(Y.min(), 0.0, atol=1e-9)
    assert np.isclose(Y.max(), 5.0, atol=1e-9)


def test_azimuth_rotation():
    """A horizontal core at azimuth=90 is the same as one at azimuth=0
    rotated -90 degrees about z (mapping +Y to +X)."""
    p1 = _make_params(plunge_deg=0.0, azimuth_deg=0.0)
    p2 = _make_params(plunge_deg=0.0, azimuth_deg=90.0)
    X1, Y1, Z1 = cylinder_surface_points(p1)
    X2, Y2, Z2 = cylinder_surface_points(p2)
    # Rotation that maps +Y to +X is (x, y, z) -> (y, -x, z).
    np.testing.assert_allclose(X2, Y1, atol=1e-9)
    np.testing.assert_allclose(Y2, -X1, atol=1e-9)
    np.testing.assert_allclose(Z2, Z1, atol=1e-9)


def test_unrolled_theta_zero_is_east_of_trend():
    """At theta=0 and s=0 (collar), the surface point sits at distance r
    in the east-of-trend direction. For a horizontal core trending east
    (azimuth=90), east-of-east is south, so the offset is -y."""
    p = _make_params(
        collar_x=0.0,
        collar_y=0.0,
        collar_z=0.0,
        azimuth_deg=90.0,
        plunge_deg=0.0,
        length=5.0,
        diameter=0.4,
        n_axial=3,
        n_circ=5,
    )
    X, Y, Z = cylinder_surface_points(p)
    r = 0.4 / 2
    np.testing.assert_allclose(X[0, 0], 0.0, atol=1e-9)
    np.testing.assert_allclose(Y[0, 0], -r, atol=1e-9)
    np.testing.assert_allclose(Z[0, 0], 0.0, atol=1e-9)


def test_plunge_zero_does_not_blow_up():
    """Horizontal core does not produce NaN values."""
    p = _make_params(plunge_deg=0.0, azimuth_deg=37.0)
    X, Y, Z = cylinder_surface_points(p)
    assert not np.any(np.isnan(X))
    assert not np.any(np.isnan(Y))
    assert not np.any(np.isnan(Z))


def test_sample_layers_on_cylinder_uses_initial_z_at():
    """One sampled point's layer index matches what we get from calling
    initial_z_at and layer_index_from_z directly on its world coordinates.
    Pins that sample_layers_on_cylinder routes through the same primitives
    the rest of the project uses."""
    p = _make_params(
        collar_x=0.0,
        collar_y=0.0,
        collar_z=1.0,
        azimuth_deg=30.0,
        plunge_deg=60.0,
        length=3.0,
        diameter=0.5,
        n_axial=4,
        n_circ=4,
    )
    f1 = FoldParameters(A=2.0, B=1.0, dip_dir=0.0, dip=90.0, rake=0.0)
    f2 = FoldParameters(A=1.5, B=1.2, dip_dir=45.0, dip=70.0, rake=10.0)
    X, Y, Z, layer_idx, Z0 = sample_layers_on_cylinder(
        p, f1, f2, n_layers=5, extent=5.0
    )

    i, j = 1, 2
    expected_z0 = initial_z_at(
        np.array([X[i, j]]),
        np.array([Y[i, j]]),
        np.array([Z[i, j]]),
        f1,
        f2,
    )
    expected_idx = layer_index_from_z(expected_z0, n_layers=5, extent=5.0)
    assert int(layer_idx[i, j]) == int(expected_idx[0])
    np.testing.assert_allclose(Z0[i, j], expected_z0[0], atol=1e-9)


def test_sample_layers_on_cylinder_layer_indices_in_range():
    """All layer_idx values are in [0, n_layers) when n_layers equals the
    palette size (5)."""
    p = _make_params(
        collar_x=0.0,
        collar_y=0.0,
        collar_z=0.0,
        azimuth_deg=0.0,
        plunge_deg=90.0,
        length=5.0,
        diameter=0.4,
        n_axial=10,
        n_circ=10,
    )
    f1 = FoldParameters(A=2.0, B=1.0, dip_dir=0.0, dip=90.0, rake=0.0)
    f2 = FoldParameters(A=1.5, B=1.2, dip_dir=45.0, dip=70.0, rake=10.0)
    _, _, _, layer_idx, _ = sample_layers_on_cylinder(
        p, f1, f2, n_layers=5, extent=5.0
    )
    assert np.all(layer_idx >= 0)
    assert np.all(layer_idx < 5)
