import numpy as np
import pytest

from superposed_folds.transforms import (
    dipdir_dip_rake_to_rotation_angles,
    rotate_xyz,
)


def test_rotation_identity():
    X = np.array([1.0, 2.0])
    Y = np.array([0.0, 1.0])
    Z = np.array([0.0, 0.5])
    Xr, Yr, Zr = rotate_xyz(X, Y, Z, alpha_x=0.0, alpha_y=0.0, alpha_z=0.0)
    np.testing.assert_allclose(Xr, X)
    np.testing.assert_allclose(Yr, Y)
    np.testing.assert_allclose(Zr, Z)


def test_rotation_round_trip():
    rng = np.random.default_rng(0)
    X, Y, Z = rng.standard_normal((3, 8))
    a, b, c = 0.4, -0.7, 1.1
    X1, Y1, Z1 = rotate_xyz(X, Y, Z, a, b, c)
    X2, Y2, Z2 = rotate_xyz(X1, Y1, Z1, a, b, c, inverse=True)
    np.testing.assert_allclose(X2, X, atol=1e-12)
    np.testing.assert_allclose(Y2, Y, atol=1e-12)
    np.testing.assert_allclose(Z2, Z, atol=1e-12)


def test_dipdir_dip_rake_at_standard_orientation():
    # UCD F1 default: 000/90, rake 0 → all rotation angles 0
    ax, ay, az = dipdir_dip_rake_to_rotation_angles(dip_dir=0.0, dip=90.0, rake=0.0)
    assert ax == pytest.approx(0.0)
    assert ay == pytest.approx(0.0)
    assert az == pytest.approx(0.0)


def test_dipdir_dip_rake_horizontal_axial_plane():
    # dip 0 → alpha_x = 90° (in radians)
    ax, _, _ = dipdir_dip_rake_to_rotation_angles(dip_dir=45.0, dip=0.0, rake=0.0)
    assert ax == pytest.approx(np.pi / 2, abs=1e-6)


def test_rotation_around_z_known_outcome():
    """Rotating the unit X vector by +90° around Z must give (0, -1, 0).

    The round-trip test catches sign flips that compose to identity; this
    pins down the rotation direction itself.
    """
    Xr, Yr, Zr = rotate_xyz(
        np.array([1.0]), np.array([0.0]), np.array([0.0]),
        alpha_x=0.0, alpha_y=0.0, alpha_z=np.pi / 2,
    )
    np.testing.assert_allclose([Xr[0], Yr[0], Zr[0]], [0.0, -1.0, 0.0], atol=1e-12)


def test_rotation_around_x_known_outcome():
    """Rotating the unit Y vector by +90° around X must give (0, 0, -1)."""
    Xr, Yr, Zr = rotate_xyz(
        np.array([0.0]), np.array([1.0]), np.array([0.0]),
        alpha_x=np.pi / 2, alpha_y=0.0, alpha_z=0.0,
    )
    np.testing.assert_allclose([Xr[0], Yr[0], Zr[0]], [0.0, 0.0, -1.0], atol=1e-12)
