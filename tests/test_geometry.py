import math

import numpy as np

from superposed_folds.geometry import (
    FoldParameters,
    apply_fold,
    apply_superposed_fold,
    initial_z_at,
    make_layer_stack,
)


def test_fold_parameters_defaults_constant_volume():
    p = FoldParameters(A=3.0, B=1.0, dip_dir=0.0, dip=90.0, rake=0.0)
    assert p.C == 1.0  # default C = 1/B


def test_fold_parameters_explicit_C():
    p = FoldParameters(A=2.0, B=2.0, C=0.7, dip_dir=10.0, dip=80.0, rake=15.0)
    assert p.C == 0.7


def test_fold_parameters_amplitude_property():
    p = FoldParameters(A=3.0, B=2.0, dip_dir=0.0, dip=90.0, rake=0.0)
    assert math.isclose(p.amplitude, 6.0)


def test_apply_fold_zero_amplitude_is_pure_strain():
    # A=0 → pure plane-strain (no folding)
    p = FoldParameters(A=0.0, B=2.0, C=0.5, dip_dir=0.0, dip=90.0, rake=0.0)
    X = np.array([1.0, 2.0, 3.0])
    Y = np.array([0.5, 1.5, 2.5])
    Z = np.array([0.0, 0.0, 0.0])
    Xf, Yf, Zf = apply_fold(X, Y, Z, p)
    np.testing.assert_allclose(Xf, X)
    np.testing.assert_allclose(Yf, 0.5 * Y)
    np.testing.assert_allclose(Zf, 2.0 * Z)


def test_apply_fold_pure_fold_b_one_is_volume_preserving():
    # B=1, C=1, A=3 → pure folding, volume preserved
    p = FoldParameters(A=3.0, B=1.0, C=1.0, dip_dir=0.0, dip=90.0, rake=0.0)
    X = np.linspace(0, 2 * np.pi, 5)
    Y = np.linspace(0, 2 * np.pi, 5)
    Z = np.zeros(5)
    Xf, Yf, Zf = apply_fold(X, Y, Z, p)
    np.testing.assert_allclose(Xf, X)
    np.testing.assert_allclose(Yf, Y)
    np.testing.assert_allclose(Zf, 3.0 * np.sin(Y))


def test_initial_z_at_inverts_apply_fold_for_single_event():
    # When F2 is the identity (A=0, B=1, C=1, no rotation),
    # initial_z should reduce to the inverse of a single F1 fold.
    f1 = FoldParameters(A=2.0, B=1.0, C=1.0, dip_dir=0.0, dip=90.0, rake=0.0)
    f2_identity = FoldParameters(A=0.0, B=1.0, C=1.0, dip_dir=0.0, dip=90.0, rake=0.0)

    Y0 = np.linspace(0.1, 5.0, 6)  # avoid Y=0 where forward is degenerate
    Z0 = np.linspace(-1.0, 1.0, 6)
    X0 = np.zeros_like(Y0)

    Xf, Yf, Zf = apply_fold(X0, Y0, Z0, f1)
    Z_recovered = initial_z_at(Xf, Yf, Zf, f1, f2_identity)
    np.testing.assert_allclose(Z_recovered, Z0, atol=1e-9)


def test_apply_superposed_with_identity_f2_equals_apply_fold():
    f1 = FoldParameters(A=2.0, B=1.0, C=1.0, dip_dir=0.0, dip=90.0, rake=0.0)
    f2_identity = FoldParameters(A=0.0, B=1.0, C=1.0, dip_dir=0.0, dip=90.0, rake=0.0)

    rng = np.random.default_rng(0)
    X, Y, Z = rng.uniform(-3, 3, size=(3, 20))

    Xf1, Yf1, Zf1 = apply_fold(X, Y, Z, f1)
    Xs, Ys, Zs = apply_superposed_fold(X, Y, Z, f1, f2_identity)
    np.testing.assert_allclose(Xs, Xf1, atol=1e-9)
    np.testing.assert_allclose(Ys, Yf1, atol=1e-9)
    np.testing.assert_allclose(Zs, Zf1, atol=1e-9)


def test_forward_inverse_round_trip():
    f1 = FoldParameters(A=1.5, B=1.0, dip_dir=0.0, dip=90.0, rake=0.0)
    f2 = FoldParameters(A=1.5, B=1.0, dip_dir=90.0, dip=90.0, rake=0.0)
    rng = np.random.default_rng(1)
    X, Y, Z = rng.uniform(0.5, 2.5, size=(3, 30))
    Xf, Yf, Zf = apply_superposed_fold(X, Y, Z, f1, f2)
    Z_back = initial_z_at(Xf, Yf, Zf, f1, f2)
    np.testing.assert_allclose(Z_back, Z, atol=1e-6)


def test_apply_superposed_is_non_commutative():
    """Swapping f1 and f2 must change the result for a non-symmetric pair.

    This is the whole point of `superposed`: order matters. A regression that
    silently made the two events commute would invalidate every preset.

    Note: Type-1 (perpendicular vertical axes, equal amplitudes) happens to
    be commutative up to a 90° rotation, so this test deliberately uses
    different amplitudes and a non-perpendicular F2 axial plane.
    """
    f1 = FoldParameters(A=2.0, B=1.0, dip_dir=0.0, dip=90.0, rake=0.0)
    f2 = FoldParameters(A=4.0, B=1.0, dip_dir=45.0, dip=60.0, rake=20.0)

    rng = np.random.default_rng(2)
    X, Y, Z = rng.uniform(-3, 3, size=(3, 50))

    Xa, Ya, Za = apply_superposed_fold(X, Y, Z, f1, f2)
    Xb, Yb, Zb = apply_superposed_fold(X, Y, Z, f2, f1)

    diff = max(
        float(np.max(np.abs(Xa - Xb))),
        float(np.max(np.abs(Ya - Yb))),
        float(np.max(np.abs(Za - Zb))),
    )
    assert diff > 1e-3


def test_make_layer_stack_shape_and_extent():
    layers = make_layer_stack(n_layers=3, extent=5.0, n_grid=16)
    # Three layers, each (16, 16, 3)-shaped grid
    assert len(layers) == 3
    for X, Y, Z in layers:
        assert X.shape == (16, 16)
        assert Y.shape == (16, 16)
        assert Z.shape == (16, 16)
        # Extent ±5.0
        assert np.isclose(X.min(), -5.0)
        assert np.isclose(X.max(),  5.0)
    # Layers stacked in z, evenly spaced
    z_levels = sorted({float(layer[2][0, 0]) for layer in layers})
    diffs = np.diff(z_levels)
    np.testing.assert_allclose(diffs, diffs[0])
