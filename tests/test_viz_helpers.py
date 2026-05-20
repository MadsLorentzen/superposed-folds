"""Pin the discrete layer-binning helper used by both the 2D map and the
unrolled drill-core view."""

import math

import numpy as np

from superposed_folds.geometry import FoldParameters, make_layer_stack
from superposed_folds.layers import LAYER_COLORS, N_LAYERS, layer_index_from_z
from superposed_folds.viz import fig_3d_block_diagram, fig_3d_stack


def test_layer_index_from_z_periodic_wrap():
    """Z values that span multiple periods of the layer stack all wrap into
    [0, len(LAYER_COLORS))."""
    n_layers = 5
    extent = 5.0
    # Bin centers at z = -2.5, -1.25, 0.0, 1.25, 2.5; spacing = 1.25.
    # These z values cover several periods up and down.
    z = np.array([-2.5, 0.0, 2.5, 3.75, 5.0, -3.75, -5.0, 12.5, -12.5])
    indices = layer_index_from_z(z, n_layers=n_layers, extent=extent)

    n_colors = len(LAYER_COLORS)
    assert indices.dtype.kind == "i"
    assert np.all(indices >= 0)
    assert np.all(indices < n_colors)


def test_layer_index_from_z_centers_align_with_make_layer_stack():
    """Z values exactly at horizon-center heights bin to 0, 1, 2, ... in
    order. This pins the visual contract that 2D map bands line up with 3D
    layer-stack colors."""
    n_layers = 5
    extent = 5.0
    layers = make_layer_stack(n_layers=n_layers, extent=extent, n_grid=4)
    z_centers = np.array([Z[0, 0] for (_X, _Y, Z) in layers])

    indices = layer_index_from_z(z_centers, n_layers=n_layers, extent=extent)

    assert list(indices) == [0, 1, 2, 3, 4]


def _grid_size(fig):
    """Pull the (rows, cols) grid size from the first layer Surface trace."""
    surfaces = [t for t in fig.data if type(t).__name__ == "Surface"]
    return surfaces[0].z.shape


def test_fig_3d_stack_default_wavelength_uses_caller_n_grid():
    """At default λ=2π, the auto-scaler should not bump n_grid up: the
    caller's value (64 here) is already finer than 8 pts/wavelength."""
    f = FoldParameters(A=2.0, B=1.0, dip_dir=0.0, dip=90.0, rake=0.0)
    fig = fig_3d_stack(f, f, n_grid=64, extent=5.0)
    assert _grid_size(fig) == (64, 64)


def test_fig_3d_stack_short_wavelength_auto_scales_n_grid():
    """At λ=1.0 km with extent=5.0, we have 10 wavelengths across the 2*extent
    span; 8 pts/wave needs ≥80 grid points. Auto-scaler must bump up from 48."""
    f = FoldParameters(
        A=2.0, B=1.0, dip_dir=0.0, dip=90.0, rake=0.0, wavelength=1.0
    )
    fig = fig_3d_stack(f, f, n_grid=48, extent=5.0)
    n, _ = _grid_size(fig)
    assert n >= 80, f"expected n_grid ≥ 80 at λ=1.0, got {n}"


def test_fig_3d_stack_caps_n_grid_at_256():
    """At a pathologically small λ the auto-scaler must stop at 256 to keep
    the browser-side Plotly render responsive."""
    f = FoldParameters(
        A=2.0, B=1.0, dip_dir=0.0, dip=90.0, rake=0.0, wavelength=0.1
    )
    fig = fig_3d_stack(f, f, n_grid=48, extent=5.0)
    n, _ = _grid_size(fig)
    assert n == 256, f"expected n_grid capped at 256, got {n}"


def test_fig_3d_stack_unused_math_default_matches_two_pi():
    """Sanity: the FoldParameters default wavelength is exactly 2π so that
    the MATLAB parity case (sin(Y)) is reproduced bit-for-bit. Keeping this
    test next to the auto-scaler so we notice if the default ever drifts."""
    f = FoldParameters(A=1.0, B=1.0, dip_dir=0.0, dip=90.0, rake=0.0)
    assert f.wavelength == 2 * math.pi


def _block_surface_traces(fig):
    """Pull only the Surface traces (cube faces), excluding any
    Scatter3d / Cone traces from the orientation cues."""
    return [t for t in fig.data if type(t).__name__ == "Surface"]


def test_fig_3d_block_diagram_has_six_faces():
    """Cube view returns exactly six Surface traces (one per face)."""
    f = FoldParameters(A=2.0, B=1.0, dip_dir=0.0, dip=90.0, rake=0.0)
    fig = fig_3d_block_diagram(f, f, n_face_grid=64, extent=5.0)
    surfaces = _block_surface_traces(fig)
    assert len(surfaces) == 6, f"expected 6 cube faces, got {len(surfaces)}"


def test_fig_3d_block_diagram_face_planes():
    """Each face sits at the correct cube plane: one coordinate constant,
    the other two spanning their full ranges."""
    f = FoldParameters(A=2.0, B=1.0, dip_dir=0.0, dip=90.0, rake=0.0)
    extent = 5.0
    z_half = extent / 2.0
    fig = fig_3d_block_diagram(f, f, n_face_grid=32, extent=extent)
    surfaces = _block_surface_traces(fig)

    found = set()
    for s in surfaces:
        X, Y, Z = np.asarray(s.x), np.asarray(s.y), np.asarray(s.z)
        if np.allclose(X, X.flat[0]):
            found.add(("x", float(X.flat[0])))
        elif np.allclose(Y, Y.flat[0]):
            found.add(("y", float(Y.flat[0])))
        elif np.allclose(Z, Z.flat[0]):
            found.add(("z", float(Z.flat[0])))

    expected = {
        ("x", -extent), ("x", extent),
        ("y", -extent), ("y", extent),
        ("z", -z_half), ("z", z_half),
    }
    assert found == expected, f"face planes {found!r} != expected {expected!r}"


def test_fig_3d_block_diagram_uses_layer_color_scale():
    """All cube faces share the same discrete LAYER_COLORS colorscale and
    the cmin/cmax span the full layer-index range."""
    f = FoldParameters(A=2.0, B=1.0, dip_dir=0.0, dip=90.0, rake=0.0)
    fig = fig_3d_block_diagram(f, f, n_face_grid=32, extent=5.0)
    for s in _block_surface_traces(fig):
        unique_colors = []
        for _stop, color in s.colorscale:
            if color not in unique_colors:
                unique_colors.append(color)
        assert unique_colors == LAYER_COLORS, (
            f"face colorscale colors {unique_colors!r} != {LAYER_COLORS!r}"
        )
        assert s.cmin == -0.5
        assert s.cmax == N_LAYERS - 0.5


def test_fig_3d_block_diagram_layer_indices_in_range():
    """surfacecolor on every face is integer-valued and in [0, N_LAYERS)."""
    f = FoldParameters(A=2.0, B=1.0, dip_dir=0.0, dip=90.0, rake=0.0)
    fig = fig_3d_block_diagram(f, f, n_face_grid=32, extent=5.0)
    for s in _block_surface_traces(fig):
        sc = np.asarray(s.surfacecolor)
        assert sc.dtype.kind in ("i", "u"), f"non-integer surfacecolor on {s.name}"
        assert sc.min() >= 0
        assert sc.max() < N_LAYERS


def test_fig_3d_block_diagram_short_wavelength_auto_scales():
    """At lambda=1.0 with extent=5.0, the auto-scaler should push the
    per-face grid above the caller's requested 48 (same formula as
    fig_3d_stack: ceil(16 * extent / lam_min) = ceil(80) = 80)."""
    f = FoldParameters(
        A=2.0, B=1.0, dip_dir=0.0, dip=90.0, rake=0.0, wavelength=1.0
    )
    fig = fig_3d_block_diagram(f, f, n_face_grid=48, extent=5.0)
    n_rows, n_cols = np.asarray(_block_surface_traces(fig)[0].surfacecolor).shape
    assert n_rows >= 80 and n_cols >= 80, (
        f"expected per-face grid >= 80 at lambda=1.0, got ({n_rows}, {n_cols})"
    )


def test_fig_3d_block_diagram_caps_grid_at_256():
    """At a pathologically small lambda the auto-scaler caps n_face_grid
    at 256 to keep the browser-side render responsive."""
    f = FoldParameters(
        A=2.0, B=1.0, dip_dir=0.0, dip=90.0, rake=0.0, wavelength=0.1
    )
    fig = fig_3d_block_diagram(f, f, n_face_grid=48, extent=5.0)
    n_rows, n_cols = np.asarray(_block_surface_traces(fig)[0].surfacecolor).shape
    assert n_rows == 256 and n_cols == 256
