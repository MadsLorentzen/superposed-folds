"""Pin the discrete layer-binning helper used by both the 2D map and the
unrolled drill-core view."""

import numpy as np

from superposed_folds.geometry import make_layer_stack
from superposed_folds.viz import _LAYER_COLORS, layer_index_from_z


def test_layer_index_from_z_periodic_wrap():
    """Z values that span multiple periods of the layer stack all wrap into
    [0, len(_LAYER_COLORS))."""
    n_layers = 5
    extent = 5.0
    # Bin centers at z = -2.5, -1.25, 0.0, 1.25, 2.5; spacing = 1.25.
    # These z values cover several periods up and down.
    z = np.array([-2.5, 0.0, 2.5, 3.75, 5.0, -3.75, -5.0, 12.5, -12.5])
    indices = layer_index_from_z(z, n_layers=n_layers, extent=extent)

    n_colors = len(_LAYER_COLORS)
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
