"""Layer-stack color contract used by every superposed-fold view.

Single source of truth for the discrete-color palette and the layer-binning
helper. The 3D forward-mapped surfaces, the 2D inverse-mapped interference
map, and the unrolled drill-core strip all read from here so the layer-to-
color mapping is identical across views. Changing `LAYER_COLORS` propagates
through the default `n_layers` in `make_layer_stack`, every figure builder,
and the Streamlit app.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

LAYER_COLORS: list[str] = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd"]
N_LAYERS: int = len(LAYER_COLORS)


def layer_index_from_z(
    z_array: NDArray[np.floating],
    n_layers: int,
    extent: float,
) -> NDArray[np.integer]:
    """Round per-point initial-z values to integer layer indices and wrap
    them periodically into the available palette.

    Bin centers are `np.linspace(-extent/2, extent/2, n_layers)`, matching
    the horizon heights produced by `make_layer_stack` with default
    `z_span = extent`. The result wraps modulo `len(LAYER_COLORS)` so the
    same colors used in the 3D viewer cycle through the 2D map and the
    unrolled drill-core strip.
    """
    z_levels = np.linspace(-extent / 2.0, extent / 2.0, n_layers)
    layer_spacing = z_levels[1] - z_levels[0] if n_layers > 1 else 1.0
    layer_index = np.round((z_array - z_levels[0]) / layer_spacing)
    return layer_index.astype(int) % len(LAYER_COLORS)
