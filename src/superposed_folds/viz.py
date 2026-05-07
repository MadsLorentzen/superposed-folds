"""Plotly + matplotlib figure builders for the Streamlit app."""

from __future__ import annotations

from typing import TYPE_CHECKING

import matplotlib.pyplot as plt
import mplstereonet  # noqa: F401  (registers 'stereonet' projection with matplotlib)
import numpy as np
import plotly.graph_objects as go
from numpy.typing import NDArray

from .geometry import (
    FoldParameters,
    apply_superposed_fold,
    initial_z_at,
    make_layer_stack,
)

if TYPE_CHECKING:
    from .cylinder import DrillCoreParameters

_LAYER_COLORS = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd"]


def _discrete_colorscale(colors: list[str]) -> list[list[float | str]]:
    """Build a Plotly colorscale that renders crisp bands rather than gradients.

    Each color gets one equal-width slice; paired stops at each band boundary
    prevent Plotly from interpolating between adjacent colors.
    """
    n = len(colors)
    scale: list[list[float | str]] = []
    for i, color in enumerate(colors):
        scale.append([i / n, color])
        scale.append([(i + 1) / n, color])
    scale[-1][0] = 1.0  # avoid floating-point drift on the upper bound
    return scale


def layer_index_from_z(
    z_array: NDArray[np.floating],
    n_layers: int,
    extent: float,
) -> NDArray[np.integer]:
    """Round per-point initial-z values to integer layer indices and wrap
    them periodically into the available palette.

    Bin centers are `np.linspace(-extent/2, extent/2, n_layers)`, matching
    the horizon heights produced by `make_layer_stack` with default
    `z_span = extent`. The result wraps modulo `len(_LAYER_COLORS)` so the
    same five colors used in the 3D viewer cycle through the 2D map and
    the unrolled drill-core strip.
    """
    z_levels = np.linspace(-extent / 2.0, extent / 2.0, n_layers)
    layer_spacing = z_levels[1] - z_levels[0] if n_layers > 1 else 1.0
    layer_index = np.round((z_array - z_levels[0]) / layer_spacing)
    n_colors = len(_LAYER_COLORS)
    return layer_index.astype(int) % n_colors


def fig_3d_stack(
    f1: FoldParameters,
    f2: FoldParameters,
    *,
    n_layers: int = 5,
    n_grid: int = 64,
    extent: float = 5.0,
) -> go.Figure:
    """Render the 3D superposed-fold stack as `n_layers` colored surfaces."""
    fig = go.Figure()
    for (X0, Y0, Z0), color in zip(
        make_layer_stack(n_layers=n_layers, extent=extent, n_grid=n_grid),
        _LAYER_COLORS,
        strict=False,
    ):
        Xf, Yf, Zf = apply_superposed_fold(X0, Y0, Z0, f1, f2)
        fig.add_trace(
            go.Surface(
                x=Xf,
                y=Yf,
                z=Zf,
                colorscale=[[0, color], [1, color]],
                showscale=False,
                opacity=0.85,
                name=f"layer z₀={Z0[0, 0]:+.2f}",
            )
        )
    fig.update_layout(
        scene=dict(
            xaxis_title="X (East)",
            yaxis_title="Y (North)",
            zaxis_title="Z",
            aspectmode="data",
            annotations=[
                dict(
                    showarrow=False,
                    x=0.0,
                    y=extent,
                    z=0.0,
                    text="N",
                    font=dict(size=14, color="black"),
                )
            ],
        ),
        margin=dict(l=0, r=0, t=0, b=0),
        height=520,
    )
    return fig


def fig_2d_interference(
    f1: FoldParameters,
    f2: FoldParameters,
    *,
    n_layers: int = 5,
    n_grid: int = 200,
    extent: float = 5.0,
    z_section: float = 0.0,
) -> go.Figure:
    """Render the 2D interference pattern at z = `z_section`.

    For each (X, Y) on a grid, compute `initial_z_at`, then color by the
    discrete layer that initial z falls into.
    """
    xs = np.linspace(-extent, extent, n_grid)
    ys = np.linspace(-extent, extent, n_grid)
    X, Y = np.meshgrid(xs, ys, indexing="xy")
    Z = np.full_like(X, z_section)

    Z0 = initial_z_at(X, Y, Z, f1, f2)
    layer_index_wrapped = layer_index_from_z(Z0, n_layers, extent)
    n_colors = len(_LAYER_COLORS)

    fig = go.Figure(
        data=go.Heatmap(
            x=xs,
            y=ys,
            z=layer_index_wrapped,
            colorscale=_discrete_colorscale(_LAYER_COLORS),
            zmin=-0.5,
            zmax=n_colors - 0.5,
            showscale=False,
        )
    )
    fig.update_layout(
        xaxis=dict(title="X (East)", scaleanchor="y", scaleratio=1),
        yaxis=dict(title="Y (North)"),
        # Generous top margin so the heatmap aligns vertically with the 3D
        # scene next to it (Plotly's 3D scene reserves more top padding than
        # a default heatmap, otherwise the 2D content sits visibly higher).
        margin=dict(l=10, r=10, t=50, b=10),
        height=420,
        annotations=[
            dict(
                xref="paper",
                yref="paper",
                x=0.02,
                y=0.98,
                text="↑ N",
                showarrow=False,
                font=dict(size=14, color="black"),
            )
        ],
    )
    return fig


def fig_stereonet(f1: FoldParameters, f2: FoldParameters) -> plt.Figure:
    """Lower-hemisphere equal-area stereonet (Schmidt) showing the F1 and F2
    axial planes (great circles) and fold-axis lineations (markers).

    Built on `mplstereonet`. Returns a matplotlib `Figure` for rendering with
    Streamlit's `st.pyplot()`.
    """
    fig, ax = plt.subplots(figsize=(4, 4), subplot_kw={"projection": "stereonet"})

    for fold, color, label in [(f1, "#1f77b4", "F1"), (f2, "#d62728", "F2")]:
        strike = (fold.dip_dir - 90.0) % 360.0
        # UCD rake conventions: range -90..+90, positive = anti-clockwise from
        # dip direction. With strike chosen by the right-hand rule (90° CCW
        # from dip direction), this matches mplstereonet's rake angle: 0 =
        # along strike, 90 = down-dip, 180 = along opposite strike. Since the
        # fold axis is an undirected line, we fold negative UCD rakes into
        # the [0, 180) range -- this preserves the *line* on the stereonet
        # but loses the visual distinction between, say, rake=-27 and
        # rake=+153. We compensate by annotating the original signed rake
        # numerically next to each fold-axis marker below.
        rake_mpl = fold.rake % 180.0

        ax.plane(strike, fold.dip, color=color, linewidth=2, label=f"{label} axial plane")
        ax.rake(
            strike, fold.dip, rake_mpl,
            color=color, marker="o", markersize=8, linestyle="None",
            label=f"{label} fold axis (rake={fold.rake:+.0f}°)",
        )

    ax.grid(linestyle=":", linewidth=0.5)
    ax.set_azimuth_ticks(range(0, 360, 90), labels=["N", "E", "S", "W"])
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, -0.15), ncol=2, fontsize=8)
    return fig


def fig_3d_drill_core_trace(
    X: NDArray[np.floating],
    Y: NDArray[np.floating],
    Z: NDArray[np.floating],
    layer_idx: NDArray[np.integer],
) -> go.Surface:
    """Plotly Surface trace for a drill core embedded in the 3D layer
    stack. Uses the same discrete colorscale as the 2D map and the layer
    surfaces, so colors match across all three views.

    Returns the trace, not a Figure. The Streamlit app composes it onto
    the cached layer-stack figure with `add_trace` so changes to drill-core
    parameters do not invalidate the layer-surface cache.
    """
    n_colors = len(_LAYER_COLORS)
    return go.Surface(
        x=X,
        y=Y,
        z=Z,
        surfacecolor=layer_idx,
        colorscale=_discrete_colorscale(_LAYER_COLORS),
        cmin=-0.5,
        cmax=n_colors - 0.5,
        showscale=False,
        opacity=1.0,
        name="drill core",
    )


def fig_2d_drill_core_unrolled(
    layer_idx: NDArray[np.integer],
    length: float,
) -> go.Figure:
    """Heatmap of the drill core unrolled flat: depth from collar on the
    Y axis (0 at top, increasing downward to `length`), circumferential
    angle on the X axis (0 to 360 degrees). Reuses the same `layer_idx`
    array as the 3D trace, and the same discrete palette.
    """
    n_axial, n_circ = layer_idx.shape
    theta_deg = np.linspace(0.0, 360.0, n_circ)
    depth = np.linspace(0.0, length, n_axial)
    n_colors = len(_LAYER_COLORS)
    fig = go.Figure(
        data=go.Heatmap(
            x=theta_deg,
            y=depth,
            z=layer_idx,
            colorscale=_discrete_colorscale(_LAYER_COLORS),
            zmin=-0.5,
            zmax=n_colors - 0.5,
            showscale=False,
        )
    )
    fig.update_layout(
        xaxis=dict(title="θ around core (°)"),
        # Depth increases downward; reverse the y-axis so the collar is on top.
        yaxis=dict(title="depth from collar", autorange="reversed"),
        margin=dict(l=10, r=10, t=30, b=40),
        height=260,
    )
    return fig


def drill_core_map_overlay_traces(p: DrillCoreParameters) -> list[go.Scatter]:
    """Traces that draw the drill core on the 2D interference map at z=0:
    a collar marker, the projected line from collar to toe, and a toe
    marker. For nearly-vertical cores (plunge >= 89 degrees) returns just
    the collar marker (the projected line collapses to a point).
    """
    az = np.deg2rad(p.azimuth_deg)
    pl = np.deg2rad(p.plunge_deg)
    trend_x = np.sin(az) * np.cos(pl)
    trend_y = np.cos(az) * np.cos(pl)
    toe_x = p.collar_x + p.length * trend_x
    toe_y = p.collar_y + p.length * trend_y

    collar_marker = go.Scatter(
        x=[p.collar_x],
        y=[p.collar_y],
        mode="markers",
        marker=dict(size=10, color="black", symbol="circle"),
        name="collar",
        showlegend=False,
        hoverinfo="skip",
    )

    if p.plunge_deg >= 89.0:
        return [collar_marker]

    line = go.Scatter(
        x=[p.collar_x, toe_x],
        y=[p.collar_y, toe_y],
        mode="lines",
        line=dict(color="black", width=2),
        name="core",
        showlegend=False,
        hoverinfo="skip",
    )
    toe_marker = go.Scatter(
        x=[toe_x],
        y=[toe_y],
        mode="markers",
        marker=dict(size=8, color="black", symbol="x"),
        name="toe",
        showlegend=False,
        hoverinfo="skip",
    )
    return [line, collar_marker, toe_marker]
