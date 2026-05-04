"""Plotly + matplotlib figure builders for the Streamlit app."""

from __future__ import annotations

import matplotlib.pyplot as plt
import mplstereonet  # noqa: F401  (registers 'stereonet' projection with matplotlib)
import numpy as np
import plotly.graph_objects as go

from .geometry import (
    FoldParameters,
    apply_superposed_fold,
    initial_z_at,
    make_layer_stack,
)

_LAYER_COLORS = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd"]


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
            xaxis_title="X",
            yaxis_title="Y",
            zaxis_title="Z",
            aspectmode="data",
        ),
        margin=dict(l=0, r=0, t=0, b=0),
        height=420,
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
    # Bin against the same horizon centers used by `make_layer_stack`, so the
    # 2D map's colored bands line up with the 3D viewer's colored layers.
    z_levels = np.linspace(-extent / 2, extent / 2, n_layers)
    layer_spacing = z_levels[1] - z_levels[0] if n_layers > 1 else 1.0
    layer_index = np.round((Z0 - z_levels[0]) / layer_spacing)

    fig = go.Figure(
        data=go.Heatmap(
            x=xs,
            y=ys,
            z=layer_index,
            colorscale="Viridis",
            showscale=False,
        )
    )
    fig.update_layout(
        xaxis=dict(title="X", scaleanchor="y", scaleratio=1),
        yaxis=dict(title="Y"),
        margin=dict(l=10, r=10, t=10, b=10),
        height=320,
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
