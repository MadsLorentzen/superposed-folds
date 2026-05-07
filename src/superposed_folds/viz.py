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
    z_arrays: list[NDArray[np.floating]] = []
    for (X0, Y0, Z0), color in zip(
        make_layer_stack(n_layers=n_layers, extent=extent, n_grid=n_grid),
        _LAYER_COLORS,
        strict=False,
    ):
        Xf, Yf, Zf = apply_superposed_fold(X0, Y0, Z0, f1, f2)
        z_arrays.append(Zf)
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

    # Compute a z-axis range that fits the actual folded layer surfaces
    # (which can extend well beyond +/-extent for high-amplitude folds)
    # while staying at least as wide as +/-extent so the drill-core
    # cylinder, which the user can place anywhere in [-extent, extent]
    # along z, also fits. The 5% pad keeps surfaces from touching the
    # box edge. The range is recomputed only when the cached figure is
    # rebuilt (i.e. when F1/F2 changes); toggling layer-surface
    # visibility from the app does not retrigger autorange because each
    # axis has autorange=False.
    all_z = np.concatenate([z.ravel() for z in z_arrays])
    z_data_half = float(max(abs(all_z.min()), abs(all_z.max())))
    z_half = max(z_data_half, extent) * 1.05

    # 3D north arrow (shaft + cone arrowhead), placed at the back-top-left
    # corner of the scene: x = -extent (lowest X), z near z_half (highest
    # Z), shaft running in +Y so the tip points toward the highest Y.
    arrow_color = "black"
    arrow_x = -extent
    arrow_z = z_half * 0.95
    arrow_shaft_y_start = extent * 0.40
    arrow_shaft_y_end = extent * 0.70
    arrow_size = extent * 0.18
    fig.add_trace(
        go.Scatter3d(
            x=[arrow_x, arrow_x],
            y=[arrow_shaft_y_start, arrow_shaft_y_end],
            z=[arrow_z, arrow_z],
            mode="lines",
            line=dict(color=arrow_color, width=10),
            name="north",
            showlegend=False,
            hoverinfo="skip",
        )
    )
    fig.add_trace(
        go.Cone(
            x=[arrow_x],
            y=[arrow_shaft_y_end],
            z=[arrow_z],
            u=[0.0],
            v=[arrow_size],
            w=[0.0],
            sizemode="absolute",
            sizeref=arrow_size,
            anchor="tail",
            showscale=False,
            colorscale=[[0, arrow_color], [1, arrow_color]],
            name="north",
            showlegend=False,
            hoverinfo="skip",
        )
    )

    fig.update_layout(
        scene=dict(
            xaxis=dict(
                title="X (East, km)", range=[-extent, extent], autorange=False
            ),
            yaxis=dict(
                title="Y (North, km)", range=[-extent, extent], autorange=False
            ),
            zaxis=dict(
                title="Z (km)", range=[-z_half, z_half], autorange=False
            ),
            # Data aspect: the scene proportions follow the explicit
            # axis ranges. With z half-width >= x/y half-width, the
            # scene will appear taller than wide for high-amplitude
            # folds, which is geologically faithful (no vertical
            # exaggeration).
            aspectmode="data",
            # Explicit scene.uirevision (in addition to the layout-level
            # one below) so 3D camera/pan/zoom state survives toggling
            # the drill-core enable and layer-surfaces visibility
            # checkboxes, which both change the figure's trace list.
            uirevision="3d-scene-locked",
            # Pin a default camera so the first render is deterministic
            # and uirevision has something concrete to preserve when the
            # user pans or rotates.
            camera=dict(eye=dict(x=1.4, y=1.4, z=1.0)),
            annotations=[
                dict(
                    showarrow=False,
                    x=arrow_x,
                    y=extent * 0.95,
                    z=arrow_z,
                    text="<b>N</b>",
                    font=dict(size=20, color="black"),
                    bgcolor="rgba(255, 255, 255, 0.85)",
                    bordercolor="black",
                    borderwidth=1,
                    borderpad=3,
                )
            ],
        ),
        # Preserve camera position across reruns when only visibility
        # toggles change; the value is constant so user pan/rotate is kept.
        uirevision="3d-scene-locked",
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
        xaxis=dict(title="X (East, km)", scaleanchor="y", scaleratio=1),
        yaxis=dict(title="Y (North, km)"),
        # Generous top margin so the heatmap aligns vertically with the 3D
        # scene next to it (Plotly's 3D scene reserves more top padding than
        # a default heatmap, otherwise the 2D content sits visibly higher).
        margin=dict(l=10, r=10, t=50, b=10),
        height=420,
        # Pin the legend inside the plot area at the top right. Without
        # this, adding drill-core overlay traces would shrink the plot
        # area to make room for an outside legend and the 2D map would
        # visually shrink whenever drill core is enabled.
        legend=dict(
            yanchor="top",
            y=0.98,
            xanchor="right",
            x=0.98,
            bgcolor="rgba(255, 255, 255, 0.85)",
            bordercolor="black",
            borderwidth=1,
            font=dict(size=11),
        ),
        annotations=[
            dict(
                xref="paper",
                yref="paper",
                x=0.05,
                y=0.95,
                text="<b>↑ N</b>",
                showarrow=False,
                font=dict(size=22, color="black"),
                bgcolor="rgba(255, 255, 255, 0.85)",
                bordercolor="black",
                borderwidth=1,
                borderpad=4,
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
    inside_mask: NDArray[np.bool_] | None = None,
    fade_opacity: float = 0.15,
) -> list[go.Surface]:
    """Plotly Surface traces for a drill core embedded in the 3D layer
    stack. Uses the same discrete colorscale as the 2D map and the layer
    surfaces, so colors match across all three views.

    When `inside_mask` is given (a 1D bool array of length `n_axial`),
    the cylinder is split into segments along its axial direction.
    Rows where the mask is True render at full opacity (the cylinder is
    sampling the model's original layer stack); rows where the mask is
    False render at `fade_opacity` (the cylinder is in the model's
    periodic continuation, above or below the original layer stack).
    When `fade_opacity <= 0`, outside segments are dropped entirely
    rather than rendered as invisible traces. When `inside_mask` is
    None, returns a single full-opacity trace.

    Returns a list of 0 to 3 Surface traces. The Streamlit app composes
    them onto the cached layer-stack figure with `add_trace` so changes
    to drill-core parameters do not invalidate the layer-surface cache.
    """
    n_colors = len(_LAYER_COLORS)
    common = dict(
        colorscale=_discrete_colorscale(_LAYER_COLORS),
        cmin=-0.5,
        cmax=n_colors - 0.5,
        showscale=False,
    )

    def _outside_trace(Xs, Ys, Zs, ls):
        if fade_opacity <= 0.0:
            return None
        return go.Surface(
            x=Xs, y=Ys, z=Zs,
            surfacecolor=ls,
            opacity=fade_opacity,
            name="drill core (outside layer stack)",
            **common,
        )

    def _inside_trace(Xs, Ys, Zs, ls):
        return go.Surface(
            x=Xs, y=Ys, z=Zs,
            surfacecolor=ls,
            opacity=1.0,
            name="drill core",
            **common,
        )

    if inside_mask is None:
        return [_inside_trace(X, Y, Z, layer_idx)]

    inside = np.asarray(inside_mask, dtype=bool)

    if bool(inside.all()):
        return [_inside_trace(X, Y, Z, layer_idx)]
    if not bool(inside.any()):
        outside = _outside_trace(X, Y, Z, layer_idx)
        return [outside] if outside is not None else []

    # Split into contiguous inside/outside segments. Each segment
    # includes one row of overlap with the next so the rendering stays
    # visually continuous (Plotly Surface needs at least 2 axial rows).
    state_changes = np.flatnonzero(np.diff(inside.astype(int))) + 1
    boundaries = np.concatenate(([0], state_changes, [len(inside)]))

    traces: list[go.Surface] = []
    for i in range(len(boundaries) - 1):
        start = int(boundaries[i])
        end = int(boundaries[i + 1])
        is_inside = bool(inside[start])
        end_with_overlap = min(end + 1, len(inside))
        Xs = X[start:end_with_overlap]
        Ys = Y[start:end_with_overlap]
        Zs = Z[start:end_with_overlap]
        ls = layer_idx[start:end_with_overlap]
        if Xs.shape[0] < 2:
            continue
        if is_inside:
            traces.append(_inside_trace(Xs, Ys, Zs, ls))
        else:
            outside = _outside_trace(Xs, Ys, Zs, ls)
            if outside is not None:
                traces.append(outside)
    return traces


def fig_2d_drill_core_unrolled(
    layer_idx: NDArray[np.integer],
    length: float,
    inside_mask: NDArray[np.bool_] | None = None,
    fade_opacity: float = 0.15,
) -> go.Figure:
    """Heatmap of the drill core unrolled flat: depth from collar on the
    Y axis (0 at top, increasing downward to `length`), circumferential
    angle on the X axis (0 to 360 degrees). Reuses the same `layer_idx`
    array as the 3D trace, and the same discrete palette.

    When `inside_mask` is given, the strip is split into two heatmap
    layers: cells where the mask is True render at full opacity
    (cylinder is sampling the model's original layer stack); cells
    where the mask is False render at `fade_opacity` (cylinder is in
    the model's periodic continuation, above or below the original
    layer stack). Each layer NaN-masks the other so cells appear in
    exactly one of the two traces.

    `inside_mask` may be either 1D of shape `(n_axial,)` (one
    inside/outside decision per axial row, broadcast to all theta
    columns; gives a flat horizontal boundary) or 2D of shape
    `(n_axial, n_circ)` (per-cell decision; the boundary follows the
    layer-stack topology, which is the geometrically faithful choice).
    """
    n_axial, n_circ = layer_idx.shape
    theta_deg = np.linspace(0.0, 360.0, n_circ)
    depth = np.linspace(0.0, length, n_axial)
    n_colors = len(_LAYER_COLORS)
    z_float = layer_idx.astype(float)

    common = dict(
        x=theta_deg,
        y=depth,
        colorscale=_discrete_colorscale(_LAYER_COLORS),
        zmin=-0.5,
        zmax=n_colors - 0.5,
        showscale=False,
    )

    fig = go.Figure()
    if inside_mask is None:
        fig.add_trace(go.Heatmap(z=z_float, **common))
    else:
        inside = np.asarray(inside_mask, dtype=bool)
        # Broadcast 1D (per-row) to 2D (per-cell) so downstream code
        # handles both shapes uniformly.
        if inside.ndim == 1:
            inside_2d = np.broadcast_to(inside[:, None], layer_idx.shape)
        else:
            inside_2d = inside
        if bool(inside_2d.all()):
            fig.add_trace(go.Heatmap(z=z_float, **common))
        elif not bool(inside_2d.any()):
            if fade_opacity > 0.0:
                fig.add_trace(
                    go.Heatmap(z=z_float, opacity=fade_opacity, **common)
                )
        else:
            z_inside = np.where(inside_2d, z_float, np.nan)
            fig.add_trace(go.Heatmap(z=z_inside, **common))
            if fade_opacity > 0.0:
                z_outside = np.where(~inside_2d, z_float, np.nan)
                fig.add_trace(
                    go.Heatmap(z=z_outside, opacity=fade_opacity, **common)
                )

    fig.update_layout(
        xaxis=dict(title="θ around core (°)"),
        # Depth increases downward; reverse the y-axis so the collar is on top.
        yaxis=dict(title="depth from collar (km)", autorange="reversed"),
        margin=dict(l=10, r=10, t=30, b=40),
        height=520,
    )
    return fig


def drill_core_map_overlay_traces(p: DrillCoreParameters) -> list[go.Scatter]:
    """Traces that draw the drill core on the 2D interference map at z=0:
    a collar marker, the projected line from collar to toe, and a toe
    marker. For nearly-vertical cores (plunge >= 89 degrees) returns just
    the collar marker (the projected line collapses to a point).

    All traces opt into the legend so the user can identify the black
    line as the drill plan and the markers as collar/toe.
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
        name="drill collar",
        showlegend=True,
        hoverinfo="skip",
    )

    if p.plunge_deg >= 89.0:
        return [collar_marker]

    line = go.Scatter(
        x=[p.collar_x, toe_x],
        y=[p.collar_y, toe_y],
        mode="lines",
        line=dict(color="black", width=2),
        name="drill plan (collar to toe)",
        showlegend=True,
        hoverinfo="skip",
    )
    toe_marker = go.Scatter(
        x=[toe_x],
        y=[toe_y],
        mode="markers",
        marker=dict(size=8, color="black", symbol="x"),
        name="drill toe",
        showlegend=True,
        hoverinfo="skip",
    )
    return [line, collar_marker, toe_marker]
