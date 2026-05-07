"""Streamlit app for the superposed-folds toolkit.

Run locally with:

    uv run streamlit run streamlit_app.py
"""

from __future__ import annotations

import hashlib
import inspect

import plotly.graph_objects as go
import streamlit as st

from superposed_folds import (
    ALL_PRESETS,
    END_MEMBERS,
    OBLIQUE_ROTATION,
    PARALLEL_ROTATION,
    DrillCoreParameters,
    FoldParameters,
    classify_nearest,
    drill_core_map_overlay_traces,
    fig_2d_drill_core_unrolled,
    fig_2d_interference,
    fig_3d_drill_core_trace,
    fig_3d_stack,
    fig_stereonet,
    sample_layers_on_cylinder,
)
from superposed_folds import cylinder as _cylinder_module
from superposed_folds import viz as _viz_module
from superposed_folds.references import REFERENCES

# Cache-busting fingerprint: changes whenever viz.py or cylinder.py source
# changes. Streamlit's `@st.cache_data` keys cached entries by the wrapper
# function's own source code plus the arg values, so edits to functions
# called from the wrapper (e.g. `fig_2d_interference` in viz.py or
# `sample_layers_on_cylinder` in cylinder.py) don't invalidate the cache
# on their own, including when their return shape changes. Passing this
# fingerprint as an argument makes any change to either module invalidate
# every cached figure or data tuple on the next module reload.
_VIZ_SOURCE_FINGERPRINT = hashlib.md5(
    inspect.getsource(_viz_module).encode("utf-8")
    + inspect.getsource(_cylinder_module).encode("utf-8")
).hexdigest()

st.set_page_config(page_title="Superposed Folds", layout="wide")

st.markdown(
    """
    # Superposed Folds: interactive playground

    A Python port of Martin Schöpfer's UCD MATLAB resource, implementing the
    Ramsay & Lisle (2000) plane-strain superposed-folding equations. Pick a
    Grasemann (2004) preset on the left or move the sliders to explore custom
    configurations. The 3D view shows a stack of horizons; the 2D map below
    shows the interference pattern at z = 0.
    """
)

# ----- Preset state -------------------------------------------------------

_PRESET_BY_NAME = {p.name: p for p in ALL_PRESETS}
_PRESET_GROUPS = {
    "End members": [p.name for p in END_MEMBERS],
    "Parallel rotation": [p.name for p in PARALLEL_ROTATION],
    "Oblique rotation": [p.name for p in OBLIQUE_ROTATION],
}


def _set_sliders_from_preset(preset_name: str) -> None:
    """Snap every slider to the chosen preset's values."""
    p = _PRESET_BY_NAME[preset_name]
    st.session_state["A1"] = p.f1.A
    st.session_state["B1"] = p.f1.B
    st.session_state["A2"] = p.f2.A
    st.session_state["B2"] = p.f2.B
    st.session_state["dipdir2"] = p.f2.dip_dir
    st.session_state["dip2"] = p.f2.dip
    st.session_state["rake2"] = p.f2.rake


if "preset_name" not in st.session_state:
    _initial = "Type 1 (dome-and-basin)"
    st.session_state["preset_name"] = _initial
    _set_sliders_from_preset(_initial)


_DRILL_CORE_DEFAULTS: dict[str, float | int | bool] = {
    "drill_core_enabled": False,
    "show_layer_surfaces": True,
    "collar_x": 0.0,
    "collar_y": 0.0,
    "collar_z": 2.5,
    "azimuth": 0.0,
    "plunge": 90.0,
    "core_length": 5.0,
    "core_diameter": 0.4,
}

# Hardcoded resolution for the 2D map and the unrolled core section.
# Both used to be user-controlled sliders; we settled on 400 as a single
# fixed value that gives a smooth render without being too slow.
_MAP_RESOLUTION = 400
_CORE_RESOLUTION = 400
for _key, _value in _DRILL_CORE_DEFAULTS.items():
    if _key not in st.session_state:
        st.session_state[_key] = _value


def _on_preset_change() -> None:
    _set_sliders_from_preset(st.session_state["preset_name"])


# ----- Cached figure builders ---------------------------------------------

@st.cache_data(show_spinner=False, max_entries=128)
def _cached_fig_3d_layers(
    f1: FoldParameters, f2: FoldParameters, _viz_fingerprint: str
):
    return fig_3d_stack(f1, f2, n_grid=48)


@st.cache_data(show_spinner=False, max_entries=128)
def _cached_fig_2d_map(
    f1: FoldParameters, f2: FoldParameters, n_grid: int, _viz_fingerprint: str
):
    return fig_2d_interference(f1, f2, n_grid=n_grid)


@st.cache_data(show_spinner=False, max_entries=128)
def _cached_fig_stereonet(
    f1: FoldParameters, f2: FoldParameters, _viz_fingerprint: str
):
    return fig_stereonet(f1, f2)


@st.cache_data(show_spinner=False, max_entries=128)
def _cached_drill_core_data(
    f1: FoldParameters,
    f2: FoldParameters,
    core: DrillCoreParameters,
    n_layers: int,
    extent: float,
    _viz_fingerprint: str,
):
    """Cached: world-frame X, Y, Z and layer indices on the drill-core
    surface. Cache key includes both fold parameters and the full drill-core
    parametrization, so changing F1/F2 sliders or any drill-core slider
    correctly invalidates the cache while leaving the layer-surface cache
    intact."""
    return sample_layers_on_cylinder(
        core, f1, f2, n_layers=n_layers, extent=extent
    )




# ----- Sidebar (outside any fragment) -------------------------------------
#
# Preset picker on top; static content (classification + refs) gets filled
# in inside the fragment. Sidebar reruns when a preset is picked (full page
# rerun); does not flicker on slider movement.

with st.sidebar:
    st.subheader("Preset")
    grouped_names: list[str] = [n for names in _PRESET_GROUPS.values() for n in names]
    st.selectbox(
        "Choose a preset",
        options=grouped_names,
        key="preset_name",
        on_change=_on_preset_change,
        label_visibility="collapsed",
    )
    _sidebar_classification = st.empty()
    _sidebar_references = st.empty()


# ----- Main fragment: sliders + figures -----------------------------------
#
# Sliders live here (not in sidebar) so they can be inside @st.fragment.
# Streamlit forbids st.sidebar inside fragments; this is the documented
# workaround. Slider movement reruns only this function; the page header
# and sidebar preset picker stay put.

@st.fragment
def _interactive_panel() -> None:
    with st.expander("Parameters", expanded=True):
        col_a, col_b, col_c, col_d = st.columns(4)
        with col_a:
            st.slider("F1 amplitude (A1)", 0.0, 6.0, step=0.1, key="A1")
            st.slider("F1 stretch (B1)", 0.5, 2.0, step=0.05, key="B1")
        with col_b:
            st.slider("F2 amplitude (A2)", 0.0, 6.0, step=0.1, key="A2")
            st.slider("F2 stretch (B2)", 0.5, 2.0, step=0.05, key="B2")
        with col_c:
            st.slider("F2 dip direction (°)", 0.0, 360.0, step=1.0, key="dipdir2")
            st.slider("F2 axial-plane dip (°)", 0.0, 90.0, step=1.0, key="dip2")
        with col_d:
            st.slider("F2 rake (axis pitch, °)", -90.0, 90.0, step=1.0, key="rake2")

    with st.expander("Drill core", expanded=False):
        dc_toggle_a, dc_toggle_b = st.columns(2)
        with dc_toggle_a:
            st.checkbox(
                "Enable drill core",
                key="drill_core_enabled",
                help=(
                    "The cylinder samples the layered fold model anywhere "
                    "you place it. Parts that lie above or below the "
                    "visualized horizons are rendered at reduced opacity "
                    "to flag they are showing the model's periodic "
                    "continuation rather than crossing a drawn surface."
                ),
            )
        with dc_toggle_b:
            st.checkbox(
                "Show layer surfaces in 3D viewer",
                key="show_layer_surfaces",
                help=(
                    "Hide the folded layer surfaces in the 3D viewer to "
                    "see the drill core on its own. Has no effect when "
                    "the drill core is disabled."
                ),
            )
        collar_cols = st.columns(3)
        with collar_cols[0]:
            st.slider("Collar X (km)", -5.0, 5.0, step=0.1, key="collar_x")
        with collar_cols[1]:
            st.slider("Collar Y (km)", -5.0, 5.0, step=0.1, key="collar_y")
        with collar_cols[2]:
            st.slider("Collar Z (km)", -5.0, 5.0, step=0.1, key="collar_z")
        orient_cols = st.columns(4)
        with orient_cols[0]:
            st.slider("Azimuth (°)", 0.0, 360.0, step=1.0, key="azimuth")
        with orient_cols[1]:
            st.slider("Plunge (°)", 0.0, 90.0, step=1.0, key="plunge")
        with orient_cols[2]:
            st.slider(
                "Core length (km)", 0.5, 10.0, step=0.1, key="core_length"
            )
        with orient_cols[3]:
            st.slider(
                "Core diameter", 0.05, 1.5, step=0.05, key="core_diameter",
                help="Visualization probe diameter, not a real drill bit size.",
            )

    A1 = st.session_state["A1"]
    B1 = st.session_state["B1"]
    A2 = st.session_state["A2"]
    B2 = st.session_state["B2"]
    dipdir2 = st.session_state["dipdir2"]
    dip2 = st.session_state["dip2"]
    rake2 = st.session_state["rake2"]

    f1 = FoldParameters(A=A1, B=B1, dip_dir=0.0, dip=90.0, rake=0.0)
    f2 = FoldParameters(A=A2, B=B2, dip_dir=dipdir2, dip=dip2, rake=rake2)

    drill_core_enabled = bool(st.session_state["drill_core_enabled"])
    if drill_core_enabled:
        core = DrillCoreParameters(
            collar_x=float(st.session_state["collar_x"]),
            collar_y=float(st.session_state["collar_y"]),
            collar_z=float(st.session_state["collar_z"]),
            azimuth_deg=float(st.session_state["azimuth"]),
            plunge_deg=float(st.session_state["plunge"]),
            length=float(st.session_state["core_length"]),
            diameter=float(st.session_state["core_diameter"]),
            n_axial=_CORE_RESOLUTION,
            n_circ=_CORE_RESOLUTION,
        )
        Xc, Yc, Zc, layer_idx_c, Z0c = _cached_drill_core_data(
            f1, f2, core, 5, 5.0, _VIZ_SOURCE_FINGERPRINT
        )
    else:
        core = None
        Xc = Yc = Zc = layer_idx_c = Z0c = None

    matched = classify_nearest(f1, f2, tol=0.05)
    if matched is not None:
        type_label = matched.name
        explainer = matched.explainer
    else:
        # Off-preset: tell the user what they're closest to so they can
        # snap back if they want.
        nearest = classify_nearest(f1, f2, tol=float("inf"))
        nearest_name = nearest.name if nearest else "(none)"
        type_label = "Custom configuration"
        explainer = (
            "You've moved the sliders away from a canonical Grasemann (2004) "
            f"preset. Closest preset: **{nearest_name}**. Pick it from the "
            "dropdown above to snap back."
        )

    # Update the sidebar placeholders that were created outside this fragment.
    # The .empty() containers from before are addressable from here; writing
    # to them replaces their contents on every fragment rerun.
    with _sidebar_classification.container():
        st.markdown(f"### {type_label}")
        st.write(explainer)
        f1_orientation = FoldParameters(
            A=3.0, B=1.0, dip_dir=0.0, dip=90.0, rake=0.0
        )
        f2_orientation = FoldParameters(
            A=3.0, B=1.0, dip_dir=dipdir2, dip=dip2, rake=rake2
        )
        st.pyplot(
            _cached_fig_stereonet(
                f1_orientation, f2_orientation, _VIZ_SOURCE_FINGERPRINT
            ),
            width="stretch",
        )
    with _sidebar_references.container():
        st.markdown("### References")
        for r in REFERENCES:
            if r["url"]:
                st.markdown(f"- [{r['label']}]({r['url']})")
            else:
                st.markdown(f"- {r['label']}")

    col_3d, col_2d = st.columns([3, 2])
    with col_3d:
        st.markdown(
            "<h5 style='text-align: center; margin-bottom: 0.25rem'>"
            "3D fold model</h5>",
            unsafe_allow_html=True,
        )
        fig_3d = _cached_fig_3d_layers(f1, f2, _VIZ_SOURCE_FINGERPRINT)
        if drill_core_enabled:
            # Shallow-copy the cached figure before mutating, otherwise
            # add_trace and trace.visible writes would mutate the cached
            # object and pollute future cache hits.
            fig_3d = go.Figure(fig_3d)
            if not bool(st.session_state["show_layer_surfaces"]):
                # Hide only the layer Surface traces; keep the north
                # arrow (Scatter3d shaft + Cone head) visible. The
                # drill-core surface is added below and so isn't
                # affected here.
                for trace in fig_3d.data:
                    if isinstance(trace, go.Surface):
                        trace.visible = False
            # A cylinder vertex is "in the original layer stack" iff its
            # initial-z value rounds to one of the n_layers bin centers
            # (i.e. its raw layer index falls in [0, n_layers-1] before the
            # mod-wrap). That range corresponds to Z0 within
            # [-extent/2 - spacing/2, extent/2 + spacing/2]. Outside this
            # band, the cylinder is sampling the model's periodic
            # continuation; fade it.
            n_layers = 5
            extent = 5.0
            spacing = extent / (n_layers - 1) if n_layers > 1 else extent
            half_band = extent / 2.0 + spacing / 2.0
            in_stack_per_vertex = (Z0c >= -half_band) & (Z0c <= half_band)
            # 3D Surface only supports per-trace opacity, so collapse the
            # per-vertex mask to a per-row decision (majority vote across
            # theta) for the 3D split.
            inside_row = in_stack_per_vertex.mean(axis=1) >= 0.5
            for core_trace in fig_3d_drill_core_trace(
                Xc, Yc, Zc, layer_idx_c, inside_mask=inside_row
            ):
                fig_3d.add_trace(core_trace)
        st.plotly_chart(fig_3d, width="stretch", key="fig3d")
    with col_2d:
        st.markdown(
            "<h5 style='text-align: center; margin-bottom: 0.25rem'>"
            "Map view: 2D interference pattern (z = 0)</h5>",
            unsafe_allow_html=True,
        )
        fig_2d = _cached_fig_2d_map(
            f1, f2, _MAP_RESOLUTION, _VIZ_SOURCE_FINGERPRINT
        )
        if drill_core_enabled:
            fig_2d = go.Figure(fig_2d)
            for trace in drill_core_map_overlay_traces(core):
                fig_2d.add_trace(trace)
        st.plotly_chart(fig_2d, width="stretch", key="fig2d")

    if drill_core_enabled:
        # Constrain the unrolled strip to the middle ~50% of page width so
        # it reads as a vertical column rather than a wide thin band.
        _, col_unroll, _ = st.columns([1, 2, 1])
        with col_unroll:
            st.markdown(
                "<h5 style='text-align: center; margin-bottom: 0.25rem'>"
                "Unrolled drill-core section</h5>",
                unsafe_allow_html=True,
            )
            # Pass the 2D per-vertex mask so the unrolled strip's fade
            # boundary follows the layer-stack topology (wavy) instead
            # of a flat horizontal cutoff.
            st.plotly_chart(
                fig_2d_drill_core_unrolled(
                    layer_idx_c,
                    core.length,
                    inside_mask=in_stack_per_vertex,
                ),
                width="stretch",
                key="figunrolled",
            )


_interactive_panel()
