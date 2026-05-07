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
    DrillCoreParameters,
    END_MEMBERS,
    OBLIQUE_ROTATION,
    PARALLEL_ROTATION,
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
from superposed_folds import viz as _viz_module
from superposed_folds.references import REFERENCES

# Cache-busting fingerprint: changes whenever viz.py's source changes.
# Streamlit's `@st.cache_data` keys cached entries by the wrapper function's
# own source code plus the arg values, so edits to functions called from the
# wrapper (e.g. `fig_2d_interference` in viz.py) don't invalidate the cache
# on their own. Passing this fingerprint as an argument makes any change to
# viz.py invalidate every cached figure on the next module reload.
_VIZ_SOURCE_FINGERPRINT = hashlib.md5(
    inspect.getsource(_viz_module).encode("utf-8")
).hexdigest()

st.set_page_config(page_title="Superposed Folds", layout="wide")

st.markdown(
    """
    # Superposed Folds — interactive playground

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
    "collar_x": 0.0,
    "collar_y": 0.0,
    "collar_z": 2.5,
    "azimuth": 0.0,
    "plunge": 90.0,
    "core_length": 5.0,
    "core_diameter": 0.4,
    "map_resolution": 120,
}
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
# workaround. Slider movement reruns only this function — the page header
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
            st.slider(
                "Map resolution (px)",
                min_value=80,
                max_value=400,
                step=20,
                key="map_resolution",
            )

    with st.expander("Drill core", expanded=False):
        st.checkbox("Enable drill core", key="drill_core_enabled")
        dc_col_a, dc_col_b, dc_col_c = st.columns(3)
        with dc_col_a:
            st.slider("Collar X", -5.0, 5.0, step=0.1, key="collar_x")
            st.slider("Collar Y", -5.0, 5.0, step=0.1, key="collar_y")
            st.slider("Collar Z", -5.0, 5.0, step=0.1, key="collar_z")
        with dc_col_b:
            st.slider("Azimuth (°)", 0.0, 360.0, step=1.0, key="azimuth")
            st.slider("Plunge (°)", 0.0, 90.0, step=1.0, key="plunge")
        with dc_col_c:
            st.slider("Core length", 0.5, 10.0, step=0.1, key="core_length")
            st.slider(
                "Core diameter", 0.05, 1.5, step=0.05, key="core_diameter"
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
        )
        Xc, Yc, Zc, layer_idx_c = _cached_drill_core_data(
            f1, f2, core, 5, 5.0, _VIZ_SOURCE_FINGERPRINT
        )
    else:
        core = None
        Xc = Yc = Zc = layer_idx_c = None

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
            f"preset. Closest preset: **{nearest_name}** — pick it from the "
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
        fig_3d = _cached_fig_3d_layers(f1, f2, _VIZ_SOURCE_FINGERPRINT)
        if drill_core_enabled:
            # Shallow-copy the cached figure before mutating, otherwise
            # add_trace would mutate the cached object and pollute future
            # cache hits.
            fig_3d = go.Figure(fig_3d)
            fig_3d.add_trace(
                fig_3d_drill_core_trace(Xc, Yc, Zc, layer_idx_c)
            )
        st.plotly_chart(fig_3d, width="stretch", key="fig3d")
    with col_2d:
        fig_2d = _cached_fig_2d_map(
            f1, f2, int(st.session_state["map_resolution"]), _VIZ_SOURCE_FINGERPRINT
        )
        if drill_core_enabled:
            fig_2d = go.Figure(fig_2d)
            for trace in drill_core_map_overlay_traces(core):
                fig_2d.add_trace(trace)
        st.plotly_chart(fig_2d, width="stretch", key="fig2d")

    if drill_core_enabled:
        st.plotly_chart(
            fig_2d_drill_core_unrolled(layer_idx_c, core.length),
            width="stretch",
            key="figunrolled",
        )


_interactive_panel()
