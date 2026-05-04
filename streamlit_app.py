"""Streamlit app for the superposed-folds toolkit.

Run locally with:

    uv run streamlit run streamlit_app.py
"""

from __future__ import annotations

import streamlit as st

from superposed_folds import (
    ALL_PRESETS,
    END_MEMBERS,
    OBLIQUE_ROTATION,
    PARALLEL_ROTATION,
    FoldParameters,
    classify_nearest,
    fig_2d_interference,
    fig_3d_stack,
    fig_stereonet,
)
from superposed_folds.references import REFERENCES

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


def _on_preset_change() -> None:
    _set_sliders_from_preset(st.session_state["preset_name"])


# ----- Cached figure builders ---------------------------------------------

@st.cache_data(show_spinner=False, max_entries=128)
def _cached_fig_3d(
    A1: float, B1: float, dipdir2: float, dip2: float, rake2: float, A2: float, B2: float,
):
    f1 = FoldParameters(A=A1, B=B1, dip_dir=0.0, dip=90.0, rake=0.0)
    f2 = FoldParameters(A=A2, B=B2, dip_dir=dipdir2, dip=dip2, rake=rake2)
    return fig_3d_stack(f1, f2, n_grid=48)


@st.cache_data(show_spinner=False, max_entries=128)
def _cached_fig_2d(
    A1: float, B1: float, dipdir2: float, dip2: float, rake2: float, A2: float, B2: float,
):
    f1 = FoldParameters(A=A1, B=B1, dip_dir=0.0, dip=90.0, rake=0.0)
    f2 = FoldParameters(A=A2, B=B2, dip_dir=dipdir2, dip=dip2, rake=rake2)
    return fig_2d_interference(f1, f2, n_grid=120)


@st.cache_data(show_spinner=False, max_entries=128)
def _cached_fig_stereonet(dipdir2: float, dip2: float, rake2: float):
    f1 = FoldParameters(A=3.0, B=1.0, dip_dir=0.0, dip=90.0, rake=0.0)
    f2 = FoldParameters(A=3.0, B=1.0, dip_dir=dipdir2, dip=dip2, rake=rake2)
    return fig_stereonet(f1, f2)


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

    A1 = st.session_state["A1"]
    B1 = st.session_state["B1"]
    A2 = st.session_state["A2"]
    B2 = st.session_state["B2"]
    dipdir2 = st.session_state["dipdir2"]
    dip2 = st.session_state["dip2"]
    rake2 = st.session_state["rake2"]

    f1 = FoldParameters(A=A1, B=B1, dip_dir=0.0, dip=90.0, rake=0.0)
    f2 = FoldParameters(A=A2, B=B2, dip_dir=dipdir2, dip=dip2, rake=rake2)

    matched = classify_nearest(f1, f2, tol=0.05)
    type_label = matched.name if matched else "Custom"
    explainer = (
        matched.explainer
        if matched
        else "Slider state does not match a canonical Grasemann (2004) configuration."
    )

    # Update the sidebar placeholders that were created outside this fragment.
    # The .empty() containers from before are addressable from here; writing
    # to them replaces their contents on every fragment rerun.
    with _sidebar_classification.container():
        st.markdown(f"### {type_label}")
        st.write(explainer)
        st.pyplot(_cached_fig_stereonet(dipdir2, dip2, rake2), width="stretch")
    with _sidebar_references.container():
        st.markdown("### References")
        for r in REFERENCES:
            if r["url"]:
                st.markdown(f"- [{r['label']}]({r['url']})")
            else:
                st.markdown(f"- {r['label']}")

    col_3d, col_2d = st.columns(2)
    with col_3d:
        st.plotly_chart(
            _cached_fig_3d(A1, B1, dipdir2, dip2, rake2, A2, B2),
            width="stretch",
            key="fig3d",
        )
    with col_2d:
        st.plotly_chart(
            _cached_fig_2d(A1, B1, dipdir2, dip2, rake2, A2, B2),
            width="stretch",
            key="fig2d",
        )


_interactive_panel()
