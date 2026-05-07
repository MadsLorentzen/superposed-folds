# Drill-core simulation and resolution slider

**Status:** Approved design, ready for implementation plan.
**Branch:** `feat/drill-core`
**Date:** 2026-05-07

## Background

Martin Schöpfer (UCD Fault Analysis Group, original author of the MATLAB
papermodel resource this project ports) reviewed the deployed Streamlit app
and suggested two additions:

1. A resolution slider, because the 2D interference plot has visible
   jaggedness at the default `n_grid=120` and could be smoother.
2. A cylindrical "drill core" view, because real drill cores are
   cylindrical and that would add value to the tool. He had wanted to
   build this in the original MATLAB version but never got to it.

This document captures the design we agreed for both, plus the small
refactors we will land alongside them.

## Goals

- Let the user generate the same superposed-fold interference pattern on
  the curved surface of a cylindrical drill core, viewing it both
  3D-embedded in the existing layer stack and unrolled flat.
- Let the user smooth the 2D interference plot with a resolution slider.
- Make the geological orientation of the model legible, since adding an
  azimuth slider for the core requires a compass anchor in the views that
  do not currently have one.

## Non-goals

- No new dependencies. Cylinder math stays in numpy; figures reuse the
  existing plotly and matplotlib stacks.
- No new MATLAB parity tests. The drill core reuses `initial_z_at`, which
  is already parity-tested.
- No 1D core log view. A purely vertical 1D log throws away most of the
  interference signal and would not show what makes the core interesting.
- No end caps on the cylinder. Caps would just repeat a circular slice
  of the 2D map and add visual clutter.
- No refactor of `geometry.py`, `transforms.py`, `classification.py`, or
  `references.py`.
- No marimo or `archive/` cleanup unrelated to this feature.

## User-facing decisions

| Decision | Choice | Rationale |
|---|---|---|
| Drill-core views | Both 3D cylinder embedded in the layer stack and a separate unrolled 2D strip | Schöpfer wanted core context; the embedded cylinder gives spatial intuition, the unrolled view is the geologically meaningful artifact |
| Drill-core controls | Collar (x, y, z) + plunge + azimuth + length + diameter | Standard geological parametrization; reuses the dipdir/dip mental model already in the app |
| Resolution slider | One slider, 2D interference map only, range 80 to 400 with step 20, default 120 | The 2D map is where coarseness is visible; the 3D mesh at 48 is fine and bumping it is much more expensive |
| Orientation cues | Axis relabels (`X (East)`, `Y (North)`) plus a small "↑ N" annotation on the 2D map and a "N" text marker at the +Y wall of the 3D scene | Labels alone are easy to miss when dragging the azimuth slider; a discrete arrow draws the eye |
| Drill-core map overlay | Collar marker + line + toe marker projected onto the 2D map's z=0 plane | Standard drill-plan convention; makes both azimuth and plunge visible from the map. Vertical cores collapse to a single collar dot |
| UI layout | Resolution slider drops into the empty col_d row 2 of the existing Parameters expander. Drill-core controls live in their own collapsible expander, collapsed by default. Unrolled strip appears as a new full-width row below the existing 3D + 2D row, only when drill core is enabled | Default view stays identical to today; the feature reveals itself only when the user opts in |

## Module layout

```
src/superposed_folds/
  geometry.py        (unchanged)
  transforms.py      (unchanged)
  classification.py  (unchanged)
  cylinder.py        (new)
  viz.py             (refactor + 2 new figure builders + overlay helper)
  references.py      (unchanged)
  __init__.py        (re-export new public names)
```

### `cylinder.py` (new)

Owns three things:

#### `DrillCoreParameters` (frozen dataclass)

| Field | Type | Initial Streamlit value | Slider range | Meaning |
|---|---|---|---|---|
| `collar_x` | float | 0.0 | -5.0..+5.0 | World X of the borehole top |
| `collar_y` | float | 0.0 | -5.0..+5.0 | World Y of the borehole top |
| `collar_z` | float | 2.5 | -5.0..+5.0 | World Z of the borehole top (initial = top of the default layer stack) |
| `azimuth_deg` | float | 0.0 | 0..360 | Compass bearing the core trends toward |
| `plunge_deg` | float | 90.0 | 0..90 | Angle below horizontal (90 = straight down) |
| `length` | float | 5.0 | 0.5..10.0 | Core length in world units |
| `diameter` | float | 0.4 | 0.05..1.5 | Cylinder diameter |
| `n_axial` | int | 200 | (internal) | Samples along the core axis |
| `n_circ` | int | 200 | (internal) | Samples around the circumference |

The dataclass itself has all user-controlled fields *required* (no
defaults), matching `FoldParameters`. `n_axial` and `n_circ` have
defaults of 200 since they are internal-only. The "Initial Streamlit
value" column above shows what `st.session_state` is initialized to on
first run; the slider ranges are also UI-only. Numeric values are
written for the current model `extent = 5.0`; if `extent` ever becomes
configurable the Streamlit initialization scales these values rather
than the dataclass holding the dependency.

`n_axial` and `n_circ` are not user-facing in v1. We hard-code them and
revisit only if performance becomes a problem. Schöpfer's resolution
request was about the 2D map, not the core surface.

#### `cylinder_surface_points(p: DrillCoreParameters) -> tuple[X, Y, Z]`

Returns three `(n_axial, n_circ)` numpy arrays of world coordinates.

Construction:
1. Build the cylinder in a local frame where the core axis is along
   local +z' and the circle is in local x'-y'. With axial parameter
   `s in [0, length]` (depth into the hole, 0 at collar, `length` at
   toe) and angular parameter `theta in [0, 2*pi]`:
   - local point = (r * cos(theta), r * sin(theta), +s) where
     r = diameter / 2. The cylinder extends from local origin (collar)
     to local (0, 0, length) (toe) along +z'.
2. Rotate local frame to world. The local +z' axis must point along
   the core's down-plunge unit vector in world coordinates so that
   increasing `s` moves down-plunge from the collar:
   ```
   trend = (sin(azimuth) * cos(plunge),    # +X = East
            cos(azimuth) * cos(plunge),    # +Y = North
           -sin(plunge))                   # +Z = up; plunge points down
   ```
3. Choose `e_x_local` perpendicular to `trend` in the horizontal plane,
   pointing east-of-trend (deterministic up to a fixed convention; pinned
   by a test).
4. Rotate, then translate by collar.

The "east-of-trend" convention pins where theta=0 lands on the unrolled
view, so that the unrolled strip's horizontal axis has a consistent
geological meaning across different orientations.

Edge case: `plunge = 0` (horizontal core) is allowed. The east-of-trend
basis remains well-defined. The slider range is 0..90.

#### `sample_layers_on_cylinder(p, f1, f2, n_layers, extent) -> tuple[X, Y, Z, layer_idx]`

1. `X, Y, Z = cylinder_surface_points(p)`
2. `Z0 = initial_z_at(X, Y, Z, f1, f2)`
3. `layer_idx = layer_index_from_z(Z0, n_layers, extent)` (the helper
   extracted from `viz.py`).

Returns all four arrays so the figure builders can use them directly:
the 3D Surface trace needs (X, Y, Z, layer_idx); the unrolled 2D Heatmap
needs only `layer_idx` indexed by axial and circumferential parameters.

### `viz.py` (refactor + additions)

#### Refactor

- Extract the discrete layer-binning logic (currently inline at
  `viz.py:97-106`) into `layer_index_from_z(z_array, n_layers, extent)
  -> int_array`. This is the function that rounds Z values to integer
  layer indices and wraps them modulo `len(_LAYER_COLORS)`. The unrolled
  drill-core view reuses this function; that is why we extract it.
- Behavior is unchanged. The two existing tests we add for it pin the
  visual contract that 2D bands line up with 3D layers.

#### New: `fig_3d_drill_core_trace(X, Y, Z, layer_idx) -> go.Surface`

Returns a single Plotly trace, not a Figure. Takes the precomputed
arrays from `sample_layers_on_cylinder` so the same data tuple can feed
both the 3D and the unrolled views without recomputation. The Streamlit
app composes the trace onto the cached layer-stack figure with
`add_trace` so changes to drill-core parameters do not invalidate the
layer-surface cache.

The trace uses the same discrete colorscale as the 2D map so the
embedded core's bands match the 2D map's bands and the 3D layer stack's
bands.

#### New: `fig_2d_drill_core_unrolled(layer_idx, length) -> go.Figure`

Heatmap with axial depth on the Y axis (depth = 0 at collar, increasing
downward to `length`) and circumferential angle on the X axis (0..360
degrees). `layer_idx` is the same array used by the 3D trace. Same
discrete colorscale.

#### New: `drill_core_map_overlay_traces(p) -> list[go.Trace]`

Returns the (typically 2 or 3) traces that draw the drill core on the 2D
interference map: collar marker, projected line, toe marker. For
plunge >= 89 degrees, returns just the collar marker (the line collapses
to a point).

Composed onto the cached 2D map figure with `add_trace` outside the
cache, same pattern as the 3D cylinder.

#### Orientation cues

- `fig_3d_stack`: scene xaxis title `X (East)`, yaxis title `Y (North)`,
  zaxis title `Z`. Add a `scene.annotations` entry with text "N" at the
  +Y wall of the model (positioned at `(0, +extent, 0)`).
- `fig_2d_interference`: xaxis title `X (East)`, yaxis title `Y (North)`.
  Add a paper-coordinate annotation "↑ N" near the top-left corner.
- The stereonet already has compass labels. No change.

### `streamlit_app.py` (extended)

#### State additions

```
drill_core_enabled : bool   = False
collar_x, collar_y, collar_z = 0.0, 0.0, 2.5
azimuth, plunge              = 0.0, 90.0
core_length, core_diameter   = 5.0, 0.4
map_resolution               = 120
```

The drill-core fields are *not* touched by `_set_sliders_from_preset`;
they persist across preset changes since they are independent of the
F1/F2 fold parameters.

#### Layout

```
[ Header markdown                         ]   unchanged
[ Parameters expander, expanded by default ]
   col_a:  F1 amplitude, F1 stretch
   col_b:  F2 amplitude, F2 stretch
   col_c:  F2 dip direction, F2 axial-plane dip
   col_d:  F2 rake, Map resolution            <- new slider in row 2
[ Drill core expander, collapsed by default ]   new
   - Enable drill core (checkbox)
   - collar_x, collar_y, collar_z sliders
   - azimuth, plunge sliders
   - length, diameter sliders
[ 3D stack | 2D map ]                         existing columns; both
                                              now carry orientation cues
                                              and (when enabled)
                                              drill-core overlays
                                              composed on top
[ Unrolled drill-core strip, full-width ]     new; only rendered when
                                              drill_core_enabled
```

#### Cached builders (refactored to take dataclasses)

```
_cached_fig_3d_layers(f1, f2, _viz_fingerprint)              -> Figure
_cached_fig_2d_map(f1, f2, n_grid, _viz_fingerprint)         -> Figure
_cached_fig_stereonet(f1, f2, _viz_fingerprint)              -> Figure
_cached_drill_core_data(f1, f2, core, n_layers, extent,
                        _viz_fingerprint)                    -> (X, Y, Z, layer_idx)   # new
```

Frozen dataclasses hash cleanly under `@st.cache_data`. The drill-core
cache returns the sampled-and-binned data tuple, not a figure, so both
the 3D trace builder and the unrolled-strip figure builder consume the
same precomputed arrays.

#### Composition pattern

For the 3D scene:

```
fig = _cached_fig_3d_layers(f1, f2, fingerprint)             # cached on F1/F2
if drill_core_enabled:
    X, Y, Z, layer_idx = _cached_drill_core_data(
        f1, f2, core, n_layers, extent, fingerprint
    )                                                         # cached on (F1, F2, core)
    fig = go.Figure(fig)                                      # shallow copy
    fig.add_trace(fig_3d_drill_core_trace(X, Y, Z, layer_idx))
st.plotly_chart(fig, ...)
```

For the 2D map, similar: call `_cached_fig_2d_map`, copy it, then
`add_trace` the collar/line/toe overlay traces from
`drill_core_map_overlay_traces(core)`. The unrolled strip figure is
built directly from the same `(layer_idx, core.length)` tuple.

This keeps the layer surfaces cached on F1/F2 only, while drill-core
changes only do the incremental work of resampling the cylinder and
adding the new traces.

## Test plan

### `tests/test_cylinder.py` (new)

| Test | What it pins down |
|---|---|
| `test_drill_core_parameters_defaults` | Frozen dataclass round-trip, defaults, sensible bounds |
| `test_cylinder_surface_points_shape` | Returns three `(n_axial, n_circ)` arrays |
| `test_vertical_core_at_origin` | Plunge=90, collar=(0,0,0): all points within `diameter/2` of axis, z in [-length, 0] |
| `test_horizontal_core_along_north` | Plunge=0, azimuth=0: axis is +Y; points lie within `diameter/2` of the y-axis, y in [0, length] |
| `test_azimuth_rotation` | Plunge=0, azimuth=90 vs 0: second cylinder is the first rotated 90 degrees around z |
| `test_unrolled_theta_zero_is_east_of_trend` | Pins the east-of-trend convention so we do not silently flip it |
| `test_sample_layers_on_cylinder_uses_initial_z_at` | One known sample point's `initial_z` matches the value returned in `Z0` |
| `test_sample_layers_on_cylinder_layer_indices_in_range` | All `layer_idx` values are in `[0, n_layers)` |
| `test_plunge_zero_does_not_blow_up` | Horizontal core does not NaN |

### `tests/test_viz_helpers.py` (new)

| Test | What it pins down |
|---|---|
| `test_layer_index_from_z_periodic_wrap` | Z values across multiple periods all wrap into `[0, n_colors)` |
| `test_layer_index_from_z_centers_align_with_make_layer_stack` | Z values exactly at layer-center heights bin to 0, 1, 2, ... in order. Pins the visual contract that 2D bands line up with 3D layers |

### Existing test files

- `tests/test_geometry.py`: no change.
- `tests/test_classification.py`: no change.
- `tests/test_transforms.py`: no change.
- `tests/test_matlab_parity.py`: no change. The drill-core path reuses
  `initial_z_at`, which is already parity-tested.

### What we do not unit-test

- `streamlit_app.py`. Consistent with current repo posture; verified by
  manual QA in the running app.
- The new figure builders (Plotly trace constructors). Visual code is
  verified by eye, same as the existing `fig_3d_stack` etc.

### Manual QA gate before merge

Launch `uv run streamlit run streamlit_app.py`. Verify:

1. Default view is identical to current main (drill core off, all other
   panels look the same).
2. Resolution slider visibly smooths the 2D map.
3. Orientation cues render on both 2D map and 3D scene.
4. Toggling drill core on:
   - Cylinder appears embedded in the 3D scene.
   - Map-view overlay (collar, line, toe) appears on the 2D map.
   - Unrolled strip row appears below.
5. Dragging F1/F2 sliders updates the cylinder coloring and the unrolled
   strip in lockstep with the layer stack and the 2D map.
6. Dragging drill-core sliders updates only the drill-core artifacts.
7. Vertical core (plunge=90): map overlay shows just the collar dot.
8. Horizontal core (plunge=0): map overlay shows a full line; cylinder
   visibly horizontal in 3D.
9. Switching presets does not reset drill-core slider values.

## Branch and commit plan

Branch: `feat/drill-core` from current `main` (commit `0eec6a3`).

Commits, each leaving the branch in a working state:

1. `docs: add design spec for drill-core feature and resolution slider`
   This file. Sets context for everything that follows.

2. `refactor: extract layer_index_from_z helper; cached builders take dataclasses`
   Pure refactor, no app behavior change. Adds
   `tests/test_viz_helpers.py`. `uv run pytest` passes; manual smoke
   test in the app confirms zero behavior change.

3. `feat(cylinder): add DrillCoreParameters, surface points, layer sampling`
   Adds `src/superposed_folds/cylinder.py` and `tests/test_cylinder.py`.
   Library-only addition; no app integration yet. `uv run pytest`
   passes.

4. `feat(viz): add drill-core figure builders and orientation cues`
   Adds `fig_3d_drill_core_trace`, `fig_2d_drill_core_unrolled`,
   `drill_core_map_overlay_traces`, plus axis relabels and the N
   annotation in `viz.py`. Re-exports new public names from
   `__init__.py`.

5. `feat(app): add drill-core expander, resolution slider, unrolled strip row`
   Threads the new figure builders through `streamlit_app.py`. The
   feature is now end-to-end usable. Manual QA per the gate above.

6. PR and merge.

## What this plan deliberately does not do

- No new dependencies.
- No 1D core log view.
- No end caps on the cylinder.
- No refactor of `geometry.py`, `transforms.py`, `classification.py`,
  `references.py`.
- No changes to MATLAB parity snapshots.
- No marimo cleanup or other unrelated archive work.

## Open questions

None at design-approval time. Any open question that surfaces during
implementation goes back through brainstorming before code lands.
