"""superposed-folds: interactive visualization of superposed-fold geometries."""

from .classification import (
    ALL_PRESETS,
    END_MEMBERS,
    OBLIQUE_ROTATION,
    PARALLEL_ROTATION,
    Preset,
    classify_nearest,
)
from .cylinder import (
    DrillCoreParameters,
    cylinder_surface_points,
    sample_layers_on_cylinder,
)
from .geometry import (
    FoldParameters,
    apply_fold,
    apply_superposed_fold,
    initial_z_at,
    make_layer_stack,
)
from .layers import LAYER_COLORS, N_LAYERS, layer_index_from_z
from .viz import (
    drill_core_map_overlay_traces,
    fig_2d_drill_core_unrolled,
    fig_2d_interference,
    fig_3d_block_diagram,
    fig_3d_drill_core_trace,
    fig_3d_stack,
    fig_stereonet,
)

__version__ = "0.1.0.dev0"

__all__ = [
    "ALL_PRESETS",
    "END_MEMBERS",
    "LAYER_COLORS",
    "N_LAYERS",
    "OBLIQUE_ROTATION",
    "PARALLEL_ROTATION",
    "DrillCoreParameters",
    "FoldParameters",
    "Preset",
    "apply_fold",
    "apply_superposed_fold",
    "classify_nearest",
    "cylinder_surface_points",
    "drill_core_map_overlay_traces",
    "fig_2d_drill_core_unrolled",
    "fig_2d_interference",
    "fig_3d_block_diagram",
    "fig_3d_drill_core_trace",
    "fig_3d_stack",
    "fig_stereonet",
    "initial_z_at",
    "layer_index_from_z",
    "make_layer_stack",
    "sample_layers_on_cylinder",
]
