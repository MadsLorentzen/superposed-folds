"""superposed-folds: interactive visualization of superposed-fold geometries."""

from .classification import (
    ALL_PRESETS,
    END_MEMBERS,
    OBLIQUE_ROTATION,
    PARALLEL_ROTATION,
    Preset,
    classify_nearest,
)
from .geometry import (
    FoldParameters,
    apply_fold,
    apply_superposed_fold,
    initial_z_at,
    make_layer_stack,
)
from .viz import fig_2d_interference, fig_3d_stack, fig_stereonet

__version__ = "0.1.0.dev0"

__all__ = [
    "ALL_PRESETS",
    "END_MEMBERS",
    "OBLIQUE_ROTATION",
    "PARALLEL_ROTATION",
    "FoldParameters",
    "Preset",
    "apply_fold",
    "apply_superposed_fold",
    "classify_nearest",
    "fig_2d_interference",
    "fig_3d_stack",
    "fig_stereonet",
    "initial_z_at",
    "make_layer_stack",
]
