"""21 canonical Grasemann (2004) preset configurations.

F2 (dip-direction, dip, rake) triples are taken verbatim from the UCD
papermodel pages (filename pattern `F1_0_90_0_F2_<dipdir>_<dip>_<rake>`):

- End Members: ``Superposed_PM_EndMembers.html``
- Parallel rotation: ``Superposed_PM_Transitional_ParallelRotation.html``
- Oblique rotation: ``Superposed_PM_Transitional_ObliqueRotation.html``

F1 is fixed at the UCD reference orientation (000/90, rake 0) for every
preset, matching every papermodel on those pages.
"""

from __future__ import annotations

from dataclasses import dataclass

from .geometry import FoldParameters


@dataclass(frozen=True)
class Preset:
    id: str
    name: str
    explainer: str
    f1: FoldParameters
    f2: FoldParameters


_F1 = FoldParameters(A=3.0, B=1.0, dip_dir=0.0, dip=90.0, rake=0.0)


def _f2(dip_dir: float, dip: float, rake: float) -> FoldParameters:
    return FoldParameters(A=3.0, B=1.0, dip_dir=dip_dir, dip=dip, rake=rake)


# ----- End members --------------------------------------------------------

END_MEMBERS: list[Preset] = [
    Preset(
        id="type-0_1",
        name="Type 0₁",
        explainer=(
            "F2 axis parallel to F1 axis (x₁). No interference pattern is "
            "generated; the F1 fold is preserved unchanged but a lineation "
            "contained within the first fold is folded by F2."
        ),
        f1=_F1,
        f2=_f2(dip_dir=90.0, dip=0.0, rake=0.0),
    ),
    Preset(
        id="type-0_2",
        name="Type 0₂",
        explainer=(
            "F2 axis parallel to the normal to F1's axial plane (x₂). No "
            "interference pattern in section; folds a lineation within F1."
        ),
        f1=_F1,
        f2=_f2(dip_dir=0.0, dip=90.0, rake=90.0),
    ),
    Preset(
        id="type-0_3",
        name="Type 0₃ (= Type 0)",
        explainer=(
            "F2 axis parallel to the shear direction in F1's axial plane "
            "(x₃). Coaxial superposition; amplitudes simply add."
        ),
        f1=_F1,
        f2=_f2(dip_dir=0.0, dip=90.0, rake=0.0),
    ),
    Preset(
        id="type-1",
        name="Type 1 (dome-and-basin)",
        explainer=(
            "F2 axial plane perpendicular to F1's, both vertical with "
            "horizontal axes 90° apart. Produces the classic 'egg-carton' "
            "dome-and-basin interference pattern."
        ),
        f1=_F1,
        f2=_f2(dip_dir=90.0, dip=90.0, rake=0.0),
    ),
    Preset(
        id="type-2",
        name="Type 2 (mushroom)",
        explainer=(
            "F2 axial plane perpendicular to F1's, with F2 axis vertical. "
            "Produces the 'mushroom' or 'crescent' interference pattern."
        ),
        f1=_F1,
        f2=_f2(dip_dir=90.0, dip=90.0, rake=90.0),
    ),
    Preset(
        id="type-3",
        name="Type 3 (hook)",
        explainer=(
            "F2 axial plane horizontal (refolds F1 in section). Produces "
            "the 'hook' interference pattern."
        ),
        f1=_F1,
        f2=_f2(dip_dir=0.0, dip=0.0, rake=0.0),
    ),
]


# ----- Parallel rotation (45° around an F2 reference axis) ----------------

# Pairs from `Superposed_PM_Transitional_ParallelRotation.html`. F2 triples
# are read off the image filenames (UCD canonical values; no calibration).
_PARALLEL_PAIRS: list[tuple[str, str, tuple[float, float, float]]] = [
    ("1",   "0_1", (90.0, 45.0, 0.0)),
    ("2",   "0_2", (45.0, 90.0, 90.0)),
    ("3",   "0_3", (0.0,  45.0, 0.0)),
    ("1",   "2",   (90.0, 90.0, 45.0)),
    ("0_2", "0_3", (0.0,  90.0, 45.0)),
    ("3",   "0_1", (45.0, 0.0,  0.0)),
    ("2",   "3",   (90.0, 45.0, 90.0)),
    ("0_1", "0_2", (0.0,  45.0, 90.0)),
    ("1",   "0_3", (45.0, 90.0, 0.0)),
]


def _build_parallel(a: str, b: str, f2_orientation: tuple[float, float, float]) -> Preset:
    end_a = next(p for p in END_MEMBERS if p.id == f"type-{a}")
    end_b = next(p for p in END_MEMBERS if p.id == f"type-{b}")
    dipdir, dip, rake = f2_orientation
    return Preset(
        id=f"parallel-{a}-to-{b}",
        name=f"{end_a.name} ↔ {end_b.name} (parallel)",
        explainer=(
            f"45° parallel rotation halfway between {end_a.name} and "
            f"{end_b.name}, around one of the F2 reference axes."
        ),
        f1=_F1,
        f2=_f2(dip_dir=dipdir, dip=dip, rake=rake),
    )


PARALLEL_ROTATION: list[Preset] = [_build_parallel(a, b, o) for a, b, o in _PARALLEL_PAIRS]


# ----- Oblique rotation (60° around the 045/35 body-diagonal axis) --------

# Pairs from `Superposed_PM_Transitional_ObliqueRotation.html`.
_OBLIQUE_PAIRS: list[tuple[str, str, tuple[float, float, float]]] = [
    ("1",   "3",   (297.0, 48.0,  27.0)),
    ("3",   "0_2", (153.0, 48.0,  63.0)),
    ("1",   "0_2", (45.0,  71.0,  45.0)),
    ("2",   "0_3", (45.0,  71.0, -45.0)),
    ("0_1", "0_3", (153.0, 48.0, -27.0)),
    ("2",   "0_1", (297.0, 48.0, -63.0)),
]


def _build_oblique(a: str, b: str, f2_orientation: tuple[float, float, float]) -> Preset:
    end_a = next(p for p in END_MEMBERS if p.id == f"type-{a}")
    end_b = next(p for p in END_MEMBERS if p.id == f"type-{b}")
    dipdir, dip, rake = f2_orientation
    return Preset(
        id=f"oblique-{a}-to-{b}",
        name=f"{end_a.name} ↔ {end_b.name} (oblique)",
        explainer=(
            f"60° rotation around the body-diagonal axis (045/35) between "
            f"{end_a.name} and {end_b.name}."
        ),
        f1=_F1,
        f2=_f2(dip_dir=dipdir, dip=dip, rake=rake),
    )


OBLIQUE_ROTATION: list[Preset] = [_build_oblique(a, b, o) for a, b, o in _OBLIQUE_PAIRS]


ALL_PRESETS: list[Preset] = [*END_MEMBERS, *PARALLEL_ROTATION, *OBLIQUE_ROTATION]


# ----- classify_nearest ---------------------------------------------------

def _params_distance(a: FoldParameters, b: FoldParameters) -> float:
    """Euclidean-ish distance between two parameter sets, normalized.

    Orientation differences are scaled by 90° so they're commensurate with
    one degree of slider movement. Amplitude (`A`) and stretch (`B`, `C`)
    differences are kept *unscaled* on purpose: with the default `tol=0.05`
    in the app, this means any meaningful slider nudge on amplitude or
    stretch produces "Custom" rather than snapping to a preset. That is the
    intended UX — moving a slider should move the user out of the preset.
    """
    return float(
        (a.A - b.A) ** 2
        + (a.B - b.B) ** 2
        + ((a.C or 0.0) - (b.C or 0.0)) ** 2
        + ((a.dip_dir - b.dip_dir) / 90.0) ** 2
        + ((a.dip - b.dip) / 90.0) ** 2
        + ((a.rake - b.rake) / 90.0) ** 2
    )


def classify_nearest(
    f1: FoldParameters, f2: FoldParameters, tol: float = 0.01
) -> Preset | None:
    """Return the closest preset within `tol`, or None ('Custom') otherwise."""
    distances = [
        (_params_distance(f1, p.f1) + _params_distance(f2, p.f2), p) for p in ALL_PRESETS
    ]
    distances.sort(key=lambda kv: kv[0])
    best_dist, best_preset = distances[0]
    return best_preset if best_dist <= tol else None
