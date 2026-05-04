from superposed_folds.classification import (
    ALL_PRESETS,
    END_MEMBERS,
    OBLIQUE_ROTATION,
    PARALLEL_ROTATION,
    Preset,
    classify_nearest,
)
from superposed_folds.geometry import FoldParameters


def test_end_members_count():
    assert len(END_MEMBERS) == 6


def test_end_member_ids_unique_and_named():
    ids = {p.id for p in END_MEMBERS}
    expected = {"type-0_1", "type-0_2", "type-0_3", "type-1", "type-2", "type-3"}
    assert ids == expected
    for p in END_MEMBERS:
        assert isinstance(p, Preset)
        assert p.name
        assert p.explainer
        assert p.f1.dip_dir == 0.0 and p.f1.dip == 90.0 and p.f1.rake == 0.0


def test_parallel_rotation_count_and_ids():
    # Pairs from Superposed_PM_Transitional_ParallelRotation.html.
    assert len(PARALLEL_ROTATION) == 9
    ids = {p.id for p in PARALLEL_ROTATION}
    expected_pairs = {
        ("1",   "0_1"),
        ("2",   "0_2"),
        ("3",   "0_3"),
        ("1",   "2"),
        ("0_2", "0_3"),
        ("3",   "0_1"),
        ("2",   "3"),
        ("0_1", "0_2"),
        ("1",   "0_3"),
    }
    expected_ids = {f"parallel-{a}-to-{b}" for a, b in expected_pairs}
    assert ids == expected_ids


def test_oblique_rotation_count_and_ids():
    # Pairs from Superposed_PM_Transitional_ObliqueRotation.html.
    assert len(OBLIQUE_ROTATION) == 6
    ids = {p.id for p in OBLIQUE_ROTATION}
    expected_pairs = {
        ("1",   "3"),
        ("3",   "0_2"),
        ("1",   "0_2"),
        ("2",   "0_3"),
        ("0_1", "0_3"),
        ("2",   "0_1"),
    }
    expected_ids = {f"oblique-{a}-to-{b}" for a, b in expected_pairs}
    assert ids == expected_ids


def test_all_presets_total():
    assert len(ALL_PRESETS) == 21


def test_classify_nearest_returns_self_for_each_preset():
    for preset in ALL_PRESETS:
        match = classify_nearest(preset.f1, preset.f2, tol=1e-6)
        assert match is not None and match.id == preset.id


def test_classify_nearest_returns_none_when_far():
    far_f1 = FoldParameters(A=99.0, B=7.7, dip_dir=11.0, dip=22.0, rake=33.0)
    far_f2 = FoldParameters(A=99.0, B=7.7, dip_dir=44.0, dip=55.0, rake=66.0)
    assert classify_nearest(far_f1, far_f2, tol=1e-3) is None
