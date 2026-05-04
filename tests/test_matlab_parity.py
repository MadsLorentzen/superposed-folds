"""Verify Python port matches the original UCD MATLAB script numerically."""

from pathlib import Path

import numpy as np
import pytest

from superposed_folds.classification import ALL_PRESETS
from superposed_folds.geometry import initial_z_at

DATA_DIR = Path(__file__).parent / "data"

PRESET_FILE_MAP = {
    "parity_type_0_3.npz": "type-0_3",
    "parity_type_1.npz": "type-1",
    "parity_type_2.npz": "type-2",
    "parity_type_3.npz": "type-3",
}


@pytest.mark.parametrize("filename,preset_id", PRESET_FILE_MAP.items())
def test_python_matches_matlab(filename: str, preset_id: str) -> None:
    snapshot_path = DATA_DIR / filename
    if not snapshot_path.exists():
        pytest.skip(f"snapshot {filename} not generated yet (see tests/data/README.md)")

    data = np.load(snapshot_path)
    X = data["X"]
    Y = data["Y"]
    Z = data["Z"]
    Z_initial_matlab = data["Z_initial"]

    preset = next(p for p in ALL_PRESETS if p.id == preset_id)
    Z_initial_python = initial_z_at(X, Y, Z, preset.f1, preset.f2)

    np.testing.assert_allclose(Z_initial_python, Z_initial_matlab, atol=1e-6, rtol=1e-6)
