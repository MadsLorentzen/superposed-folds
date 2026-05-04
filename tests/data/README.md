# Parity snapshots

These `.npz` files are reference outputs from the original UCD MATLAB script
`fold_interference_pattern.m`, used by `tests/test_matlab_parity.py` to verify
that the Python port produces numerically identical results.

## Regenerating

The MATLAB code is **not** redistributed in this repo. To regenerate the
snapshots:

1. Place the UCD MATLAB code (downloaded from
   `https://www.fault-analysis-group.ucd.ie/SuperPosedFolds/SPPM/SUPERPOSED_FOLDING_SCRIPT.zip`)
   in some local directory, e.g. `~/Desktop/github_local/superposed-folds-reference/`.
2. Copy `tests/data/generate_parity.m` into that directory.
3. From inside that directory, run `octave generate_parity.m` (or open in MATLAB).
   This produces `parity_*.mat`.
4. Move the `.mat` files into this `tests/data/` directory.
5. From the repo root, run `uv run python tests/data/mat_to_npz.py`.
6. Delete the `.mat` files. Commit only the `.npz` files.

The `.npz` files are small (~10 KB each) and ship with the repo.
