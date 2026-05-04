"""One-shot conversion from Octave .mat snapshots to .npz for pytest."""

from pathlib import Path

import numpy as np
from scipy.io import loadmat


def main() -> None:
    here = Path(__file__).parent
    for mat in here.glob("parity_*.mat"):
        data = loadmat(mat)
        out = here / mat.with_suffix(".npz").name
        np.savez(
            out,
            X=data["X"],
            Y=data["Y"],
            Z=data["Z"],
            Z_initial=data["Zini"],
        )
        print(f"wrote {out}")


if __name__ == "__main__":
    main()
