# superposed-folds

Interactive Python toolkit for visualizing superposed folds. A modern Python
port of Martin Schöpfer's MATLAB papermodel resource at the
[UCD Fault Analysis Group](https://www.fault-analysis-group.ucd.ie/SuperPosedFolds/),
implementing the plane-strain superposed-folding equations from Ramsay & Lisle
(2000) and the extended Grasemann et al. (2004) classification.

[![Open in Streamlit](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://superposed-folds.streamlit.app/)

![demo](assets/demo.gif)

Pick a Grasemann (2004) preset on the left or move the sliders. The 3D fold
stack, the 2D interference map at z = 0, the stereonet, and the classification
readout all update together.

## What's inside

- A small Python library (`superposed_folds`) implementing:
  - `FoldParameters` and the Ramsay & Lisle plane-strain equations
  - Forward (`apply_superposed_fold`) and inverse (`initial_z_at`) maps
  - All 21 canonical Grasemann (2004) preset configurations
  - Plotly figure builders: 3D stack, 2D interference map, stereonet
- A Streamlit app (`streamlit_app.py`) for the interactive playground.
- pytest suite with parity checks against the original UCD MATLAB code.

## Install

```bash
git clone <repo>
cd superposed-folds
uv venv
uv pip install -e ".[app,dev]"
```

## Run the web app locally

```bash
uv run streamlit run streamlit_app.py
```

## Run the tests

```bash
uv run pytest
```

## Credits and references

- **Original MATLAB resource and educational content**: Martin Schöpfer, UCD Fault Analysis Group ([page](https://www.fault-analysis-group.ucd.ie/SuperPosedFolds/))
- **Plane-strain equations**: Ramsay, J. G. and Lisle, R. J. (2000) *The Techniques of Modern Structural Geology, Volume 3*. Academic Press, p. 955.
- **Classification (original)**: Ramsay, J. G. (1962) *J. Geol.* 70, 466-481.
- **Classification (extended)**: Grasemann, B. et al. (2004) *J. Geol.* 112, 119-125.
- Project surfaced via the [Software Underground](https://softwareunderground.org) Mattermost.

## License

MIT — see `LICENSE`.
