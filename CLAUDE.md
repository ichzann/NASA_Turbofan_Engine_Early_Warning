# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

Exploratory analysis on the NASA CMAPSS turbofan engine degradation dataset (`behrad3d/nasa-cmaps` on Kaggle). Goal: early-warning / anomaly detection on engine sensor readings. Work lives primarily in `engine_fail.ipynb`; `defs.py` holds helpers pulled out of the notebook.

## Data

- Dataset is fetched at runtime via `kagglehub.dataset_download("behrad3d/nasa-cmaps")`, which returns a local path. The `CMaps/` subdirectory contains `train_FD00{1..4}.txt` and `test_FD00{1..4}.txt`, whitespace-separated, no header.
- Column 0 is the engine unit id, column 1 is the cycle index, columns 2+ are operational settings and sensor channels. The notebook groups by column `0` (engine id) when computing per-engine aggregates.

## Architecture notes

- `defs.py::pca_plot` scales columns `[2:]` with `StandardScaler`, fits a 2-component `PCA`, then plots per-engine min/max PCA coordinates as green/red engine-id labels. Known issues to be aware of when editing:
  - It references `df_train` (global) instead of its parameter `df_iso_train` on line 2 — the function only works when a `df_train` global exists.
  - Cell 1 of the notebook imports it as `from C import pca_plot` / `import C`; the module is actually `defs.py`. Fix to `from defs import pca_plot` when touching this.
  - Imports (`StandardScaler`, `PCA`, `pd`, `plt`) are expected to already be in scope from the notebook; `defs.py` itself does not import them.

## Running

Open `engine_fail.ipynb` in Jupyter and run top-to-bottom. Cell 0 downloads the dataset; subsequent cells depend on the `path` variable it sets. Switch engine subset by changing `document_nr` in cell 2 (values 1–4 correspond to the four FD00x fault-mode datasets).

Dependencies used: `kagglehub`, `pandas`, `numpy`, `matplotlib`, `scikit-learn`. No requirements file or test suite exists yet.
