# A-BLAST: Accelerated Broad Learning System–Transformer for Long-Horizon Water-Related Time-Series Forecasting

This repository contains the **final MethodsX reproducibility package** for **A-BLAST (Accelerated Broad Learning System–Transformer)**, a component-wise hybrid method for long-horizon forecasting of water-related time series.

A-BLAST combines:

- history-only Seasonal-Trend decomposition using Loess (STL),
- Broad Learning System (BLS) random feature mapping,
- seeded fixed attention and Transformer-based feature transformation,
- BLS–Transformer feature fusion,
- nonlinear enhancement nodes, and
- closed-form ridge regression for analytical output-weight estimation.

The repository supports the manuscript:

> **A-BLAST: Accelerated Broad Learning System–Transformer Hybrid Method for Long-Horizon Water-Related Time-Series Forecasting**

## Final validation protocol

The locked v3.2 evaluation uses:

- chronological **70% / 15% / 15%** train / validation / test partitioning;
- **365-day lookback** and **STL period 365**;
- forecast horizons **96, 192, 336, and 720 days**;
- **5 random seeds**: 11, 22, 33, 44, 55;
- **5 pre-specified rolling forecast origins per horizon**;
- model fitting without future/test information;
- history-only STL recomputed at each forecast origin;
- a fixed final model across all five test origins; and
- repeated, synchronized timing on an NVIDIA Tesla T4 runtime.

The exact protocol is stored in [`PROTOCOL_LOCK.json`](PROTOCOL_LOCK.json).

## Final validation summary

Across the eight dataset–horizon settings, A-BLAST achieved the lowest:

- **MAE in 5/8 settings**, and
- **MSE in 6/8 settings**.

In matched rolling-origin evaluations, the median MAE change of A-BLAST was:

- **−7.61% vs BLS**, with 128/200 wins; and
- **−4.87% vs Transformer**, with 116/200 wins.

A-BLAST training was approximately **4.77×–9.53× faster than the standalone Transformer** in the final experiments. BLS remained the fastest model overall; A-BLAST is therefore presented as an **accuracy–efficiency compromise**, not as the universally fastest or most accurate method.

Full tables are available in [`results/tables/`](results/tables/).

## Repository structure

```text
A-BLAST-water-demand/
├── README.md
├── UPLOAD_TO_GITHUB.md
├── RELEASE_NOTES_v1.0.0-methodsx.md
├── REPRODUCIBILITY.md
├── DATA_SOURCES.md
├── CITATION.cff
├── LICENSE
├── requirements.txt
├── PROTOCOL_LOCK.json
├── run_all.py
├── src/
├── notebooks/
├── results/
│   ├── tables/
│   └── raw_results/
├── figures/
├── reproducibility/
├── data/raw/
├── docs/
└── scripts/
```

## One-click reproduction in Google Colab

Open:

`notebooks/ABLAST_MethodsX_ONECLICK_v3_2_FINAL_ROLLING_ORIGIN.ipynb`

and choose **Runtime → Run all**.

The notebook installs dependencies, mounts Google Drive once, downloads and hashes the public datasets, runs the leakage-safe rolling-origin evaluation, autosaves checkpoints, resumes after interruptions, generates the manuscript tables/figures, and stores the final results ZIP in Google Drive.

The only unavoidable interaction is Google Drive authorization.

## Command-line rerun

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python run_all.py --out A-BLAST_MethodsX_RESULTS_v3_2
```

Use `--quick` only for a synthetic software smoke test. **Synthetic smoke-test output must never be used as manuscript evidence.**

## Datasets

Raw data are **not redistributed**. The one-click code acquires the exact public sources automatically and records SHA-256 hashes.

### Chennai Water Management

- Target: total reservoir water level, summed across POONDI, CHOLAVARAM, REDHILLS, and CHEMBARAMBAKKAM
- Unit: million cubic feet (mcft)
- Final series: 2004-01-01 to 2019-09-16, 5,738 daily observations
- SHA-256: `03c2add98d00f74b13b3caabb958e8e8b4e81dd5016b71bf04c839b09de32269`

### Melbourne Water Supply

- Target: `supplied_ML`
- Unit: megalitres (ML)
- Final series: 1890-12-27 to 2020-02-28, 47,180 daily observations
- SHA-256: `9dbce9b80e4edf5323ce74bb7c59db2ca56a89471b7a3999f77c1b138b112204`

See [`DATA_SOURCES.md`](DATA_SOURCES.md) and [`reproducibility/dataset_profiles.json`](reproducibility/dataset_profiles.json).

## Reproducibility artifacts

The repository preserves per-seed/per-origin predictions and metrics, timing, selected hyperparameters, dataset profiles/checksums, environment metadata, the full run log, and the exact source-code snapshot used by the final run.

See [`REPRODUCIBILITY.md`](REPRODUCIBILITY.md).

## Notes on interpretation

The final evaluation intentionally avoids claims of universal superiority. A-BLAST improves the accuracy–efficiency trade-off in most tested settings, but the effect varies by dataset and horizon. The ablation study also shows that attention weighting and enhancement-node contributions are dataset-dependent.

## Citation

Before the MethodsX article receives final bibliographic details, cite the software using [`CITATION.cff`](CITATION.cff). Add the article DOI after publication.

## License

Code is released under the **MIT License**. Source datasets retain their original terms and are not redistributed here.
