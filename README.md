# A-BLAST: Accelerated Broad Learning System–Transformer for Long-Term Water Demand Prediction

This repository provides the implementation and validation assets for **A-BLAST (Accelerated Broad Learning System–Transformer)**, a component-wise hybrid forecasting method for long-term water demand prediction.

A-BLAST combines:

- Seasonal-Trend decomposition using Loess (STL) as a preliminary component separation step,
- Broad Learning System (BLS) random feature mapping,
- attention-based feature weighting,
- Transformer-based feature transformation,
- nonlinear enhancement nodes, and
- closed-form ridge regression for efficient output-weight estimation.

The repository is prepared to support the MethodsX manuscript:

> A-BLAST: Accelerated Broad Learning System–Transformer Hybrid Model for Long-Term Water Demand Prediction

## Repository structure

```text
A-BLAST-water-demand/
├── README.md
├── requirements.txt
├── LICENSE
├── CITATION.cff
├── src/
│   ├── __init__.py
│   ├── ablast_model.py
│   ├── preprocessing.py
│   ├── evaluation.py
│   └── visualization.py
├── scripts/
│   └── generate_validation_artifacts.py
├── results/
│   ├── forecasting_performance.tsv
│   ├── computational_efficiency.tsv
│   └── accuracy_efficiency.tsv
├── figures/
│   ├── graphical_abstract.png
│   ├── accuracy_efficiency_plane.png
│   ├── comparative_boxplots.png
│   └── forecast_visualization_2x3.png
├── data/
│   ├── raw/
│   │   └── README.md
│   └── processed/
└── notebooks/
    └── README.md
```

## Main workflow

The implemented workflow follows the manuscript pipeline:

```text
Original water-related time series
→ STL decomposition
→ trend and seasonal components
→ component-wise A-BLAST modeling
→ predicted trend + predicted seasonal
→ reconstructed final forecast
→ evaluation using MSE, MAE, and computational time
```

## Quick start

Create an environment and install dependencies:

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

Regenerate validation tables and figures from the saved results:

```bash
python scripts/generate_validation_artifacts.py
```

This script produces:

- paired statistical comparison,
- accuracy-efficiency plot,
- comparative boxplots, and
- summary efficiency table.

## Data

Raw datasets are not bundled in this repository. Place the downloaded datasets in `data/raw/` and update the paths in your experiment scripts or notebooks.

Recommended raw files:

- Chennai Water Management dataset files,
- Melbourne Water Supply dataset file.

The processed result tables used in the manuscript are provided in `results/` for reproducibility of the validation figures.

## Reproducibility note

The included `src/` implementation contains a reusable A-BLAST estimator and utility functions. The `results/` directory contains the manuscript-level validation outputs used to generate the tables and figures.

For full experiment reruns, add the original Colab notebooks to `notebooks/` or implement dataset-specific runner scripts using the modules in `src/`.

## Citation

If you use this code, please cite the associated MethodsX article once published. A placeholder citation file is included as `CITATION.cff` and should be updated after publication.

## License

This repository is released under the MIT License. Update the license if your institution or publisher requires a different license.
