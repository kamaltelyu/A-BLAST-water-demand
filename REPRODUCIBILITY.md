# Reproducibility guide

This repository corresponds to the **locked v3.2 fixed-model rolling-origin protocol** used for the final manuscript validation.

## Experimental design

- chronological split: 70% train / 15% validation / 15% test
- lookback: 365 days
- STL period: 365 days, robust fitting enabled
- horizons: 96, 192, 336, 720 days
- seeds: 11, 22, 33, 44, 55
- test origins: 5 pre-specified rolling origins per horizon
- final model: fitted once on train+validation and held fixed across the five test origins
- each rolling-origin input uses only observations strictly before that origin
- residual STL component is not modeled separately

Canonical configuration: `PROTOCOL_LOCK.json` and `reproducibility/experiment_config.json`.

## Models

### BLS baseline
Direct multi-output BLS-style model with random feature mapping, enhancement nodes, and ridge output-weight estimation.

### Transformer baseline
Gradient-trained standalone Transformer direct forecaster evaluated with the same split, history window, horizons, and forecast origins.

### A-BLAST
History-only STL → BLS feature mapping → seeded fixed attention/Transformer-based transformation → feature fusion → enhancement nodes → closed-form ridge regression → future trend/seasonal reconstruction.

The Transformer-based block inside A-BLAST is a fixed seeded transformation rather than a separately gradient-trained forecasting network.

## Final runtime environment

- GPU: NVIDIA Tesla T4 (14.563 GB)
- CPU: x86_64, 2 logical CPUs
- RAM: 12.671 GB
- Python: 3.13.15
- PyTorch: 2.11.0+cu130
- CUDA: 13.0
- NumPy: 2.1.3
- pandas: 2.2.3
- scikit-learn: 1.6.1
- statsmodels: 0.15.0
- SciPy: 1.16.3
- Matplotlib: 3.10.0

Exact environment: `reproducibility/environment.json`; package snapshot: `reproducibility/pip_freeze.txt`.

## Timing protocol

Training and inference are measured separately. CUDA synchronization and repeated inference timing are used. Common STL preprocessing is excluded from model-specific timing because it is shared across models.

## Integrity and provenance

Dataset metadata and SHA-256 hashes:

- `reproducibility/download_manifest.json`
- `reproducibility/dataset_profiles.json`

Exact final source-code snapshot:

- `reproducibility/code_snapshot/`

## Final evidence

- manuscript tables: `results/tables/`
- per-seed/per-origin outputs: `results/raw_results/`
- figures: `figures/`
- run log: `reproducibility/run_log.txt`
