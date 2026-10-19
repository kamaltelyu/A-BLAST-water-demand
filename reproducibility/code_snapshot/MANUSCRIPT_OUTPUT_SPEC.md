# Manuscript output specification — v3.2 FINAL

## Table 1 — Rolling-origin forecasting performance
For each dataset/horizon/model, MSE, MAE, and RMSE are averaged across the five seeds within each pre-specified test origin. The manuscript reports mean ± SD across the five temporal origins.

## Table 2 — Computational efficiency
Training time is measured once per fitted dataset × horizon × model × seed model. Inference time is measured at all five origins after warm-up and CUDA synchronization. Common STL preprocessing is excluded from model timing because it is identical across models.

## Table 3 — Unit-free paired robustness
Comparisons are matched by dataset × horizon × origin × seed. Reported change is `100 × (A-BLAST error − baseline error) / baseline error`; negative values favor A-BLAST. This avoids invalid pooling of mcft and ML errors. No independence-based p-value is claimed.

## Table 4 — Rolling-origin ablation
At H=96, BLS baseline and Full A-BLAST are reused from the main experiment. `w/o attention`, `w/o Transformer`, and `w/o enhancement` are fitted under the same protocol and evaluated at all five origins.

## Figures
- Fig. 2: two dataset-specific accuracy–efficiency panels.
- Fig. 3: 2×3 dataset-specific boxplots; errors never pooled across units.
- Fig. 4: 2×3 forecast visualization at H=96 and the pre-specified central origin O3; mean forecast ±1 SD across seeds.
- Fig. S1: relative MAE across all rolling origins, using Transformer=1 within each dataset/horizon/origin.
