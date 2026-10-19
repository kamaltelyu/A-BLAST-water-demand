# A-BLAST MethodsX v3.2 FINAL — autosave/resume rolling-origin rerun

Use the notebook in `notebooks/ABLAST_MethodsX_ONECLICK_v3_2_FINAL_ROLLING_ORIGIN.ipynb` and choose **Runtime → Run all**.

## What changed from v3.1

v3.2 keeps the leakage-safe 70/15/15 chronological split, 365-day lookback, horizons 96/192/336/720, five seeds, automatic dataset acquisition, Google Drive autosave, and resume-safe checkpoints. The final test evaluation is now a **fixed-model rolling-origin backtest with five temporal forecast origins per horizon**.

For every dataset × horizon × model × seed:

1. hyperparameters are selected without test data;
2. the final model is fitted once on train+validation;
3. five evenly spaced feasible forecast origins are pre-specified inside the held-out test interval;
4. the fitted model remains fixed for all five origins;
5. for each origin, STL and the 365-day input window use only observations strictly before that origin;
6. earlier test observations may appear as *past observed inputs* at later origins, but test observations are never used to refit the model.

This provides temporal robustness without multiplying model-training cost by five.

## Persistence

The Colab notebook writes directly to:

`MyDrive/A-BLAST_MethodsX_Rerun_v3_2_FINAL/A-BLAST_MethodsX_RESULTS_v3_2/`

Final ZIP:

`MyDrive/A-BLAST_MethodsX_Rerun_v3_2_FINAL/A-BLAST_MethodsX_RESULTS_v3_2.zip`

If the runtime disconnects, open the same notebook and choose **Run all** again. A completed dataset × horizon × model × seed checkpoint contains metrics for all five origins, predictions for all five horizons, and one timing record. Completed checkpoints are skipped automatically.

## Final manuscript outputs

- Table 1: rolling-origin MSE/MAE/RMSE. Seeds are averaged within each origin; values are then mean ± SD across five temporal origins.
- Table 2: training and inference efficiency. Training is timed once per fitted model. Common STL preprocessing is excluded because it is shared by all models.
- Table 3: unit-free paired percentage comparison, matched by dataset × horizon × origin × seed. This avoids mixing Chennai mcft and Melbourne ML errors.
- Table 4: rolling-origin architectural ablation at H=96.
- Table 5: exact experimental setup, environment, seeds, and rolling-origin protocol.
- Table 6: exact dataset profile and SHA-256.
- Fig. 2: separate Chennai and Melbourne accuracy–efficiency panels.
- Fig. 3: separate Chennai/Melbourne rows for MAE, MSE, and model time.
- Fig. 4: pre-specified central rolling origin (O3) at H=96, mean forecast ±1 SD across seeds.
- Fig. S1: relative MAE across all rolling origins, normalized to Transformer within each dataset/horizon/origin.

## One unavoidable interaction

Google Drive authorization is required once. There are no dataset-path prompts and no `input()` calls.
