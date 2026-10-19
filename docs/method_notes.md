# A-BLAST method notes — final v3.2

The final validation uses a history-only, leakage-safe, fixed-model rolling-origin protocol. The original series is split chronologically, decomposed using STL without future observations, and modeled component-wise using trend and seasonal components. A-BLAST applies BLS feature mapping, fixed seeded attention/Transformer-based feature transformation, feature fusion, enhancement nodes, and closed-form ridge regression. Future trend and seasonal forecasts are summed to reconstruct the final forecast.

Canonical executable definition: `src/pipeline.py`, `PROTOCOL_LOCK.json`, and `reproducibility/code_snapshot/`.
