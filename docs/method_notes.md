# A-BLAST method notes

A-BLAST is a component-wise hybrid forecasting method.

1. The original water-related time series is decomposed into trend, seasonal, and residual components using STL.
2. The trend and seasonal components are modeled separately.
3. For each component, BLS feature nodes are generated through random nonlinear mapping.
4. Attention-based weighting emphasizes informative BLS feature nodes.
5. A Transformer-based feature transformation block enriches temporal representation.
6. BLS and Transformer-transformed features are fused.
7. Enhancement nodes expand nonlinear representation.
8. Output weights are computed using closed-form ridge regression.
9. Final forecast is reconstructed as predicted trend plus predicted seasonal component.

The residual component is not modeled separately in the current implementation.
