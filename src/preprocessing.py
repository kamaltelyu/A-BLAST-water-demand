"""Preprocessing utilities for component-wise forecasting."""

from __future__ import annotations

import numpy as np
import pandas as pd
from statsmodels.tsa.seasonal import STL


def stl_decompose(series: pd.Series, period: int = 365, robust: bool = True) -> pd.DataFrame:
    """Decompose a univariate series into trend, seasonal, and residual components."""
    clean = pd.Series(series).astype(float).interpolate().ffill().bfill()
    result = STL(clean, period=period, robust=robust).fit()
    return pd.DataFrame(
        {
            "observed": clean.values,
            "trend": result.trend,
            "seasonal": result.seasonal,
            "resid": result.resid,
        },
        index=clean.index,
    )


def make_lag_features(values: np.ndarray, lags: int = 2) -> tuple[np.ndarray, np.ndarray]:
    """Create simple lagged features for tabular forecasting models."""
    values = np.asarray(values, dtype=float).ravel()
    if lags < 1:
        raise ValueError("lags must be at least 1")
    x, y = [], []
    for i in range(lags, len(values)):
        x.append(values[i - lags : i])
        y.append(values[i])
    return np.asarray(x), np.asarray(y)


def train_test_by_horizon(x: np.ndarray, y: np.ndarray, horizon: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Use the last horizon samples as the test set."""
    if horizon <= 0 or horizon >= len(y):
        raise ValueError("horizon must be positive and smaller than the number of samples")
    return x[:-horizon], x[-horizon:], y[:-horizon], y[-horizon:]
