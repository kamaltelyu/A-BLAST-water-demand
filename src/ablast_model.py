"""A-BLAST model implementation.

This module provides a lightweight implementation of the A-BLAST workflow used
in the MethodsX manuscript. The implementation follows the manuscript logic:

component input -> BLS feature mapping -> attention-based weighting ->
Transformer-based feature transformation -> BLS-Transformer fusion ->
enhancement nodes -> closed-form ridge regression.

The Transformer block is used as a feature transformation module. Output weights
are estimated analytically using ridge regression.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
import torch
from sklearn.preprocessing import StandardScaler
from torch import nn


ArrayLike = np.ndarray


def _sigmoid(x: ArrayLike) -> ArrayLike:
    return 1.0 / (1.0 + np.exp(-np.clip(x, -60, 60)))


def _softmax(x: ArrayLike, axis: int = 1) -> ArrayLike:
    x = x - np.max(x, axis=axis, keepdims=True)
    ex = np.exp(x)
    return ex / np.sum(ex, axis=axis, keepdims=True)


class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding."""

    def __init__(self, d_model: int, max_len: int = 10000) -> None:
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        if d_model > 1:
            pe[:, 1::2] = torch.cos(position * div_term[: pe[:, 1::2].shape[1]])
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:, : x.size(1), :]


@dataclass
class ABLASTRegressor:
    """Accelerated Broad Learning System-Transformer regressor.

    Parameters
    ----------
    n_feature_nodes:
        Number of BLS feature nodes.
    n_enhancement_nodes:
        Number of BLS enhancement nodes.
    d_model:
        Transformer feature dimension.
    n_heads:
        Number of self-attention heads.
    n_layers:
        Number of Transformer encoder layers.
    ridge_lambda:
        Ridge regularization value for closed-form output-weight estimation.
    random_state:
        Seed for reproducibility.
    """

    n_feature_nodes: int = 100
    n_enhancement_nodes: int = 50
    d_model: int = 4
    n_heads: int = 2
    n_layers: int = 1
    ridge_lambda: float = 1e-5
    random_state: int = 42

    def __post_init__(self) -> None:
        self.scaler_x = StandardScaler()
        self.scaler_y = StandardScaler()
        self.rng = np.random.default_rng(self.random_state)
        self._is_fitted = False

    def _init_parameters(self, n_features: int) -> None:
        self.W_f = self.rng.normal(0.0, 1.0, size=(n_features, self.n_feature_nodes))
        self.b_f = self.rng.normal(0.0, 0.1, size=(self.n_feature_nodes,))
        self.W_a = self.rng.normal(0.0, 1.0, size=(self.n_feature_nodes, self.n_feature_nodes))
        self.b_a = self.rng.normal(0.0, 0.1, size=(self.n_feature_nodes,))
        self.W_e = self.rng.normal(0.0, 1.0, size=(self.n_feature_nodes, self.n_enhancement_nodes))
        self.b_e = self.rng.normal(0.0, 0.1, size=(self.n_enhancement_nodes,))

        torch.manual_seed(self.random_state)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.d_model,
            nhead=max(1, min(self.n_heads, self.d_model)),
            dim_feedforward=max(8, 2 * self.d_model),
            dropout=0.0,
            batch_first=True,
            activation="relu",
        )
        self.positional_encoding = PositionalEncoding(self.d_model)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=self.n_layers)
        self.transformer.eval()

    def _feature_mapping(self, x_norm: ArrayLike) -> ArrayLike:
        return _sigmoid(x_norm @ self.W_f + self.b_f)

    def _attention_weighting(self, h1: ArrayLike) -> Tuple[ArrayLike, ArrayLike]:
        scores = h1 @ self.W_a + self.b_a
        attn = _softmax(scores, axis=1)
        return h1 * attn, attn

    def _transformer_transform(self, h_weighted: ArrayLike) -> ArrayLike:
        # Select the most informative feature nodes based on average activation.
        importance = np.mean(np.abs(h_weighted), axis=0)
        idx = np.argsort(importance)[-self.d_model :]
        selected = h_weighted[:, idx]
        if selected.shape[1] < self.d_model:
            selected = np.pad(selected, ((0, 0), (0, self.d_model - selected.shape[1])))

        x_tensor = torch.tensor(selected[None, :, :], dtype=torch.float32)
        with torch.no_grad():
            x_tensor = self.positional_encoding(x_tensor)
            h_t = self.transformer(x_tensor).squeeze(0).cpu().numpy()

        # Align Transformer output to the BLS feature-node dimension.
        if h_t.shape[1] < self.n_feature_nodes:
            h_t = np.pad(h_t, ((0, 0), (0, self.n_feature_nodes - h_t.shape[1])))
        elif h_t.shape[1] > self.n_feature_nodes:
            h_t = h_t[:, : self.n_feature_nodes]
        return h_t

    def _design_matrix(self, x: ArrayLike) -> ArrayLike:
        x_norm = self.scaler_x.transform(np.asarray(x, dtype=float))
        h1 = self._feature_mapping(x_norm)
        h_weighted, _ = self._attention_weighting(h1)
        h_t = self._transformer_transform(h_weighted)
        h_c = (h1 + h_t) / 2.0
        h_e = _sigmoid(h_c @ self.W_e + self.b_e)
        return np.concatenate([h_c, h_e], axis=1)

    def fit(self, x: ArrayLike, y: ArrayLike) -> "ABLASTRegressor":
        x = np.asarray(x, dtype=float)
        y = np.asarray(y, dtype=float).reshape(-1, 1)
        x_norm = self.scaler_x.fit_transform(x)
        y_norm = self.scaler_y.fit_transform(y)
        self._init_parameters(x.shape[1])

        h1 = self._feature_mapping(x_norm)
        h_weighted, _ = self._attention_weighting(h1)
        h_t = self._transformer_transform(h_weighted)
        h_c = (h1 + h_t) / 2.0
        h_e = _sigmoid(h_c @ self.W_e + self.b_e)
        z = np.concatenate([h_c, h_e], axis=1)

        identity = np.eye(z.shape[1])
        self.beta = np.linalg.solve(
            z.T @ z + self.ridge_lambda * identity,
            z.T @ y_norm,
        )
        self._is_fitted = True
        return self

    def predict(self, x: ArrayLike) -> ArrayLike:
        if not self._is_fitted:
            raise RuntimeError("ABLASTRegressor must be fitted before prediction.")
        z = self._design_matrix(x)
        y_norm_pred = z @ self.beta
        return self.scaler_y.inverse_transform(y_norm_pred).ravel()


@dataclass
class BLSRegressor:
    """Simple Broad Learning System baseline with closed-form ridge regression."""

    n_feature_nodes: int = 100
    n_enhancement_nodes: int = 50
    ridge_lambda: float = 1e-5
    random_state: int = 42

    def __post_init__(self) -> None:
        self.scaler_x = StandardScaler()
        self.scaler_y = StandardScaler()
        self.rng = np.random.default_rng(self.random_state)
        self._is_fitted = False

    def fit(self, x: ArrayLike, y: ArrayLike) -> "BLSRegressor":
        x = np.asarray(x, dtype=float)
        y = np.asarray(y, dtype=float).reshape(-1, 1)
        x_norm = self.scaler_x.fit_transform(x)
        y_norm = self.scaler_y.fit_transform(y)

        self.W_f = self.rng.normal(size=(x.shape[1], self.n_feature_nodes))
        self.b_f = self.rng.normal(0.0, 0.1, size=(self.n_feature_nodes,))
        h1 = _sigmoid(x_norm @ self.W_f + self.b_f)

        self.W_e = self.rng.normal(size=(self.n_feature_nodes, self.n_enhancement_nodes))
        self.b_e = self.rng.normal(0.0, 0.1, size=(self.n_enhancement_nodes,))
        h_e = _sigmoid(h1 @ self.W_e + self.b_e)
        z = np.concatenate([h1, h_e], axis=1)

        identity = np.eye(z.shape[1])
        self.beta = np.linalg.solve(z.T @ z + self.ridge_lambda * identity, z.T @ y_norm)
        self._is_fitted = True
        return self

    def predict(self, x: ArrayLike) -> ArrayLike:
        if not self._is_fitted:
            raise RuntimeError("BLSRegressor must be fitted before prediction.")
        x_norm = self.scaler_x.transform(np.asarray(x, dtype=float))
        h1 = _sigmoid(x_norm @ self.W_f + self.b_f)
        h_e = _sigmoid(h1 @ self.W_e + self.b_e)
        z = np.concatenate([h1, h_e], axis=1)
        y_norm_pred = z @ self.beta
        return self.scaler_y.inverse_transform(y_norm_pred).ravel()


@dataclass
class ComponentWiseForecaster:
    """Fit separate models to trend and seasonal components and reconstruct forecasts."""

    trend_model: object
    seasonal_model: object

    def fit(self, x_trend: ArrayLike, y_trend: ArrayLike, x_seasonal: ArrayLike, y_seasonal: ArrayLike):
        self.trend_model.fit(x_trend, y_trend)
        self.seasonal_model.fit(x_seasonal, y_seasonal)
        return self

    def predict(self, x_trend: ArrayLike, x_seasonal: ArrayLike) -> ArrayLike:
        trend_pred = self.trend_model.predict(x_trend)
        seasonal_pred = self.seasonal_model.predict(x_seasonal)
        return trend_pred + seasonal_pred
