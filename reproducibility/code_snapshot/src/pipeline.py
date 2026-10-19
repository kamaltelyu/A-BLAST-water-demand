from __future__ import annotations

import json
import math
import os
import platform
import random
import time
from datetime import datetime, timezone
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.preprocessing import StandardScaler
from statsmodels.tsa.seasonal import STL


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    try:
        torch.use_deterministic_algorithms(True, warn_only=True)
    except Exception:
        pass


def sync_if_cuda(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def warmup_device(device: torch.device) -> None:
    if device.type == "cuda":
        x = torch.ones((64, 64), device=device)
        for _ in range(3):
            x = x @ x
        torch.cuda.synchronize(device)


def timed_call(fn, device: torch.device):
    sync_if_cuda(device)
    t0 = time.perf_counter()
    out = fn()
    sync_if_cuda(device)
    return out, time.perf_counter() - t0


def timed_inference(fn, device: torch.device, warmup: int = 3, repeats: int = 20):
    for _ in range(max(0, warmup)):
        _ = fn()
        sync_if_cuda(device)
    vals = []
    out = None
    for _ in range(max(1, repeats)):
        out, dt = timed_call(fn, device)
        vals.append(dt)
    arr = np.asarray(vals, dtype=float)
    return out, {
        "median_s": float(np.median(arr)),
        "mean_s": float(np.mean(arr)),
        "sd_s": float(np.std(arr, ddof=1)) if len(arr) > 1 else 0.0,
        "n_repeats": int(len(arr)),
    }


def environment_info() -> Dict[str, object]:
    info: Dict[str, object] = {
        "platform": platform.platform(),
        "os": platform.system(),
        "os_release": platform.release(),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "device": "cuda" if torch.cuda.is_available() else "cpu",
        "cuda_version": torch.version.cuda,
        "cpu": platform.processor() or platform.machine(),
    }
    try:
        import psutil
        info["ram_total_gb"] = round(psutil.virtual_memory().total / (1024**3), 3)
        info["cpu_logical_count"] = psutil.cpu_count(logical=True)
        info["cpu_physical_count"] = psutil.cpu_count(logical=False)
    except Exception:
        pass
    if torch.cuda.is_available():
        info["gpu"] = torch.cuda.get_device_name(0)
        try:
            props = torch.cuda.get_device_properties(0)
            info["gpu_total_memory_gb"] = round(props.total_memory / (1024**3), 3)
        except Exception:
            pass
    for mod, key in [("sklearn", "scikit_learn"), ("statsmodels", "statsmodels"), ("scipy", "scipy"), ("matplotlib", "matplotlib")]:
        try:
            m = __import__(mod)
            info[key] = m.__version__
        except Exception:
            pass
    return info


def validate_daily_index(y: pd.Series, dataset_name: str) -> Dict[str, object]:
    idx = pd.DatetimeIndex(y.index)
    if not idx.is_monotonic_increasing:
        raise ValueError(f"{dataset_name}: index is not chronological")
    if idx.has_duplicates:
        raise ValueError(f"{dataset_name}: duplicate dates remain after loader")
    full = pd.date_range(idx.min().normalize(), idx.max().normalize(), freq="D")
    missing_dates = full.difference(idx.normalize())
    return {
        "start": str(idx.min().date()),
        "end": str(idx.max().date()),
        "n": int(len(y)),
        "n_missing_calendar_days": int(len(missing_dates)),
        "first_missing_calendar_days": [str(x.date()) for x in missing_dates[:10]],
    }


def load_chennai(levels_csv: str) -> pd.Series:
    df = pd.read_csv(levels_csv)
    if "Date" not in df.columns:
        raise ValueError("Chennai levels file must contain a 'Date' column.")
    df["Date"] = pd.to_datetime(df["Date"], dayfirst=True, errors="raise")
    reservoirs = ["POONDI", "CHOLAVARAM", "REDHILLS", "CHEMBARAMBAKKAM"]
    missing = [c for c in reservoirs if c not in df.columns]
    if missing:
        raise ValueError(f"Missing Chennai reservoir columns: {missing}")
    df = df.sort_values("Date").drop_duplicates("Date").set_index("Date")
    vals = df[reservoirs].apply(pd.to_numeric, errors="raise")
    y = vals.sum(axis=1, min_count=len(reservoirs)).astype(float)
    y.name = "Total_Water"
    if y.isna().any():
        raise ValueError("Chennai target contains missing values; this protocol does not impute targets.")
    return y


def load_melbourne(csv_path: str) -> pd.Series:
    df = pd.read_csv(csv_path)
    lower = {c.lower(): c for c in df.columns}
    date_col = lower.get("recorddate")
    target_col = lower.get("supplied_ml")
    if date_col is None or target_col is None:
        raise ValueError("Melbourne file must contain recorddate and supplied_ML columns.")
    df[date_col] = pd.to_datetime(df[date_col], errors="raise")
    df = df.sort_values(date_col).drop_duplicates(date_col).set_index(date_col)
    y = pd.to_numeric(df[target_col], errors="raise").astype(float)
    y.name = "supplied_ML"
    if y.isna().any():
        raise ValueError("Melbourne target contains missing values; this protocol does not impute targets.")
    return y


def chronological_split(y: pd.Series, train_ratio=0.70, val_ratio=0.15):
    n = len(y)
    n_train = int(math.floor(n * train_ratio))
    n_val = int(math.floor(n * val_ratio))
    train = y.iloc[:n_train].copy()
    val = y.iloc[n_train:n_train+n_val].copy()
    test = y.iloc[n_train+n_val:].copy()
    if min(len(train), len(val), len(test)) == 0:
        raise ValueError("Empty chronological partition.")
    return train, val, test


def stl_history_only(y_history: pd.Series, period: int = 365, robust: bool = True) -> pd.DataFrame:
    if len(y_history) < max(2 * period, period + 10):
        raise ValueError(f"Need at least ~2 seasonal cycles for STL(period={period}); got {len(y_history)} points.")
    result = STL(y_history.astype(float), period=period, robust=robust).fit()
    return pd.DataFrame({"trend": result.trend, "seasonal": result.seasonal, "resid": result.resid}, index=y_history.index)


def make_supervised(component_df: pd.DataFrame, lookback: int, horizon: int, max_samples: int = 5000):
    arr = component_df[["trend", "seasonal"]].to_numpy(dtype=np.float64)
    n_possible = len(arr) - lookback - horizon + 1
    if n_possible <= 0:
        raise ValueError(f"Not enough history: n={len(arr)}, lookback={lookback}, horizon={horizon}")
    starts = np.arange(n_possible)
    if max_samples and len(starts) > max_samples:
        starts = np.unique(np.linspace(0, n_possible - 1, max_samples, dtype=int))
    X = np.stack([arr[s:s+lookback] for s in starts], axis=0)
    y_t = np.stack([arr[s+lookback:s+lookback+horizon, 0] for s in starts], axis=0)
    y_s = np.stack([arr[s+lookback:s+lookback+horizon, 1] for s in starts], axis=0)
    return X, y_t, y_s


def last_history_window(component_df: pd.DataFrame, lookback: int) -> np.ndarray:
    if len(component_df) < lookback:
        raise ValueError("History shorter than lookback window")
    return component_df[["trend", "seasonal"]].iloc[-lookback:].to_numpy(dtype=np.float64)[None, ...]


class MultiOutputRidge:
    def __init__(self, ridge_lambda: float):
        self.ridge_lambda = ridge_lambda
        self.beta = None

    def fit(self, Z: np.ndarray, Y: np.ndarray):
        eye = np.eye(Z.shape[1], dtype=np.float64)
        self.beta = np.linalg.solve(Z.T @ Z + self.ridge_lambda * eye, Z.T @ Y)
        return self

    def predict(self, Z: np.ndarray):
        return Z @ self.beta


class BLSDirect:
    def __init__(self, horizon: int, feature_nodes=64, enhancement_nodes=32, ridge_lambda=1e-3, seed=42):
        self.horizon = horizon
        self.feature_nodes = feature_nodes
        self.enhancement_nodes = enhancement_nodes
        self.ridge_lambda = ridge_lambda
        self.seed = seed
        self.x_scaler = StandardScaler()
        self.y_scaler = StandardScaler()
        self.Wf = None
        self.We = None
        self.ridge = None

    @staticmethod
    def sigmoid(x):
        x = np.clip(x, -40, 40)
        return 1.0 / (1.0 + np.exp(-x))

    def _features(self, X):
        flat = X.reshape(len(X), -1)
        Xs = self.x_scaler.transform(flat)
        H1 = self.sigmoid(Xs @ self.Wf)
        H2 = self.sigmoid(H1 @ self.We)
        return np.hstack([H1, H2])

    def fit(self, X, Y):
        rng = np.random.default_rng(self.seed)
        flat = X.reshape(len(X), -1)
        Xs = self.x_scaler.fit_transform(flat)
        Ys = self.y_scaler.fit_transform(Y)
        self.Wf = rng.standard_normal((Xs.shape[1], self.feature_nodes)) / np.sqrt(max(1, Xs.shape[1]))
        H1 = self.sigmoid(Xs @ self.Wf)
        self.We = rng.standard_normal((self.feature_nodes, self.enhancement_nodes)) / np.sqrt(self.feature_nodes)
        H2 = self.sigmoid(H1 @ self.We)
        Z = np.hstack([H1, H2])
        self.ridge = MultiOutputRidge(self.ridge_lambda).fit(Z, Ys)
        return self

    def predict(self, X):
        Ys = self.ridge.predict(self._features(X))
        return self.y_scaler.inverse_transform(Ys)


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 4096):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float32) * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x):
        return x + self.pe[:, :x.size(1)]


class TransformerNet(nn.Module):
    def __init__(self, input_dim=2, d_model=32, n_heads=4, num_layers=1, ff_dim=64, dropout=0.1, horizon=96):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, d_model)
        self.pos = PositionalEncoding(d_model)
        layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=n_heads, dim_feedforward=ff_dim,
                                           dropout=dropout, batch_first=True, activation="gelu")
        self.encoder = nn.TransformerEncoder(layer, num_layers=num_layers)
        self.head = nn.Linear(d_model, horizon)

    def forward(self, x):
        z = self.pos(self.input_proj(x))
        z = self.encoder(z)
        return self.head(z.mean(dim=1))


class TransformerDirect:
    def __init__(self, horizon, d_model=32, n_heads=4, num_layers=1, ff_dim=64, dropout=0.1,
                 lr=1e-3, batch_size=64, max_epochs=50, patience=5, seed=42, device=None):
        self.horizon = horizon
        self.kw = dict(d_model=d_model, n_heads=n_heads, num_layers=num_layers, ff_dim=ff_dim, dropout=dropout)
        self.lr = lr
        self.batch_size = batch_size
        self.max_epochs = max_epochs
        self.patience = patience
        self.seed = seed
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.x_scaler = StandardScaler()
        self.y_scaler = StandardScaler()
        self.model = None
        self.epochs_trained = 0

    def _scale_X_fit(self, X):
        n, l, c = X.shape
        return self.x_scaler.fit_transform(X.reshape(-1, c)).reshape(n, l, c)

    def _scale_X(self, X):
        n, l, c = X.shape
        return self.x_scaler.transform(X.reshape(-1, c)).reshape(n, l, c)

    def fit(self, X, Y, X_val=None, Y_val=None):
        set_seed(self.seed)
        Xs = self._scale_X_fit(X)
        Ys = self.y_scaler.fit_transform(Y)
        self.model = TransformerNet(input_dim=X.shape[-1], horizon=self.horizon, **self.kw).to(self.device)
        opt = optim.Adam(self.model.parameters(), lr=self.lr)
        loss_fn = nn.MSELoss()
        Xt = torch.tensor(Xs, dtype=torch.float32)
        Yt = torch.tensor(Ys, dtype=torch.float32)
        if X_val is not None and Y_val is not None and len(X_val) > 0:
            Xv = torch.tensor(self._scale_X(X_val), dtype=torch.float32, device=self.device)
            Yv = torch.tensor(self.y_scaler.transform(Y_val), dtype=torch.float32, device=self.device)
        else:
            Xv = Yv = None
        best_state = None
        best_val = float("inf")
        bad = 0
        for epoch in range(self.max_epochs):
            self.model.train()
            g = torch.Generator().manual_seed(self.seed + epoch)
            perm = torch.randperm(len(Xt), generator=g)
            for start in range(0, len(Xt), self.batch_size):
                idx = perm[start:start+self.batch_size]
                xb = Xt[idx].to(self.device)
                yb = Yt[idx].to(self.device)
                opt.zero_grad(set_to_none=True)
                loss = loss_fn(self.model(xb), yb)
                loss.backward()
                opt.step()
            self.epochs_trained = epoch + 1
            if Xv is not None:
                self.model.eval()
                with torch.no_grad():
                    val_loss = float(loss_fn(self.model(Xv), Yv).item())
                if val_loss < best_val - 1e-8:
                    best_val = val_loss
                    bad = 0
                    best_state = {k: v.detach().cpu().clone() for k, v in self.model.state_dict().items()}
                else:
                    bad += 1
                    if bad >= self.patience:
                        break
        if best_state is not None:
            self.model.load_state_dict(best_state)
        self.model.eval()
        return self

    def predict(self, X):
        self.model.eval()
        Xs = self._scale_X(X)
        with torch.no_grad():
            pred = self.model(torch.tensor(Xs, dtype=torch.float32, device=self.device)).cpu().numpy()
        return self.y_scaler.inverse_transform(pred)


class FixedAttentionTransformer(nn.Module):
    def __init__(self, d_model=64, n_heads=4, num_layers=1, ff_dim=128, dropout=0.1, use_attention=True, use_transformer=True):
        super().__init__()
        self.use_attention = use_attention
        self.use_transformer = use_transformer
        self.feature_attention = nn.Linear(d_model, d_model) if use_attention else None
        self.pos = PositionalEncoding(d_model)
        if use_transformer:
            layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=n_heads, dim_feedforward=ff_dim,
                                               dropout=dropout, batch_first=True, activation="gelu")
            self.encoder = nn.TransformerEncoder(layer, num_layers=num_layers)
        else:
            self.encoder = None

    def forward(self, h):
        if self.use_attention:
            weights = torch.softmax(self.feature_attention(h), dim=-1)
            h = h * weights
        if self.use_transformer:
            h = self.pos(h)
            h = self.encoder(h)
        return h


class ABLASTDirect:
    def __init__(self, horizon, feature_nodes=64, enhancement_nodes=32, n_heads=4, num_layers=1,
                 ff_dim=128, dropout=0.1, ridge_lambda=1e-3, seed=42, device=None,
                 use_attention=True, use_transformer=True, use_enhancement=True):
        self.horizon = horizon
        self.feature_nodes = feature_nodes
        self.enhancement_nodes = enhancement_nodes
        self.n_heads = n_heads
        self.num_layers = num_layers
        self.ff_dim = ff_dim
        self.dropout = dropout
        self.ridge_lambda = ridge_lambda
        self.seed = seed
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.use_attention = use_attention
        self.use_transformer = use_transformer
        self.use_enhancement = use_enhancement
        if feature_nodes % n_heads != 0:
            raise ValueError("feature_nodes must be divisible by n_heads")
        self.x_scaler = StandardScaler()
        self.y_scaler = StandardScaler()
        self.Wf = None
        self.We = None
        self.fixed_transform = None
        self.ridge = None

    @staticmethod
    def sigmoid(x):
        x = np.clip(x, -40, 40)
        return 1.0 / (1.0 + np.exp(-x))

    def _fit_scale_X(self, X):
        n, l, c = X.shape
        return self.x_scaler.fit_transform(X.reshape(-1, c)).reshape(n, l, c)

    def _scale_X(self, X):
        n, l, c = X.shape
        return self.x_scaler.transform(X.reshape(-1, c)).reshape(n, l, c)

    def _representation(self, Xs):
        H1 = self.sigmoid(np.einsum("nlc,cf->nlf", Xs, self.Wf))
        H1_t = torch.tensor(H1, dtype=torch.float32, device=self.device)
        self.fixed_transform.eval()
        with torch.no_grad():
            Ht_seq = self.fixed_transform(H1_t)
            Ht = Ht_seq.mean(dim=1).cpu().numpy()
        H1_pool = H1.mean(axis=1)
        Hc = 0.5 * (H1_pool + Ht) if self.use_transformer or self.use_attention else H1_pool
        if self.use_enhancement:
            He = self.sigmoid(Hc @ self.We)
            return np.hstack([Hc, He])
        return Hc

    def fit(self, X, Y):
        set_seed(self.seed)
        rng = np.random.default_rng(self.seed)
        Xs = self._fit_scale_X(X)
        Ys = self.y_scaler.fit_transform(Y)
        self.Wf = rng.standard_normal((X.shape[-1], self.feature_nodes)) / np.sqrt(max(1, X.shape[-1]))
        self.We = rng.standard_normal((self.feature_nodes, self.enhancement_nodes)) / np.sqrt(self.feature_nodes)
        set_seed(self.seed + 1000)
        self.fixed_transform = FixedAttentionTransformer(
            d_model=self.feature_nodes, n_heads=self.n_heads, num_layers=self.num_layers,
            ff_dim=self.ff_dim, dropout=self.dropout,
            use_attention=self.use_attention, use_transformer=self.use_transformer,
        ).to(self.device)
        for p in self.fixed_transform.parameters():
            p.requires_grad_(False)
        self.fixed_transform.eval()
        Z = self._representation(Xs)
        self.ridge = MultiOutputRidge(self.ridge_lambda).fit(Z, Ys)
        return self

    def predict(self, X):
        Xs = self._scale_X(X)
        Z = self._representation(Xs)
        Ys = self.ridge.predict(Z)
        return self.y_scaler.inverse_transform(Ys)


@dataclass
class ExperimentConfig:
    train_ratio: float = 0.70
    val_ratio: float = 0.15
    lookback: int = 365
    stl_period: int = 365
    stl_robust: bool = True
    horizons: Tuple[int, ...] = (96, 192, 336, 720)
    seeds: Tuple[int, ...] = (11, 22, 33, 44, 55)
    n_test_origins: int = 5
    max_train_samples: int = 5000
    ridge_grid: Tuple[float, ...] = (1e-5, 1e-3, 1e-1)
    feature_nodes: int = 64
    enhancement_nodes: int = 32
    n_heads: int = 4
    num_layers: int = 1
    ff_dim: int = 128
    dropout: float = 0.1
    transformer_d_model: int = 32
    transformer_heads: int = 4
    transformer_layers: int = 1
    transformer_ff_dim: int = 64
    transformer_dropout: float = 0.1
    transformer_lr: float = 1e-3
    transformer_batch_size: int = 64
    transformer_max_epochs: int = 50
    transformer_patience: int = 5
    selection_seed: int = 42
    inference_warmup: int = 3
    inference_repeats: int = 20
    ablation_horizon: int = 96


def raw_metrics(y_true: np.ndarray, y_pred: np.ndarray):
    return {
        "mse": float(mean_squared_error(y_true, y_pred)),
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
    }


def combine_component_predictions(t, s):
    return np.asarray(t).reshape(-1) + np.asarray(s).reshape(-1)


def build_model(model_name: str, horizon: int, ridge_lambda: float, seed: int, cfg: ExperimentConfig, device,
                ablation_variant: str | None = None):
    if model_name == "BLS":
        return BLSDirect(horizon, cfg.feature_nodes, cfg.enhancement_nodes, ridge_lambda, seed)
    if model_name == "A-BLAST":
        flags = {"use_attention": True, "use_transformer": True, "use_enhancement": True}
        if ablation_variant == "w/o attention":
            flags["use_attention"] = False
        elif ablation_variant == "w/o Transformer":
            flags["use_transformer"] = False
        elif ablation_variant == "w/o enhancement":
            flags["use_enhancement"] = False
        return ABLASTDirect(
            horizon, cfg.feature_nodes, cfg.enhancement_nodes, cfg.n_heads, cfg.num_layers,
            cfg.ff_dim, cfg.dropout, ridge_lambda, seed, device, **flags
        )
    if model_name == "Transformer":
        return TransformerDirect(
            horizon, cfg.transformer_d_model, cfg.transformer_heads, cfg.transformer_layers,
            cfg.transformer_ff_dim, cfg.transformer_dropout, cfg.transformer_lr,
            cfg.transformer_batch_size, cfg.transformer_max_epochs, cfg.transformer_patience,
            seed, device
        )
    raise KeyError(model_name)


def fit_two_components(model_name, X, yt, ys, horizon, ridge_lambda, seed, cfg, device,
                       X_val=None, yt_val=None, ys_val=None, ablation_variant=None):
    mt = build_model(model_name, horizon, ridge_lambda, seed, cfg, device, ablation_variant)
    ms = build_model(model_name, horizon, ridge_lambda, seed + 10000, cfg, device, ablation_variant)
    if model_name == "Transformer":
        mt.fit(X, yt, X_val, yt_val)
        ms.fit(X, ys, X_val, ys_val)
    else:
        mt.fit(X, yt)
        ms.fit(X, ys)
    return mt, ms


def select_ridge_lambda(model_name, Xtr, yt, ys, Xval_origin, yval_raw, horizon, cfg, device):
    """Validation-only ridge selection. Transformer architecture is fixed a priori."""
    if model_name == "Transformer":
        return 0.0
    best = None
    for lam in cfg.ridge_grid:
        mt, ms = fit_two_components(model_name, Xtr, yt, ys, horizon, lam, cfg.selection_seed, cfg, device)
        pred = combine_component_predictions(mt.predict(Xval_origin)[0], ms.predict(Xval_origin)[0])
        score = mean_absolute_error(yval_raw[:horizon], pred)
        if best is None or score < best[0]:
            best = (score, lam)
    return float(best[1])


def _prepare(dataset_name: str, y: pd.Series, cfg: ExperimentConfig):
    train, val, test = chronological_split(y, cfg.train_ratio, cfg.val_ratio)
    if max(cfg.horizons) > len(val) or max(cfg.horizons) > len(test):
        raise ValueError(
            f"Validation/test partition too short for max horizon {max(cfg.horizons)}: val={len(val)}, test={len(test)}"
        )
    train_comp = stl_history_only(train, cfg.stl_period, cfg.stl_robust)
    trainval = pd.concat([train, val])
    trainval_comp = stl_history_only(trainval, cfg.stl_period, cfg.stl_robust)
    test_start_idx = len(trainval)
    return train, val, test, train_comp, trainval, trainval_comp, test_start_idx


def rolling_origin_indices(n_total: int, test_start_idx: int, horizon: int, n_origins: int) -> List[int]:
    """Evenly spaced forecast starts within the test interval.

    Later origins may use earlier test observations as *past observed inputs*, but no target at or after
    the current forecast start is used for preprocessing, fitting, tuning, or feature construction.
    Models remain fixed after fitting on train+validation; there is no test-period refitting.
    """
    last_start = n_total - horizon
    if last_start < test_start_idx:
        raise ValueError(f"No feasible test origin for H={horizon}")
    span = last_start - test_start_idx
    if span < n_origins - 1:
        raise ValueError(
            f"Need at least {n_origins} distinct origins for H={horizon}; feasible start span is only {span} days"
        )
    arr = np.linspace(test_start_idx, last_start, n_origins)
    idx = np.rint(arr).astype(int)
    # Enforce strict monotonicity in the unlikely event of rounding collisions.
    for i in range(1, len(idx)):
        if idx[i] <= idx[i-1]:
            idx[i] = idx[i-1] + 1
    if idx[-1] > last_start:
        shift = idx[-1] - last_start
        idx -= shift
    if len(np.unique(idx)) != n_origins:
        raise RuntimeError("Could not construct distinct rolling origins")
    return [int(x) for x in idx]


def build_origin_cache(y: pd.Series, test_start_idx: int, horizon: int, cfg: ExperimentConfig):
    origins = rolling_origin_indices(len(y), test_start_idx, horizon, cfg.n_test_origins)
    out = []
    for j, pos in enumerate(origins, start=1):
        history = y.iloc[:pos]
        comp = stl_history_only(history, cfg.stl_period, cfg.stl_robust)
        X_origin = last_history_window(comp, cfg.lookback)
        truth = y.iloc[pos:pos+horizon].to_numpy(dtype=float)
        dates = y.index[pos:pos+horizon]
        out.append({
            "origin_id": f"O{j}",
            "origin_number": j,
            "origin_index": int(pos),
            "origin_last_observed_date": str(pd.Timestamp(y.index[pos-1]).date()),
            "forecast_start": str(pd.Timestamp(dates[0]).date()),
            "forecast_end": str(pd.Timestamp(dates[-1]).date()),
            "X_origin": X_origin,
            "y_true": truth,
            "dates": dates,
        })
    return out


def _atomic_csv(df: pd.DataFrame, path: Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    df.to_csv(tmp, index=False)
    os.replace(tmp, path)


def _atomic_json(obj, path: Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def _safe_read_csv(path: Path, columns=None) -> pd.DataFrame:
    path = Path(path)
    if not path.exists() or path.stat().st_size == 0:
        return pd.DataFrame(columns=columns)
    try:
        return pd.read_csv(path)
    except Exception:
        backup = path.with_name(path.name + ".corrupt")
        try:
            path.replace(backup)
        except Exception:
            pass
        return pd.DataFrame(columns=columns)


def _summarize_metrics(df: pd.DataFrame) -> pd.DataFrame:
    """Temporal summary: average seeds within each origin, then mean±SD across origins."""
    if df.empty:
        return pd.DataFrame()
    origin = (df.groupby(["dataset", "horizon", "model", "origin_id"], as_index=False)
              .agg(mse=("mse", "mean"), mae=("mae", "mean"), rmse=("rmse", "mean")))
    out = (origin.groupby(["dataset", "horizon", "model"], as_index=False)
           .agg(mse_mean=("mse", "mean"), mse_sd=("mse", "std"),
                mae_mean=("mae", "mean"), mae_sd=("mae", "std"),
                rmse_mean=("rmse", "mean"), rmse_sd=("rmse", "std"),
                n_origins=("origin_id", "nunique")))
    counts = (df.groupby(["dataset", "horizon", "model"], as_index=False)
              .size().rename(columns={"size": "n_seed_origin_runs"}))
    return out.merge(counts, on=["dataset", "horizon", "model"], how="left")


def _summarize_timing(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame()
    return (df.groupby(["dataset", "horizon", "model"], as_index=False)
            .agg(training_time_mean_s=("training_time_s", "mean"),
                 training_time_sd_s=("training_time_s", "std"),
                 prediction_time_mean_s=("prediction_time_mean_across_origins_s", "mean"),
                 prediction_time_sd_s=("prediction_time_mean_across_origins_s", "std"),
                 total_time_mean_s=("total_time_s", "mean"),
                 total_time_sd_s=("total_time_s", "std"),
                 n_training_runs=("seed", "nunique")))


def _dataset_checkpoint_status(out_dir: Path, cfg: ExperimentConfig, metrics: pd.DataFrame, preds: pd.DataFrame, timing: pd.DataFrame) -> None:
    total = len(cfg.horizons) * 3 * len(cfg.seeds)
    completed = 0
    items = []
    for H in cfg.horizons:
        for model_name in ["BLS", "Transformer", "A-BLAST"]:
            for seed in cfg.seeds:
                if metrics.empty:
                    mcount = 0
                else:
                    mcount = int(((metrics["horizon"] == H) & (metrics["model"] == model_name) & (metrics["seed"] == seed)).sum())
                if preds.empty:
                    pcount = 0
                else:
                    pcount = int(((preds["horizon"] == H) & (preds["model"] == model_name) & (preds["seed"] == seed)).sum())
                if timing.empty:
                    tcount = 0
                else:
                    tcount = int(((timing["horizon"] == H) & (timing["model"] == model_name) & (timing["seed"] == seed)).sum())
                ok = bool(mcount == cfg.n_test_origins and pcount == H * cfg.n_test_origins and tcount == 1)
                if ok:
                    completed += 1
                items.append({"horizon": int(H), "model": model_name, "seed": int(seed), "complete": ok})
    _atomic_json({
        "updated_utc": datetime.now(timezone.utc).isoformat(),
        "completed_main_runs": completed,
        "total_main_runs": total,
        "remaining_main_runs": total - completed,
        "rolling_test_origins_per_horizon": cfg.n_test_origins,
        "items": items,
    }, out_dir / "checkpoint_status.json")


def run_dataset(dataset_name: str, y: pd.Series, out_dir: str, cfg: ExperimentConfig):
    """Fixed-model rolling-origin evaluation with resume-safe checkpoints.

    For each horizon/model/seed, the model is fitted exactly once on train+validation history.
    It is then evaluated at five pre-specified, evenly spaced forecast origins in the held-out test
    interval. At later origins, previously observed test values may enter the historical input window,
    as they would in deployment, but the fitted model is never updated using test observations.
    """
    import logging
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    warmup_device(device)
    index_check = validate_daily_index(y, dataset_name)
    if index_check["n_missing_calendar_days"] != 0:
        raise ValueError(f"{dataset_name}: {index_check['n_missing_calendar_days']} missing calendar days detected. Protocol aborts rather than imputes.")

    train, val, test, train_comp, trainval, trainval_comp, test_start_idx = _prepare(dataset_name, y, cfg)

    metrics_path = out_dir / "metrics_by_seed_origin.csv"
    preds_path = out_dir / "predictions_by_seed_origin.csv"
    timing_path = out_dir / "timing_by_seed.csv"
    summary_path = out_dir / "metrics_summary.csv"
    timing_summary_path = out_dir / "timing_summary.csv"
    selected_path = out_dir / "selected_hyperparameters.json"
    metrics = _safe_read_csv(metrics_path)
    preds = _safe_read_csv(preds_path)
    timing = _safe_read_csv(timing_path)
    if selected_path.exists():
        try:
            selected = json.loads(selected_path.read_text(encoding="utf-8"))
        except Exception:
            selected = {}
    else:
        selected = {}

    origin_manifest = {}
    origin_cache = {}
    for H in cfg.horizons:
        cache = build_origin_cache(y, test_start_idx, H, cfg)
        origin_cache[H] = cache
        origin_manifest[str(H)] = [{k: v for k, v in o.items() if k not in {"X_origin", "y_true", "dates"}} for o in cache]

    _atomic_json(asdict(cfg), out_dir / "experiment_config.json")
    _atomic_json(environment_info(), out_dir / "environment.json")
    _atomic_json({
        "dataset": dataset_name,
        "n_total": len(y),
        "index_check": index_check,
        "train": {"n": len(train), "start": str(train.index[0].date()), "end": str(train.index[-1].date())},
        "validation": {"n": len(val), "start": str(val.index[0].date()), "end": str(val.index[-1].date())},
        "test": {"n": len(test), "start": str(test.index[0].date()), "end": str(test.index[-1].date())},
        "rolling_origins": origin_manifest,
        "fit_policy": "One final model fit per dataset × horizon × model × seed on train+validation; no test-period refitting.",
        "leakage_guard": "For every forecast origin, STL is fitted only to observations strictly before that origin. Future target values are never used in preprocessing, tuning, fitting, or feature construction.",
    }, out_dir / "split_manifest.json")
    _dataset_checkpoint_status(out_dir, cfg, metrics, preds, timing)

    for H in cfg.horizons:
        # Validation-only selection uses training history and the first H observations of validation.
        Xtr_sel, yt_sel, ys_sel = make_supervised(train_comp, cfg.lookback, H, cfg.max_train_samples)
        Xval_origin = last_history_window(train_comp, cfg.lookback)
        yval_raw = val.to_numpy(dtype=float)[:H]

        # Final fit uses train+validation only. The fitted model is held fixed across all test origins.
        Xtr, yt, ys = make_supervised(trainval_comp, cfg.lookback, H, cfg.max_train_samples)
        origins = origin_cache[H]

        for model_name in ["BLS", "Transformer", "A-BLAST"]:
            sel_key = f"{H}:{model_name}"
            if sel_key in selected and "ridge_lambda" in selected[sel_key]:
                lam = float(selected[sel_key]["ridge_lambda"])
            else:
                logging.info("%s H=%s %s: selecting locked hyperparameter on validation data...", dataset_name, H, model_name)
                lam = select_ridge_lambda(model_name, Xtr_sel, yt_sel, ys_sel, Xval_origin, yval_raw, H, cfg, device)
                selected[sel_key] = {"ridge_lambda": lam}
                _atomic_json(selected, selected_path)

            for seed in cfg.seeds:
                if metrics.empty:
                    mm = pd.Series([], dtype=bool)
                    mcount = 0
                else:
                    mm = ((metrics["dataset"] == dataset_name) & (metrics["horizon"] == H) &
                          (metrics["model"] == model_name) & (metrics["seed"] == seed))
                    mcount = int(mm.sum())
                if preds.empty:
                    pm = pd.Series([], dtype=bool)
                    pcount = 0
                else:
                    pm = ((preds["dataset"] == dataset_name) & (preds["horizon"] == H) &
                          (preds["model"] == model_name) & (preds["seed"] == seed))
                    pcount = int(pm.sum())
                if timing.empty:
                    tm = pd.Series([], dtype=bool)
                    tcount = 0
                else:
                    tm = ((timing["dataset"] == dataset_name) & (timing["horizon"] == H) &
                          (timing["model"] == model_name) & (timing["seed"] == seed))
                    tcount = int(tm.sum())
                if mcount == cfg.n_test_origins and pcount == H * cfg.n_test_origins and tcount == 1:
                    logging.info("RESUME skip complete: %s | H=%s | %s | seed=%s", dataset_name, H, model_name, seed)
                    continue

                if not metrics.empty and len(mm) == len(metrics):
                    metrics = metrics.loc[~mm].copy()
                if not preds.empty and len(pm) == len(preds):
                    preds = preds.loc[~pm].copy()
                if not timing.empty and len(tm) == len(timing):
                    timing = timing.loc[~tm].copy()

                logging.info("RUN: %s | H=%s | %s | seed=%s | %s rolling origins", dataset_name, H, model_name, seed, cfg.n_test_origins)
                if model_name == "Transformer":
                    cut = max(1, int(len(Xtr) * 0.9))
                    Xfit, ytfit, ysfit = Xtr[:cut], yt[:cut], ys[:cut]
                    Xes, ytes, yses = Xtr[cut:], yt[cut:], ys[cut:]
                    fit_fn = lambda: fit_two_components(model_name, Xfit, ytfit, ysfit, H, lam, seed, cfg, device, Xes, ytes, yses)
                else:
                    fit_fn = lambda: fit_two_components(model_name, Xtr, yt, ys, H, lam, seed, cfg, device)
                (mt, ms), train_time = timed_call(fit_fn, device)

                metric_rows = []
                pred_rows = []
                pred_medians = []
                for o in origins:
                    def infer():
                        pt = mt.predict(o["X_origin"])[0]
                        ps = ms.predict(o["X_origin"])[0]
                        return pt, ps
                    (pt, ps), tinfo = timed_inference(infer, device, cfg.inference_warmup, cfg.inference_repeats)
                    pred_medians.append(tinfo["median_s"])
                    py = combine_component_predictions(pt, ps)
                    mets = raw_metrics(o["y_true"], py)
                    metric_rows.append({
                        "dataset": dataset_name, "horizon": H, "model": model_name, "seed": seed,
                        "origin_id": o["origin_id"], "origin_number": o["origin_number"],
                        "origin_index": o["origin_index"], "origin_last_observed_date": o["origin_last_observed_date"],
                        "forecast_start": o["forecast_start"], "forecast_end": o["forecast_end"],
                        "mse": mets["mse"], "mae": mets["mae"], "rmse": mets["rmse"],
                        "prediction_time_median_s": tinfo["median_s"],
                        "prediction_time_mean_s": tinfo["mean_s"],
                        "prediction_time_repeat_sd_s": tinfo["sd_s"],
                        "prediction_repeats": tinfo["n_repeats"],
                        "ridge_lambda": lam,
                    })
                    for i, dt in enumerate(o["dates"]):
                        pred_rows.append({
                            "dataset": dataset_name, "horizon": H, "model": model_name, "seed": seed,
                            "origin_id": o["origin_id"], "origin_number": o["origin_number"],
                            "origin_last_observed_date": o["origin_last_observed_date"],
                            "date": str(pd.Timestamp(dt).date()), "step": i + 1,
                            "y_true": float(o["y_true"][i]), "y_pred": float(py[i]),
                            "trend_pred": float(pt[i]), "seasonal_pred": float(ps[i]),
                        })

                pred_medians_arr = np.asarray(pred_medians, dtype=float)
                timing_row = {
                    "dataset": dataset_name, "horizon": H, "model": model_name, "seed": seed,
                    "training_time_s": train_time,
                    "prediction_time_mean_across_origins_s": float(pred_medians_arr.mean()),
                    "prediction_time_sd_across_origins_s": float(pred_medians_arr.std(ddof=1)) if len(pred_medians_arr) > 1 else 0.0,
                    "total_time_s": float(train_time + pred_medians_arr.mean()),
                    "n_origins": cfg.n_test_origins,
                    "ridge_lambda": lam,
                    "n_fit_windows": len(Xtr),
                    "epochs_trend": getattr(mt, "epochs_trained", None),
                    "epochs_seasonal": getattr(ms, "epochs_trained", None),
                }
                metrics = pd.concat([metrics, pd.DataFrame(metric_rows)], ignore_index=True)
                preds = pd.concat([preds, pd.DataFrame(pred_rows)], ignore_index=True)
                timing = pd.concat([timing, pd.DataFrame([timing_row])], ignore_index=True)
                _atomic_csv(metrics, metrics_path)
                _atomic_csv(preds, preds_path)
                _atomic_csv(timing, timing_path)
                _atomic_csv(_summarize_metrics(metrics), summary_path)
                _atomic_csv(_summarize_timing(timing), timing_summary_path)
                _atomic_json(selected, selected_path)
                _dataset_checkpoint_status(out_dir, cfg, metrics, preds, timing)
                logging.info("CHECKPOINT saved: %s | H=%s | %s | seed=%s | origins=%s", dataset_name, H, model_name, seed, cfg.n_test_origins)

    summary = _summarize_metrics(metrics)
    timing_summary = _summarize_timing(timing)
    _atomic_csv(metrics, metrics_path)
    _atomic_csv(preds, preds_path)
    _atomic_csv(timing, timing_path)
    _atomic_csv(summary, summary_path)
    _atomic_csv(timing_summary, timing_summary_path)
    _dataset_checkpoint_status(out_dir, cfg, metrics, preds, timing)
    return metrics, summary, timing, timing_summary


def _summarize_ablation(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame()
    origin = (df.groupby(["dataset", "horizon", "variant", "origin_id"], as_index=False)
              .agg(mse=("mse", "mean"), mae=("mae", "mean"), rmse=("rmse", "mean")))
    out = (origin.groupby(["dataset", "horizon", "variant"], as_index=False)
           .agg(mse_mean=("mse", "mean"), mse_sd=("mse", "std"),
                mae_mean=("mae", "mean"), mae_sd=("mae", "std"),
                rmse_mean=("rmse", "mean"), rmse_sd=("rmse", "std"),
                n_origins=("origin_id", "nunique")))
    counts = (df.groupby(["dataset", "horizon", "variant"], as_index=False)
              .size().rename(columns={"size": "n_seed_origin_runs"}))
    return out.merge(counts, on=["dataset", "horizon", "variant"], how="left")


def run_ablation(dataset_name: str, y: pd.Series, out_dir: str, cfg: ExperimentConfig, selected_hparams: Dict[str, dict]):
    """Rolling-origin ablation at H=96. Main BLS and Full A-BLAST results are reused; only three removal variants are newly fitted."""
    import logging
    out_dir = Path(out_dir)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train, val, test, train_comp, trainval, trainval_comp, test_start_idx = _prepare(dataset_name, y, cfg)
    H = cfg.ablation_horizon
    if H not in cfg.horizons:
        raise ValueError("ablation_horizon must be one of configured horizons")
    Xtr, yt, ys = make_supervised(trainval_comp, cfg.lookback, H, cfg.max_train_samples)
    origins = build_origin_cache(y, test_start_idx, H, cfg)
    lam = float(selected_hparams.get(f"{H}:A-BLAST", {}).get("ridge_lambda", cfg.ridge_grid[0]))
    path = out_dir / "ablation_by_seed_origin.csv"
    summary_path = out_dir / "ablation_summary.csv"
    df = _safe_read_csv(path)

    # Reuse exact main-experiment metrics for the two reference variants.
    main_path = out_dir / "metrics_by_seed_origin.csv"
    main = _safe_read_csv(main_path)
    if not main.empty:
        reuse = []
        for model_name, variant in [("BLS", "BLS baseline"), ("A-BLAST", "Full A-BLAST")]:
            s = main[(main.horizon == H) & (main.model == model_name)].copy()
            if not s.empty:
                reuse.append(s[["dataset", "horizon", "seed", "origin_id", "origin_number", "mse", "mae", "rmse"]].assign(variant=variant))
        if reuse:
            reused = pd.concat(reuse, ignore_index=True)
            if df.empty:
                df = reused
            else:
                keys = set(zip(df.dataset, df.horizon, df.variant, df.seed, df.origin_id))
                reused = reused[[ (r.dataset, r.horizon, r.variant, r.seed, r.origin_id) not in keys for r in reused.itertuples() ]]
                df = pd.concat([df, reused], ignore_index=True)
            _atomic_csv(df, path)

    variants = ["w/o attention", "w/o Transformer", "w/o enhancement"]
    for variant in variants:
        for seed in cfg.seeds:
            if df.empty:
                count = 0
                mask = pd.Series([], dtype=bool)
            else:
                mask = ((df["dataset"] == dataset_name) & (df["horizon"] == H) &
                        (df["variant"] == variant) & (df["seed"] == seed))
                count = int(mask.sum())
            if count == cfg.n_test_origins:
                logging.info("RESUME skip ablation: %s | %s | seed=%s", dataset_name, variant, seed)
                continue
            if not df.empty and len(mask) == len(df):
                df = df.loc[~mask].copy()
            logging.info("ABLATION RUN: %s | %s | seed=%s | %s origins", dataset_name, variant, seed, cfg.n_test_origins)
            mt, ms = fit_two_components("A-BLAST", Xtr, yt, ys, H, lam, seed, cfg, device, ablation_variant=variant)
            rows = []
            for o in origins:
                py = combine_component_predictions(mt.predict(o["X_origin"])[0], ms.predict(o["X_origin"])[0])
                mets = raw_metrics(o["y_true"], py)
                rows.append({
                    "dataset": dataset_name, "horizon": H, "variant": variant, "seed": seed,
                    "origin_id": o["origin_id"], "origin_number": o["origin_number"], **mets,
                })
            df = pd.concat([df, pd.DataFrame(rows)], ignore_index=True)
            _atomic_csv(df, path)
            _atomic_csv(_summarize_ablation(df), summary_path)
            completed_groups = (df.groupby(["variant", "seed"]).size() == cfg.n_test_origins).sum()
            total_groups = 5 * len(cfg.seeds)
            _atomic_json({
                "updated_utc": datetime.now(timezone.utc).isoformat(),
                "completed_ablation_seed_groups": int(completed_groups),
                "total_ablation_seed_groups": int(total_groups),
                "remaining_ablation_seed_groups": int(total_groups - completed_groups),
                "rolling_test_origins_per_group": cfg.n_test_origins,
                "note": "BLS baseline and Full A-BLAST are reused from the main experiment; three removal variants are newly fitted.",
            }, out_dir / "ablation_checkpoint_status.json")
            logging.info("ABLATION CHECKPOINT saved: %s | %s | seed=%s", dataset_name, variant, seed)

    summ = _summarize_ablation(df)
    _atomic_csv(df, path)
    _atomic_csv(summ, summary_path)
    return df, summ
