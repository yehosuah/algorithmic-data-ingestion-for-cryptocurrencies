from __future__ import annotations
from pathlib import Path
from typing import List, Tuple, Optional, Sequence, Dict

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler


def load_parquet_dataset(path: str | Path, *, drop_duplicates: bool = True) -> pd.DataFrame:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Dataset not found: {p}")
    df = pd.read_parquet(p)
    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
        if drop_duplicates:
            df = df.sort_values("timestamp").drop_duplicates(subset=["timestamp"], keep="last")
    return df.reset_index(drop=True)


def sliding_windows(
    df: pd.DataFrame,
    *,
    window: int = 32,
    series_cols: Optional[Sequence[str]] = None,
    y_col: str = "y_dir",
    standardize_per_window: bool = True,
) -> Tuple[np.ndarray, np.ndarray, pd.Series, List[str], StandardScaler | None]:
    """
    Build a 3D tensor (N, C, L) for TCN/CNN models from sequential features.
    Returns (X, y, ts, used_cols, scaler)
    - X: float32 array, shape (N, C, L)
    - y: int64 array, shape (N,)
    - ts: timestamps for the window end row
    - used_cols: the feature channels used
    - scaler: optional StandardScaler applied across dataset channels
    """
    if series_cols is None:
        # Default to market micro-structure oriented channels
        # Assumes dataset built via `build_market_features`
        default_cols = [
            "ret_1", "logret_1", "rvol_5", "rvol_20", "macd", "macd_signal_9", "rsi_14", "hl_spread", "oi_obv",
        ]
        series_cols = [c for c in default_cols if c in df.columns]
        if not series_cols:
            raise ValueError("No suitable series columns found in dataset for TCN windows")

    vals = df[series_cols].astype(float).values
    y = df[y_col].astype(int).values if y_col in df.columns else None
    ts = pd.to_datetime(df["timestamp"], utc=True) if "timestamp" in df.columns else pd.Series(index=df.index, data=pd.NaT)

    # Global scaler across channels (fit on in-sample portion externally if needed)
    scaler: Optional[StandardScaler] = None
    try:
        scaler = StandardScaler(with_mean=True, with_std=True)
        vals = scaler.fit_transform(vals)
    except Exception:
        scaler = None

    n, c = vals.shape
    if n < window + 1:
        raise ValueError(f"Not enough rows for window={window}")

    # Build windows with stride 1
    L = window
    N = n - L
    X = np.empty((N, c, L), dtype=np.float32)
    for i in range(N):
        seg = vals[i:i+L, :].T  # (C, L)
        if standardize_per_window:
            # Per-window standardization for stability (channel-wise)
            m = seg.mean(axis=1, keepdims=True)
            s = seg.std(axis=1, keepdims=True) + 1e-6
            seg = (seg - m) / s
        X[i] = seg

    y_out = y[L:] if y is not None else np.array([], dtype=np.int64)
    ts_out = ts.iloc[L:] if len(ts) else pd.Series(dtype="datetime64[ns, UTC]")
    return X, y_out.astype(np.int64), ts_out.reset_index(drop=True), list(series_cols), scaler


def select_market_features(df: pd.DataFrame, *, exclude: Optional[Sequence[str]] = None) -> Tuple[pd.DataFrame, pd.Series, List[str]]:
    """
    Extract tabular features for tree models, dropping non-feature columns.
    """
    non_feat = {"timestamp", "dt", "symbol", "exchange", "timeframe", "feature_version", "close", "ret_next", "y_dir"}
    if exclude:
        non_feat |= set(exclude)
    feat_cols = [c for c in df.columns if c not in non_feat]
    X = df[feat_cols].astype(float)
    y = df["y_dir"].astype(int) if "y_dir" in df.columns else pd.Series(index=df.index, dtype=int)
    return X, y, feat_cols


def ensure_labels(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if "ret_next" not in out.columns and "close" in out.columns:
        out["ret_next"] = out["close"].pct_change().shift(-1)
    if "y_dir" not in out.columns and "ret_next" in out.columns:
        out["y_dir"] = (out["ret_next"] > 0).astype(int)
    if "timestamp" in out.columns:
        out = out.iloc[:-1].copy()  # drop last unlabeled row
    return out

