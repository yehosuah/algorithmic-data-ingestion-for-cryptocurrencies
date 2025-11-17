from __future__ import annotations
import json
from pathlib import Path
from typing import List, Tuple, Optional, Sequence, Dict, Any

import yaml

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
            subset = ["timestamp", "symbol"] if "symbol" in df.columns else ["timestamp"]
            df = (
                df.sort_values(subset)
                .drop_duplicates(subset=subset, keep="last")
            )
    return df.reset_index(drop=True)


def sanitize_market_dataset(
    df: pd.DataFrame,
    *,
    price_outlier_factor: float = 10.0,
    logret_cap: float = 0.25,
    verbose: bool = False,
) -> pd.DataFrame:
    required = {"timestamp", "symbol", "close"}
    if not required.issubset(df.columns):
        return df
    out = df.copy()
    out = out.dropna(subset=["timestamp", "symbol", "close"])
    out["timestamp"] = pd.to_datetime(out["timestamp"], utc=True)
    out["close"] = out["close"].astype(float)
    out = out[out["close"] > 0].copy()

    if price_outlier_factor and price_outlier_factor > 0:
        medians = out.groupby("symbol")["close"].transform("median").abs()
        lower = medians / price_outlier_factor
        upper = medians * price_outlier_factor
        mask = out["close"].between(lower, upper)
        dropped = len(out) - int(mask.sum())
        if dropped and verbose:
            print(
                f"[Sanitize] Dropped {dropped} rows outside ±{price_outlier_factor}x median close"
            )
        out = out[mask].copy()

    out = out.sort_values(["symbol", "timestamp"]).reset_index(drop=True)
    grouped = out.groupby("symbol", group_keys=False)

    out["ret_1"] = grouped["close"].pct_change()

    def _logret(series: pd.Series) -> pd.Series:
        return np.log(series.replace(0.0, np.nan)).diff()

    logret = grouped["close"].transform(_logret)
    if logret_cap and logret_cap > 0:
        logret = logret.clip(lower=-logret_cap, upper=logret_cap)
    out["logret_1"] = logret

    for window, col in ((5, "rvol_5"), (20, "rvol_20")):
        out[col] = grouped["logret_1"].transform(
            lambda s, w=window: s.rolling(w, min_periods=w).std()
        )

    out[["ret_1", "logret_1", "rvol_5", "rvol_20"]] = out[
        ["ret_1", "logret_1", "rvol_5", "rvol_20"]
    ].fillna(0.0)

    cols_to_drop = [col for col in ("ret_next", "y_dir") if col in out.columns]
    if cols_to_drop:
        out = out.drop(columns=cols_to_drop)

    return out.sort_values("timestamp").reset_index(drop=True)


def sliding_windows(
    df: pd.DataFrame,
    *,
    window: int = 32,
    series_cols: Optional[Sequence[str]] = None,
    y_col: str = "y_dir",
    standardize_per_window: bool = True,
    stride: int = 1,
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
        if "base_prob" in df.columns:
            default_cols.append("base_prob")
        series_cols = [c for c in default_cols if c in df.columns]
        if not series_cols:
            raise ValueError("No suitable series columns found in dataset for TCN windows")

    series_df = df[series_cols].astype(float).replace([np.inf, -np.inf], np.nan)
    # Forward/backward fill to handle rolling-feature warmups; fall back to zeros if still missing
    series_df = series_df.ffill().bfill().fillna(0.0)
    vals = series_df.values
    if not np.isfinite(vals).all():
        raise ValueError("Series columns contain non-finite values after preprocessing; inspect dataset")
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
    if stride < 1:
        raise ValueError("stride must be >= 1")

    # Build windows with stride 1
    L = window
    starts = list(range(0, n - L, stride))
    N = len(starts)
    X = np.empty((N, c, L), dtype=np.float32)
    y_buf: List[int] = []
    ts_buf: List[pd.Timestamp] = []
    for idx, start in enumerate(starts):
        seg = vals[start:start+L, :].T  # (C, L)
        if standardize_per_window:
            # Per-window standardization for stability (channel-wise)
            m = seg.mean(axis=1, keepdims=True)
            s = seg.std(axis=1, keepdims=True) + 1e-6
            seg = (seg - m) / s
        X[idx] = seg
        if y is not None:
            y_buf.append(int(y[start + L]))
        if len(ts):
            ts_buf.append(ts.iloc[start + L])

    y_out = np.array(y_buf, dtype=np.int64) if y is not None else np.array([], dtype=np.int64)
    if ts_buf:
        ts_out = pd.Series(ts_buf, dtype="datetime64[ns, UTC]")
    else:
        ts_out = pd.Series(dtype="datetime64[ns, UTC]")
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


def load_canonical_contract(path: str | Path) -> Dict[str, Any]:
    """
    Load the canonical training contract (YAML/JSON) generated in task 7.1.
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Contract not found: {p}")
    text = p.read_text()
    if p.suffix.lower() in {".json"}:
        contract = json.loads(text)
    else:
        contract = yaml.safe_load(text)
    contract = contract or {}
    contract["_contract_path"] = str(p)
    return contract


def _resolve_dataset_path(contract: Dict[str, Any]) -> Path:
    ds = contract.get("dataset", {}) if isinstance(contract, dict) else {}
    raw_path = ds.get("path")
    if raw_path is None:
        raise KeyError("Dataset path missing in contract under dataset.path")
    base = Path(contract.get("_contract_path", "")).expanduser()
    root = base.parent if base.exists() else Path(".")
    ds_path = Path(raw_path)
    candidates = []
    if not ds_path.is_absolute():
        # Prefer project-root relative (e.g., data/...) even if contract is under configs/
        candidates.append((root.parent / ds_path).resolve())
        candidates.append((root / ds_path).resolve())
        candidates.append(ds_path.resolve())
    else:
        candidates.append(ds_path)
    for c in candidates:
        if c.exists():
            return c
    # Fallback to first candidate for error message
    return candidates[0] if candidates else ds_path


def load_training_dataset(contract: Dict[str, Any]) -> pd.DataFrame:
    """
    Load the canonical dataset defined by the contract and attach metadata.

    The returned DataFrame is sorted by timestamp & symbol and exposes
    feature_cols, label_col and regime_cols via ``df.attrs``.
    """
    ds_path = _resolve_dataset_path(contract)
    df = load_parquet_dataset(ds_path)
    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    sort_cols = [c for c in ["symbol", "timestamp"] if c in df.columns]
    if sort_cols:
        df = df.sort_values(sort_cols).reset_index(drop=True)

    feature_cols = contract.get("features", {}).get("core", []) if isinstance(contract, dict) else []
    label_col = contract.get("labels", {}).get("primary") if isinstance(contract, dict) else None
    regime_cols = []
    for dim in contract.get("regimes", {}).get("dimensions", []):
        col = dim.get("column")
        if col:
            regime_cols.append(col)

    df.attrs["feature_cols"] = feature_cols
    df.attrs["label_col"] = label_col
    df.attrs["regime_cols"] = regime_cols
    df.attrs["contract"] = contract
    return df
