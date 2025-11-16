from __future__ import annotations

from collections import defaultdict
from typing import Callable, Dict, Iterable, List

import numpy as np
import pandas as pd


"""
Registry-based feature builder for market_multi_3symbol_1m.parquet.
- Each callable takes the full dataframe (sorted by time, grouped by symbol)
  and returns a Series aligned to the input index.
- Rolling logic is strictly backward-looking to avoid leakage.
"""

FeatureFn = Callable[[pd.DataFrame], pd.Series]


def _symbol_keys(df: pd.DataFrame) -> pd.Series:
    if "symbol" in df.columns:
        return df["symbol"]
    return pd.Series(["all"] * len(df), index=df.index)


def _close_series(df: pd.DataFrame) -> pd.Series:
    if "close" in df.columns:
        return pd.to_numeric(df["close"], errors="coerce")
    raise ValueError("Dataset must contain 'close' for feature computation")


def _aligned(df: pd.DataFrame, series: pd.Series, name: str) -> pd.Series:
    s = pd.Series(series.values, index=df.index, name=name)
    return s


def _log_return_by_symbol(close: pd.Series, keys: pd.Series, periods: int) -> pd.Series:
    return close.groupby(keys).transform(lambda s: np.log(s.replace(0, np.nan)).diff(periods=periods))


def _rolling_sum(series: pd.Series, keys: pd.Series, window: int) -> pd.Series:
    return series.groupby(keys).transform(lambda s: s.rolling(window, min_periods=1).sum())


def _rolling_std(series: pd.Series, keys: pd.Series, window: int, min_periods: int = 2) -> pd.Series:
    return series.groupby(keys).transform(lambda s: s.rolling(window, min_periods=min_periods).std())


def feat_log_return_1m(df: pd.DataFrame) -> pd.Series:
    close = _close_series(df)
    keys = _symbol_keys(df)
    vals = _log_return_by_symbol(close, keys, periods=1)
    return _aligned(df, vals, "feat_log_return_1m")


def feat_log_return_5m(df: pd.DataFrame) -> pd.Series:
    close = _close_series(df)
    keys = _symbol_keys(df)
    vals = _log_return_by_symbol(close, keys, periods=5)
    return _aligned(df, vals, "feat_log_return_5m")


def feat_log_return_15m(df: pd.DataFrame) -> pd.Series:
    close = _close_series(df)
    keys = _symbol_keys(df)
    vals = _log_return_by_symbol(close, keys, periods=15)
    return _aligned(df, vals, "feat_log_return_15m")


def feat_rolling_return_15m(df: pd.DataFrame) -> pd.Series:
    close = _close_series(df)
    keys = _symbol_keys(df)
    vals = _rolling_sum(_log_return_by_symbol(close, keys, periods=1), keys, window=15)
    return _aligned(df, vals, "feat_rolling_return_15m")


def feat_rolling_return_1h(df: pd.DataFrame) -> pd.Series:
    close = _close_series(df)
    keys = _symbol_keys(df)
    vals = _rolling_sum(_log_return_by_symbol(close, keys, periods=1), keys, window=60)
    return _aligned(df, vals, "feat_rolling_return_1h")


def feat_realized_vol_15m(df: pd.DataFrame) -> pd.Series:
    close = _close_series(df)
    keys = _symbol_keys(df)
    logret1 = _log_return_by_symbol(close, keys, periods=1)
    vals = _rolling_std(logret1, keys, window=15, min_periods=2)
    return _aligned(df, vals, "feat_realized_vol_15m")


def feat_realized_vol_1h(df: pd.DataFrame) -> pd.Series:
    close = _close_series(df)
    keys = _symbol_keys(df)
    logret1 = _log_return_by_symbol(close, keys, periods=1)
    vals = _rolling_std(logret1, keys, window=60, min_periods=2)
    return _aligned(df, vals, "feat_realized_vol_1h")


def feat_vol_of_vol_1h(df: pd.DataFrame) -> pd.Series:
    vol_1h = feat_realized_vol_1h(df)
    keys = _symbol_keys(df)
    vals = _rolling_std(vol_1h, keys, window=60, min_periods=5)
    return _aligned(df, vals, "feat_vol_of_vol_1h")


def feat_spread_bps(df: pd.DataFrame) -> pd.Series:
    if "bid" in df.columns and "ask" in df.columns:
        mid = (pd.to_numeric(df["bid"], errors="coerce") + pd.to_numeric(df["ask"], errors="coerce")) / 2.0
        spread = pd.to_numeric(df["ask"], errors="coerce") - pd.to_numeric(df["bid"], errors="coerce")
        vals = (spread / mid.replace(0.0, np.nan)) * 1e4
    elif "hl_spread" in df.columns:
        vals = pd.to_numeric(df["hl_spread"], errors="coerce") * 1e4
    else:
        vals = pd.Series(np.nan, index=df.index)
    return _aligned(df, vals, "feat_spread_bps")


def feat_rolling_volume_15m(df: pd.DataFrame) -> pd.Series:
    if "volume" in df.columns:
        proxy = pd.to_numeric(df["volume"], errors="coerce")
    elif "oi_obv" in df.columns:
        proxy = pd.to_numeric(df["oi_obv"], errors="coerce").diff().abs()
    else:
        proxy = pd.Series(np.nan, index=df.index)
    keys = _symbol_keys(df)
    vals = _rolling_sum(proxy, keys, window=15)
    return _aligned(df, vals, "feat_rolling_volume_15m")


def feat_turnover_proxy_1h(df: pd.DataFrame) -> pd.Series:
    close = _close_series(df)
    if "volume" in df.columns:
        vol = pd.to_numeric(df["volume"], errors="coerce")
        proxy = close * vol
    elif "oi_obv" in df.columns:
        proxy = close * pd.to_numeric(df["oi_obv"], errors="coerce").abs()
    else:
        proxy = pd.Series(np.nan, index=df.index)
    keys = _symbol_keys(df)
    vals = proxy.groupby(keys).transform(lambda s: s.rolling(60, min_periods=1).mean())
    return _aligned(df, vals, "feat_turnover_proxy_1h")


def feat_relative_strength_vs_basket(df: pd.DataFrame) -> pd.Series:
    if "timestamp" not in df.columns:
        return pd.Series(np.nan, index=df.index, name="feat_relative_strength_vs_basket")
    close = _close_series(df)
    keys = _symbol_keys(df)
    logret_5 = _log_return_by_symbol(close, keys, periods=5)
    tmp = pd.DataFrame({"logret_5": logret_5, "timestamp": pd.to_datetime(df["timestamp"], utc=True), "symbol": keys})
    grouped_time = tmp.groupby("timestamp")
    count = grouped_time["logret_5"].transform("count")
    total = grouped_time["logret_5"].transform("sum")
    others_mean = (total - tmp["logret_5"]) / count.add(-1).replace(0, np.nan)
    rel = tmp["logret_5"] - others_mean
    return _aligned(df, rel, "feat_relative_strength_vs_basket")


def feat_cross_corr_15m(df: pd.DataFrame) -> pd.Series:
    if "timestamp" not in df.columns:
        return pd.Series(np.nan, index=df.index, name="feat_cross_corr_15m")
    close = _close_series(df)
    keys = _symbol_keys(df)
    logret_1 = _log_return_by_symbol(close, keys, periods=1)
    tmp = pd.DataFrame({"logret_1": logret_1, "timestamp": pd.to_datetime(df["timestamp"], utc=True), "symbol": keys})
    symbols = sorted(tmp["symbol"].dropna().unique().tolist())
    benchmark = symbols[0] if symbols else None
    if benchmark is None:
        return pd.Series(np.nan, index=df.index, name="feat_cross_corr_15m")
    bench_series = tmp[tmp["symbol"] == benchmark].set_index("timestamp")["logret_1"]
    out = pd.Series(np.nan, index=df.index, name="feat_cross_corr_15m")
    window = 15
    for sym in symbols:
        idx = tmp.index[tmp["symbol"] == sym]
        sym_df = tmp.loc[idx, ["timestamp", "logret_1"]].set_index("timestamp")
        aligned = sym_df.join(bench_series, how="left", lsuffix="_sym", rsuffix="_bench")
        corr = aligned["logret_1_sym"].rolling(window, min_periods=5).corr(aligned["logret_1_bench"])
        out.loc[idx] = corr.values
    return out


def feat_vol_bucket_id_raw(df: pd.DataFrame) -> pd.Series:
    vol = feat_realized_vol_1h(df)
    return _aligned(df, vol, "feat_vol_bucket_id_raw")


def feat_liquidity_metric_raw(df: pd.DataFrame) -> pd.Series:
    liq = feat_rolling_volume_15m(df)
    return _aligned(df, liq, "feat_liquidity_metric_raw")


def feat_hour_of_day_sin(df: pd.DataFrame) -> pd.Series:
    if "timestamp" not in df.columns:
        return pd.Series(np.nan, index=df.index, name="feat_hour_of_day_sin")
    ts = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
    minutes = ts.dt.hour * 60 + ts.dt.minute
    vals = np.sin(2 * np.pi * minutes / (24 * 60))
    return _aligned(df, pd.Series(vals, index=df.index), "feat_hour_of_day_sin")


def feat_hour_of_day_cos(df: pd.DataFrame) -> pd.Series:
    if "timestamp" not in df.columns:
        return pd.Series(np.nan, index=df.index, name="feat_hour_of_day_cos")
    ts = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
    minutes = ts.dt.hour * 60 + ts.dt.minute
    vals = np.cos(2 * np.pi * minutes / (24 * 60))
    return _aligned(df, pd.Series(vals, index=df.index), "feat_hour_of_day_cos")


def feat_day_of_week_sin(df: pd.DataFrame) -> pd.Series:
    if "timestamp" not in df.columns:
        return pd.Series(np.nan, index=df.index, name="feat_day_of_week_sin")
    ts = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
    dow = ts.dt.dayofweek.astype(float)
    vals = np.sin(2 * np.pi * dow / 7.0)
    return _aligned(df, pd.Series(vals, index=df.index), "feat_day_of_week_sin")


def feat_day_of_week_cos(df: pd.DataFrame) -> pd.Series:
    if "timestamp" not in df.columns:
        return pd.Series(np.nan, index=df.index, name="feat_day_of_week_cos")
    ts = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
    dow = ts.dt.dayofweek.astype(float)
    vals = np.cos(2 * np.pi * dow / 7.0)
    return _aligned(df, pd.Series(vals, index=df.index), "feat_day_of_week_cos")


FEATURE_REGISTRY: Dict[str, FeatureFn] = {
    # Returns / trend
    "feat_log_return_1m": feat_log_return_1m,
    "feat_log_return_5m": feat_log_return_5m,
    "feat_log_return_15m": feat_log_return_15m,
    "feat_rolling_return_15m": feat_rolling_return_15m,
    "feat_rolling_return_1h": feat_rolling_return_1h,
    # Volatility
    "feat_realized_vol_15m": feat_realized_vol_15m,
    "feat_realized_vol_1h": feat_realized_vol_1h,
    "feat_vol_of_vol_1h": feat_vol_of_vol_1h,
    # Microstructure / liquidity
    "feat_spread_bps": feat_spread_bps,
    "feat_rolling_volume_15m": feat_rolling_volume_15m,
    "feat_turnover_proxy_1h": feat_turnover_proxy_1h,
    # Cross-asset
    "feat_relative_strength_vs_basket": feat_relative_strength_vs_basket,
    "feat_cross_corr_15m": feat_cross_corr_15m,
    # Regime proto
    "feat_vol_bucket_id_raw": feat_vol_bucket_id_raw,
    "feat_liquidity_metric_raw": feat_liquidity_metric_raw,
    # Calendar
    "feat_hour_of_day_sin": feat_hour_of_day_sin,
    "feat_hour_of_day_cos": feat_hour_of_day_cos,
    "feat_day_of_week_sin": feat_day_of_week_sin,
    "feat_day_of_week_cos": feat_day_of_week_cos,
}


def apply_feature_registry(df: pd.DataFrame, features: Iterable[str] | None = None) -> pd.DataFrame:
    names = list(features) if features else list(FEATURE_REGISTRY.keys())
    computed: Dict[str, pd.Series] = {}
    for name in names:
        func = FEATURE_REGISTRY.get(name)
        if func is None:
            raise KeyError(f"Feature {name} not found in registry")
        computed[name] = func(df)
    feat_df = pd.DataFrame(computed, index=df.index)
    return pd.concat([df.reset_index(drop=True), feat_df.reset_index(drop=True)], axis=1)


def list_feature_families() -> Dict[str, List[str]]:
    families = defaultdict(list)
    families.update(
        {
            "trend": [
                "feat_log_return_1m",
                "feat_log_return_5m",
                "feat_log_return_15m",
                "feat_rolling_return_15m",
                "feat_rolling_return_1h",
            ],
            "vol": [
                "feat_realized_vol_15m",
                "feat_realized_vol_1h",
                "feat_vol_of_vol_1h",
            ],
            "microstructure": [
                "feat_spread_bps",
                "feat_rolling_volume_15m",
                "feat_turnover_proxy_1h",
            ],
            "cross_asset": [
                "feat_relative_strength_vs_basket",
                "feat_cross_corr_15m",
            ],
            "regime": [
                "feat_vol_bucket_id_raw",
                "feat_liquidity_metric_raw",
            ],
            "calendar": [
                "feat_hour_of_day_sin",
                "feat_hour_of_day_cos",
                "feat_day_of_week_sin",
                "feat_day_of_week_cos",
            ],
        }
    )
    return families


__all__ = ["FEATURE_REGISTRY", "apply_feature_registry", "list_feature_families"]
