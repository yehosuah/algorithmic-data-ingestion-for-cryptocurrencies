from __future__ import annotations
import numpy as np
import pandas as pd
from typing import Sequence

_EPS = 1e-9


def _rolling(series: pd.Series, window: int, func: str, *, min_periods: int | None = None) -> pd.Series:
    if min_periods is None:
        min_periods = 1
    roll = series.rolling(window, min_periods=min_periods)
    if func == "mean":
        return roll.mean()
    if func == "std":
        return roll.std()
    if func == "max":
        return roll.max()
    if func == "min":
        return roll.min()
    raise ValueError(f"Unsupported rolling func={func}")


def augment_market_features(df: pd.DataFrame, *, inplace: bool = False) -> pd.DataFrame:
    """Create additional tabular features for tree models.

    The transformation is intentionally lightweight so it can run inline during training
    without requiring a rebuild of the historical parquet datasets.
    """
    out = df if inplace else df.copy()

    def ensure_cols(cols: Sequence[str]) -> bool:
        return all(c in out.columns for c in cols)

    if "ret_1" in out.columns:
        ret1 = out["ret_1"].astype(float)
        out["ret_1_abs"] = ret1.abs()
        out["ret_1_sq"] = ret1 * ret1
        out["ret_mean_5"] = _rolling(ret1, 5, "mean")
        out["ret_mean_20"] = _rolling(ret1, 20, "mean")
        out["ret_mean_60"] = _rolling(ret1, 60, "mean")
        out["ret_mean_120"] = _rolling(ret1, 120, "mean")
        out["ret_std_20"] = _rolling(ret1, 20, "std")
        out["ret_std_60"] = _rolling(ret1, 60, "std")
        out["ret_std_120"] = _rolling(ret1, 120, "std")
        out["ret_z_20"] = (ret1 - out["ret_mean_20"]) / (out["ret_std_20"].abs() + _EPS)
        out["ret_z_60"] = (ret1 - out["ret_mean_60"]) / (out["ret_std_60"].abs() + _EPS)
        out["ret_z_120"] = (ret1 - out["ret_mean_120"]) / (out["ret_std_120"].abs() + _EPS)
        out["ret_mom_5_20"] = out["ret_mean_5"] - out["ret_mean_20"]
        out["ret_mom_20_60"] = out["ret_mean_20"] - out["ret_mean_60"]

    if "logret_1" in out.columns:
        logr = out["logret_1"].astype(float)
        out["logret_std_50"] = _rolling(logr, 50, "std")
        out["logret_std_120"] = _rolling(logr, 120, "std")

    if ensure_cols(["macd", "macd_signal_9"]):
        macd = out["macd"].astype(float)
        macd_sig = out["macd_signal_9"].astype(float)
        out["macd_hist"] = macd - macd_sig
        out["macd_hist_abs"] = out["macd_hist"].abs()
        out["macd_hist_slope"] = out["macd_hist"].diff().fillna(0.0)

    if ensure_cols(["ema_12", "ema_26"]):
        ema12 = out["ema_12"].astype(float)
        ema26 = out["ema_26"].astype(float)
        out["ema_ratio"] = ema12 / (ema26.abs() + _EPS) - 1.0

    if ensure_cols(["rvol_5", "rvol_20"]):
        rv5 = out["rvol_5"].astype(float)
        rv20 = out["rvol_20"].astype(float)
        out["rvol_ratio"] = rv5 / (rv20.abs() + _EPS)
        out["rvol_delta"] = rv5 - rv20
        out["rvol_z_50"] = (rv5 - _rolling(rv5, 50, "mean")) / (_rolling(rv5, 50, "std").abs() + _EPS)

    if "rsi_14" in out.columns:
        rsi = out["rsi_14"].astype(float)
        out["rsi_centered"] = rsi - 50.0

    if "hl_spread" in out.columns:
        spread = out["hl_spread"].astype(float)
        spread_ma = _rolling(spread, 20, "mean")
        spread_std = _rolling(spread, 20, "std")
        out["hl_spread_z"] = (spread - spread_ma) / (spread_std.abs() + _EPS)
        out["hl_spread_mean_60"] = _rolling(spread, 60, "mean")
        out["hl_spread_trend"] = spread_ma - _rolling(spread, 60, "mean")

    if "oi_obv" in out.columns:
        obv = out["oi_obv"].astype(float)
        out["obv_diff"] = obv.diff().fillna(0.0)
        out["obv_z_50"] = (obv - _rolling(obv, 50, "mean")) / (_rolling(obv, 50, "std").abs() + _EPS)
        out["obv_slope_20"] = out["obv_diff"].rolling(20, min_periods=1).mean()

    if "timestamp" in out.columns:
        ts = pd.to_datetime(out["timestamp"], utc=True)
        minutes = ts.dt.hour * 60 + ts.dt.minute
        out["tod_sin"] = np.sin(2 * np.pi * minutes / (24 * 60))
        out["tod_cos"] = np.cos(2 * np.pi * minutes / (24 * 60))
        dow = ts.dt.dayofweek.astype(float)
        out["dow_sin"] = np.sin(2 * np.pi * dow / 7.0)
        out["dow_cos"] = np.cos(2 * np.pi * dow / 7.0)

    if "symbol" in out.columns:
        sym_series = out["symbol"].astype(str)
        if "hl_spread" in out.columns:
            spread = out["hl_spread"].astype(float)
            out["sym_spread_median"] = spread.groupby(sym_series).transform("median")
            out["sym_spread_q90"] = spread.groupby(sym_series).transform(lambda s: s.quantile(0.9))
            median_by_symbol = spread.groupby(sym_series).median()
            rank_map = median_by_symbol.rank(method="dense")
            out["sym_liquidity_rank"] = sym_series.map(rank_map)
            denom = out["sym_spread_q90"].abs() + _EPS
            out["sym_spread_ratio"] = spread / denom
        if "rvol_20" in out.columns:
            rvol = out["rvol_20"].astype(float)
            out["sym_rvol_median"] = rvol.groupby(sym_series).transform("median")
            out["sym_rvol_q90"] = rvol.groupby(sym_series).transform(lambda s: s.quantile(0.9))
            rvol_median = rvol.groupby(sym_series).median()
            rvol_rank_map = rvol_median.rank(method="dense")
            out["sym_vol_rank"] = sym_series.map(rvol_rank_map)
            denom = out["sym_rvol_q90"].abs() + _EPS
            out["sym_rvol_ratio"] = rvol / denom
        unique_symbols = sym_series.dropna().unique()
        for sym in unique_symbols:
            prefix = sym.split("/", 1)[0].replace("-", "_").replace(" ", "_").lower()
            col_name = f"sym_{prefix}"
            if col_name not in out.columns:
                out[col_name] = (sym_series == sym).astype(float)

    # Replace remaining NaNs coming from rolling windows with 0 so downstream models stay robust
    new_cols = [c for c in out.columns if c not in df.columns]
    if new_cols:
        out[new_cols] = out[new_cols].fillna(0.0)
    return out
