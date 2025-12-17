from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd

from labels.cost_model import estimate_transaction_cost_bps


def shift_future_prices(df: pd.DataFrame, horizon_minutes: int) -> pd.Series:
    if "close" not in df.columns:
        raise ValueError("close column required for label generation")
    close = pd.to_numeric(df["close"], errors="coerce")
    if "symbol" in df.columns:
        fut = close.groupby(df["symbol"]).shift(-horizon_minutes)
    else:
        fut = close.shift(-horizon_minutes)
    fut.name = "future_price"
    return fut


def shift_past_prices(df: pd.DataFrame, horizon_minutes: int) -> pd.Series:
    """
    Shift close prices backward in time (lookback) by `horizon_minutes` bars.

    This is inference-safe, unlike `shift_future_prices`, because it relies only on
    information that already occurred.
    """
    if "close" not in df.columns:
        raise ValueError("close column required for label/feature generation")
    close = pd.to_numeric(df["close"], errors="coerce")
    if "symbol" in df.columns:
        past = close.groupby(df["symbol"]).shift(horizon_minutes)
    else:
        past = close.shift(horizon_minutes)
    past.name = "past_price"
    return past


def _net_return(df: pd.DataFrame, horizon_minutes: int) -> pd.Series:
    future_price = shift_future_prices(df, horizon_minutes)
    close = pd.to_numeric(df["close"], errors="coerce")
    future_ret = (future_price - close) / close.replace(0.0, np.nan)
    costs = df.apply(estimate_transaction_cost_bps, axis=1) / 10000.0
    return future_ret - costs


def _past_net_return(df: pd.DataFrame, horizon_minutes: int) -> pd.Series:
    past_price = shift_past_prices(df, horizon_minutes)
    close = pd.to_numeric(df["close"], errors="coerce")
    past_ret = (close - past_price) / past_price.replace(0.0, np.nan)
    costs = df.apply(estimate_transaction_cost_bps, axis=1) / 10000.0
    return past_ret - costs


def generate_directional_label(df: pd.DataFrame, horizon_minutes: int) -> pd.Series:
    future_price = shift_future_prices(df, horizon_minutes)
    close = pd.to_numeric(df["close"], errors="coerce")
    future_ret = (future_price - close) / close.replace(0.0, np.nan)
    label = (future_ret > 0).astype("Int64")
    label.name = f"directional_{horizon_minutes}m"
    # drop rows without future info
    label.iloc[-horizon_minutes:] = pd.NA
    return label


def generate_directional_lookback_feature(df: pd.DataFrame, horizon_minutes: int) -> pd.Series:
    """
    Inference-safe proxy for `generate_directional_label` using lookback returns.

    Produces a 0/1 signal based on whether the past `horizon_minutes` return was positive.
    """
    past_price = shift_past_prices(df, horizon_minutes)
    close = pd.to_numeric(df["close"], errors="coerce")
    past_ret = (close - past_price) / past_price.replace(0.0, np.nan)
    feature = (past_ret > 0).astype("Int64")
    feature.name = f"directional_{horizon_minutes}m"
    feature.iloc[:horizon_minutes] = pd.NA
    return feature


def generate_cost_adjusted_label(df: pd.DataFrame, horizon_minutes: int) -> pd.Series:
    net_ret = _net_return(df, horizon_minutes)
    label = (net_ret > 0).astype("Int64")
    label.name = f"cost_adjusted_{horizon_minutes}m"
    label.iloc[-horizon_minutes:] = pd.NA
    return label


def generate_cost_adjusted_lookback_feature(df: pd.DataFrame, horizon_minutes: int) -> pd.Series:
    """
    Inference-safe proxy for `generate_cost_adjusted_label` using lookback net returns.
    """
    net_ret = _past_net_return(df, horizon_minutes)
    feature = (net_ret > 0).astype("Int64")
    feature.name = f"cost_adjusted_{horizon_minutes}m"
    feature.iloc[:horizon_minutes] = pd.NA
    return feature


def generate_meta_label(df: pd.DataFrame, horizon_minutes: int, base_signal_col: str, edge_threshold: float) -> pd.Series:
    if base_signal_col not in df.columns:
        raise ValueError(f"Base signal column {base_signal_col} missing")
    base_signal = pd.to_numeric(df[base_signal_col], errors="coerce")
    net_ret = _net_return(df, horizon_minutes)
    meta = ((base_signal > 0) & (net_ret > edge_threshold)).astype("Int64")
    meta.name = f"meta_{horizon_minutes}m_{base_signal_col}"
    meta.iloc[-horizon_minutes:] = pd.NA
    return meta


def generate_meta_lookback_feature(
    df: pd.DataFrame,
    horizon_minutes: int,
    base_signal_col: str,
    edge_threshold: float,
) -> pd.Series:
    """
    Inference-safe proxy for `generate_meta_label` using lookback net returns.
    """
    if base_signal_col not in df.columns:
        raise ValueError(f"Base signal column {base_signal_col} missing")
    base_signal = pd.to_numeric(df[base_signal_col], errors="coerce")
    net_ret = _past_net_return(df, horizon_minutes)
    meta = ((base_signal > 0) & (net_ret > edge_threshold)).astype("Int64")
    meta.name = f"meta_{horizon_minutes}m_{base_signal_col}"
    meta.iloc[:horizon_minutes] = pd.NA
    return meta


def generate_continuous_return_label(df: pd.DataFrame, horizon_minutes: int) -> pd.Series:
    cont = _net_return(df, horizon_minutes)
    cont.name = f"net_return_{horizon_minutes}m"
    cont.iloc[-horizon_minutes:] = pd.NA
    return cont


def generate_continuous_return_lookback_feature(df: pd.DataFrame, horizon_minutes: int) -> pd.Series:
    """
    Inference-safe proxy for `generate_continuous_return_label` using lookback net returns.
    """
    cont = _past_net_return(df, horizon_minutes)
    cont.name = f"net_return_{horizon_minutes}m"
    cont.iloc[:horizon_minutes] = pd.NA
    return cont


__all__ = [
    "shift_future_prices",
    "shift_past_prices",
    "generate_directional_label",
    "generate_directional_lookback_feature",
    "generate_cost_adjusted_label",
    "generate_cost_adjusted_lookback_feature",
    "generate_meta_label",
    "generate_meta_lookback_feature",
    "generate_continuous_return_label",
    "generate_continuous_return_lookback_feature",
    "_net_return",
    "_past_net_return",
]
