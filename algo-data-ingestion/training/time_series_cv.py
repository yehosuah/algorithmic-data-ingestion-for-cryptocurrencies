from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, TypedDict

import numpy as np
import pandas as pd

from .metrics import equity_curve, summary_stats
from .thresholds import select_prob_threshold


class TimeSeriesSplitConfig(TypedDict, total=False):
    n_splits: int
    train_window: Optional[str]  # e.g. "180D" or None for expanding
    val_window: str  # e.g. "30D"
    test_window: Optional[str]  # optional, for final evaluation
    min_gap: Optional[str]  # optional, gap between train/val to avoid leakage
    expanding: bool  # True: expanding window, False: rolling window
    step: Optional[str]  # optional, override step between validation starts


@dataclass
class SplitResult:
    train_idx: np.ndarray
    val_idx: np.ndarray
    test_idx: Optional[np.ndarray] = None


def _parse_window(value: Optional[str]) -> Optional[pd.Timedelta]:
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return pd.Timedelta(float(value), unit="D")
    return pd.Timedelta(value)


def make_time_series_splits(
    df: pd.DataFrame,
    time_col: str,
    config: TimeSeriesSplitConfig,
) -> List[Dict[str, np.ndarray]]:
    """
    Given a sorted DataFrame and a config, return a list of split dicts:
      {
        'train_idx': np.ndarray,
        'val_idx': np.ndarray,
        # 'test_idx': optional, for final split
      }
    Uses time-based boundaries; no leakage.
    """
    if time_col not in df.columns:
        raise KeyError(f"time_col {time_col} missing from DataFrame")
    df_sorted = (
        df.sort_values(time_col)
        .reset_index(drop=False)
        .rename(columns={"index": "orig_index"})
    )
    ts = pd.to_datetime(df_sorted[time_col], utc=True)
    n_splits = int(config.get("n_splits", 3))
    val_delta = _parse_window(config.get("val_window"))
    if val_delta is None:
        raise ValueError("val_window is required for time-series splits")
    train_delta = _parse_window(config.get("train_window"))
    min_gap = _parse_window(config.get("min_gap")) or pd.Timedelta(0)
    step_delta = _parse_window(config.get("step")) or val_delta
    test_delta = _parse_window(config.get("test_window"))
    expanding = bool(config.get("expanding", train_delta is None))
    if not expanding and train_delta is None:
        raise ValueError("train_window must be provided when expanding=False")

    start_time = ts.min()
    end_time = ts.max()
    splits: List[Dict[str, np.ndarray]] = []

    base_train_delta = train_delta or val_delta
    first_val_start = start_time + base_train_delta + min_gap
    val_start = first_val_start
    split_idx = 0
    last_val_end: Optional[pd.Timestamp] = None
    while split_idx < n_splits and val_start < end_time:
        val_end = val_start + val_delta
        if val_end > end_time:
            val_end = end_time
        train_end = val_start - min_gap
        train_start = start_time if expanding or train_delta is None else train_end - train_delta
        if train_start < start_time:
            train_start = start_time

        train_mask = (ts >= train_start) & (ts < train_end)
        val_mask = (ts >= val_start) & (ts < val_end)

        if not val_mask.any() or not train_mask.any():
            val_start = val_start + step_delta
            split_idx += 1
            continue

        split = {
            "train_idx": df_sorted.loc[train_mask, "orig_index"].to_numpy(),
            "val_idx": df_sorted.loc[val_mask, "orig_index"].to_numpy(),
        }
        splits.append(split)
        last_val_end = val_end
        val_start = val_start + step_delta
        split_idx += 1

    if test_delta is not None and splits and last_val_end is not None:
        test_start = last_val_end + min_gap
        test_end = test_start + test_delta
        test_mask = (ts >= test_start) & (ts < test_end)
        if test_mask.any():
            splits[-1]["test_idx"] = df_sorted.loc[test_mask, "orig_index"].to_numpy()

    return splits


def evaluate_by_regime(
    df: pd.DataFrame,
    y_true: np.ndarray,
    y_proba: np.ndarray,
    regime_col: str,
    *,
    ret_col: str = "ret_next",
    cost_bps: float = 5.0,
    long_only: bool = False,
    min_hold_bars: int = 1,
) -> Dict[str, Dict[str, float]]:
    """
    Compute metrics (P&L net, Sharpe, hit-rate, etc.) per regime.
    """
    if len(y_true) != len(y_proba):
        raise ValueError("y_true and y_proba must have same length")
    if regime_col not in df.columns:
        raise KeyError(f"Regime column {regime_col} missing from DataFrame")
    ret_series = df[ret_col] if ret_col in df.columns else pd.Series(0.0, index=df.index)
    y_series = pd.Series(y_true, index=df.index)
    prob_series = pd.Series(y_proba, index=df.index)
    res: Dict[str, Dict[str, float]] = {}
    for regime, g in df.groupby(regime_col):
        prob = prob_series.loc[g.index]
        y_reg = y_series.loc[g.index].to_numpy()
        thr, _ = select_prob_threshold(ret_series.loc[g.index], prob, cost_bps=cost_bps, long_only=long_only, min_hold_bars=min_hold_bars)
        eq = equity_curve(
            ret_series.loc[g.index],
            prob,
            threshold=thr,
            cost_bps=cost_bps,
            long_only=long_only,
            min_hold_bars=min_hold_bars,
        )
        stats = summary_stats(eq)
        hit_rate = float(np.mean(((prob >= 0.5) & (y_reg == 1)) | ((prob < 0.5) & (y_reg == 0)))) if len(y_reg) else 0.0
        stats.update(
            {
                "pnl_net": float(eq["pnl"].sum()),
                "hit_rate": hit_rate,
                "threshold": thr,
                "count": int(len(prob)),
            }
        )
        res[str(regime)] = stats
    return res


def get_split_data(
    df: pd.DataFrame,
    split: Dict[str, np.ndarray],
    feature_cols: List[str],
    label_col: str,
    *,
    seq_params: Optional[Dict[str, Any]] = None,
) -> Any:
    """
    Given df and a split dict:
      - Extract X_train, y_train, X_val, y_val
    For sequence models, this delegates to build_sequences().
    """
    if seq_params:
        from .sequence_builder import build_sequences

        seq_len = int(seq_params.get("seq_len", 32))
        horizon = int(seq_params.get("horizon", 1))
        group_col = seq_params.get("group_col", "symbol")
        stride = int(seq_params.get("seq_stride", 1))
        X_all, y_all, idx = build_sequences(
            df,
            feature_cols,
            label_col,
            seq_len=seq_len,
            horizon=horizon,
            group_col=group_col,
            stride=stride,
            return_index=True,
        )
        train_mask = np.isin(idx, split["train_idx"])
        val_mask = np.isin(idx, split["val_idx"])
        X_train, y_train = X_all[train_mask], y_all[train_mask]
        X_val, y_val = X_all[val_mask], y_all[val_mask]
        return X_train, y_train, X_val, y_val, idx

    train_idx = split["train_idx"]
    val_idx = split["val_idx"]
    X_train = df.loc[train_idx, feature_cols]
    y_train = df.loc[train_idx, label_col]
    X_val = df.loc[val_idx, feature_cols]
    y_val = df.loc[val_idx, label_col]
    return X_train, y_train, X_val, y_val
