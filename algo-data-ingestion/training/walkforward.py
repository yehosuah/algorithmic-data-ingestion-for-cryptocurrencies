from __future__ import annotations
import pandas as pd
from typing import Iterator, Tuple


def time_folds(
    df: pd.DataFrame,
    n_folds: int = 6,
    embargo_minutes: int = 60,
    *,
    scheme: str = "even",
) -> Iterator[Tuple[pd.Index, pd.Index]]:
    """
    Yield (train_idx, val_idx) pairs for time-ordered walk-forward folds.
    Embargo removes a gap around fold boundaries to avoid leakage.
    """
    df = df.sort_values("timestamp").reset_index(drop=True)
    ts = pd.to_datetime(df["timestamp"], utc=True)
    n = len(df)

    if scheme == "calendar_month":
        # Drop tz then convert to monthly Periods; use .dt after tz_localize
        months = ts.dt.tz_localize(None).dt.to_period("M").astype(str)
        df = df.assign(_month=months)
        unique_months = list(dict.fromkeys(df["_month"].tolist()))  # preserve order
        # Use last n_folds months as validation folds if specified; otherwise all but the first
        start_idx = max(1, len(unique_months) - n_folds) if n_folds is not None else 1
        months_for_val = unique_months[start_idx:]
        for m in months_for_val:
            val_mask = df["_month"] == m
            if not val_mask.any():
                continue
            val_idx = df.index[val_mask]
            val_start = int(val_idx.min())
            val_end = int(val_idx.max()) + 1  # exclusive end
            if embargo_minutes > 0:
                emb_bars = embargo_minutes
                emb_start = max(0, val_start - emb_bars)
                emb_end = min(n, val_end + emb_bars)
                train_mask = (df.index < emb_start) | (df.index >= emb_end)
            else:
                train_mask = ~val_mask
            yield (df.index[train_mask], df.index[val_mask])
    else:
        # Default: even-sized folds over index
        fold_size = n // (n_folds + 1)
        for k in range(1, n_folds + 1):
            val_start = k * fold_size
            val_end = (k + 1) * fold_size if k < n_folds else n
            val_mask = (df.index >= val_start) & (df.index < val_end)

            if embargo_minutes > 0:
                emb_bars = embargo_minutes  # 1 bar ~ 1 minute
                emb_start = max(0, val_start - emb_bars)
                emb_end = min(n, val_end + emb_bars)
                train_mask = (df.index < emb_start) | (df.index >= emb_end)
            else:
                train_mask = ~val_mask

            yield (df.index[train_mask], df.index[val_mask])
