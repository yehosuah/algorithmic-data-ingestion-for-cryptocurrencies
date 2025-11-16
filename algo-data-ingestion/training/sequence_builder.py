from __future__ import annotations

from typing import List, Tuple, Optional

import numpy as np
import pandas as pd


def build_sequences(
    df: pd.DataFrame,
    feature_cols: List[str],
    label_col: str,
    seq_len: int,
    horizon: int,
    *,
    group_col: str = "symbol",
    return_index: bool = False,
) -> Tuple[np.ndarray, np.ndarray] | Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Create leak-free sequences (X_seq, y) grouped by symbol.

    X_seq shape: (num_samples, seq_len, n_features)
    y shape: (num_samples,)
    """
    if seq_len < 1:
        raise ValueError("seq_len must be >= 1")
    if horizon < 0:
        raise ValueError("horizon must be >= 0")
    missing = [c for c in feature_cols if c not in df.columns]
    if missing:
        raise KeyError(f"Missing feature columns for sequence build: {missing}")
    if label_col not in df.columns:
        raise KeyError(f"Label column {label_col} not found in dataset")

    groups = [("", df)] if group_col not in df.columns else df.groupby(group_col)
    X_buf, y_buf, idx_buf = [], [], []
    for _, g in groups:
        g = g.sort_values("timestamp").reset_index(drop=False).rename(columns={"index": "orig_index"})
        feats = (
            g[feature_cols]
            .astype(float)
            .replace([np.inf, -np.inf], np.nan)
            .fillna(method="ffill")
            .fillna(method="bfill")
            .fillna(0.0)
            .to_numpy()
        )
        labels = g[label_col].to_numpy()
        n = len(g)
        limit = n - seq_len - horizon + 1
        if limit <= 0:
            continue
        for start in range(limit):
            end = start + seq_len
            target = end - 1 + horizon
            X_buf.append(feats[start:end])
            y_buf.append(labels[target])
            idx_buf.append(int(g.loc[target, "orig_index"]))

    X_seq = np.asarray(X_buf, dtype=float)
    y_seq = np.asarray(y_buf)
    idx_arr = np.asarray(idx_buf, dtype=int)
    if return_index:
        return X_seq, y_seq, idx_arr
    return X_seq, y_seq
