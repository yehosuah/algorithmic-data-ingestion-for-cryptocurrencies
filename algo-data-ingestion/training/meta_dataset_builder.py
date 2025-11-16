from __future__ import annotations

from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd


def build_meta_dataset(
    base_model_preds: Dict[str, np.ndarray],
    y_true: np.ndarray,
    regimes: Optional[np.ndarray] = None,
) -> Tuple[pd.DataFrame, np.ndarray]:
    """
    Build a meta-learning dataset from OOS base predictions.
    """
    if not base_model_preds:
        raise ValueError("base_model_preds cannot be empty")
    base_df = pd.DataFrame({k: np.asarray(v).reshape(-1) for k, v in base_model_preds.items()})
    y_series = pd.Series(np.asarray(y_true).reshape(-1), name="y_true")
    if len(y_series) != len(base_df):
        min_len = min(len(y_series), len(base_df))
        base_df = base_df.iloc[:min_len]
        y_series = y_series.iloc[:min_len]
    feats = base_df.copy()
    if regimes is not None:
        reg_series = pd.Series(regimes).astype(str)
        if len(reg_series) != len(feats):
            reg_series = reg_series.iloc[: len(feats)]
        reg_ohe = pd.get_dummies(reg_series, prefix="regime")
        feats = pd.concat([feats, reg_ohe], axis=1)
    mask = np.isfinite(feats).all(axis=1) & np.isfinite(y_series)
    X_meta = feats.loc[mask].reset_index(drop=True)
    y_meta = y_series.loc[mask].to_numpy(dtype=int)
    return X_meta, y_meta
